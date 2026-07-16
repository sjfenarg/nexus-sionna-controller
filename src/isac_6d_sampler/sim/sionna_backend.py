from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Callable
from xml.etree import ElementTree

import numpy as np

from isac_6d_sampler.core.antenna_patterns import register_sionna_antenna_patterns
from isac_6d_sampler.core.model import ChannelMode, DYNAMIC_SCENE_OBJECT_NAMES, SPEED_OF_LIGHT_M_PER_S, SimulationRequest
from isac_6d_sampler.core.validation import validate_request
from isac_6d_sampler.sim.channel import render_channel_samples
from isac_6d_sampler.sim.power import dbm_to_watt, field_amplitude_from_dbm

from .materials import assign_calibrated_materials
from .planner import LinkPlan, TimeframePlan, build_simulation_plan, orientation_for, position_for
from .results import LinkResult, SimulationResult, TimeframeResult

ProgressCallback = Callable[[int, int, str], None]
JOURNAL_HORN_PATTERN = "isac_horn_77_81"


@dataclass(frozen=True, slots=True)
class LinkBatch:
    links: tuple[LinkPlan, ...]
    rx_devices: tuple
    tx_devices: tuple
    los: bool


@dataclass(frozen=True, slots=True)
class TimeframeLinkRef:
    timeframe_index: int
    timeframe: TimeframePlan
    link: LinkPlan


@dataclass(frozen=True, slots=True)
class TimeframeLinkBatch:
    refs: tuple[TimeframeLinkRef, ...]
    rx_devices: tuple[tuple[int, TimeframePlan, object], ...]
    tx_devices: tuple[tuple[int, TimeframePlan, object], ...]
    los: bool


class SionnaSimulator:
    """Sionna RT backend with the package boundary kept small and lazy-loaded."""

    def simulate(self, request: SimulationRequest, progress: ProgressCallback | None = None) -> SimulationResult:
        scene_design = request.scene
        scene_design.ensure_defaults()
        validate_request(request)
        mitsuba_variant = _configure_mitsuba_variant(request.sionna.use_gpu)
        try:
            from sionna.rt import load_scene
        except ModuleNotFoundError as exc:
            raise RuntimeError("Sionna is not installed; run with --dry-run for schema validation") from exc

        subbands = [band.vector() for band in request.bands]
        frequency_vector = np.concatenate(subbands)
        with _scene_xml_with_configured_dynamic_objects(scene_design) as scene_path:
            scene = load_scene(str(scene_path), merge_shapes=request.sionna.merge_shapes)
        scene.frequency = float(np.mean(frequency_vector))
        scene.bandwidth = float(frequency_vector[-1] - frequency_vector[0])
        assign_calibrated_materials(scene, scene_design.name)

        plan = build_simulation_plan(scene_design)
        result_frames: list[TimeframeResult] = []
        batch_size = max(1, int(request.sionna.batch_timeframes))
        tf_idx = 0
        while tf_idx < len(plan.timeframes):
            chunk = _next_batchable_timeframe_chunk(plan.timeframes, tf_idx, batch_size)
            if progress:
                end = tf_idx + len(chunk) - 1
                label = f"{tf_idx}" if tf_idx == end else f"{tf_idx}-{end}"
                progress(
                    _last_completed_timeframe_index(tf_idx, chunk),
                    len(plan.timeframes),
                    f"Solving Sionna timeframe(s) {label}",
                )
            self._apply_object_positions(scene, scene_design, chunk[0])
            result_frames.extend(
                self._solve_timeframe_chunk(
                    scene=scene,
                    timeframes=chunk,
                    start_index=tf_idx,
                    f_vector=frequency_vector,
                    request=request,
                    progress=progress,
                    progress_total=len(plan.timeframes),
                )
            )
            tf_idx += len(chunk)

        return SimulationResult(
            scenario_name=scene_design.name,
            sample_id=request.sample_id,
            frequency_vector_hz=frequency_vector[None, :],
            subband_vectors_hz=[v[None, :] for v in subbands],
            timeframes=result_frames,
            metadata={
                "backend": "sionna",
                "channel_mode": request.channel_mode.value,
                "use_gpu_requested": request.sionna.use_gpu,
                "mitsuba_variant": mitsuba_variant,
                "tx_power_dbm": float(request.sionna.tx_power_dbm),
                "tx_power_w": dbm_to_watt(request.sionna.tx_power_dbm),
            },
        )

    def _solve_timeframe_chunk(
        self,
        scene,
        timeframes: tuple[TimeframePlan, ...],
        start_index: int,
        f_vector: np.ndarray,
        request: SimulationRequest,
        progress: ProgressCallback | None = None,
        progress_total: int | None = None,
    ) -> list[TimeframeResult]:
        frame_links = self._solve_all_links_for_timeframes(
            scene=scene,
            timeframes=timeframes,
            start_index=start_index,
            f_vector=f_vector,
            request=request,
            progress=progress,
            progress_total=progress_total or (start_index + len(timeframes)),
        )
        return [
            TimeframeResult(
                name=timeframe.name,
                links=frame_links[offset],
                metadata={
                    **timeframe.metadata,
                    "device_positions": timeframe.device_positions,
                    "object_positions": timeframe.object_positions,
                    "device_orientations": timeframe.device_orientations,
                    "object_orientations": timeframe.object_orientations,
                },
            )
            for offset, timeframe in enumerate(timeframes)
        ]

    def _solve_all_links(
        self,
        scene,
        timeframe: TimeframePlan,
        f_vector: np.ndarray,
        request: SimulationRequest,
    ) -> list[LinkResult]:
        return self._solve_all_links_for_timeframes(
            scene=scene,
            timeframes=(timeframe,),
            start_index=0,
            f_vector=f_vector,
            request=request,
        )[0]

    def _solve_all_links_for_timeframes(
        self,
        scene,
        timeframes: tuple[TimeframePlan, ...],
        start_index: int,
        f_vector: np.ndarray,
        request: SimulationRequest,
        progress: ProgressCallback | None = None,
        progress_total: int | None = None,
    ) -> list[list[LinkResult]]:
        from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter

        register_sionna_antenna_patterns()
        frame_links: list[list[LinkResult]] = [[] for _ in timeframes]
        solver = PathSolver()

        batches = _build_timeframe_link_batches(timeframes, start_index, request.sionna.los)
        for batch_index, batch in enumerate(batches):
            cleanup_names: list[str] = []
            representative = batch.refs[0].link
            self._configure_arrays(
                scene,
                PlanarArray,
                representative,
                frequency_hz=float(np.mean(f_vector)),
            )
            rx_local = _local_device_index(batch.rx_devices, batch.refs, side="rx")
            tx_local = _local_device_index(batch.tx_devices, batch.refs, side="tx")

            try:
                if progress is not None:
                    progress(
                        start_index,
                        progress_total or (start_index + len(timeframes)),
                        _format_sionna_batch_progress(batch, batch_index, len(batches)),
                    )
                for idx, (tf_abs_idx, timeframe, tx_device) in enumerate(batch.tx_devices):
                    tx_name = f"TX_{idx}_tf{tf_abs_idx}_{tx_device.id}"
                    tx_position = position_for(tx_device, timeframe)
                    if _tx_is_only_monostatic_in_batch(batch.refs, tf_abs_idx, tx_device.id):
                        tx_position = _offset_monostatic_tx_position(tx_position, float(np.mean(f_vector)))
                    transmitter = Transmitter(
                        name=tx_name,
                        position=_vector3_for_mitsuba(tx_position),
                        orientation=_vector3_for_mitsuba(orientation_for(tx_device, timeframe)),
                        power_dbm=float(request.sionna.tx_power_dbm),
                    )
                    scene.add(transmitter)
                    cleanup_names.append(tx_name)
                for idx, (tf_abs_idx, timeframe, rx_device) in enumerate(batch.rx_devices):
                    rx_name = f"RX_{idx}_tf{tf_abs_idx}_{rx_device.id}"
                    receiver = Receiver(
                        name=rx_name,
                        position=_vector3_for_mitsuba(position_for(rx_device, timeframe)),
                        orientation=_vector3_for_mitsuba(orientation_for(rx_device, timeframe)),
                    )
                    scene.add(receiver)
                    cleanup_names.append(rx_name)

                paths = solver(
                    scene=scene,
                    samples_per_src=request.sionna.samples_per_src,
                    max_num_paths_per_src=request.sionna.max_num_paths_per_src or 1_000_000,
                    max_depth=request.sionna.max_depth,
                    los=batch.los,
                    specular_reflection=request.sionna.specular_reflection,
                    diffuse_reflection=request.sionna.diffuse_reflection,
                    refraction=request.sionna.refraction,
                    synthetic_array=request.sionna.synthetic_array,
                    seed=request.sionna.seed if request.sionna.seed is not None else 42,
                )
                for ref in batch.refs:
                    link = ref.link
                    h, path_delays, path_coefficients = _paths_to_link_data(
                        paths,
                        f_vector,
                        request,
                        link,
                        rx_local_index=rx_local[(ref.timeframe_index, link.rx.id)],
                        tx_local_index=tx_local[(ref.timeframe_index, link.tx.id)],
                    )
                    frame_links[ref.timeframe_index - start_index].append(
                        LinkResult(
                            rx_index=link.rx_index,
                            tx_index=link.tx_index,
                            rx_id=link.rx.id,
                            tx_id=link.tx.id,
                            h=h,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            metadata={
                                "path_delays_s": path_delays,
                                "path_coefficients": path_coefficients,
                            },
                        )
                    )
            finally:
                for name in cleanup_names:
                    if name in scene.transmitters or name in scene.receivers:
                        scene.remove(name)
        return frame_links

    def _configure_arrays(self, scene, planar_array_cls, link: LinkPlan, frequency_hz: float) -> None:
        scene.tx_array = planar_array_cls(
            num_rows=link.tx.panel.rows,
            num_cols=link.tx.panel.cols,
            vertical_spacing=_spacing_m_to_wavelengths(link.tx.panel.vertical_spacing_m, frequency_hz),
            horizontal_spacing=_spacing_m_to_wavelengths(link.tx.panel.horizontal_spacing_m, frequency_hz),
            pattern=_sionna_array_pattern(link.tx.panel.pattern),
            polarization=link.tx.panel.polarization,
        )
        scene.rx_array = planar_array_cls(
            num_rows=link.rx.panel.rows,
            num_cols=link.rx.panel.cols,
            vertical_spacing=_spacing_m_to_wavelengths(link.rx.panel.vertical_spacing_m, frequency_hz),
            horizontal_spacing=_spacing_m_to_wavelengths(link.rx.panel.horizontal_spacing_m, frequency_hz),
            pattern=_sionna_array_pattern(link.rx.panel.pattern),
            polarization=link.rx.panel.polarization,
        )

    def _apply_object_positions(self, scene, design, timeframe: TimeframePlan) -> None:
        for obj in design.objects:
            if obj.object_name in scene.objects:
                scene.objects[obj.object_name].position = _vector3_for_mitsuba(
                    timeframe.object_positions.get(obj.id, obj.position)
                )
                scene.objects[obj.object_name].orientation = _vector3_for_mitsuba(
                    timeframe.object_orientations.get(obj.id, obj.orientation_rad)
                )


@contextmanager
def _scene_xml_with_configured_dynamic_objects(design):
    """Load-time filter for optional dynamic scene meshes.

    Sionna scene geometry cannot be removed after ``load_scene``. Optional objects
    such as the car must therefore be excluded from the XML before loading.
    """
    source = Path(design.scenario_path)
    configured = {obj.object_name for obj in design.objects}
    excluded = set(DYNAMIC_SCENE_OBJECT_NAMES) - configured
    if not excluded:
        yield source
        return
    filtered = _filtered_scene_xml(source, excluded)
    if filtered is None:
        yield source
        return
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".xml",
        prefix=f"{source.stem}_filtered_",
        dir=source.parent,
        delete=False,
    ) as tmp:
        tmp.write(filtered)
        tmp_path = Path(tmp.name)
    try:
        yield tmp_path
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _filtered_scene_xml(source: Path, excluded_object_names: set[str]) -> bytes | None:
    try:
        tree = ElementTree.parse(source)
    except ElementTree.ParseError:
        return None
    root = tree.getroot()
    _remove_matching_shapes(root, excluded_object_names)
    ElementTree.indent(tree, space="\t")
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=False)


def _remove_matching_shapes(root, excluded_object_names: set[str]) -> None:
    for parent in root.iter():
        for child in list(parent):
            if child.tag != "shape":
                continue
            name = child.attrib.get("name") or child.attrib.get("id")
            if name in excluded_object_names:
                parent.remove(child)


def _paths_to_link_data(
    paths,
    frequency_vector: np.ndarray,
    request: SimulationRequest,
    link: LinkPlan,
    rx_local_index: int = 0,
    tx_local_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, tau = paths.cir(normalize_delays=False, out_type="numpy")
    delays = _slice_link_array(np.asarray(tau), rx_local_index, tx_local_index)
    coeffs = _slice_link_array(np.asarray(a), rx_local_index, tx_local_index)
    if coeffs.ndim > 0 and coeffs.shape[-1] == 1 and coeffs.ndim == delays.ndim + 1:
        coeffs = np.squeeze(coeffs, axis=-1)
    coeffs = coeffs * field_amplitude_from_dbm(request.sionna.tx_power_dbm)
    if request.channel_mode == ChannelMode.FREQUENCY_DOMAIN:
        h = _paths_cfr_chunked(paths, np.asarray(tau), frequency_vector)
        h = _slice_link_array(np.asarray(h), rx_local_index, tx_local_index)
        h = h * field_amplitude_from_dbm(request.sionna.tx_power_dbm)
        return (
            _reshape_h_for_reference(h, link).astype(np.complex64),
            _reshape_path_delays_for_reference(delays, link).astype(np.float64),
            _reshape_path_coefficients_for_reference(coeffs, link).astype(np.complex64),
        )
    h = render_channel_samples(delays, coeffs, frequency_vector, request.channel_mode)
    return (
        _reshape_h_for_reference(h, link).astype(np.complex64),
        _reshape_path_delays_for_reference(delays, link).astype(np.float64),
        _reshape_path_coefficients_for_reference(coeffs, link).astype(np.complex64),
    )


def _paths_cfr_chunked(paths, tau: np.ndarray, frequency_vector: np.ndarray, max_entries: int = 250_000_000):
    frequencies = np.asarray(frequency_vector, dtype=np.float64).reshape(-1)
    if frequencies.size == 0:
        return np.zeros((*np.asarray(tau).shape[:-1], 0), dtype=np.complex64)
    path_entries = max(1, int(np.prod(np.asarray(tau).shape)))
    chunk_size = max(1, min(frequencies.size, int(max_entries) // path_entries))
    chunks = []
    for start in range(0, frequencies.size, chunk_size):
        stop = min(start + chunk_size, frequencies.size)
        chunks.append(
            np.asarray(
                paths.cfr(
                    frequencies[start:stop],
                    normalize_delays=False,
                    normalize=False,
                    out_type="numpy",
                ),
                dtype=np.complex64,
            )
        )
    return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=-1)


def _configure_mitsuba_variant(use_gpu: bool) -> str:
    import mitsuba as mi

    if use_gpu:
        try:
            mi.set_variant("cuda_ad_mono_polarized", "llvm_ad_mono_polarized")
        except ImportError:
            mi.set_variant("llvm_ad_mono_polarized")
    else:
        mi.set_variant("llvm_ad_mono_polarized")
    return str(mi.variant())


def _vector3_for_mitsuba(values) -> list[float]:
    return [float(values[0]), float(values[1]), float(values[2])]


def _sionna_array_pattern(pattern: str) -> str:
    return pattern


def _tx_is_only_monostatic_in_batch(
    refs: tuple[TimeframeLinkRef, ...],
    timeframe_index: int,
    tx_id: str,
) -> bool:
    matching = [
        ref
        for ref in refs
        if ref.timeframe_index == timeframe_index and ref.link.tx.id == tx_id
    ]
    return bool(matching) and all(ref.link.is_monostatic for ref in matching)


def _offset_monostatic_tx_position(values, frequency_hz: float) -> tuple[float, float, float]:
    wavelength_m = SPEED_OF_LIGHT_M_PER_S / float(frequency_hz)
    return (float(values[0]) + wavelength_m / 4.0, float(values[1]), float(values[2]))


def _spacing_m_to_wavelengths(spacing_m: float, frequency_hz: float) -> float:
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    return float(spacing_m) * float(frequency_hz) / SPEED_OF_LIGHT_M_PER_S


def _reshape_h_for_reference(h: np.ndarray, link: LinkPlan) -> np.ndarray:
    samples = h.shape[-1]
    rx_ant = link.rx.panel.element_count
    tx_ant = link.tx.panel.element_count

    if h.ndim == 5 and h.shape[0] == 1 and h.shape[2] == 1:
        h = h[0, :, 0, :, :]
    elif h.ndim == 4 and h.shape[0] == 1:
        h = h[0, :, :, :]
    elif h.ndim == 2:
        h = h.reshape(rx_ant, tx_ant, samples)

    if h.shape[:2] != (rx_ant, tx_ant):
        h = h.reshape(rx_ant, tx_ant, samples)

    return h.reshape(
        link.rx.panel.rows,
        link.rx.panel.cols,
        link.tx.panel.rows,
        link.tx.panel.cols,
        1,
        samples,
    )


def _reshape_path_delays_for_reference(delays: np.ndarray, link: LinkPlan) -> np.ndarray:
    path_count = delays.shape[-1]
    rx_ant = link.rx.panel.element_count
    tx_ant = link.tx.panel.element_count

    if delays.ndim == 4 and delays.shape[0] == 1 and delays.shape[2] == 1:
        delays = delays[0, :, 0, :]
    elif delays.ndim == 3 and delays.shape[0] == 1:
        delays = delays[0, :, :]
    elif delays.ndim == 1:
        delays = delays.reshape(rx_ant, tx_ant, path_count)

    if delays.shape[:2] != (rx_ant, tx_ant):
        delays = delays.reshape(rx_ant, tx_ant, path_count)

    return delays.reshape(
        link.rx.panel.rows,
        link.rx.panel.cols,
        link.tx.panel.rows,
        link.tx.panel.cols,
        1,
        path_count,
    )


def _reshape_path_coefficients_for_reference(coeffs: np.ndarray, link: LinkPlan) -> np.ndarray:
    path_count = coeffs.shape[-1]
    rx_ant = link.rx.panel.element_count
    tx_ant = link.tx.panel.element_count

    if coeffs.ndim == 4 and coeffs.shape[0] == 1 and coeffs.shape[2] == 1:
        coeffs = coeffs[0, :, 0, :]
    elif coeffs.ndim == 3 and coeffs.shape[0] == 1:
        coeffs = coeffs[0, :, :]
    elif coeffs.ndim == 1:
        coeffs = coeffs.reshape(rx_ant, tx_ant, path_count)

    if coeffs.shape[:2] != (rx_ant, tx_ant):
        coeffs = coeffs.reshape(rx_ant, tx_ant, path_count)

    return coeffs.reshape(
        link.rx.panel.rows,
        link.rx.panel.cols,
        link.tx.panel.rows,
        link.tx.panel.cols,
        1,
        path_count,
    )


def _build_link_batches(links: tuple[LinkPlan, ...], request_los: bool) -> list[LinkBatch]:
    grouped: dict[tuple, list[LinkPlan]] = {}
    for link in links:
        key = (
            _panel_key(link.rx),
            _panel_key(link.tx),
            bool(request_los and not link.is_monostatic),
        )
        grouped.setdefault(key, []).append(link)

    batches: list[LinkBatch] = []
    for (_, _, los), batch_links in grouped.items():
        rx_devices = _unique_devices(link.rx for link in batch_links)
        tx_devices = _unique_devices(link.tx for link in batch_links)
        batches.append(
            LinkBatch(
                links=tuple(batch_links),
                rx_devices=rx_devices,
                tx_devices=tx_devices,
                los=los,
            )
        )
    return batches


def _format_sionna_batch_progress(batch: TimeframeLinkBatch, batch_index: int, batch_count: int) -> str:
    links = ", ".join(
        f"{ref.link.rx.id}<-{ref.link.tx.id}"
        f" ({ref.link.rx.panel.element_count}x{ref.link.tx.panel.element_count})"
        for ref in batch.refs[:4]
    )
    if len(batch.refs) > 4:
        links += f", +{len(batch.refs) - 4} more"
    propagation = "LOS+reflections" if batch.los else "reflections only"
    return f"Sionna link batch {batch_index + 1}/{batch_count}: {links}; {propagation}"


def _next_batchable_timeframe_chunk(
    timeframes: tuple[TimeframePlan, ...],
    start: int,
    requested_size: int,
) -> tuple[TimeframePlan, ...]:
    requested_size = max(1, requested_size)
    first = timeframes[start]
    chunk = [first]
    for timeframe in timeframes[start + 1 : start + requested_size]:
        if not _same_object_state(first, timeframe):
            break
        chunk.append(timeframe)
    return tuple(chunk)


def _last_completed_timeframe_index(start: int, chunk: tuple[TimeframePlan, ...]) -> int:
    return start + len(chunk) - 1


def _same_object_state(left: TimeframePlan, right: TimeframePlan) -> bool:
    return (
        _state_dict_equal(left.object_positions, right.object_positions)
        and _state_dict_equal(left.object_orientations, right.object_orientations)
    )


def _state_dict_equal(left: dict, right: dict) -> bool:
    if left.keys() != right.keys():
        return False
    return all(np.allclose(left[key], right[key]) for key in left)


def _build_timeframe_link_batches(
    timeframes: tuple[TimeframePlan, ...],
    start_index: int,
    request_los: bool,
) -> list[TimeframeLinkBatch]:
    grouped: dict[tuple, list[TimeframeLinkRef]] = {}
    for offset, timeframe in enumerate(timeframes):
        timeframe_index = start_index + offset
        for link in timeframe.links:
            key = (
                _panel_key(link.rx),
                _panel_key(link.tx),
                bool(request_los and not link.is_monostatic),
            )
            grouped.setdefault(key, []).append(
                TimeframeLinkRef(timeframe_index=timeframe_index, timeframe=timeframe, link=link)
            )

    batches: list[TimeframeLinkBatch] = []
    for (_, _, los), refs in grouped.items():
        for split_refs in _split_horn_refs(tuple(refs)):
            for split_refs in _split_pose_colliding_refs(tuple(split_refs)):
                rx_devices = _unique_timeframe_devices(
                    (ref.timeframe_index, ref.timeframe, ref.link.rx) for ref in split_refs
                )
                tx_devices = _unique_timeframe_devices(
                    (ref.timeframe_index, ref.timeframe, ref.link.tx) for ref in split_refs
                )
                batches.append(
                    TimeframeLinkBatch(
                        refs=tuple(split_refs),
                        rx_devices=rx_devices,
                        tx_devices=tx_devices,
                        los=los,
                    )
                )
    return batches


def _split_horn_refs(refs: tuple[TimeframeLinkRef, ...]) -> list[list[TimeframeLinkRef]]:
    split: list[list[TimeframeLinkRef]] = []
    regular: list[TimeframeLinkRef] = []
    for ref in refs:
        if _link_uses_journal_horn(ref.link):
            split.append([ref])
        else:
            regular.append(ref)
    if regular:
        split.append(regular)
    return split


def _link_uses_journal_horn(link: LinkPlan) -> bool:
    return link.rx.panel.pattern == JOURNAL_HORN_PATTERN or link.tx.panel.pattern == JOURNAL_HORN_PATTERN


def _split_pose_colliding_refs(
    refs: tuple[TimeframeLinkRef, ...],
) -> list[list[TimeframeLinkRef]]:
    """Avoid batching same-pose devices on the same side of a Sionna solve.

    Sionna/Dr.Jit can return missing paths or crash when separate logical devices occupy the
    same pose as multiple transmitters or multiple receivers in one PathSolver call. Keep batching
    where possible, but split those role-wise pose collisions into separate solves.
    """
    groups: list[list[TimeframeLinkRef]] = []
    group_keys: list[tuple[dict[tuple, str], dict[tuple, str]]] = []
    for ref in refs:
        rx_key = _role_pose_key(ref.timeframe, ref.link.rx)
        tx_key = _role_pose_key(ref.timeframe, ref.link.tx)
        for idx, (rx_keys, tx_keys) in enumerate(group_keys):
            if (
                rx_keys.get(rx_key, ref.link.rx.id) == ref.link.rx.id
                and tx_keys.get(tx_key, ref.link.tx.id) == ref.link.tx.id
            ):
                groups[idx].append(ref)
                rx_keys[rx_key] = ref.link.rx.id
                tx_keys[tx_key] = ref.link.tx.id
                break
        else:
            groups.append([ref])
            group_keys.append(({rx_key: ref.link.rx.id}, {tx_key: ref.link.tx.id}))
    return groups


def _role_pose_key(timeframe: TimeframePlan, device) -> tuple:
    return (
        _rounded_vector(position_for(device, timeframe)),
        _rounded_vector(orientation_for(device, timeframe)),
    )


def _panel_key(device) -> tuple:
    panel = device.panel
    return (
        panel.rows,
        panel.cols,
        panel.pattern,
        panel.polarization,
        panel.vertical_spacing_m,
        panel.horizontal_spacing_m,
        _rounded_vector(device.orientation_rad),
    )


def _unique_devices(devices) -> tuple:
    by_id = {}
    for device in devices:
        by_id.setdefault(device.id, device)
    return tuple(by_id.values())


def _unique_timeframe_devices(devices) -> tuple[tuple[int, TimeframePlan, object], ...]:
    by_key = {}
    for timeframe_index, timeframe, device in devices:
        by_key.setdefault(
            _timeframe_device_state_key(timeframe, device),
            (timeframe_index, timeframe, device),
        )
    return tuple(by_key.values())


def _local_device_index(
    devices: tuple[tuple[int, TimeframePlan, object], ...],
    refs: tuple[TimeframeLinkRef, ...],
    side: str,
) -> dict[tuple[int, str], int]:
    state_to_index = {
        _timeframe_device_state_key(timeframe, device): idx
        for idx, (_, timeframe, device) in enumerate(devices)
    }
    local: dict[tuple[int, str], int] = {}
    for ref in refs:
        device = ref.link.rx if side == "rx" else ref.link.tx
        local[(ref.timeframe_index, device.id)] = state_to_index[
            _timeframe_device_state_key(ref.timeframe, device)
        ]
    return local


def _timeframe_device_state_key(timeframe: TimeframePlan, device) -> tuple:
    return (
        device.id,
        _rounded_vector(position_for(device, timeframe)),
        _rounded_vector(orientation_for(device, timeframe)),
    )


def _rounded_vector(values) -> tuple[float, float, float]:
    return tuple(round(float(value), 9) for value in values)


def _slice_link_array(values: np.ndarray, rx_index: int, tx_index: int) -> np.ndarray:
    if values.ndim >= 5:
        return values[rx_index, :, tx_index, ...]
    if values.ndim >= 3:
        return values[rx_index, tx_index, ...]
    return values

from __future__ import annotations

from collections.abc import Iterable
import math

from .antenna_patterns import antenna_pattern_spec
from .estimates import estimate_request_size
from .model import (
    AntennaPanel,
    BaseStation,
    ChannelMode,
    SimulationRequest,
    TrajectorySpec,
    UserEquipment,
)
from .scenarios import load_scenario_asset
from .trajectories import sample_trajectory

_POLARIZATIONS = {"V", "H", "VH", "cross"}
_TRAJECTORY_KINDS = {"static", "linear", "polyline", "curve"}
_TRAJECTORY_EASINGS = {"linear", "smoothstep"}
_DELAY_BIN_CHANNEL_MODES = {
    ChannelMode.COHERENT_PER_BIN,
    ChannelMode.PDP_BINNED,
    ChannelMode.PDP_IFFT_GRIDDED,
    ChannelMode.PDP_IFFT_EXACT,
}


def validate_request(request: SimulationRequest) -> None:
    """Raise ``ValueError`` with actionable messages for invalid simulation requests."""
    errors: list[str] = []
    scene = request.scene

    if not scene.base_stations:
        errors.append("At least one BS is required")
    if not scene.user_equipments and not scene.radiomap.enabled:
        errors.append("At least one UE is required unless radiomap mode is enabled")

    device_ids = _active_device_ids(request)
    _validate_unique_ids("device", device_ids, errors)
    _validate_unique_ids("object", (obj.id for obj in scene.objects), errors)

    for idx, bs in enumerate(scene.base_stations):
        _validate_device(f"base_stations[{idx}]", bs, errors)
    for idx, ue in enumerate(scene.user_equipments):
        _validate_device(f"user_equipments[{idx}]", ue, errors)
        _validate_trajectory(f"user_equipments[{idx}].trajectory", ue.trajectory, errors)
    for idx, obj in enumerate(scene.objects):
        if not obj.id:
            errors.append(f"objects[{idx}].id must not be empty")
        if not obj.object_name:
            errors.append(f"objects[{idx}].object_name must not be empty")
        _validate_trajectory(f"objects[{idx}].trajectory", obj.trajectory, errors)
    _validate_object_targets(request, errors)

    if scene.radiomap.enabled:
        _validate_device("radiomap.ue_template", scene.radiomap.ue_template, errors)
        _validate_trajectory(
            "radiomap.ue_template.trajectory",
            scene.radiomap.ue_template.trajectory,
            errors,
        )

    band_vectors = []
    if not request.bands:
        errors.append("At least one frequency band is required")
    for idx, band in enumerate(request.bands):
        try:
            band_vectors.append(band.vector())
        except ValueError as exc:
            errors.append(f"bands[{idx}]: {exc}")
    _validate_channel_frequency_grid(request, band_vectors, errors)

    if request.sionna.samples_per_src < 1:
        errors.append("sionna.samples_per_src must be at least 1")
    if not math.isfinite(request.sionna.tx_power_dbm):
        errors.append("sionna.tx_power_dbm must be finite")
    if request.sionna.max_depth < 0:
        errors.append("sionna.max_depth must be non-negative")
    if (
        request.sionna.max_num_paths_per_src is not None
        and request.sionna.max_num_paths_per_src < 1
    ):
        errors.append("sionna.max_num_paths_per_src must be null or at least 1")
    if request.sionna.batch_timeframes < 1:
        errors.append("sionna.batch_timeframes must be at least 1")
    if request.sionna.max_timeframes < 1:
        errors.append("sionna.max_timeframes must be at least 1")
    try:
        estimated_timeframes = estimate_request_size(request).timeframe_count
    except ValueError as exc:
        errors.append(f"radiomap: {exc}")
        estimated_timeframes = None
    if (
        estimated_timeframes is not None
        and request.sionna.max_timeframes >= 1
        and estimated_timeframes > request.sionna.max_timeframes
    ):
        errors.append(
            f"Estimated timeframe count {estimated_timeframes} exceeds "
            f"sionna.max_timeframes {request.sionna.max_timeframes}"
        )

    if errors:
        raise ValueError("Invalid simulation request: " + "; ".join(errors))


def _validate_device(prefix: str, device: BaseStation | UserEquipment, errors: list[str]) -> None:
    if not device.id:
        errors.append(f"{prefix}.id must not be empty")
    _validate_panel(f"{prefix}.panel", device.panel, errors)


def _validate_panel(prefix: str, panel: AntennaPanel, errors: list[str]) -> None:
    if panel.rows < 1:
        errors.append(f"{prefix}.rows must be at least 1")
    if panel.cols < 1:
        errors.append(f"{prefix}.cols must be at least 1")
    if panel.vertical_spacing_m < 0.0:
        errors.append(f"{prefix}.vertical_spacing_m must be non-negative")
    if panel.horizontal_spacing_m < 0.0:
        errors.append(f"{prefix}.horizontal_spacing_m must be non-negative")
    if panel.polarization not in _POLARIZATIONS:
        errors.append(f"{prefix}.polarization must be one of {sorted(_POLARIZATIONS)}")
    try:
        antenna_pattern_spec(panel.pattern)
    except ValueError as exc:
        errors.append(f"{prefix}.pattern: {exc}")
    try:
        antenna_pattern_spec(panel.element_diagram)
    except ValueError as exc:
        errors.append(f"{prefix}.element_diagram: {exc}")
    if panel.element_diagram != panel.pattern:
        errors.append(f"{prefix}.element_diagram must match {prefix}.pattern")


def _validate_trajectory(prefix: str, trajectory: TrajectorySpec, errors: list[str]) -> None:
    if trajectory.kind not in _TRAJECTORY_KINDS:
        errors.append(f"{prefix}.kind must be one of {sorted(_TRAJECTORY_KINDS)}")
        return
    if trajectory.samples < 1:
        errors.append(f"{prefix}.samples must be at least 1")
    if trajectory.easing not in _TRAJECTORY_EASINGS:
        errors.append(f"{prefix}.easing must be one of {sorted(_TRAJECTORY_EASINGS)}")
    if not 0.0 <= trajectory.start_static_fraction < 1.0:
        errors.append(f"{prefix}.start_static_fraction must be in [0, 1)")
    if not 0.0 <= trajectory.end_static_fraction < 1.0:
        errors.append(f"{prefix}.end_static_fraction must be in [0, 1)")
    if trajectory.start_static_fraction + trajectory.end_static_fraction >= 1.0:
        errors.append(f"{prefix} static fractions must sum to less than 1")
    if not trajectory.points:
        errors.append(f"{prefix}.points must contain at least one point")
    if trajectory.orientation_rad_points and len(trajectory.orientation_rad_points) not in {1, len(trajectory.points)}:
        errors.append(f"{prefix}.orientation_rad_points must contain one point or match trajectory points")
    if trajectory.kind == "linear" and len(trajectory.points) < 2:
        errors.append(f"{prefix}.points must contain at least two points for linear trajectories")
    if trajectory.kind == "polyline" and len(trajectory.points) < 2:
        errors.append(f"{prefix}.points must contain at least two points for polyline trajectories")
    if trajectory.kind == "curve":
        if len(trajectory.points) < 2:
            errors.append(f"{prefix}.points must contain at least two points for curve trajectories")
        if trajectory.bezier_handles and len(trajectory.bezier_handles) != len(trajectory.points):
            errors.append(f"{prefix}.bezier_handles must contain one in/out pair per trajectory point")
    try:
        sample_trajectory(trajectory)
    except ValueError as exc:
        errors.append(f"{prefix}: {exc}")


def _validate_object_targets(request: SimulationRequest, errors: list[str]) -> None:
    scenario_path = request.scene.scenario_path
    if not request.scene.objects or not scenario_path.exists():
        return
    asset = load_scenario_asset(scenario_path)
    available = set(asset.object_names) | {mesh.name for mesh in asset.object_meshes}
    if not available:
        return
    for idx, obj in enumerate(request.scene.objects):
        if obj.object_name not in available:
            preview = ", ".join(sorted(available)[:8])
            errors.append(
                f"objects[{idx}].object_name '{obj.object_name}' is not present in "
                f"{scenario_path}; available objects include: {preview}"
            )


def _validate_channel_frequency_grid(request: SimulationRequest, band_vectors: list, errors: list[str]) -> None:
    if request.channel_mode not in _DELAY_BIN_CHANNEL_MODES:
        return
    if len(band_vectors) != 1:
        errors.append(
            f"channel_mode {request.channel_mode.value} requires one contiguous uniformly sampled "
            "frequency band for delay-bin output; use frequency_domain or cir_paths for multi-band requests"
        )


def _active_device_ids(request: SimulationRequest) -> list[str]:
    scene = request.scene
    if scene.radiomap.enabled:
        template_id = scene.radiomap.ue_template.id
        radiomap_id = "ue_radiomap" if template_id == "rm_ue" else template_id
        return [radiomap_id, *(entity.id for entity in scene.base_stations)]
    return [
        *(entity.id for entity in scene.user_equipments),
        *(entity.id for entity in scene.base_stations),
    ]


def _validate_unique_ids(label: str, ids: Iterable[str], errors: list[str]) -> None:
    seen: set[str] = set()
    for entity_id in ids:
        if not entity_id:
            continue
        if entity_id in seen:
            errors.append(f"Duplicate {label} id: {entity_id}")
        seen.add(entity_id)

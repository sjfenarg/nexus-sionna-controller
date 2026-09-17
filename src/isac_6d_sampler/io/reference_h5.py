from __future__ import annotations

import json
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from isac_6d_sampler.core.antenna_patterns import antenna_pattern_spec
from isac_6d_sampler.core.model import AntennaPanel, ChannelMode, SimulationRequest, TrajectorySpec
from isac_6d_sampler.core.trajectories import sample_radiomap_grid
from isac_6d_sampler.sim.channel import IFFT_GRIDDED_DELAY_OVERSAMPLING
from isac_6d_sampler.sim.materials import (
    calibrated_material_specs,
    material_properties_at_frequency,
    material_spec_for_object_name,
    tuned_diffuse_scattering_coefficient_at_frequency,
)
from isac_6d_sampler.sim.planner import planned_user_equipments
from isac_6d_sampler.sim.power import dbm_to_watt
from isac_6d_sampler.sim.results import SimulationResult
from isac_6d_sampler.io.schema import validate_reference_h5


class ReferenceH5Writer:
    """Write samples using the hierarchy observed in ``output/reference_sample.h5``."""

    def __init__(self, validate_output: bool = True):
        self.validate_output = validate_output

    def write(self, path: Path, request: SimulationRequest, result: SimulationResult) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        scene = request.scene
        scene.ensure_defaults()
        with h5py.File(path, "w") as h5:
            scenarios = h5.require_group("scenarios")
            scenario_group = scenarios.require_group(result.scenario_name)
            scenario_group.require_group("metadata").attrs["description"] = scene.description
            sample_group = scenario_group.require_group(result.sample_id)
            self._write_sample_metadata(sample_group, request, result)
            self._write_parameters(sample_group, request, result)
            self._write_timeframes(sample_group, result)
        if self.validate_output:
            errors = validate_reference_h5(path)
            if errors:
                raise ValueError("Generated HDF5 failed schema validation: " + "; ".join(errors))
        return path

    def _write_sample_metadata(self, sample_group, request: SimulationRequest, result: SimulationResult) -> None:
        meta = sample_group.require_group("metadata")
        meta.attrs["description"] = request.scene.description
        meta.attrs["created_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        meta.attrs["generator"] = "isac-6d-sampler"
        meta.attrs["request_json"] = json.dumps(request.to_dict())
        for key, value in result.metadata.items():
            meta.attrs[key] = _attr_value(value)

    def _write_parameters(self, sample_group, request: SimulationRequest, result: SimulationResult) -> None:
        params = sample_group.require_group("parameters")
        user_equipments = planned_user_equipments(request.scene)
        params.attrs["n_bs"] = len(request.scene.base_stations)
        params.attrs["n_ue"] = len(user_equipments)
        params.attrs["channel_mode"] = request.channel_mode.value
        params.attrs["samples_per_src"] = request.sionna.samples_per_src
        params.attrs["max_depth"] = request.sionna.max_depth
        params.attrs["diffuse_reflection"] = request.sionna.diffuse_reflection
        _replace_dataset(params, "f_vector", result.frequency_vector_hz.astype(np.float64))

        subbands = params.require_group("subband_f_vectors")
        for idx, vector in enumerate(result.subband_vectors_hz):
            _replace_dataset(subbands, f"subband_f_vector_{idx}", vector.astype(np.float64))

        vna = params.require_group("vna_params")
        vna.attrs["n_subbands"] = len(result.subband_vectors_hz)
        vna.attrs["n_sweeps"] = 1
        vna.attrs["ifbw"] = 10_000.0
        starts = np.array([float(v.reshape(-1)[0]) for v in result.subband_vectors_hz], dtype=np.float64)
        stops = np.array([float(v.reshape(-1)[-1]) for v in result.subband_vectors_hz], dtype=np.float64)
        points = np.array([v.size for v in result.subband_vectors_hz], dtype=np.float64)
        _replace_dataset(vna, "subbands_f_start", starts)
        _replace_dataset(vna, "subbands_f_stop", stops)
        _replace_dataset(vna, "subbands_points", points)
        self._write_frequency_bands(params, request)
        self._write_sionna_params(params, request)
        self._write_channel_params(params, request, result)
        self._write_material_params(params, result)
        self._write_device_params(params, request, user_equipments)
        self._write_link_params(params, request, result, user_equipments)

        antennas = params.require_group("antenna_params")
        for idx, bs in enumerate(request.scene.base_stations):
            self._write_antenna(antennas.require_group(f"bs{idx}"), bs.panel, bs.position, bs.orientation_rad, bs.id)
        for idx, ue in enumerate(user_equipments):
            self._write_antenna(antennas.require_group(f"ue{idx}"), ue.panel, ue.position, ue.orientation_rad, ue.id)

        objects = params.require_group("object_params")
        for idx, obj in enumerate(request.scene.objects):
            obj_group = objects.require_group(f"object{idx}")
            obj_group.attrs["id"] = obj.id
            obj_group.attrs["object_name"] = obj.object_name
            material = material_spec_for_object_name(obj.object_name)
            if material is not None:
                obj_group.attrs["material_prefix"] = material.object_prefix
                obj_group.attrs["material_name"] = material.name
            _replace_dataset(obj_group, "position", np.asarray(obj.position, dtype=np.float64))
            _replace_dataset(obj_group, "orientation", np.asarray(obj.orientation_rad, dtype=np.float64))
        self._write_trajectory_params(params, request, user_equipments)
        self._write_radiomap_params(params, request, user_equipments)

    def _write_frequency_bands(self, params, request: SimulationRequest) -> None:
        bands = params.require_group("frequency_bands")
        for idx, band in enumerate(request.bands):
            group = bands.require_group(f"band{idx}")
            group.attrs["name"] = band.name
            group.attrs["start_hz"] = float(band.start_hz)
            group.attrs["stop_hz"] = float(band.stop_hz)
            group.attrs["points"] = int(band.points)

    def _write_sionna_params(self, params, request: SimulationRequest) -> None:
        sionna_group = params.require_group("sionna_params")
        for field in fields(request.sionna):
            sionna_group.attrs[field.name] = _attr_value(getattr(request.sionna, field.name))

    def _write_channel_params(self, params, request: SimulationRequest, result: SimulationResult) -> None:
        group = params.require_group("channel_params")
        f_vector = np.asarray(result.frequency_vector_hz, dtype=np.float64).reshape(-1)
        group.attrs["mode"] = request.channel_mode.value
        group.attrs["tx_power_dbm"] = float(request.sionna.tx_power_dbm)
        group.attrs["tx_power_w"] = dbm_to_watt(request.sionna.tx_power_dbm)
        group.attrs["frequency_points"] = int(f_vector.size)
        if f_vector.size:
            group.attrs["frequency_start_hz"] = float(f_vector[0])
            group.attrs["frequency_stop_hz"] = float(f_vector[-1])
        bandwidth_hz = float(f_vector[-1] - f_vector[0]) if f_vector.size > 1 else 0.0
        group.attrs["bandwidth_hz"] = bandwidth_hz

        if request.channel_mode == ChannelMode.FREQUENCY_DOMAIN:
            group.attrs["sample_axis"] = "frequency_hz"
            group.attrs["value_quantity"] = "complex_frequency_response"
            _replace_dataset(group, "sample_coordinates", f_vector)
        elif request.channel_mode == ChannelMode.CIR_PATHS:
            group.attrs["sample_axis"] = "path_index"
            group.attrs["value_quantity"] = "complex_path_coefficient"
            group.attrs["sample_coordinate_note"] = "per-link path index; delays are not gridded"
        else:
            group.attrs["sample_axis"] = "delay_bin_s"
            group.attrs["value_quantity"] = (
                "linear_power" if request.channel_mode == ChannelMode.PDP_BINNED else "complex_delay_response"
            )
            if request.channel_mode == ChannelMode.PDP_IFFT_EXACT:
                group.attrs["ifft_renderer"] = "exact_nonuniform_frequency_response"
            elif request.channel_mode == ChannelMode.PDP_IFFT_GRIDDED:
                group.attrs["ifft_renderer"] = "oversampled_delay_grid"
                group.attrs["delay_oversampling"] = int(IFFT_GRIDDED_DELAY_OVERSAMPLING)
            samples = int(f_vector.size)
            delay_bin_width_s = 1.0 / max(bandwidth_hz, 1.0)
            group.attrs["delay_bin_width_s"] = delay_bin_width_s
            group.attrs["delay_window_s"] = samples * delay_bin_width_s
            _replace_dataset(
                group,
                "sample_coordinates",
                np.arange(samples, dtype=np.float64) * delay_bin_width_s,
            )

    def _write_material_params(self, params, result: SimulationResult) -> None:
        group = params.require_group("material_params")
        representative_frequency_hz = float(np.mean(np.asarray(result.frequency_vector_hz, dtype=np.float64)))
        group.attrs["source"] = "itu-r-p2040-3_frequency_evaluated_with_project_tuning"
        specs = calibrated_material_specs()
        group.attrs["representative_frequency_hz"] = representative_frequency_hz
        group.attrs["n_materials"] = len(specs)
        for idx, spec in enumerate(specs):
            material_group = group.require_group(f"material{idx}")
            material_group.attrs["object_prefix"] = spec.object_prefix
            material_group.attrs["name"] = spec.name
            relative_permittivity, conductivity = material_properties_at_frequency(spec, representative_frequency_hz)
            material_group.attrs["relative_permittivity"] = float(relative_permittivity)
            material_group.attrs["conductivity"] = float(conductivity)
            material_group.attrs["scattering_coefficient"] = float(tuned_diffuse_scattering_coefficient_at_frequency(spec, representative_frequency_hz))
            material_group.attrs["scattering_pattern"] = spec.scattering_pattern
            material_group.attrs["alpha_r"] = float(spec.alpha_r)
            material_group.attrs["source"] = spec.source

    def _write_link_params(self, params, request: SimulationRequest, result: SimulationResult, user_equipments) -> None:
        group = params.require_group("link_params")
        links = result.timeframes[0].links if result.timeframes else []
        group.attrs["n_links"] = len(links)
        ue_ids = {ue.id for ue in user_equipments}
        bs_ids = {bs.id for bs in request.scene.base_stations}
        for link in links:
            name = f"rx{link.rx_index}_tx{link.tx_index}"
            link_group = group.require_group(name)
            rx_type = _entity_type(link.rx_id, ue_ids, bs_ids)
            tx_type = _entity_type(link.tx_id, ue_ids, bs_ids)
            link_group.attrs["dataset_name"] = name
            link_group.attrs["rx_index"] = int(link.rx_index)
            link_group.attrs["tx_index"] = int(link.tx_index)
            link_group.attrs["rx_id"] = link.rx_id
            link_group.attrs["tx_id"] = link.tx_id
            link_group.attrs["rx_device_group"] = f"device{link.rx_index}"
            link_group.attrs["tx_device_group"] = f"device{link.tx_index}"
            link_group.attrs["rx_entity_type"] = rx_type
            link_group.attrs["tx_entity_type"] = tx_type
            link_group.attrs["direction"] = _link_direction(rx_type, tx_type, link.rx_id == link.tx_id)
            link_group.attrs["is_monostatic"] = bool(link.rx_index == link.tx_index)

    def _write_device_params(self, params, request: SimulationRequest, user_equipments) -> None:
        group = params.require_group("device_params")
        group.attrs["index_order"] = "ues_first_then_base_stations"
        group.attrs["n_devices"] = len(user_equipments) + len(request.scene.base_stations)
        for index, ue in enumerate(user_equipments):
            device_group = group.require_group(f"device{index}")
            device_group.attrs["index"] = int(index)
            device_group.attrs["entity_id"] = ue.id
            device_group.attrs["entity_type"] = "ue"
            device_group.attrs["antenna_group"] = f"antenna_params/ue{index}"
        offset = len(user_equipments)
        for bs_index, bs in enumerate(request.scene.base_stations):
            index = offset + bs_index
            device_group = group.require_group(f"device{index}")
            device_group.attrs["index"] = int(index)
            device_group.attrs["entity_id"] = bs.id
            device_group.attrs["entity_type"] = "bs"
            device_group.attrs["antenna_group"] = f"antenna_params/bs{bs_index}"

    def _write_trajectory_params(self, params, request: SimulationRequest, user_equipments) -> None:
        trajectories = params.require_group("trajectory_params")
        ue_group = trajectories.require_group("ues")
        for idx, ue in enumerate(user_equipments):
            group = ue_group.require_group(f"ue{idx}")
            group.attrs["entity_id"] = ue.id
            self._write_trajectory(group, ue.trajectory)

        object_group = trajectories.require_group("objects")
        for idx, obj in enumerate(request.scene.objects):
            group = object_group.require_group(f"object{idx}")
            group.attrs["entity_id"] = obj.id
            group.attrs["object_name"] = obj.object_name
            self._write_trajectory(group, obj.trajectory)

    def _write_trajectory(self, group, trajectory: TrajectorySpec) -> None:
        group.attrs["kind"] = trajectory.kind
        group.attrs["samples"] = int(trajectory.samples)
        group.attrs["start_static_fraction"] = float(trajectory.start_static_fraction)
        group.attrs["end_static_fraction"] = float(trajectory.end_static_fraction)
        group.attrs["easing"] = trajectory.easing
        _replace_dataset(group, "control_points", np.asarray(trajectory.points, dtype=np.float64))
        if trajectory.bezier_handles:
            _replace_dataset(group, "bezier_handles", np.asarray(trajectory.bezier_handles, dtype=np.float64))
        if trajectory.orientation_rad_points:
            _replace_dataset(
                group,
                "orientation_rad_control_points",
                np.asarray(trajectory.orientation_rad_points, dtype=np.float64),
            )

    def _write_radiomap_params(self, params, request: SimulationRequest, user_equipments) -> None:
        radiomap = request.scene.radiomap
        group = params.require_group("radiomap_params")
        group.attrs["enabled"] = bool(radiomap.enabled)
        group.attrs["x_min"] = float(radiomap.x_min)
        group.attrs["x_max"] = float(radiomap.x_max)
        group.attrs["y_min"] = float(radiomap.y_min)
        group.attrs["y_max"] = float(radiomap.y_max)
        group.attrs["x_spacing"] = float(radiomap.x_spacing)
        group.attrs["y_spacing"] = float(radiomap.y_spacing)
        group.attrs["height"] = float(radiomap.height)
        group.attrs["ue_template_id"] = user_equipments[0].id if radiomap.enabled and user_equipments else radiomap.ue_template.id
        if radiomap.enabled:
            positions = sample_radiomap_grid(radiomap)
            xs = np.unique(positions[:, 0])
            ys = np.unique(positions[:, 1])
            group.attrs["x_points"] = int(xs.size)
            group.attrs["y_points"] = int(ys.size)
            _replace_dataset(group, "x_coordinates", xs.astype(np.float64))
            _replace_dataset(group, "y_coordinates", ys.astype(np.float64))
            _replace_dataset(group, "positions", positions.astype(np.float64))

    def _write_antenna(self, group, panel: AntennaPanel, position, orientation, name: str) -> None:
        spec = antenna_pattern_spec(panel.pattern)
        group.attrs["ant_name"] = name
        group.attrs["ant_type"] = panel.element_diagram
        group.attrs["pattern"] = panel.pattern
        group.attrs["pattern_label"] = spec.label
        group.attrs["pattern_source"] = spec.source
        if spec.max_gain_db is not None:
            group.attrs["pattern_max_gain_db"] = float(spec.max_gain_db)
        if spec.h_3db_beamwidth_deg is not None:
            group.attrs["h_3db_beamwidth_deg"] = float(spec.h_3db_beamwidth_deg)
        if spec.v_3db_beamwidth_deg is not None:
            group.attrs["v_3db_beamwidth_deg"] = float(spec.v_3db_beamwidth_deg)
        group.attrs["num_rows"] = panel.rows
        group.attrs["num_cols"] = panel.cols
        group.attrs["vertical_spacing"] = panel.vertical_spacing_m
        group.attrs["horizontal_spacing"] = panel.horizontal_spacing_m
        group.attrs["vertical_spacing_units"] = "meter"
        group.attrs["horizontal_spacing_units"] = "meter"
        group.attrs["ant_height"] = float(position[2])
        group.attrs["polarization"] = panel.polarization
        group.attrs["h_pol_vector"] = "NONE" if panel.h_pol_vector is None else json.dumps(panel.h_pol_vector)
        _replace_dataset(group, "ant_coord", np.asarray(position[:2], dtype=np.float64))
        group.attrs["ant_orientation_units"] = "degrees"
        group.attrs["ant_orientation_rad_units"] = "radians"
        _replace_dataset(group, "ant_orientation", np.rad2deg(np.asarray(orientation[:2], dtype=np.float64)))
        _replace_dataset(group, "ant_orientation_rad", np.asarray(orientation, dtype=np.float64))
        _replace_dataset(group, "v_pol_vector", np.asarray(panel.v_pol_vector, dtype=np.float64))

    def _write_timeframes(self, sample_group, result: SimulationResult) -> None:
        timeframes_group = sample_group.require_group("timeframes")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        channel_sample_axis = _sample_axis_for_channel_mode(result.metadata.get("channel_mode"))
        write_path_metadata = result.metadata.get("channel_mode") != ChannelMode.FREQUENCY_DOMAIN.value
        for timeframe in result.timeframes:
            tf_group = timeframes_group.require_group(timeframe.name)
            h_group = tf_group.require_group("h")
            tau_group = tf_group.require_group("tau") if write_path_metadata else None
            a_group = tf_group.require_group("a") if write_path_metadata else None
            timestamp_group = tf_group.require_group("timestamps")
            self._write_timeframe_attrs(tf_group, timeframe.metadata)
            tf_group.require_group("parameters").attrs["n_channels"] = len(timeframe.links)
            self._write_timeframe_positions(tf_group, timeframe.metadata)
            self._write_timeframe_orientations(tf_group, timeframe.metadata)
            for link in timeframe.links:
                name = f"rx{link.rx_index}_tx{link.tx_index}"
                h_dataset = _replace_dataset(
                    h_group,
                    name,
                    np.asarray(link.h, dtype=np.complex64),
                    compression="gzip",
                    compression_opts=4,
                )
                _write_link_dataset_attrs(h_dataset, link, "channel", sample_axis=channel_sample_axis)
                if write_path_metadata and "path_delays_s" in link.metadata:
                    tau_dataset = _replace_dataset(
                        tau_group,
                        name,
                        np.asarray(link.metadata["path_delays_s"], dtype=np.float64),
                        compression="gzip",
                        compression_opts=4,
                    )
                    _write_link_dataset_attrs(tau_dataset, link, "path_delay", sample_axis="path_index")
                    tau_dataset.attrs["value_units"] = "seconds"
                if write_path_metadata and "path_coefficients" in link.metadata:
                    a_dataset = _replace_dataset(
                        a_group,
                        name,
                        np.asarray(link.metadata["path_coefficients"], dtype=np.complex64),
                        compression="gzip",
                        compression_opts=4,
                    )
                    _write_link_dataset_attrs(a_dataset, link, "path_coefficient", sample_axis="path_index")
                timestamp_shape = np.asarray(link.h).shape[:-1]
                timestamp = link.timestamp or datetime.now(timezone.utc).isoformat()
                timestamp_data = np.full(timestamp_shape, timestamp, dtype=object)
                timestamp_dataset = _replace_dataset(timestamp_group, name, timestamp_data, dtype=string_dtype)
                _write_link_dataset_attrs(timestamp_dataset, link, "timestamp")

    def _write_timeframe_attrs(self, tf_group, metadata: dict) -> None:
        structured_keys = {
            "device_positions",
            "object_positions",
            "device_orientations",
            "object_orientations",
        }
        for key, value in metadata.items():
            if key not in structured_keys:
                tf_group.attrs[key] = _attr_value(value)

    def _write_timeframe_positions(self, tf_group, metadata: dict) -> None:
        positions_group = tf_group.require_group("positions")
        for group_name, values in (
            ("devices", metadata.get("device_positions", {})),
            ("objects", metadata.get("object_positions", {})),
        ):
            group = positions_group.require_group(group_name)
            for entity_id, position in values.items():
                _replace_dataset(group, entity_id, np.asarray(position, dtype=np.float64))

    def _write_timeframe_orientations(self, tf_group, metadata: dict) -> None:
        orientations_group = tf_group.require_group("orientations")
        for group_name, values in (
            ("devices", metadata.get("device_orientations", {})),
            ("objects", metadata.get("object_orientations", {})),
        ):
            group = orientations_group.require_group(group_name)
            for entity_id, orientation in values.items():
                _replace_dataset(group, entity_id, np.asarray(orientation, dtype=np.float64))


def default_output_path(request: SimulationRequest, now: datetime | None = None) -> Path:
    now = now or datetime.now()
    prefix = "radiomap" if request.scene.radiomap.enabled else "scene"
    stamp = now.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(request.output_dir)
    candidate = output_dir / f"{prefix}_{stamp}.h5"
    if not candidate.exists():
        return candidate
    for suffix in range(1, 1000):
        candidate = output_dir / f"{prefix}_{stamp}_{suffix:03d}.h5"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a free output filename for timestamp {stamp}")


def _replace_dataset(group, name: str, data, **kwargs):
    if name in group:
        del group[name]
    return group.create_dataset(name, data=data, **kwargs)


def _write_link_dataset_attrs(dataset, link, role: str, sample_axis: str | None = None) -> None:
    dataset.attrs["dataset_role"] = role
    dataset.attrs["dataset_name"] = f"rx{link.rx_index}_tx{link.tx_index}"
    dataset.attrs["rx_index"] = int(link.rx_index)
    dataset.attrs["tx_index"] = int(link.tx_index)
    dataset.attrs["rx_id"] = link.rx_id
    dataset.attrs["tx_id"] = link.tx_id
    dataset.attrs["is_monostatic"] = bool(link.rx_index == link.tx_index)
    labels = ["rx_row", "rx_col", "tx_row", "tx_col", "polarization"]
    if sample_axis is not None:
        labels = [*labels, sample_axis]
        dataset.attrs["sample_axis"] = sample_axis
    dataset.attrs["axis_labels"] = json.dumps(labels)


def _attr_value(value):
    if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
        return value
    return json.dumps(value)


def _entity_type(entity_id: str, ue_ids: set[str], bs_ids: set[str]) -> str:
    if entity_id in ue_ids:
        return "ue"
    if entity_id in bs_ids:
        return "bs"
    return "unknown"


def _link_direction(rx_type: str, tx_type: str, same_entity: bool) -> str:
    if same_entity:
        return "monostatic"
    if rx_type == "ue" and tx_type == "bs":
        return "bs_to_ue"
    if rx_type == "bs" and tx_type == "ue":
        return "ue_to_bs"
    return f"{tx_type}_to_{rx_type}"


def _sample_axis_for_channel_mode(channel_mode: str | None) -> str:
    if channel_mode == ChannelMode.FREQUENCY_DOMAIN.value:
        return "frequency_hz"
    if channel_mode == ChannelMode.CIR_PATHS.value:
        return "path_index"
    return "delay_bin_s"

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from isac_6d_sampler.core.antenna_patterns import available_pattern_names
from isac_6d_sampler.core.config_io import read_request, template_request, write_request
from isac_6d_sampler.core.estimates import estimate_request_size, format_request_estimate
from isac_6d_sampler.core.frequencies import parse_band_specs
from isac_6d_sampler.core.model import (
    BaseStation,
    ChannelMode,
    DynamicObject,
    SimulationRequest,
    TrajectorySpec,
    UserEquipment,
)
from isac_6d_sampler.core.trajectory_specs import build_trajectory_spec
from isac_6d_sampler.core.validation import validate_request
from isac_6d_sampler.io.reference_h5 import ReferenceH5Writer, default_output_path
from isac_6d_sampler.io.schema import validate_reference_h5
from isac_6d_sampler.sim.dry_run import DryRunSimulator
from isac_6d_sampler.sim.sionna_backend import SionnaSimulator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate ISAC 6D channel samples")
    parser.add_argument("--config", type=Path, help="JSON simulation request to load")
    parser.add_argument("--write-template", type=Path, help="Write an editable JSON request template and exit")
    parser.add_argument("--validate-h5", type=Path, help="Validate an existing reference-style HDF5 file and exit")
    parser.add_argument("--plan-only", action="store_true", help="Validate and print the planned run size without simulating")
    parser.add_argument("--dry-run", action="store_true", help="Generate deterministic schema-valid data")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument("--scenario", type=Path, help="Override Sionna XML scenario path")
    parser.add_argument("--sample-id", help="Override sample id, e.g. s000")
    parser.add_argument("--channel-mode", choices=[mode.value for mode in ChannelMode])
    parser.add_argument("--bs-pattern", choices=list(available_pattern_names()))
    parser.add_argument("--ue-pattern", choices=list(available_pattern_names()))
    parser.add_argument("--bs-spacing", metavar="VERTICAL,HORIZONTAL", help="Set BS panel spacing for all BSs")
    parser.add_argument("--ue-spacing", metavar="VERTICAL,HORIZONTAL", help="Set UE/radiomap panel spacing for all UEs")
    parser.add_argument(
        "--band",
        action="append",
        default=[],
        metavar="START_GHZ:STOP_GHZ:POINTS[:NAME]",
        help="Add a frequency band. Can be repeated.",
    )
    parser.add_argument(
        "--bs",
        action="append",
        default=[],
        metavar="ID:X,Y,Z[:YAW,PITCH,ROLL[:ROWSxCOLS]]",
        help="Add a BS, e.g. bs1:52,-19,19:0,0,0:10x10. Can be repeated.",
    )
    parser.add_argument(
        "--ue",
        action="append",
        default=[],
        metavar="ID:X,Y,Z[:YAW,PITCH,ROLL[:ROWSxCOLS]]",
        help="Add a UE, e.g. ue1:0,0,1.5:0,0,0:1x1. Can be repeated.",
    )
    parser.add_argument(
        "--ue-trajectory",
        action="append",
        default=[],
        metavar="ID:KIND:SAMPLES:POINTS",
        help="Set a UE trajectory, e.g. ue1:linear:8:0,0,1.5;5,0,1.5. Can be repeated.",
    )
    parser.add_argument("--samples-per-src", type=int)
    parser.add_argument("--tx-power-dbm", type=float, help="Transmit power in dBm used for every transmitter")
    parser.add_argument(
        "--max-num-paths-per-src",
        help="Maximum paths per source, or one of none/null/unlimited",
    )
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--los", choices=["on", "off"])
    parser.add_argument("--specular", choices=["on", "off"])
    parser.add_argument("--diffuse", choices=["on", "off"])
    parser.add_argument("--refraction", choices=["on", "off"])
    parser.add_argument("--synthetic-array", choices=["on", "off"])
    parser.add_argument("--merge-shapes", choices=["on", "off"])
    parser.add_argument("--use-gpu", choices=["on", "off"])
    parser.add_argument("--batch-timeframes", type=int)
    parser.add_argument("--max-timeframes", type=int)
    parser.add_argument("--radiomap", choices=["on", "off"], help="Enable or disable xy radiomap mode")
    parser.add_argument("--rm-x-min", type=float)
    parser.add_argument("--rm-x-max", type=float)
    parser.add_argument("--rm-y-min", type=float)
    parser.add_argument("--rm-y-max", type=float)
    parser.add_argument("--rm-x-spacing", type=float)
    parser.add_argument("--rm-y-spacing", type=float)
    parser.add_argument("--rm-height", type=float)
    parser.add_argument("--rm-rotation", type=float, help="Radiomap rotation in degrees around its center")
    parser.add_argument(
        "--object",
        action="append",
        default=[],
        metavar="ID:SCENE_OBJECT:X,Y,Z[:YAW,PITCH,ROLL]",
        help="Add a dynamic scene object, e.g. car0:CAR_obj:0,-3,0:0,0,1.57. Can be repeated.",
    )
    parser.add_argument(
        "--object-trajectory",
        action="append",
        default=[],
        metavar="ID:KIND:SAMPLES:POINTS",
        help="Set an object trajectory, e.g. car0:linear:16:0,-3,0;10,-3,0. Can be repeated.",
    )
    args = parser.parse_args(argv)

    if args.write_template:
        write_request(args.write_template, template_request())
        print(args.write_template)
        return 0
    if args.validate_h5:
        errors = validate_reference_h5(args.validate_h5)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print(f"{args.validate_h5}: valid")
        return 0

    request = read_request(args.config) if args.config else SimulationRequest()
    try:
        _apply_overrides(request, args)
    except ValueError as exc:
        parser.error(str(exc))
    request.scene.ensure_defaults()
    try:
        validate_request(request)
    except ValueError as exc:
        parser.error(str(exc))
    if args.plan_only:
        print(format_request_estimate(estimate_request_size(request)))
        return 0
    simulator = DryRunSimulator() if request.dry_run else SionnaSimulator()
    result = simulator.simulate(request, progress=lambda i, n, msg: print(f"[{i + 1}/{n}] {msg}"))
    output_path = default_output_path(request)
    print("Writing and validating HDF5 output")
    ReferenceH5Writer().write(output_path, request, result)
    print(output_path)
    return 0

def _apply_overrides(request: SimulationRequest, args: argparse.Namespace) -> None:
    if args.dry_run:
        request.dry_run = True
    if args.output_dir is not None:
        request.output_dir = args.output_dir
    if args.scenario is not None:
        request.scene.scenario_path = args.scenario
        request.scene.name = args.scenario.stem
    if args.sample_id is not None:
        request.sample_id = args.sample_id
    if args.channel_mode is not None:
        request.channel_mode = ChannelMode(args.channel_mode)
    if args.radiomap is not None:
        request.scene.radiomap.enabled = args.radiomap == "on"
    if args.rm_x_min is not None:
        request.scene.radiomap.x_min = args.rm_x_min
    if args.rm_x_max is not None:
        request.scene.radiomap.x_max = args.rm_x_max
    if args.rm_y_min is not None:
        request.scene.radiomap.y_min = args.rm_y_min
    if args.rm_y_max is not None:
        request.scene.radiomap.y_max = args.rm_y_max
    if args.rm_x_spacing is not None:
        request.scene.radiomap.x_spacing = args.rm_x_spacing
    if args.rm_y_spacing is not None:
        request.scene.radiomap.y_spacing = args.rm_y_spacing
    if args.rm_height is not None:
        request.scene.radiomap.height = args.rm_height
    if args.rm_rotation is not None:
        request.scene.radiomap.rotation_deg = args.rm_rotation
    for spec in args.bs:
        request.scene.base_stations.append(_parse_bs_spec(spec))
    for spec in args.ue:
        request.scene.user_equipments.append(_parse_ue_spec(spec))
    if args.ue_trajectory:
        if not request.scene.user_equipments and not request.scene.radiomap.enabled:
            request.scene.ensure_defaults()
        ues_by_id = {ue.id: ue for ue in request.scene.user_equipments}
        for spec in args.ue_trajectory:
            ue_id, trajectory = _parse_entity_trajectory_spec(spec, label="UE")
            if ue_id not in ues_by_id:
                raise ValueError(f"UE trajectory references unknown UE id: {ue_id}")
            ues_by_id[ue_id].trajectory = trajectory
    if args.bs_spacing is not None or args.ue_spacing is not None:
        request.scene.ensure_defaults()
    if args.bs_spacing is not None:
        vertical, horizontal = _parse_vector2(args.bs_spacing, label="BS spacing")
        for bs in request.scene.base_stations:
            bs.panel.vertical_spacing_m = vertical
            bs.panel.horizontal_spacing_m = horizontal
    if args.ue_spacing is not None:
        vertical, horizontal = _parse_vector2(args.ue_spacing, label="UE spacing")
        for ue in request.scene.user_equipments:
            ue.panel.vertical_spacing_m = vertical
            ue.panel.horizontal_spacing_m = horizontal
        request.scene.radiomap.ue_template.panel.vertical_spacing_m = vertical
        request.scene.radiomap.ue_template.panel.horizontal_spacing_m = horizontal
    if args.bs_pattern is not None or args.ue_pattern is not None:
        request.scene.ensure_defaults()
    if args.bs_pattern is not None:
        for bs in request.scene.base_stations:
            bs.panel.pattern = args.bs_pattern
            bs.panel.element_diagram = args.bs_pattern
    if args.ue_pattern is not None:
        for ue in request.scene.user_equipments:
            ue.panel.pattern = args.ue_pattern
            ue.panel.element_diagram = args.ue_pattern
        request.scene.radiomap.ue_template.panel.pattern = args.ue_pattern
        request.scene.radiomap.ue_template.panel.element_diagram = args.ue_pattern
    if args.band:
        request.bands = parse_band_specs(args.band)
    if args.samples_per_src is not None:
        request.sionna.samples_per_src = args.samples_per_src
    if args.tx_power_dbm is not None:
        request.sionna.tx_power_dbm = args.tx_power_dbm
    if args.max_num_paths_per_src is not None:
        request.sionna.max_num_paths_per_src = _parse_optional_int(
            args.max_num_paths_per_src,
            label="max-num-paths-per-src",
        )
    if args.max_depth is not None:
        request.sionna.max_depth = args.max_depth
    if args.seed is not None:
        request.sionna.seed = args.seed
    if args.los is not None:
        request.sionna.los = args.los == "on"
    if args.specular is not None:
        request.sionna.specular_reflection = args.specular == "on"
    if args.diffuse is not None:
        request.sionna.diffuse_reflection = args.diffuse == "on"
    if args.refraction is not None:
        request.sionna.refraction = args.refraction == "on"
    if args.synthetic_array is not None:
        request.sionna.synthetic_array = args.synthetic_array == "on"
    if args.merge_shapes is not None:
        request.sionna.merge_shapes = args.merge_shapes == "on"
    if args.use_gpu is not None:
        request.sionna.use_gpu = args.use_gpu == "on"
    if args.batch_timeframes is not None:
        request.sionna.batch_timeframes = args.batch_timeframes
    if args.max_timeframes is not None:
        request.sionna.max_timeframes = args.max_timeframes
    for spec in args.object:
        request.scene.objects.append(_parse_object_spec(spec))
    if args.object_trajectory:
        objects_by_id = {obj.id: obj for obj in request.scene.objects}
        for spec in args.object_trajectory:
            object_id, trajectory = _parse_entity_trajectory_spec(spec, label="Object")
            if object_id not in objects_by_id:
                raise ValueError(f"Object trajectory references unknown object id: {object_id}")
            objects_by_id[object_id].trajectory = trajectory


def _parse_bs_spec(spec: str) -> BaseStation:
    device_id, position, orientation, panel_shape = _parse_device_spec(spec, label="BS")
    bs = BaseStation(id=device_id, position=position, orientation_rad=orientation)
    if panel_shape is not None:
        bs.panel.rows, bs.panel.cols = panel_shape
    return bs


def _parse_ue_spec(spec: str) -> UserEquipment:
    device_id, position, orientation, panel_shape = _parse_device_spec(spec, label="UE")
    ue = UserEquipment(
        id=device_id,
        position=position,
        orientation_rad=orientation,
        trajectory=TrajectorySpec.static(position),
    )
    if panel_shape is not None:
        ue.panel.rows, ue.panel.cols = panel_shape
    return ue


def _parse_device_spec(
    spec: str,
    *,
    label: str,
) -> tuple[str, tuple[float, float, float], tuple[float, float, float], tuple[int, int] | None]:
    parts = spec.split(":")
    if len(parts) not in (2, 3, 4):
        raise ValueError(f"{label} specs must be ID:X,Y,Z[:YAW,PITCH,ROLL[:ROWSxCOLS]]")
    device_id, position_text = parts[:2]
    position = _parse_vector3(position_text, label=f"{label} position")
    orientation = _parse_vector3(parts[2], label=f"{label} orientation") if len(parts) >= 3 else (0.0, 0.0, 0.0)
    panel_shape = _parse_panel_shape(parts[3], label=label) if len(parts) == 4 else None
    return device_id, position, orientation, panel_shape


def _parse_object_spec(spec: str) -> DynamicObject:
    parts = spec.split(":")
    if len(parts) not in (3, 4):
        raise ValueError("Object specs must be ID:SCENE_OBJECT:X,Y,Z[:YAW,PITCH,ROLL]")
    object_id, object_name, position_text = parts[:3]
    position = _parse_vector3(position_text, label="object position")
    orientation = (
        _parse_vector3(parts[3], label="object orientation")
        if len(parts) == 4
        else (0.0, 0.0, 0.0)
    )
    return DynamicObject(
        id=object_id,
        object_name=object_name,
        position=position,
        orientation_rad=orientation,
        trajectory=TrajectorySpec.static(position),
    )


def _parse_entity_trajectory_spec(spec: str, *, label: str) -> tuple[str, TrajectorySpec]:
    parts = spec.split(":", 3)
    if len(parts) != 4:
        raise ValueError(f"{label} trajectory specs must be ID:KIND:SAMPLES:POINTS")
    entity_id, kind, samples_text, points_text = parts
    try:
        samples = int(samples_text)
    except ValueError as exc:
        raise ValueError(f"{label} trajectory samples must be an integer: {samples_text}") from exc
    return entity_id, build_trajectory_spec(kind, points_text, samples=samples)


def _parse_panel_shape(text: str, *, label: str) -> tuple[int, int]:
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"{label} panel shape must use ROWSxCOLS format")
    try:
        rows, cols = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"{label} panel shape must contain integer rows and cols: {text}") from exc
    return rows, cols


def _parse_vector3(text: str, *, label: str) -> tuple[float, float, float]:
    parts = text.split(",")
    if len(parts) != 3:
        raise ValueError(f"{label} must have x,y,z format")
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise ValueError(f"{label} must contain numeric values: {text}") from exc


def _parse_vector2(text: str, *, label: str) -> tuple[float, float]:
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError(f"{label} must have vertical,horizontal format")
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError as exc:
        raise ValueError(f"{label} must contain numeric values: {text}") from exc


def _parse_optional_int(text: str, *, label: str) -> int | None:
    if text.strip().lower() in {"none", "null", "unlimited"}:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer or one of none/null/unlimited: {text}") from exc

if __name__ == "__main__":
    raise SystemExit(main())

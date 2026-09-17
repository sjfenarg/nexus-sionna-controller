from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import (
    AntennaPanel,
    BaseStation,
    ChannelMode,
    DynamicObject,
    FrequencyBand,
    RadiomapConfig,
    SceneDesign,
    SimulationRequest,
    SionnaConfig,
    TrajectorySpec,
    UserEquipment,
)
from .antenna_patterns import sionna_pattern_name


def read_request(path: Path) -> SimulationRequest:
    with Path(path).open("r", encoding="utf-8") as fh:
        return request_from_dict(json.load(fh))


def write_request(path: Path, request: SimulationRequest) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(request.to_dict(), fh, indent=2)
        fh.write("\n")
    return path


def request_from_dict(data: dict[str, Any]) -> SimulationRequest:
    scene = _scene(data.get("scene", {}))
    bands = [_band(item) for item in data.get("bands", [])] or [FrequencyBand()]
    channel_mode = ChannelMode(data.get("channel_mode", ChannelMode.FREQUENCY_DOMAIN.value))
    sionna = _sionna(data.get("sionna", {}))
    request = SimulationRequest(
        scene=scene,
        bands=bands,
        channel_mode=channel_mode,
        sionna=sionna,
        output_dir=Path(data.get("output_dir", "output")),
        sample_id=str(data.get("sample_id", "s000")),
        dry_run=bool(data.get("dry_run", False)),
    )
    return request


def template_request() -> SimulationRequest:
    request = SimulationRequest(dry_run=True)
    request.scene.ensure_defaults()
    request.scene.objects.append(
        DynamicObject(
            id="car0",
            object_name="CAR_obj",
            position=(0.0, -3.0, 0.0),
            trajectory=TrajectorySpec.linear((0.0, -3.0, 0.0), (10.0, -3.0, 0.0), 16),
        )
    )
    request.bands = [
        FrequencyBand(name="77-79GHz", start_hz=77e9, stop_hz=79e9, points=512),
        FrequencyBand(name="79-81GHz", start_hz=79e9, stop_hz=81e9, points=512),
    ]
    return request


def _scene(data: dict[str, Any]) -> SceneDesign:
    scene = SceneDesign(
        name=str(data.get("name", "Outdoor6D_w_car")),
        scenario_path=Path(data.get("scenario_path", "scenarios/Outdoor6D_w_car.xml")),
        description=str(data.get("description", "Calibrated Outdoor 6D digital twin")),
        base_stations=[_bs(item) for item in data.get("base_stations", [])],
        user_equipments=[_ue(item) for item in data.get("user_equipments", [])],
        objects=[_object(item) for item in data.get("objects", [])],
        radiomap=_radiomap(data.get("radiomap", {})),
    )
    return scene


def _panel(data: dict[str, Any], default: AntennaPanel) -> AntennaPanel:
    pattern = sionna_pattern_name(str(data.get("pattern", data.get("element_diagram", default.pattern))))
    element_diagram = sionna_pattern_name(str(data.get("element_diagram", pattern)))
    return AntennaPanel(
        rows=int(data.get("rows", default.rows)),
        cols=int(data.get("cols", default.cols)),
        pattern=pattern,
        polarization=str(data.get("polarization", default.polarization)),
        vertical_spacing_m=float(data.get("vertical_spacing_m", default.vertical_spacing_m)),
        horizontal_spacing_m=float(data.get("horizontal_spacing_m", default.horizontal_spacing_m)),
        element_diagram=element_diagram,
        orientation_rad=_vector3(data.get("orientation_rad", default.orientation_rad)),
        v_pol_vector=_vector3(data.get("v_pol_vector", default.v_pol_vector)),
        h_pol_vector=_optional_vector3(data.get("h_pol_vector", default.h_pol_vector)),
    )


def _trajectory(data: dict[str, Any]) -> TrajectorySpec:
    return TrajectorySpec(
        kind=str(data.get("kind", "static")),
        points=[_vector3(point) for point in data.get("points", [(0.0, 0.0, 0.0)])],
        bezier_handles=[
            (_vector3(pair[0]), _vector3(pair[1]))
            for pair in data.get("bezier_handles", [])
        ],
        orientation_rad_points=[
            _vector3(point)
            for point in data.get("orientation_rad_points", data.get("orientations", []))
        ],
        samples=int(data.get("samples", 1)),
        start_static_fraction=float(data.get("start_static_fraction", 0.0)),
        end_static_fraction=float(data.get("end_static_fraction", 0.0)),
        easing=str(data.get("easing", "linear")),
    )


def _bs(data: dict[str, Any]) -> BaseStation:
    return BaseStation(
        id=str(data["id"]),
        position=_vector3(data.get("position", (0.0, 0.0, 2.0))),
        orientation_rad=_vector3(data.get("orientation_rad", (0.0, 0.0, 0.0))),
        panel=_panel(data.get("panel", {}), AntennaPanel.default_bs()),
    )


def _ue(data: dict[str, Any]) -> UserEquipment:
    position = _vector3(data.get("position", (0.0, 0.0, 1.5)))
    return UserEquipment(
        id=str(data["id"]),
        position=position,
        orientation_rad=_vector3(data.get("orientation_rad", (0.0, 0.0, 0.0))),
        panel=_panel(data.get("panel", {}), AntennaPanel.default_ue()),
        trajectory=_trajectory(data.get("trajectory", _static_trajectory_data(position))),
    )


def _object(data: dict[str, Any]) -> DynamicObject:
    position = _vector3(data.get("position", (0.0, 0.0, 0.0)))
    return DynamicObject(
        id=str(data["id"]),
        object_name=str(data.get("object_name", "CAR_obj")),
        position=position,
        orientation_rad=_vector3(data.get("orientation_rad", (0.0, 0.0, 0.0))),
        trajectory=_trajectory(data.get("trajectory", _static_trajectory_data(position))),
    )


def _radiomap(data: dict[str, Any]) -> RadiomapConfig:
    template_data = data.get("ue_template", {"id": "rm_ue"})
    return RadiomapConfig(
        enabled=bool(data.get("enabled", False)),
        x_min=float(data.get("x_min", -10.0)),
        x_max=float(data.get("x_max", 10.0)),
        y_min=float(data.get("y_min", -10.0)),
        y_max=float(data.get("y_max", 10.0)),
        x_spacing=float(data.get("x_spacing", 1.0)),
        y_spacing=float(data.get("y_spacing", 1.0)),
        height=float(data.get("height", 1.5)),
        ue_template=_ue(template_data),
    )


def _band(data: dict[str, Any]) -> FrequencyBand:
    return FrequencyBand(
        name=str(data.get("name", "band")),
        start_hz=float(data.get("start_hz", 77e9)),
        stop_hz=float(data.get("stop_hz", 81e9)),
        points=int(data.get("points", 1024)),
    )


def _sionna(data: dict[str, Any]) -> SionnaConfig:
    return SionnaConfig(
        tx_power_dbm=float(data.get("tx_power_dbm", 44.0)),
        samples_per_src=int(data.get("samples_per_src", 500_000)),
        max_depth=int(data.get("max_depth", 3)),
        max_num_paths_per_src=_optional_int(data.get("max_num_paths_per_src", 100_000)),
        seed=_optional_int(data.get("seed")),
        los=bool(data.get("los", True)),
        specular_reflection=bool(data.get("specular_reflection", True)),
        diffuse_reflection=bool(data.get("diffuse_reflection", True)),
        refraction=bool(data.get("refraction", False)),
        synthetic_array=bool(data.get("synthetic_array", False)),
        merge_shapes=bool(data.get("merge_shapes", False)),
        use_gpu=bool(data.get("use_gpu", True)),
        batch_timeframes=int(data.get("batch_timeframes", 1)),
        max_timeframes=int(data.get("max_timeframes", 100_000)),
        ue_ue_links=bool(data.get("ue_ue_links", False)),
    )


def _vector3(value: Any) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"Expected a 3D vector, got {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))


def _optional_vector3(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    return _vector3(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _static_trajectory_data(position: tuple[float, float, float]) -> dict[str, Any]:
    return {"kind": "static", "points": [position], "samples": 1}

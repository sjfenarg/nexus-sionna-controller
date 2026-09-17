from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from isac_6d_sampler.core.config_io import read_request, write_request
from isac_6d_sampler.core.model import (
    AntennaPanel,
    BaseStation,
    FrequencyBand,
    SceneDesign,
    SimulationRequest,
    TrajectorySpec,
    UserEquipment,
)
from isac_6d_sampler.core.validation import validate_request

ROOT = Path(__file__).resolve().parents[1]
H5_PATH = ROOT / "meas" / "nexus_trials.h5"
OUT_PATH = ROOT / "configs" / "hdf5_antennas_detailed_6d_powerlog_ues.json"
SCENARIO_PATH = Path("scenarios/Detailed 6D Map/Detailed 6D Map.xml")
BS_POINTING_TARGET_LATLON = (39.478980, -0.338875)
BS_HEIGHT_M = 12.5
BS_POINTING_TARGET_HEIGHT_M = 0.0

OSM_BOUNDS = {
    "minlat": 39.4774000,
    "minlon": -0.3442000,
    "maxlat": 39.4842000,
    "maxlon": -0.3348000,
}


def latlon_to_scene_xy(lat: float, lon: float) -> tuple[float, float]:
    lat0 = (OSM_BOUNDS["minlat"] + OSM_BOUNDS["maxlat"]) / 2.0
    lon0 = (OSM_BOUNDS["minlon"] + OSM_BOUNDS["maxlon"]) / 2.0
    meters_per_lat = 111_320.0
    meters_per_lon = meters_per_lat * np.cos(np.deg2rad(lat0))
    return (lon - lon0) * meters_per_lon, (lat - lat0) * meters_per_lat


def point_to_orientation(src_xyz: tuple[float, float, float], dst_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    delta = np.asarray(dst_xyz, dtype=np.float64) - np.asarray(src_xyz, dtype=np.float64)
    horizontal = float(np.hypot(delta[0], delta[1]))
    yaw = float(np.arctan2(delta[1], delta[0]))
    # GUI/Sionna local +X boresight after yaw-pitch-roll is
    # [cos(yaw)cos(pitch), sin(yaw)cos(pitch), -sin(pitch)].
    pitch = float(np.arctan2(-delta[2], max(horizontal, 1e-12)))
    return yaw, pitch, 0.0


def opposite_xy_preserve_tilt(orientation: tuple[float, float, float]) -> tuple[float, float, float]:
    yaw, pitch, roll = orientation
    yaw = float((yaw + np.pi + np.pi) % (2.0 * np.pi) - np.pi)
    return yaw, pitch, roll


def spacing_attr_to_meters(value: float) -> float:
    # The measurement HDF5 stores spacing in millimeters. The GUI/Sionna model
    # expects meters, so 10.0 means 1 cm.
    return float(value) / 1000.0


def _measurement_frequency_bands() -> list[FrequencyBand]:
    with h5py.File(H5_PATH, "r") as f:
        group = f["scenarios/nexus_trials/s001/parameters/subband_f_vectors"]
        bands = []
        for idx, name in enumerate(sorted(group.keys())):
            values = group[name][...].reshape(-1)
            start_hz = float(values[0])
            stop_hz = float(values[-1])
            step_hz = float(values[1] - values[0]) if values.size > 1 else 0.0
            center_ghz = 0.5 * (start_hz + stop_hz) / 1e9
            bandwidth_mhz = (stop_hz - start_hz) / 1e6
            bands.append(
                FrequencyBand(
                    name=f"h5_subband_{idx}_{center_ghz:.2f}GHz_{bandwidth_mhz:.2f}MHz_step_{step_hz / 1e3:.0f}kHz",
                    start_hz=start_hz,
                    stop_hz=stop_hz,
                    points=int(values.size),
                )
            )
        return bands


def main() -> None:
    with h5py.File(H5_PATH, "r") as f:
        group = f["scenarios/nexus_trials/s001/parameters/antenna_params"]
        raw = {}
        for name in sorted(group.keys()):
            lat, lon = group[name]["ant_coord"][...]
            x, y = latlon_to_scene_xy(float(lat), float(lon))
            raw[name] = {
                "lat": float(lat),
                "lon": float(lon),
                "x": float(x),
                "y": float(y),
                "attrs": dict(group[name].attrs),
                "v_pol_vector": tuple(float(v) for v in group[name]["v_pol_vector"][...]),
            }

    bs_data = raw["bs0"]
    bs_position = (bs_data["x"], bs_data["y"], BS_HEIGHT_M)
    bs_attrs = bs_data["attrs"]
    bs_panel = AntennaPanel(
        rows=int(bs_attrs.get("num_rows", 1)),
        cols=int(bs_attrs.get("num_cols", 10)),
        pattern="qom_st_2_18_omni",
        element_diagram="qom_st_2_18_omni",
        polarization="V",
        vertical_spacing_m=spacing_attr_to_meters(float(bs_attrs.get("vertical_spacing", 0.0))),
        horizontal_spacing_m=spacing_attr_to_meters(float(bs_attrs.get("horizontal_spacing", 0.0))),
        v_pol_vector=bs_data["v_pol_vector"],
    )
    bs_target_x, bs_target_y = latlon_to_scene_xy(*BS_POINTING_TARGET_LATLON)
    bs = BaseStation(
        id="bs0",
        position=bs_position,
        orientation_rad=point_to_orientation(bs_position, (bs_target_x, bs_target_y, BS_POINTING_TARGET_HEIGHT_M)),
        panel=bs_panel,
    )

    ues: list[UserEquipment] = []
    for ue_name in sorted((name for name in raw if name.startswith("ue")), key=lambda s: int(s[2:])):
        ue_idx = int(ue_name[2:])
        ue_data = raw[ue_name]
        height = 1.16 if ue_idx % 2 == 0 else 1.0
        position = (ue_data["x"], ue_data["y"], height)
        orientation = point_to_orientation(position, bs_position)
        if ue_idx in {6, 7}:
            orientation = opposite_xy_preserve_tilt(orientation)
        attrs = ue_data["attrs"]
        panel = AntennaPanel(
            rows=int(attrs.get("num_rows", 1)),
            cols=int(attrs.get("num_cols", 1)),
            pattern="powerLog",
            element_diagram="powerLog",
            polarization="V",
            vertical_spacing_m=spacing_attr_to_meters(float(attrs.get("vertical_spacing", 0.0))),
            horizontal_spacing_m=spacing_attr_to_meters(float(attrs.get("horizontal_spacing", 0.0))),
            v_pol_vector=ue_data["v_pol_vector"],
        )
        ues.append(
            UserEquipment(
                id=ue_name,
                position=position,
                orientation_rad=orientation,
                panel=panel,
                trajectory=TrajectorySpec.static(position),
            )
        )

    request = SimulationRequest(
        scene=SceneDesign(
            name="Detailed 6D Map - HDF5 antennas",
            scenario_path=SCENARIO_PATH,
            description=(
                "Detailed 6D Map loaded with antenna positions from meas/nexus_trials.h5. "
                "BS uses the measured QOM-ST-2-18 omni pattern; UEs use the measured powerLog pattern. "
                "UE6 and UE7 use opposite XY yaw while preserving the BS-pointing tilt."
            ),
            base_stations=[bs],
            user_equipments=ues,
            objects=[],
        ),
        bands=_measurement_frequency_bands(),
        sample_id="hdf5_antennas_powerlog",
        output_dir=Path("output/hdf5_antennas_powerlog"),
        dry_run=False,
    )
    validate_request(request)
    write_request(OUT_PATH, request)

    summary_path = OUT_PATH.with_suffix(".positions_summary.json")
    summary = {
        "source_h5": str(H5_PATH.relative_to(ROOT)),
        "config": str(OUT_PATH.relative_to(ROOT)),
        "scenario_path": str(SCENARIO_PATH),
        "coordinate_frame": "SceneBaker real-world XY meters converted from HDF5 lat/lon; z in meters",
        "height_policy": "BS=12.5m; even UE indices=1.16m; odd UE indices=1.0m",
        "bs_pattern": "qom_st_2_18_omni (QOM-ST-2-18-S-SG-R measured omni, 18 GHz cuts)",
        "ue_pattern": "powerLog",
        "bs_pointing_target": {
            "lat_lon": list(BS_POINTING_TARGET_LATLON),
            "scene_xyz": [bs_target_x, bs_target_y, BS_POINTING_TARGET_HEIGHT_M],
            "orientation_rad": list(bs.orientation_rad),
            "orientation_deg": list(np.rad2deg(np.asarray(bs.orientation_rad, dtype=np.float64))),
        },
        "ue6_ue7_orientation": "opposite XY yaw from BS-pointing direction, same pitch/tilt as BS-pointing direction",
        "devices": {
            "bs0": {"position": list(bs.position), "orientation_rad": list(bs.orientation_rad), "panel": asdict(bs.panel)},
            **{
                ue.id: {
                    "position": list(ue.position),
                    "orientation_rad": list(ue.orientation_rad),
                    "orientation_deg": list(np.rad2deg(np.asarray(ue.orientation_rad, dtype=np.float64))),
                    "pattern": ue.panel.pattern,
                }
                for ue in ues
            },
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Read it back to prove the GUI/config parser accepts it.
    validate_request(read_request(OUT_PATH))
    print(OUT_PATH)
    print(summary_path)


if __name__ == "__main__":
    main()

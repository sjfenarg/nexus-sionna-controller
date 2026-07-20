import json

import h5py
import numpy as np
import pytest

from isac_6d_sampler.cli import main
from isac_6d_sampler.core.antenna_patterns import ISAC_HORN_PATTERN
from isac_6d_sampler.core.config_io import read_request, template_request, write_request
from isac_6d_sampler.core.model import (
    AntennaPanel,
    BaseStation,
    ChannelMode,
    DynamicObject,
    FrequencyBand,
    RadiomapConfig,
    SimulationRequest,
    TrajectorySpec,
    UserEquipment,
)
from isac_6d_sampler.io.reference_h5 import ReferenceH5Writer
from isac_6d_sampler.sim.dry_run import DryRunSimulator


def test_request_json_round_trip(tmp_path):
    request = template_request()
    request.channel_mode = ChannelMode.COHERENT_PER_BIN
    request.sionna.seed = 123
    request.sionna.tx_power_dbm = 17.5
    request.sionna.max_num_paths_per_src = 2048
    request.sionna.los = False
    request.sionna.specular_reflection = False
    request.sionna.refraction = True
    request.sionna.synthetic_array = True
    request.sionna.merge_shapes = True
    request.sionna.use_gpu = False
    request.sionna.max_timeframes = 4096
    request.scene.objects[0].trajectory.orientation_rad_points = [(0.0, 0.0, 0.0), (0.1, 0.2, 0.3)]
    path = tmp_path / "request.json"

    write_request(path, request)
    loaded = read_request(path)

    assert loaded.channel_mode == ChannelMode.COHERENT_PER_BIN
    assert len(loaded.bands) == 2
    assert loaded.scene.objects[0].trajectory.samples == 16
    assert loaded.scene.objects[0].trajectory.orientation_rad_points == [(0.0, 0.0, 0.0), (0.1, 0.2, 0.3)]
    assert loaded.scene.base_stations[0].panel.rows == 10
    assert loaded.sionna.seed == 123
    assert loaded.sionna.tx_power_dbm == 17.5
    assert loaded.sionna.max_num_paths_per_src == 2048
    assert not loaded.sionna.los
    assert not loaded.sionna.specular_reflection
    assert loaded.sionna.refraction
    assert loaded.sionna.synthetic_array
    assert loaded.sionna.merge_shapes
    assert not loaded.sionna.use_gpu
    assert loaded.sionna.max_timeframes == 4096


def test_radiomap_config_round_trip_preserves_bounds_and_spacing(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.radiomap = RadiomapConfig(
        enabled=True,
        x_min=-2.0,
        x_max=4.0,
        y_min=-3.0,
        y_max=5.0,
        x_spacing=0.5,
        y_spacing=0.75,
        height=1.25,
    )
    path = tmp_path / "radiomap_request.json"

    write_request(path, request)
    loaded = read_request(path)

    assert loaded.scene.radiomap.enabled
    assert loaded.scene.radiomap.x_min == -2.0
    assert loaded.scene.radiomap.y_max == 5.0
    assert loaded.scene.radiomap.x_spacing == 0.5
    assert loaded.scene.radiomap.y_spacing == 0.75
    assert loaded.scene.radiomap.height == 1.25


def test_curve_trajectory_round_trip_preserves_bezier_handles(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    request.scene.user_equipments[0].trajectory = TrajectorySpec(
        kind="curve",
        points=[(0.0, 0.0, 1.5), (10.0, 0.0, 1.5)],
        bezier_handles=[
            ((0.0, 0.0, 1.5), (0.0, 5.0, 1.5)),
            ((10.0, 5.0, 1.5), (10.0, 0.0, 1.5)),
        ],
        samples=12,
    )
    path = tmp_path / "curve_request.json"

    write_request(path, request)
    loaded = read_request(path)

    trajectory = loaded.scene.user_equipments[0].trajectory
    assert trajectory.kind == "curve"
    assert trajectory.bezier_handles[0][1] == (0.0, 5.0, 1.5)


def test_rich_scene_request_round_trip_preserves_gui_design_state(tmp_path):
    request = SimulationRequest(dry_run=False, output_dir=tmp_path / "out", sample_id="rich")
    request.scene.name = "rich_scene"
    request.scene.scenario_path = tmp_path / "rich_scene.xml"
    request.scene.description = "multi entity GUI design"
    request.channel_mode = ChannelMode.FREQUENCY_DOMAIN
    request.bands = [
        FrequencyBand(name="low", start_hz=77e9, stop_hz=78e9, points=64),
        FrequencyBand(name="high", start_hz=80e9, stop_hz=81e9, points=128),
    ]
    request.sionna.tx_power_dbm = 44.0
    request.sionna.samples_per_src = 500_000
    request.sionna.max_depth = 3
    request.sionna.max_num_paths_per_src = None
    request.sionna.seed = 42
    request.sionna.synthetic_array = False

    bs0 = BaseStation(
        id="bs0",
        position=(52.353, -19.516, 19.203),
        orientation_rad=(0.1, 0.2, 0.3),
        panel=AntennaPanel(
            rows=10,
            cols=10,
            pattern=ISAC_HORN_PATTERN,
            element_diagram=ISAC_HORN_PATTERN,
            polarization="V",
            vertical_spacing_m=0.0019,
            horizontal_spacing_m=0.0019,
            orientation_rad=(0.1, 0.2, 0.3),
        ),
    )
    bs1 = BaseStation(
        id="bs1",
        position=(0.0, 20.0, 12.0),
        orientation_rad=(-0.2, 0.1, 0.0),
        panel=AntennaPanel(rows=4, cols=8, pattern="iso", element_diagram="iso", polarization="VH"),
    )

    ue0 = UserEquipment(
        id="ue0",
        position=(0.0, 0.0, 1.5),
        orientation_rad=(0.0, 0.0, 0.0),
        panel=AntennaPanel(rows=1, cols=1, pattern=ISAC_HORN_PATTERN, element_diagram=ISAC_HORN_PATTERN),
        trajectory=TrajectorySpec(
            kind="linear",
            points=[(0.0, 0.0, 1.5), (10.0, 0.0, 1.5)],
            orientation_rad_points=[(0.0, 0.0, 0.0), (0.0, 0.1, 0.0)],
            samples=25,
        ),
    )
    ue1 = UserEquipment(
        id="ue1",
        position=(1.0, 1.0, 1.5),
        orientation_rad=(0.0, 0.0, 0.4),
        panel=AntennaPanel(rows=2, cols=2, pattern="dipole", element_diagram="dipole", polarization="H"),
        trajectory=TrajectorySpec(
            kind="curve",
            points=[(1.0, 1.0, 1.5), (5.0, 4.0, 1.5), (9.0, 1.0, 1.5)],
            bezier_handles=[
                ((1.0, 1.0, 1.5), (2.0, 3.0, 1.5)),
                ((4.0, 5.0, 1.5), (6.0, 5.0, 1.5)),
                ((8.0, 3.0, 1.5), (9.0, 1.0, 1.5)),
            ],
            samples=25,
        ),
    )

    car = DynamicObject(
        id="car0",
        object_name="CAR_obj",
        position=(0.0, 0.0, 0.75),
        orientation_rad=(0.0, 0.0, 0.2),
        trajectory=TrajectorySpec(
            kind="polyline",
            points=[(0.0, 0.0, 0.75), (3.0, 1.0, 0.75), (6.0, 1.0, 0.75)],
            orientation_rad_points=[(0.0, 0.0, 0.2), (0.0, 0.0, 0.4), (0.0, 0.0, 0.6)],
            samples=25,
        ),
    )
    drone = DynamicObject(
        id="drone0",
        object_name="DRONE_obj",
        position=(-2.0, 1.0, 1.2),
        trajectory=TrajectorySpec.linear((-2.0, 1.0, 1.2), (-2.0, 4.0, 1.2), 25),
    )

    request.scene.base_stations = [bs0, bs1]
    request.scene.user_equipments = [ue0, ue1]
    request.scene.objects = [car, drone]
    request.scene.radiomap = RadiomapConfig(
        enabled=True,
        x_min=-5.0,
        x_max=5.0,
        y_min=-4.0,
        y_max=6.0,
        x_spacing=1.0,
        y_spacing=1.0,
        height=1.5,
        ue_template=UserEquipment(
            id="rm_ue",
            panel=AntennaPanel(rows=1, cols=1, pattern=ISAC_HORN_PATTERN, element_diagram=ISAC_HORN_PATTERN),
        ),
    )
    path = tmp_path / "rich_request.json"

    write_request(path, request)
    loaded = read_request(path)

    assert loaded.scene.name == "rich_scene"
    assert loaded.scene.scenario_path == tmp_path / "rich_scene.xml"
    assert [bs.id for bs in loaded.scene.base_stations] == ["bs0", "bs1"]
    assert [ue.id for ue in loaded.scene.user_equipments] == ["ue0", "ue1"]
    assert [obj.id for obj in loaded.scene.objects] == ["car0", "drone0"]
    assert loaded.scene.base_stations[0].position == (52.353, -19.516, 19.203)
    assert loaded.scene.base_stations[0].panel.pattern == ISAC_HORN_PATTERN
    assert loaded.scene.base_stations[0].panel.rows == 10
    assert loaded.scene.base_stations[1].panel.polarization == "VH"
    assert loaded.scene.user_equipments[0].trajectory.kind == "linear"
    assert loaded.scene.user_equipments[0].trajectory.orientation_rad_points[1] == (0.0, 0.1, 0.0)
    assert loaded.scene.user_equipments[1].trajectory.kind == "curve"
    assert loaded.scene.user_equipments[1].trajectory.bezier_handles[1][0] == (4.0, 5.0, 1.5)
    assert loaded.scene.objects[0].object_name == "CAR_obj"
    assert loaded.scene.objects[0].trajectory.points[2] == (6.0, 1.0, 0.75)
    assert loaded.scene.objects[1].object_name == "DRONE_obj"
    assert loaded.scene.radiomap.enabled
    assert loaded.scene.radiomap.ue_template.panel.pattern == ISAC_HORN_PATTERN
    assert loaded.sionna.tx_power_dbm == 44.0
    assert loaded.sionna.max_num_paths_per_src is None
    assert loaded.bands[1].points == 128


def test_multi_band_reference_export(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    request.bands = [
        FrequencyBand(name="low", start_hz=77e9, stop_hz=78e9, points=4),
        FrequencyBand(name="high", start_hz=80e9, stop_hz=81e9, points=5),
    ]
    result = DryRunSimulator().simulate(request)
    out = tmp_path / "multi_band.h5"
    ReferenceH5Writer().write(out, request, result)

    with h5py.File(out, "r") as h5:
        params = h5[f"scenarios/{request.scene.name}/{request.sample_id}/parameters"]
        assert params["f_vector"].shape == (1, 9)
        assert params["subband_f_vectors/subband_f_vector_0"].shape == (1, 4)
        assert params["subband_f_vectors/subband_f_vector_1"].shape == (1, 5)
        assert params["vna_params"].attrs["n_subbands"] == 2


def test_cli_template_and_multiband_dry_run(tmp_path, capsys):
    config_path = tmp_path / "template.json"
    output_dir = tmp_path / "out"

    assert main(["--write-template", str(config_path)]) == 0
    assert config_path.exists()

    assert (
        main(
            [
                "--config",
                str(config_path),
                "--dry-run",
                "--output-dir",
                str(output_dir),
                "--band",
                "77:78:4:low",
                "--band",
                "80:81:5:high",
            ]
        )
        == 0
    )
    assert len(list(output_dir.glob("scene_*.h5"))) == 1
    captured = capsys.readouterr()
    assert "Writing and validating HDF5 output" in captured.out


def test_cli_plan_only_reports_request_size_without_writing_output(tmp_path, capsys):
    output_dir = tmp_path / "out"

    assert (
        main(
            [
                "--plan-only",
                "--output-dir",
                str(output_dir),
                "--dry-run",
                "--radiomap",
                "on",
                "--rm-x-min",
                "0",
                "--rm-x-max",
                "1",
                "--rm-y-min",
                "0",
                "--rm-y-max",
                "0",
                "--rm-x-spacing",
                "1",
                "--rm-y-spacing",
                "1",
                "--object",
                "car0:CAR_obj:0,0,0",
                "--object-trajectory",
                "car0:linear:2:0,0,0;1,0,0",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert "radiomap grid: 2 x 1" in captured.out
    assert "timeframes: 4" in captured.out
    assert "links/timeframe: 3" in captured.out
    assert not output_dir.exists()


def test_cli_sionna_overrides_are_recorded_in_hdf5_request_metadata(tmp_path):
    output_dir = tmp_path / "out"

    assert (
        main(
            [
                "--dry-run",
                "--output-dir",
                str(output_dir),
                "--samples-per-src",
                "7",
                "--tx-power-dbm",
                "18.5",
                "--max-num-paths-per-src",
                "11",
                "--max-depth",
                "2",
                "--seed",
                "99",
                "--los",
                "off",
                "--specular",
                "off",
                "--diffuse",
                "off",
                "--refraction",
                "on",
                "--synthetic-array",
                "on",
                "--merge-shapes",
                "on",
                "--use-gpu",
                "off",
                "--batch-timeframes",
                "4",
                "--max-timeframes",
                "1234",
            ]
        )
        == 0
    )

    out = next(output_dir.glob("scene_*.h5"))
    with h5py.File(out, "r") as h5:
        sample = h5["scenarios/Outdoor6D_w_car/s000"]
        request_json = sample["metadata"].attrs["request_json"]
        request_data = json.loads(request_json)

    sionna = request_data["sionna"]
    assert sionna["samples_per_src"] == 7
    assert sionna["tx_power_dbm"] == 18.5
    assert sionna["max_num_paths_per_src"] == 11
    assert sionna["max_depth"] == 2
    assert sionna["seed"] == 99
    assert not sionna["los"]
    assert not sionna["specular_reflection"]
    assert not sionna["diffuse_reflection"]
    assert sionna["refraction"]
    assert sionna["synthetic_array"]
    assert sionna["merge_shapes"]
    assert not sionna["use_gpu"]
    assert sionna["batch_timeframes"] == 4
    assert sionna["max_timeframes"] == 1234


def test_cli_max_paths_unlimited_override_is_recorded_as_null(tmp_path):
    output_dir = tmp_path / "out"

    assert (
        main(
            [
                "--dry-run",
                "--output-dir",
                str(output_dir),
                "--max-num-paths-per-src",
                "unlimited",
            ]
        )
        == 0
    )

    out = next(output_dir.glob("scene_*.h5"))
    with h5py.File(out, "r") as h5:
        sample = h5["scenarios/Outdoor6D_w_car/s000"]
        request_data = json.loads(sample["metadata"].attrs["request_json"])
        assert request_data["sionna"]["max_num_paths_per_src"] is None
        assert sample["parameters/sionna_params"].attrs["max_num_paths_per_src"] == "null"


def test_cli_antenna_pattern_overrides_are_recorded(tmp_path):
    output_dir = tmp_path / "out"

    assert (
        main(
            [
                "--dry-run",
                "--output-dir",
                str(output_dir),
                "--bs-pattern",
                ISAC_HORN_PATTERN,
                "--ue-pattern",
                ISAC_HORN_PATTERN,
            ]
        )
        == 0
    )

    out = next(output_dir.glob("scene_*.h5"))
    with h5py.File(out, "r") as h5:
        sample = h5["scenarios/Outdoor6D_w_car/s000"]
        request_data = json.loads(sample["metadata"].attrs["request_json"])
        assert request_data["scene"]["base_stations"][0]["panel"]["pattern"] == ISAC_HORN_PATTERN
        assert request_data["scene"]["user_equipments"][0]["panel"]["pattern"] == ISAC_HORN_PATTERN
        assert sample["parameters/antenna_params/bs0"].attrs["pattern"] == ISAC_HORN_PATTERN
        assert sample["parameters/antenna_params/ue0"].attrs["pattern"] == ISAC_HORN_PATTERN


def test_cli_radiomap_and_object_overrides_generate_reference_timeframes(tmp_path):
    output_dir = tmp_path / "out"

    assert (
        main(
            [
                "--dry-run",
                "--output-dir",
                str(output_dir),
                "--radiomap",
                "on",
                "--rm-x-min",
                "0",
                "--rm-x-max",
                "1",
                "--rm-y-min",
                "0",
                "--rm-y-max",
                "0",
                "--rm-x-spacing",
                "1",
                "--rm-y-spacing",
                "1",
                "--rm-height",
                "1.25",
                "--object",
                "car0:CAR_obj:0,0,0:0,0,0.5",
                "--object-trajectory",
                "car0:linear:2:0,0,0;1,0,0",
            ]
        )
        == 0
    )

    out = next(output_dir.glob("radiomap_*.h5"))
    with h5py.File(out, "r") as h5:
        sample = h5["scenarios/Outdoor6D_w_car/s000"]
        request_data = json.loads(sample["metadata"].attrs["request_json"])
        assert request_data["scene"]["radiomap"]["enabled"]
        assert request_data["scene"]["radiomap"]["x_spacing"] == 1.0
        assert request_data["scene"]["user_equipments"] == []
        assert request_data["scene"]["objects"][0]["id"] == "car0"
        assert request_data["scene"]["objects"][0]["trajectory"]["samples"] == 2
        assert len(sample["timeframes"]) == 4
        assert sample["parameters"].attrs["n_ue"] == 1
        np.testing.assert_allclose(
            sample["timeframes/tf003/positions/devices/ue_radiomap"][:],
            [1.0, 0.0, 1.25],
        )
        np.testing.assert_allclose(
            sample["timeframes/tf003/positions/objects/car0"][:],
            [1.0, 0.0, 0.0],
        )


def test_cli_bs_ue_overrides_generate_multi_bs_scene(tmp_path):
    output_dir = tmp_path / "out"

    assert (
        main(
            [
                "--dry-run",
                "--output-dir",
                str(output_dir),
                "--bs",
                "bs0:0,0,3:0,0,0:2x3",
                "--bs",
                "bs1:5,0,3:0,0,0:1x2",
                "--ue",
                "ue0:0,0,1.5:0,0,0:1x1",
                "--bs-spacing",
                "0.25,0.5",
                "--ue-spacing",
                "0.1,0.2",
                "--ue-trajectory",
                "ue0:linear:3:0,0,1.5;2,0,1.5",
            ]
        )
        == 0
    )

    out = next(output_dir.glob("scene_*.h5"))
    with h5py.File(out, "r") as h5:
        sample = h5["scenarios/Outdoor6D_w_car/s000"]
        request_data = json.loads(sample["metadata"].attrs["request_json"])
        assert [bs["id"] for bs in request_data["scene"]["base_stations"]] == ["bs0", "bs1"]
        assert [ue["id"] for ue in request_data["scene"]["user_equipments"]] == ["ue0"]
        assert request_data["scene"]["base_stations"][0]["panel"]["rows"] == 2
        assert request_data["scene"]["base_stations"][0]["panel"]["cols"] == 3
        assert request_data["scene"]["base_stations"][0]["panel"]["vertical_spacing_m"] == 0.25
        assert request_data["scene"]["base_stations"][0]["panel"]["horizontal_spacing_m"] == 0.5
        assert request_data["scene"]["base_stations"][1]["panel"]["cols"] == 2
        assert request_data["scene"]["user_equipments"][0]["panel"]["vertical_spacing_m"] == 0.1
        assert request_data["scene"]["user_equipments"][0]["panel"]["horizontal_spacing_m"] == 0.2
        assert request_data["scene"]["user_equipments"][0]["trajectory"]["samples"] == 3
        assert sample["parameters"].attrs["n_bs"] == 2
        assert sample["parameters"].attrs["n_ue"] == 1
        assert sample["parameters/antenna_params/bs0"].attrs["vertical_spacing"] == 0.25
        assert sample["parameters/antenna_params/bs0"].attrs["horizontal_spacing"] == 0.5
        assert sample["parameters/antenna_params/ue0"].attrs["vertical_spacing"] == 0.1
        assert sample["parameters/antenna_params/ue0"].attrs["horizontal_spacing"] == 0.2
        assert len(sample["timeframes"]) == 3
        assert sample["timeframes/tf002/parameters"].attrs["n_channels"] == 7
        links = set(sample["timeframes/tf002/h"].keys())
        assert {"rx0_tx0", "rx0_tx1", "rx1_tx0", "rx0_tx2", "rx2_tx0", "rx1_tx1", "rx2_tx2"} == links
        assert sample["timeframes/tf002/h/rx1_tx1"].shape[:-1] == (2, 3, 1, 1, 1)
        assert sample["timeframes/tf002/h/rx2_tx2"].shape[:-1] == (1, 2, 1, 1, 1)
        np.testing.assert_allclose(
            sample["timeframes/tf002/positions/devices/ue0"][:],
            [2.0, 0.0, 1.5],
        )


def test_cli_ue_trajectory_can_target_default_ue(tmp_path):
    output_dir = tmp_path / "out"

    assert (
        main(
            [
                "--dry-run",
                "--output-dir",
                str(output_dir),
                "--ue-trajectory",
                "ue0:linear:3:0,0,1.5;2,0,1.5",
            ]
        )
        == 0
    )

    out = next(output_dir.glob("scene_*.h5"))
    with h5py.File(out, "r") as h5:
        sample = h5["scenarios/Outdoor6D_w_car/s000"]
        request_data = json.loads(sample["metadata"].attrs["request_json"])
        assert [ue["id"] for ue in request_data["scene"]["user_equipments"]] == ["ue0"]
        assert request_data["scene"]["user_equipments"][0]["trajectory"]["samples"] == 3
        assert len(sample["timeframes"]) == 3
        np.testing.assert_allclose(
            sample["timeframes/tf002/positions/devices/ue0"][:],
            [2.0, 0.0, 1.5],
        )


def test_cli_rejects_invalid_request_before_output(tmp_path):
    output_dir = tmp_path / "out"

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--dry-run",
                "--output-dir",
                str(output_dir),
                "--radiomap",
                "on",
                "--rm-x-min",
                "2",
                "--rm-x-max",
                "1",
            ]
        )

    assert exc_info.value.code == 2
    assert not output_dir.exists()

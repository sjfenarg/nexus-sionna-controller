import json
from datetime import datetime

import h5py
import numpy as np

from isac_6d_sampler.core.antenna_patterns import ISAC_HORN_PATTERN
from isac_6d_sampler.core.model import (
    ChannelMode,
    DynamicObject,
    FrequencyBand,
    RadiomapConfig,
    SimulationRequest,
    TrajectorySpec,
)
from isac_6d_sampler.io.reference_h5 import ReferenceH5Writer, default_output_path
from isac_6d_sampler.sim.channel import IFFT_GRIDDED_DELAY_OVERSAMPLING
from isac_6d_sampler.sim.dry_run import DryRunSimulator


def test_reference_writer_matches_expected_topology(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    request.scene.base_stations[0].panel.pattern = ISAC_HORN_PATTERN
    request.scene.base_stations[0].panel.element_diagram = ISAC_HORN_PATTERN
    request.scene.base_stations[0].orientation_rad = (np.pi / 2.0, np.pi, 0.25)
    request.sionna.tx_power_dbm = 20.0
    result = DryRunSimulator().simulate(request)
    out = tmp_path / "scene_test.h5"
    ReferenceH5Writer().write(out, request, result)

    with h5py.File(out, "r") as h5:
        sample = h5["scenarios"][request.scene.name][request.sample_id]
        assert "metadata" in sample
        assert "parameters/f_vector" in sample
        assert "parameters/antenna_params/bs0" in sample
        assert "parameters/antenna_params/ue0" in sample
        bs = sample["parameters/antenna_params/bs0"]
        assert bs.attrs["pattern"] == ISAC_HORN_PATTERN
        assert bs.attrs["pattern_source"] == "isac_journal"
        assert bs.attrs["pattern_max_gain_db"] == 10.3
        assert bs.attrs["h_3db_beamwidth_deg"] == 56.0
        assert bs.attrs["vertical_spacing_units"] == "meter"
        assert bs.attrs["horizontal_spacing_units"] == "meter"
        assert bs.attrs["ant_orientation_units"] == "degrees"
        assert bs.attrs["ant_orientation_rad_units"] == "radians"
        np.testing.assert_allclose(bs["ant_orientation"][:], [90.0, 180.0])
        np.testing.assert_allclose(bs["ant_orientation_rad"][:], [np.pi / 2.0, np.pi, 0.25])
        material_params = sample["parameters/material_params"]
        assert sample["parameters/sionna_params"].attrs["tx_power_dbm"] == 20.0
        assert sample["parameters/channel_params"].attrs["tx_power_dbm"] == 20.0
        np.testing.assert_allclose(sample["parameters/channel_params"].attrs["tx_power_w"], 0.1)
        assert material_params.attrs["source"] == "isac_journal"
        assert material_params.attrs["n_materials"] >= 1
        assert material_params["material0"].attrs["object_prefix"] == "BRICKS"
        device_params = sample["parameters/device_params"]
        assert device_params.attrs["index_order"] == "ues_first_then_base_stations"
        assert device_params.attrs["n_devices"] == 2
        assert device_params["device0"].attrs["entity_id"] == "ue0"
        assert device_params["device0"].attrs["entity_type"] == "ue"
        assert device_params["device0"].attrs["antenna_group"] == "antenna_params/ue0"
        assert device_params["device1"].attrs["entity_id"] == "bs0"
        assert device_params["device1"].attrs["entity_type"] == "bs"
        assert device_params["device1"].attrs["antenna_group"] == "antenna_params/bs0"
        assert "timeframes/tf000/h" in sample
        links = list(sample["timeframes/tf000/h"].keys())
        assert "rx0_tx0" in links
        assert "rx1_tx0" in links
        assert "rx0_tx1" in links
        assert sample["timeframes/tf000/h/rx0_tx1"].dtype.kind == "c"
        channel = sample["timeframes/tf000/h/rx0_tx1"]
        assert channel.attrs["dataset_role"] == "channel"
        assert channel.attrs["dataset_name"] == "rx0_tx1"
        assert channel.attrs["rx_id"] == "ue0"
        assert channel.attrs["tx_id"] == "bs0"
        assert channel.attrs["rx_index"] == 0
        assert channel.attrs["tx_index"] == 1
        assert channel.attrs["is_monostatic"] == np.False_
        assert channel.attrs["sample_axis"] == "frequency_hz"
        assert json.loads(channel.attrs["axis_labels"]) == [
            "rx_row",
            "rx_col",
            "tx_row",
            "tx_col",
            "polarization",
            "frequency_hz",
        ]
        timestamp = sample["timeframes/tf000/timestamps/rx0_tx1"]
        assert timestamp.attrs["dataset_role"] == "timestamp"
        assert timestamp.attrs["rx_id"] == "ue0"
        assert timestamp.attrs["tx_id"] == "bs0"
        assert json.loads(timestamp.attrs["axis_labels"]) == [
            "rx_row",
            "rx_col",
            "tx_row",
            "tx_col",
            "polarization",
        ]
        link_params = sample["parameters/link_params"]
        assert link_params.attrs["n_links"] == len(links)
        assert link_params["rx0_tx0"].attrs["direction"] == "monostatic"
        assert link_params["rx0_tx1"].attrs["rx_id"] == "ue0"
        assert link_params["rx0_tx1"].attrs["tx_id"] == "bs0"
        assert link_params["rx0_tx1"].attrs["rx_device_group"] == "device0"
        assert link_params["rx0_tx1"].attrs["tx_device_group"] == "device1"
        assert link_params["rx0_tx1"].attrs["rx_entity_type"] == "ue"
        assert link_params["rx0_tx1"].attrs["tx_entity_type"] == "bs"
        assert link_params["rx0_tx1"].attrs["direction"] == "bs_to_ue"
        assert link_params["rx1_tx0"].attrs["direction"] == "ue_to_bs"


def test_reference_writer_exports_timeframe_positions_for_trajectories(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    request.scene.base_stations[0].orientation_rad = (0.1, 0.2, 0.3)
    request.scene.user_equipments[0].orientation_rad = (0.4, 0.5, 0.6)
    request.scene.user_equipments[0].trajectory = TrajectorySpec.linear(
        (0.0, 0.0, 1.5),
        (2.0, 0.0, 1.5),
        3,
    )
    request.scene.objects.append(
        DynamicObject(
            id="car0",
            position=(0.0, 0.0, 0.0),
            orientation_rad=(1.0, 0.0, 0.5),
            trajectory=TrajectorySpec.linear((0.0, 0.0, 0.0), (0.0, 2.0, 0.0), 3),
        )
    )
    result = DryRunSimulator().simulate(request)
    out = tmp_path / "positions.h5"
    ReferenceH5Writer().write(out, request, result)

    with h5py.File(out, "r") as h5:
        sample = h5[f"scenarios/{request.scene.name}/{request.sample_id}"]
        assert "parameters/object_params/object0" in sample
        assert sample["parameters/object_params/object0"].attrs["material_prefix"] == "CAR"
        assert sample["parameters/object_params/object0"].attrs["material_name"] == "my_car_metal"
        assert sample["timeframes/tf002"].attrs["frame_kind"] == "scene"
        assert sample["timeframes/tf002"].attrs["scene_frame_index"] == 2
        assert sample["timeframes/tf002"].attrs["scene_frame_count"] == 3
        np.testing.assert_allclose(sample["timeframes/tf002/positions/devices/ue0"][:], [2.0, 0.0, 1.5])
        np.testing.assert_allclose(
            sample["timeframes/tf002/positions/devices/bs0"][:],
            request.scene.base_stations[0].position,
        )
        np.testing.assert_allclose(sample["timeframes/tf002/positions/objects/car0"][:], [0.0, 2.0, 0.0])
        np.testing.assert_allclose(
            sample["timeframes/tf002/orientations/devices/ue0"][:],
            [0.4, 0.5, 0.6],
        )
        np.testing.assert_allclose(
            sample["timeframes/tf002/orientations/devices/bs0"][:],
            [0.1, 0.2, 0.3],
        )
        np.testing.assert_allclose(
            sample["timeframes/tf002/orientations/objects/car0"][:],
            [1.0, 0.0, 0.5],
        )
        np.testing.assert_allclose(
            sample["parameters/trajectory_params/ues/ue0/control_points"][:],
            [[0.0, 0.0, 1.5], [2.0, 0.0, 1.5]],
        )
        assert sample["parameters/trajectory_params/ues/ue0"].attrs["kind"] == "linear"
        np.testing.assert_allclose(
            sample["parameters/trajectory_params/objects/object0/control_points"][:],
            [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        )
        assert sample["parameters/trajectory_params/objects/object0"].attrs["object_name"] == "CAR_obj"


def test_reference_writer_exports_radiomap_positions(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.radiomap = RadiomapConfig(
        enabled=True,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        x_spacing=1.0,
        y_spacing=1.0,
        height=1.25,
    )
    result = DryRunSimulator().simulate(request)
    out = tmp_path / "radiomap_positions.h5"
    ReferenceH5Writer().write(out, request, result)

    assert request.scene.user_equipments == []

    with h5py.File(out, "r") as h5:
        sample = h5[f"scenarios/{request.scene.name}/{request.sample_id}"]
        assert len(sample["timeframes"]) == 4
        assert sample["parameters"].attrs["n_ue"] == 1
        assert "parameters/antenna_params/ue0" in sample
        assert sample["parameters/radiomap_params"].attrs["ue_template_id"] == "ue_radiomap"
        request_json = json.loads(sample["metadata"].attrs["request_json"])
        assert request_json["scene"]["user_equipments"] == []
        tf003 = sample["timeframes/tf003"]
        assert tf003.attrs["frame_kind"] == "radiomap"
        assert tf003.attrs["radiomap_grid_index"] == 3
        assert tf003.attrs["radiomap_x_index"] == 1
        assert tf003.attrs["radiomap_y_index"] == 1
        assert tf003.attrs["radiomap_total_grid_points"] == 4
        assert tf003.attrs["radiomap_object_state_index"] == 0
        np.testing.assert_allclose(
            tf003["positions/devices/ue_radiomap"][:],
            [1.0, 1.0, 1.25],
        )
        np.testing.assert_allclose(
            tf003["positions/devices/bs0"][:],
            request.scene.base_stations[0].position,
        )
        radiomap = sample["parameters/radiomap_params"]
        assert bool(radiomap.attrs["enabled"]) is True
        assert radiomap.attrs["x_points"] == 2
        assert radiomap.attrs["y_points"] == 2
        np.testing.assert_allclose(radiomap["x_coordinates"][:], [0.0, 1.0])
        np.testing.assert_allclose(radiomap["y_coordinates"][:], [0.0, 1.0])
        np.testing.assert_allclose(
            radiomap["positions"][:],
            [[0.0, 0.0, 1.25], [1.0, 0.0, 1.25], [0.0, 1.0, 1.25], [1.0, 1.0, 1.25]],
        )


def test_reference_writer_exports_curve_bezier_handles(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    request.scene.user_equipments[0].trajectory = TrajectorySpec(
        kind="curve",
        points=[(0.0, 0.0, 1.5), (10.0, 0.0, 1.5)],
        bezier_handles=[
            ((0.0, 0.0, 1.5), (0.0, 5.0, 1.5)),
            ((10.0, 5.0, 1.5), (10.0, 0.0, 1.5)),
        ],
        samples=4,
    )
    result = DryRunSimulator().simulate(request)
    out = tmp_path / "curve.h5"
    ReferenceH5Writer().write(out, request, result)

    with h5py.File(out, "r") as h5:
        group = h5[f"scenarios/{request.scene.name}/{request.sample_id}/parameters/trajectory_params/ues/ue0"]
        assert group.attrs["kind"] == "curve"
        np.testing.assert_allclose(
            group["bezier_handles"][:],
            [
                [[0.0, 0.0, 1.5], [0.0, 5.0, 1.5]],
                [[10.0, 5.0, 1.5], [10.0, 0.0, 1.5]],
            ],
        )


def test_reference_writer_exports_radiomap_with_object_trajectory_positions(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.radiomap = RadiomapConfig(
        enabled=True,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=0.0,
        x_spacing=1.0,
        y_spacing=1.0,
        height=1.25,
    )
    request.scene.objects.append(
        DynamicObject(
            id="car0",
            position=(0.0, 0.0, 0.0),
            orientation_rad=(0.0, 0.0, 0.5),
            trajectory=TrajectorySpec.linear((0.0, 0.0, 0.0), (0.0, 2.0, 0.0), 2),
        )
    )
    result = DryRunSimulator().simulate(request)
    out = tmp_path / "radiomap_object_positions.h5"
    ReferenceH5Writer().write(out, request, result)

    with h5py.File(out, "r") as h5:
        sample = h5[f"scenarios/{request.scene.name}/{request.sample_id}"]
        assert len(sample["timeframes"]) == 4
        assert sample["timeframes/tf002"].attrs["radiomap_object_state_index"] == 1
        assert sample["timeframes/tf002"].attrs["radiomap_grid_index"] == 0
        np.testing.assert_allclose(
            sample["timeframes/tf001/positions/devices/ue_radiomap"][:],
            [1.0, 0.0, 1.25],
        )
        np.testing.assert_allclose(
            sample["timeframes/tf001/positions/objects/car0"][:],
            [0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            sample["timeframes/tf003/positions/objects/car0"][:],
            [0.0, 2.0, 0.0],
        )
        np.testing.assert_allclose(
            sample["timeframes/tf003/orientations/objects/car0"][:],
            [0.0, 0.0, 0.5],
        )


def test_reference_writer_exports_frequency_and_sionna_parameters(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    request.bands = [
        FrequencyBand(name="low", start_hz=77e9, stop_hz=78e9, points=4),
        FrequencyBand(name="high", start_hz=80e9, stop_hz=81e9, points=3),
    ]
    request.sionna.samples_per_src = 123
    request.sionna.max_num_paths_per_src = None
    request.sionna.diffuse_reflection = False
    result = DryRunSimulator().simulate(request)
    out = tmp_path / "params.h5"
    ReferenceH5Writer().write(out, request, result)

    with h5py.File(out, "r") as h5:
        sample = h5[f"scenarios/{request.scene.name}/{request.sample_id}"]
        bands = sample["parameters/frequency_bands"]
        assert bands["band0"].attrs["name"] == "low"
        assert bands["band0"].attrs["points"] == 4
        assert bands["band1"].attrs["start_hz"] == 80e9
        sionna = sample["parameters/sionna_params"]
        assert sionna.attrs["samples_per_src"] == 123
        assert sionna.attrs["diffuse_reflection"] == np.False_
        assert sionna.attrs["max_num_paths_per_src"] == "null"


def test_reference_writer_exports_frequency_channel_axis_metadata(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    request.bands = [FrequencyBand(name="tiny", start_hz=77e9, stop_hz=77.3e9, points=4)]
    result = DryRunSimulator().simulate(request)
    out = tmp_path / "channel_params_frequency.h5"
    ReferenceH5Writer().write(out, request, result)

    with h5py.File(out, "r") as h5:
        sample = h5[f"scenarios/{request.scene.name}/{request.sample_id}"]
        channel = sample["parameters/channel_params"]
        assert channel.attrs["mode"] == "frequency_domain"
        assert channel.attrs["sample_axis"] == "frequency_hz"
        assert channel.attrs["frequency_points"] == 4
        np.testing.assert_allclose(channel["sample_coordinates"][:], request.bands[0].vector())
        assert "tau" not in sample["timeframes/tf000"]
        assert "a" not in sample["timeframes/tf000"]


def test_reference_writer_exports_delay_bin_channel_axis_metadata(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    request.channel_mode = ChannelMode.COHERENT_PER_BIN
    request.bands = [FrequencyBand(name="tiny", start_hz=77e9, stop_hz=77.4e9, points=5)]
    result = DryRunSimulator().simulate(request)
    out = tmp_path / "channel_params_delay.h5"
    ReferenceH5Writer().write(out, request, result)

    with h5py.File(out, "r") as h5:
        sample = h5[f"scenarios/{request.scene.name}/{request.sample_id}"]
        channel = sample["parameters/channel_params"]
        assert channel.attrs["mode"] == "coherent_per_bin"
        assert channel.attrs["sample_axis"] == "delay_bin_s"
        assert channel.attrs["value_quantity"] == "complex_delay_response"
        assert channel.attrs["frequency_points"] == 5
        assert channel.attrs["delay_bin_width_s"] == 1.0 / (0.4e9)
        np.testing.assert_allclose(
            channel["sample_coordinates"][:],
            np.arange(5, dtype=np.float64) / 0.4e9,
        )


def test_reference_writer_labels_pdp_binned_values_as_linear_power(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    request.channel_mode = ChannelMode.PDP_BINNED
    request.bands = [FrequencyBand(name="tiny", start_hz=77e9, stop_hz=77.4e9, points=5)]
    result = DryRunSimulator().simulate(request)
    out = tmp_path / "channel_params_pdp_binned.h5"
    ReferenceH5Writer().write(out, request, result)

    with h5py.File(out, "r") as h5:
        sample = h5[f"scenarios/{request.scene.name}/{request.sample_id}"]
        channel = sample["parameters/channel_params"]
        assert channel.attrs["mode"] == "pdp_binned"
        assert channel.attrs["sample_axis"] == "delay_bin_s"
        assert channel.attrs["value_quantity"] == "linear_power"


def test_reference_writer_exports_ifft_renderer_metadata(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    request.channel_mode = ChannelMode.PDP_IFFT_GRIDDED
    request.bands = [FrequencyBand(name="tiny", start_hz=77e9, stop_hz=77.4e9, points=5)]
    result = DryRunSimulator().simulate(request)
    out = tmp_path / "channel_params_pdp_ifft_gridded.h5"
    ReferenceH5Writer().write(out, request, result)

    with h5py.File(out, "r") as h5:
        sample = h5[f"scenarios/{request.scene.name}/{request.sample_id}"]
        channel = sample["parameters/channel_params"]
        assert channel.attrs["mode"] == "pdp_ifft_gridded"
        assert channel.attrs["sample_axis"] == "delay_bin_s"
        assert channel.attrs["value_quantity"] == "complex_delay_response"
        assert channel.attrs["ifft_renderer"] == "oversampled_delay_grid"
        assert channel.attrs["delay_oversampling"] == IFFT_GRIDDED_DELAY_OVERSAMPLING


def test_reference_writer_exports_cir_path_delays(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    request.channel_mode = ChannelMode.CIR_PATHS
    request.scene.user_equipments[0].position = (0.0, 0.0, 1.5)
    request.scene.user_equipments[0].trajectory = TrajectorySpec.static((0.0, 0.0, 1.5))
    request.scene.base_stations[0].position = (3.0, 0.0, 1.5)
    request.bands = [FrequencyBand(name="tiny", start_hz=77e9, stop_hz=77.1e9, points=2)]
    result = DryRunSimulator().simulate(request)
    out = tmp_path / "cir_paths.h5"
    ReferenceH5Writer().write(out, request, result)

    with h5py.File(out, "r") as h5:
        sample = h5[f"scenarios/{request.scene.name}/{request.sample_id}"]
        channel = sample["parameters/channel_params"]
        assert channel.attrs["mode"] == "cir_paths"
        assert channel.attrs["sample_axis"] == "path_index"
        h = sample["timeframes/tf000/h/rx0_tx1"]
        tau = sample["timeframes/tf000/tau/rx0_tx1"]
        assert h.shape == (1, 1, 10, 10, 1, 1)
        assert tau.shape == (1, 1, 10, 10, 1, 1)
        assert tau.attrs["dataset_role"] == "path_delay"
        assert tau.attrs["rx_id"] == "ue0"
        assert tau.attrs["tx_id"] == "bs0"
        assert tau.attrs["sample_axis"] == "path_index"
        assert tau.attrs["value_units"] == "seconds"
        assert json.loads(tau.attrs["axis_labels"])[-1] == "path_index"
        np.testing.assert_allclose(tau[0, 0, 0, 0, 0, 0], 3.0 / 299_792_458.0)


def test_default_output_path_uses_timestamped_scene_name(tmp_path):
    request = SimulationRequest(output_dir=tmp_path)
    now = datetime(2026, 7, 15, 12, 30, 45)

    path = default_output_path(request, now=now)

    assert path == tmp_path / "scene_20260715_123045.h5"


def test_default_output_path_adds_suffix_instead_of_overwriting(tmp_path):
    request = SimulationRequest(output_dir=tmp_path)
    now = datetime(2026, 7, 15, 12, 30, 45)
    (tmp_path / "scene_20260715_123045.h5").write_text("existing", encoding="utf-8")
    (tmp_path / "scene_20260715_123045_001.h5").write_text("existing", encoding="utf-8")

    path = default_output_path(request, now=now)

    assert path == tmp_path / "scene_20260715_123045_002.h5"


def test_default_output_path_uses_radiomap_prefix(tmp_path):
    request = SimulationRequest(output_dir=tmp_path)
    request.scene.radiomap.enabled = True
    now = datetime(2026, 7, 15, 12, 30, 45)

    path = default_output_path(request, now=now)

    assert path == tmp_path / "radiomap_20260715_123045.h5"

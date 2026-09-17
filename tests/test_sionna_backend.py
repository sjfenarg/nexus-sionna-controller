import sys
import types

import numpy as np
import pytest

import isac_6d_sampler.sim.sionna_backend as sionna_backend
from isac_6d_sampler.core.model import (
    BaseStation,
    ChannelMode,
    DEFAULT_CENTER_FREQUENCY_HZ,
    DEFAULT_HALF_WAVELENGTH_SPACING_M,
    DynamicObject,
    RadiomapConfig,
    SceneDesign,
    SimulationRequest,
    TrajectorySpec,
    UserEquipment,
)
from isac_6d_sampler.sim.planner import LinkPlan, TimeframePlan, build_simulation_plan
from isac_6d_sampler.sim.dry_run import DryRunSimulator
from isac_6d_sampler.sim.results import LinkResult
from isac_6d_sampler.sim.sionna_backend import (
    SionnaSimulator,
    TimeframeLinkRef,
    TimeframeLinkBatch,
    _append_cached_radiomap_links,
    _build_link_batches,
    _build_timeframe_link_batches,
    _configure_mitsuba_variant,
    _external_object_mesh_paths,
    _extract_path_vertices_for_link,
    _last_completed_timeframe_index,
    _next_batchable_timeframe_chunk,
    _offset_monostatic_tx_position,
    _filtered_scene_xml,
    _object_mesh_path,
    _paths_cfr_chunked,
    _paths_to_link_data,
    _radiomap_bs_monostatic_cache_key,
    _reshape_h_for_reference,
    _scene_object_name_for_dynamic_object,
    _should_offset_monostatic_tx,
    _sionna_array_pattern,
    _slice_link_array,
    _spacing_m_to_wavelengths,
    _transpose_reciprocal_link_data,
    _tx_is_only_monostatic_in_batch,
)


def test_reshape_h_for_reference_restores_panel_rows_and_cols():
    ue = UserEquipment(id="ue0")
    bs = BaseStation(id="bs0")
    link = LinkPlan(rx_index=0, tx_index=1, rx=ue, tx=bs)
    sionna_h = np.zeros((1, 1, 1, 100, 2), dtype=np.complex64)

    reshaped = _reshape_h_for_reference(sionna_h, link)

    assert reshaped.shape == (1, 1, 10, 10, 1, 2)


def test_unconfigured_car_object_is_filtered_from_sionna_xml_before_loading(tmp_path):
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text(
        """<scene version="2.1.0">
        <shape type="ply" id="WALL_obj" name="WALL_obj"><string name="filename" value="wall.ply"/></shape>
        <shape type="ply" id="CAR_obj" name="CAR_obj"><string name="filename" value="car.ply"/></shape>
        </scene>""",
        encoding="utf-8",
    )

    filtered = _filtered_scene_xml(scene_xml, {"CAR_obj"}).decode("utf-8")

    assert "CAR_obj" not in filtered
    assert "WALL_obj" in filtered


def test_configured_car_object_is_kept_in_sionna_xml(tmp_path):
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text(
        """<scene version="2.1.0">
        <shape type="ply" id="CAR_obj" name="CAR_obj"><string name="filename" value="car.ply"/></shape>
        </scene>""",
        encoding="utf-8",
    )

    filtered = _filtered_scene_xml(scene_xml, set()).decode("utf-8")

    assert "CAR_obj" in filtered


def test_apply_object_positions_moves_configured_car_in_sionna_scene():
    car_shape = types.SimpleNamespace(position=None, orientation=None)
    scene = types.SimpleNamespace(objects={"CAR_obj": car_shape})
    design = SceneDesign(
        objects=[
            DynamicObject(
                id="car0",
                object_name="CAR_obj",
                position=(0.0, 0.0, 0.75),
                orientation_rad=(0.0, 0.0, 0.0),
            )
        ]
    )
    timeframe = TimeframePlan(
        name="tf000",
        device_positions={},
        object_positions={"car0": (2.0, 3.0, 0.75)},
        device_orientations={},
        object_orientations={"car0": (0.1, 0.2, 0.3)},
        links=(),
    )

    SionnaSimulator()._apply_object_positions(scene, design, timeframe)

    assert car_shape.position == [2.0, 3.0, 0.75]
    assert car_shape.orientation == [0.1, 0.2, 0.3]


def test_apply_object_positions_moves_external_object_by_instance_id():
    drone_shape = types.SimpleNamespace(position=None, orientation=None)
    scene = types.SimpleNamespace(objects={"drone0": drone_shape})
    design = SceneDesign(
        objects=[
            DynamicObject(
                id="drone0",
                object_name="DRONE_obj",
                position=(0.0, 0.0, 0.75),
                orientation_rad=(0.0, 0.0, 0.0),
            )
        ]
    )
    timeframe = TimeframePlan(
        name="tf000",
        device_positions={},
        object_positions={"drone0": (2.0, 3.0, 4.0)},
        device_orientations={},
        object_orientations={"drone0": (0.1, 0.2, 0.3)},
        links=(),
    )

    SionnaSimulator()._apply_object_positions(scene, design, timeframe)

    assert drone_shape.position == [2.0, 3.0, 4.0]
    assert drone_shape.orientation == [0.1, 0.2, 0.3]


def test_external_obj_meshes_are_added_to_sionna_scene(monkeypatch, tmp_path):
    scenario = tmp_path / "scene.xml"
    object_dir = tmp_path / "objects"
    object_dir.mkdir()
    mesh_path = object_dir / "DRONE_obj.obj"
    mesh_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    scenario.write_text("<scene />\n", encoding="utf-8")
    fake_rt = _FakeObjectSionnaRt()
    monkeypatch.setitem(sys.modules, "sionna", types.ModuleType("sionna"))
    monkeypatch.setitem(sys.modules, "sionna.rt", fake_rt)
    scene = _FakeEditableScene()
    design = SceneDesign(
        scenario_path=scenario,
        objects=[
            DynamicObject(id="drone0", object_name="DRONE_obj"),
            DynamicObject(id="car0", object_name="CAR_obj"),
        ],
    )

    SionnaSimulator()._add_external_object_meshes(scene, design)

    assert fake_rt.loaded_meshes == [str(mesh_path)]
    assert list(scene.objects) == ["drone0"]
    added = scene.objects["drone0"]
    assert added.mi_mesh == {"path": str(mesh_path)}
    assert added.radio_material.name == "drone0-mat"
    assert added.radio_material.conductivity == 0.01
    assert added.radio_material.relative_permittivity == 5.0


def test_external_object_mesh_path_helpers(tmp_path):
    scenario = tmp_path / "scene.xml"
    object_dir = tmp_path / "objects"
    object_dir.mkdir()
    mesh_path = object_dir / "DRONE_obj.obj"
    mesh_path.write_text("v 0 0 0\n", encoding="utf-8")
    scenario.write_text("<scene />\n", encoding="utf-8")
    design = SceneDesign(
        scenario_path=scenario,
        objects=[
            DynamicObject(id="drone0", object_name="DRONE_obj"),
            DynamicObject(id="car0", object_name="CAR_obj"),
        ],
    )

    assert _object_mesh_path(object_dir, "DRONE_obj") == mesh_path
    assert _object_mesh_path(object_dir, "DRONE_obj.obj") == mesh_path
    assert _external_object_mesh_paths(design) == [(design.objects[0], mesh_path)]
    assert _scene_object_name_for_dynamic_object(types.SimpleNamespace(objects={"drone0": object()}), design.objects[0]) == "drone0"


def test_build_link_batches_groups_default_two_ue_two_bs_scene():
    scene = SceneDesign(
        user_equipments=[UserEquipment(id="ue0"), UserEquipment(id="ue1")],
        base_stations=[BaseStation(id="bs0"), BaseStation(id="bs1")],
    )
    plan = build_simulation_plan(scene)

    batches = _build_link_batches(plan.timeframes[0].links, request_los=True)

    assert len(batches) == 4
    assert sorted(len(batch.links) for batch in batches) == [2, 2, 4, 4]
    assert sum(1 for batch in batches if batch.los) == 2


def test_timeframe_batches_group_static_object_radiomap_timeframes():
    scene = SceneDesign(
        base_stations=[BaseStation(id="bs0")],
        radiomap=RadiomapConfig(
            enabled=True,
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=0.0,
            x_spacing=1.0,
            y_spacing=1.0,
            height=1.5,
        ),
    )
    plan = build_simulation_plan(scene)

    chunk = _next_batchable_timeframe_chunk(plan.timeframes, start=0, requested_size=8)
    batches = _build_timeframe_link_batches(chunk, start_index=0, request_los=True)

    assert len(chunk) == 2
    assert sorted(len(batch.refs) for batch in batches) == [2, 2, 4]
    assert sorted(len(batch.rx_devices) for batch in batches) == [1, 2, 2]
    assert sorted(len(batch.tx_devices) for batch in batches) == [1, 1, 2]


def test_timeframe_batches_split_same_pose_devices_on_same_side():
    scene = SceneDesign(
        base_stations=[BaseStation(id="bs0", position=(0.0, 0.0, 15.0))],
        user_equipments=[
            UserEquipment(id="ue0", position=(0.0, 0.0, 0.5)),
            UserEquipment(id="ue1", position=(0.0, 0.0, 0.5)),
        ],
    )
    plan = build_simulation_plan(scene)

    batches = _build_timeframe_link_batches(plan.timeframes, start_index=0, request_los=True)

    reciprocal_batches = [
        batch
        for batch in batches
        if batch.los
        and {ref.link.rx.id for ref in batch.refs} <= {"ue0", "ue1", "bs0"}
        and {ref.link.tx.id for ref in batch.refs} <= {"ue0", "ue1", "bs0"}
    ]

    assert sorted(len(batch.refs) for batch in reciprocal_batches) == [2, 2]
    assert sorted([device[2].id for device in batch.rx_devices] for batch in reciprocal_batches) == [["ue0"], ["ue1"]]
    assert all([device[2].id for device in batch.tx_devices] == ["bs0"] for batch in reciprocal_batches)


def test_batched_progress_reports_last_completed_timeframe_index():
    scene = SceneDesign(
        base_stations=[BaseStation(id="bs0")],
        radiomap=RadiomapConfig(
            enabled=True,
            x_min=0.0,
            x_max=2.0,
            y_min=0.0,
            y_max=0.0,
            x_spacing=1.0,
            y_spacing=1.0,
            height=1.5,
        ),
    )
    plan = build_simulation_plan(scene)
    chunk = _next_batchable_timeframe_chunk(plan.timeframes, start=0, requested_size=3)

    assert len(chunk) == 3
    assert _last_completed_timeframe_index(0, chunk) == 2


def test_timeframe_batches_stop_when_object_state_changes():
    scene = SceneDesign(
        base_stations=[BaseStation(id="bs0")],
        user_equipments=[UserEquipment(id="ue0")],
        objects=[
            DynamicObject(
                id="car0",
                trajectory=TrajectorySpec.linear((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 2),
            )
        ],
    )
    plan = build_simulation_plan(scene)

    chunk = _next_batchable_timeframe_chunk(plan.timeframes, start=0, requested_size=8)

    assert len(chunk) == 1


def test_radiomap_timeframe_batches_stop_after_fixed_object_pose_grid():
    scene = SceneDesign(
        base_stations=[BaseStation(id="bs0")],
        objects=[
            DynamicObject(
                id="car0",
                trajectory=TrajectorySpec.linear((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 2),
            )
        ],
        radiomap=RadiomapConfig(
            enabled=True,
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=0.0,
            x_spacing=1.0,
            y_spacing=1.0,
            height=1.5,
        ),
    )
    plan = build_simulation_plan(scene)

    chunk = _next_batchable_timeframe_chunk(plan.timeframes, start=0, requested_size=8)

    assert len(plan.timeframes) == 4
    assert len(chunk) == 2
    assert {frame.object_positions["car0"] for frame in chunk} == {(0.0, 0.0, 0.0)}


def test_radiomap_bs_monostatic_cache_reuses_constant_scene_link():
    scene = SceneDesign(
        base_stations=[BaseStation(id="bs0")],
        radiomap=RadiomapConfig(
            enabled=True,
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=0.0,
            x_spacing=1.0,
            y_spacing=1.0,
            height=1.5,
        ),
    )
    plan = build_simulation_plan(scene)
    first_ref = next(
        TimeframeLinkRef(0, plan.timeframes[0], link)
        for link in plan.timeframes[0].links
        if link.dataset_name == "rx1_tx1"
    )
    second_ref = next(
        TimeframeLinkRef(1, plan.timeframes[1], link)
        for link in plan.timeframes[1].links
        if link.dataset_name == "rx1_tx1"
    )
    cached = LinkResult(
        rx_index=1,
        tx_index=1,
        rx_id="bs0",
        tx_id="bs0",
        h=np.ones((10, 10, 1, 1, 1, 2), dtype=np.complex64),
        metadata={"path_delays_s": np.ones((10, 10, 1, 1, 1, 1), dtype=np.float64)},
    )
    cache = {_radiomap_bs_monostatic_cache_key(first_ref): cached}
    frame_links = [[], []]
    batch = TimeframeLinkBatch(
        refs=(second_ref,),
        rx_devices=((1, plan.timeframes[1], second_ref.link.rx),),
        tx_devices=((1, plan.timeframes[1], second_ref.link.tx),),
        los=False,
    )

    assert _radiomap_bs_monostatic_cache_key(first_ref) == _radiomap_bs_monostatic_cache_key(second_ref)
    assert _append_cached_radiomap_links(frame_links, batch, start_index=0, radiomap_link_cache=cache)
    assert frame_links[1][0].rx_id == "bs0"
    assert frame_links[1][0].tx_id == "bs0"
    np.testing.assert_array_equal(frame_links[1][0].h, cached.h)
    assert frame_links[1][0].h is not cached.h
    assert frame_links[1][0].metadata["path_delays_s"] is not cached.metadata["path_delays_s"]


def test_slice_link_array_extracts_rx_tx_pair_from_sionna_layout():
    values = np.arange(2 * 3 * 4 * 5 * 6 * 1).reshape(2, 3, 4, 5, 6, 1)

    sliced = _slice_link_array(values, rx_index=1, tx_index=2)

    assert sliced.shape == (3, 5, 6, 1)
    np.testing.assert_array_equal(sliced, values[1, :, 2, :, :, :])


def test_paths_cfr_chunked_splits_large_frequency_vectors():
    paths = _FakePaths()
    tau = np.zeros((1, 2, 1, 3, 5), dtype=np.float64)
    frequencies = np.arange(11, dtype=np.float64)

    h = _paths_cfr_chunked(paths, tau, frequencies, max_entries=60)

    assert h.shape == (1, 2, 1, 3, 1, 11)
    assert paths.calls == [frequencies[0:2].tolist(), frequencies[2:4].tolist(), frequencies[4:6].tolist(), frequencies[6:8].tolist(), frequencies[8:10].tolist(), frequencies[10:11].tolist()]
    np.testing.assert_allclose(h[0, 0, 0, 0, 0], frequencies)


def test_frequency_domain_link_data_uses_cfr_and_caches_cir_for_path_viewer():
    request = SimulationRequest()
    request.channel_mode = ChannelMode.FREQUENCY_DOMAIN
    request.sionna.tx_power_dbm = 0.0
    ue = UserEquipment(id="ue0")
    bs = BaseStation(id="bs0")
    link = LinkPlan(rx_index=0, tx_index=1, rx=ue, tx=bs)
    paths = _FakeCfrOnlyPaths()

    h, path_delays, path_coefficients, path_vertices = _paths_to_link_data(
        paths,
        np.asarray([77e9, 78e9], dtype=np.float64),
        request,
        link,
    )

    assert h.shape == (1, 1, 10, 10, 1, 2)
    assert path_delays.shape == (1, 1, 10, 10, 1, 1)
    assert path_coefficients.shape == (1, 1, 10, 10, 1, 1)
    assert path_vertices is None
    assert paths.cfr_kwargs["normalize_delays"] is False
    assert paths.cfr_kwargs["normalize"] is False


def test_default_bs_spacing_maps_to_half_wavelength_for_sionna_arrays():
    assert _spacing_m_to_wavelengths(
        DEFAULT_HALF_WAVELENGTH_SPACING_M,
        DEFAULT_CENTER_FREQUENCY_HZ,
    ) == pytest.approx(0.5)


def test_configure_arrays_converts_meter_spacing_to_wavelength_multiples():
    scene = _FakeScene()
    ue = UserEquipment(id="ue0")
    bs = BaseStation(id="bs0")
    bs.panel.vertical_spacing_m = DEFAULT_HALF_WAVELENGTH_SPACING_M
    bs.panel.horizontal_spacing_m = DEFAULT_HALF_WAVELENGTH_SPACING_M * 2.0
    link = LinkPlan(rx_index=0, tx_index=1, rx=ue, tx=bs)

    SionnaSimulator()._configure_arrays(
        scene,
        _FakePlanarArray,
        link,
        frequency_hz=DEFAULT_CENTER_FREQUENCY_HZ,
    )

    assert scene.tx_array.kwargs["vertical_spacing"] == pytest.approx(0.5)
    assert scene.tx_array.kwargs["horizontal_spacing"] == pytest.approx(1.0)
    assert scene.rx_array.kwargs["vertical_spacing"] == pytest.approx(0.0)
    assert scene.rx_array.kwargs["horizontal_spacing"] == pytest.approx(0.0)


def test_monostatic_tx_position_is_offset_by_quarter_wavelength():
    offset = _offset_monostatic_tx_position((1.0, 2.0, 3.0), DEFAULT_CENTER_FREQUENCY_HZ)

    assert offset[0] == pytest.approx(1.0 + DEFAULT_HALF_WAVELENGTH_SPACING_M / 2.0)
    assert offset[1:] == (2.0, 3.0)


def test_journal_horn_pattern_is_passed_to_sionna_element_pattern_registry():
    assert _sionna_array_pattern("isac_horn_77_81", 79e9) == "isac_horn_77_81"
    assert _sionna_array_pattern("dipole", 79e9) == "dipole"


def test_horn_links_keep_planned_orientation_without_look_at_override(monkeypatch):
    fake_rt = _FakeSionnaRt()
    monkeypatch.setitem(sys.modules, "sionna", types.ModuleType("sionna"))
    monkeypatch.setitem(sys.modules, "sionna.rt", fake_rt)
    observed_link_data_kwargs = []

    def fake_paths_to_link_data(*_args, **kwargs):
        observed_link_data_kwargs.append(kwargs)
        return (
            np.zeros((1, 1, 1, 1, 1, 2), dtype=np.complex64),
            np.zeros((1, 1, 1, 1, 1, 1), dtype=np.float64),
            np.zeros((1, 1, 1, 1, 1, 1), dtype=np.complex64),
            None,
        )

    monkeypatch.setattr(
        sionna_backend,
        "_paths_to_link_data",
        fake_paths_to_link_data,
    )

    ue = UserEquipment(id="ue0", orientation_rad=(0.1, 0.2, 0.3))
    bs = BaseStation(id="bs0", orientation_rad=(0.4, 0.5, 0.6))
    ue.panel.pattern = "isac_horn_77_81"
    bs.panel.pattern = "isac_horn_77_81"
    link = LinkPlan(rx_index=1, tx_index=0, rx=bs, tx=ue)
    timeframe = TimeframePlan(
        name="tf000",
        device_positions={"ue0": (0.0, 0.0, 1.5), "bs0": (5.0, 0.0, 1.5)},
        object_positions={},
        device_orientations={"ue0": (0.11, 0.22, 0.33), "bs0": (0.44, 0.55, 0.66)},
        object_orientations={},
        links=(link,),
    )
    request = SimulationRequest()
    request.sionna.los = True
    request.sionna.tx_power_dbm = 12.5
    scene = _FakeRtScene()

    SionnaSimulator()._solve_all_links_for_timeframes(
        scene=scene,
        timeframes=(timeframe,),
        start_index=0,
        f_vector=np.asarray([77e9, 81e9], dtype=np.float64),
        request=request,
    )

    assert scene.transmitter_history[0].orientation == [0.44, 0.55, 0.66]
    assert scene.transmitter_history[0].power_dbm == 12.5
    assert scene.receiver_history[0].orientation == [0.11, 0.22, 0.33]
    assert scene.transmitter_history[0].look_at_calls == []
    assert scene.receiver_history[0].look_at_calls == []
    assert "reverse_direction" not in observed_link_data_kwargs[0]


def test_reciprocal_link_data_is_rx_tx_transpose():
    h = np.arange(1 * 2 * 3 * 4 * 1 * 5, dtype=np.float32).reshape(1, 2, 3, 4, 1, 5)
    tau = np.arange(1 * 2 * 3 * 4 * 1 * 2, dtype=np.float64).reshape(1, 2, 3, 4, 1, 2)
    a = (tau + 1j * tau).astype(np.complex64)

    h_t, tau_t, a_t = _transpose_reciprocal_link_data(h, tau, a)

    assert h_t.shape == (3, 4, 1, 2, 1, 5)
    assert tau_t.shape == (3, 4, 1, 2, 1, 2)
    np.testing.assert_array_equal(h_t, np.transpose(h, (2, 3, 0, 1, 4, 5)))
    np.testing.assert_array_equal(tau_t, np.transpose(tau, (2, 3, 0, 1, 4, 5)))
    np.testing.assert_array_equal(a_t, np.transpose(a, (2, 3, 0, 1, 4, 5)))


def test_tx_is_only_monostatic_in_batch_detects_mixed_usage():
    ue = UserEquipment(id="ue0")
    bs = BaseStation(id="bs0")
    frame = build_simulation_plan(SceneDesign(base_stations=[bs], user_equipments=[ue])).timeframes[0]
    mono = LinkPlan(rx_index=0, tx_index=0, rx=ue, tx=ue)
    bi = LinkPlan(rx_index=1, tx_index=0, rx=bs, tx=ue)

    assert _tx_is_only_monostatic_in_batch((TimeframeLinkRef(0, frame, mono),), 0, "ue0")
    assert not _tx_is_only_monostatic_in_batch(
        (TimeframeLinkRef(0, frame, mono), TimeframeLinkRef(0, frame, bi)),
        0,
        "ue0",
    )


def test_bs_monostatic_tx_is_not_offset_from_panel_center():
    ue = UserEquipment(id="ue0")
    bs = BaseStation(id="bs0")
    frame = build_simulation_plan(SceneDesign(base_stations=[bs], user_equipments=[ue])).timeframes[0]
    ue_mono = LinkPlan(rx_index=0, tx_index=0, rx=ue, tx=ue)
    bs_mono = LinkPlan(rx_index=1, tx_index=1, rx=bs, tx=bs)

    assert _should_offset_monostatic_tx((TimeframeLinkRef(0, frame, ue_mono),), 0, ue)
    assert not _should_offset_monostatic_tx((TimeframeLinkRef(0, frame, bs_mono),), 0, bs)


def test_configure_mitsuba_variant_prefers_cuda_when_requested(monkeypatch):
    fake_mi = _FakeMitsuba()
    monkeypatch.setitem(sys.modules, "mitsuba", fake_mi)

    variant = _configure_mitsuba_variant(use_gpu=True)

    assert variant == "cuda_ad_mono_polarized"
    assert fake_mi.calls == [("cuda_ad_mono_polarized", "llvm_ad_mono_polarized")]


def test_configure_mitsuba_variant_uses_cpu_when_gpu_disabled(monkeypatch):
    fake_mi = _FakeMitsuba()
    monkeypatch.setitem(sys.modules, "mitsuba", fake_mi)

    variant = _configure_mitsuba_variant(use_gpu=False)

    assert variant == "llvm_ad_mono_polarized"
    assert fake_mi.calls == [("llvm_ad_mono_polarized",)]


def test_configure_mitsuba_variant_falls_back_to_llvm(monkeypatch):
    fake_mi = _FakeMitsuba(fail_cuda=True)
    monkeypatch.setitem(sys.modules, "mitsuba", fake_mi)

    variant = _configure_mitsuba_variant(use_gpu=True)

    assert variant == "llvm_ad_mono_polarized"
    assert fake_mi.calls == [
        ("cuda_ad_mono_polarized", "llvm_ad_mono_polarized"),
        ("llvm_ad_mono_polarized",),
    ]


def test_dry_run_simulator_validates_request_before_planning_large_radiomap():
    request = SimulationRequest(dry_run=True)
    request.scene.radiomap = RadiomapConfig(
        enabled=True,
        x_min=0.0,
        x_max=1_000_000.0,
        y_min=0.0,
        y_max=0.0,
        x_spacing=1.0,
        y_spacing=1.0,
    )
    request.sionna.max_timeframes = 10

    with pytest.raises(ValueError, match="Estimated timeframe count 1000001"):
        DryRunSimulator().simulate(request)


def test_dry_run_tx_power_scales_channel_power():
    request_low = SimulationRequest(dry_run=True)
    request_low.scene.ensure_defaults()
    request_low.sionna.tx_power_dbm = 20.0
    request_high = SimulationRequest(dry_run=True)
    request_high.scene.ensure_defaults()
    request_high.sionna.tx_power_dbm = 30.0

    low = DryRunSimulator().simulate(request_low).timeframes[0].links[0].h
    high = DryRunSimulator().simulate(request_high).timeframes[0].links[0].h

    low_power = np.mean(np.abs(low) ** 2)
    high_power = np.mean(np.abs(high) ** 2)
    assert high_power / low_power == pytest.approx(10.0)


def test_sionna_simulator_validates_request_before_importing_mitsuba(monkeypatch):
    request = SimulationRequest()
    request.sionna.max_timeframes = 0
    fake_mi = _FakeMitsuba()
    monkeypatch.setitem(sys.modules, "mitsuba", fake_mi)

    with pytest.raises(ValueError, match="sionna.max_timeframes must be at least 1"):
        SionnaSimulator().simulate(request)

    assert fake_mi.calls == []


class _FakeMitsuba(types.ModuleType):
    def __init__(self, fail_cuda=False):
        super().__init__("mitsuba")
        self.fail_cuda = fail_cuda
        self.calls = []
        self._variant = None

    def set_variant(self, *variants):
        self.calls.append(variants)
        if self.fail_cuda and variants[0].startswith("cuda"):
            raise ImportError("cuda unavailable")
        self._variant = variants[0]

    def variant(self):
        return self._variant


class _FakeSionnaRt(types.ModuleType):
    def __init__(self):
        super().__init__("sionna.rt")
        self.PolarizedAntennaPattern = object
        self.PlanarArray = _FakePlanarArray
        self.Receiver = _FakeRtReceiver
        self.Transmitter = _FakeRtTransmitter
        self.PathSolver = _FakePathSolver
        self.registered_patterns = []

    def register_antenna_pattern(self, name, factory):
        self.registered_patterns.append((name, factory))


class _FakeObjectSionnaRt(types.ModuleType):
    def __init__(self):
        super().__init__("sionna.rt")
        self.loaded_meshes = []
        self.RadioMaterial = _FakeObjectRadioMaterial
        self.SceneObject = _FakeSceneObject

    def load_mesh(self, path):
        self.loaded_meshes.append(path)
        return {"path": path}


class _FakeObjectRadioMaterial:
    def __init__(self, *, name, conductivity, relative_permittivity):
        self.name = name
        self.conductivity = conductivity
        self.relative_permittivity = relative_permittivity


class _FakeSceneObject:
    def __init__(self, *, mi_mesh, radio_material, name):
        self.mi_mesh = mi_mesh
        self.radio_material = radio_material
        self.name = name


class _FakeEditableScene:
    def __init__(self):
        self.objects = {}

    def edit(self, *, add):
        for obj in add:
            self.objects[obj.name] = obj


class _FakeRtDevice:
    def __init__(self, *, name, position, orientation, power_dbm=None):
        self.name = name
        self.position = position
        self.orientation = orientation
        self.power_dbm = power_dbm
        self.look_at_calls = []

    def look_at(self, target):
        self.look_at_calls.append(target)


class _FakeRtTransmitter(_FakeRtDevice):
    pass


class _FakeRtReceiver(_FakeRtDevice):
    pass


class _FakePathSolver:
    def __call__(self, **_kwargs):
        return object()


class _FakeRtScene:
    def __init__(self):
        self.transmitters = {}
        self.receivers = {}
        self.transmitter_history = []
        self.receiver_history = []
        self.tx_array = None
        self.rx_array = None

    def add(self, device):
        if isinstance(device, _FakeRtTransmitter):
            self.transmitters[device.name] = device
            self.transmitter_history.append(device)
        elif isinstance(device, _FakeRtReceiver):
            self.receivers[device.name] = device
            self.receiver_history.append(device)
        else:
            raise TypeError(type(device))

    def remove(self, name):
        self.transmitters.pop(name, None)
        self.receivers.pop(name, None)


class _FakeScene:
    tx_array = None
    rx_array = None


class _FakePaths:
    def __init__(self):
        self.calls = []

    def cfr(self, frequencies, **_kwargs):
        frequencies = np.asarray(frequencies, dtype=np.float64)
        self.calls.append(frequencies.tolist())
        return np.broadcast_to(
            frequencies.reshape(1, 1, 1, 1, 1, -1),
            (1, 2, 1, 3, 1, frequencies.size),
        ).astype(np.complex64)


class _FakeCfrOnlyPaths:
    tau = np.zeros((1, 1, 1, 100, 1), dtype=np.float64)

    def __init__(self):
        self.cfr_kwargs = None

    def cir(self, **_kwargs):
        tau = np.zeros((1, 1, 1, 100, 1), dtype=np.float64)
        coefficients = np.ones((1, 1, 1, 100, 1, 1), dtype=np.complex64)
        return coefficients, tau

    def cfr(self, frequencies, **kwargs):
        self.cfr_kwargs = kwargs
        frequencies = np.asarray(frequencies, dtype=np.float64)
        return np.ones((1, 1, 1, 100, frequencies.size), dtype=np.complex64)


def test_path_vertex_extraction_preserves_antenna_and_path_indices():
    vertices = np.zeros((3, 1, 1, 1, 2, 2, 3), dtype=np.float32)
    interactions = np.zeros((3, 1, 1, 1, 2, 2), dtype=np.int32)
    vertices[0, 0, 0, 0, 1, 1] = [12.0, 3.0, 1.0]
    interactions[0, 0, 0, 0, 1, 1] = 1
    paths = types.SimpleNamespace(
        vertices=vertices,
        interactions=interactions,
        synthetic_array=False,
    )

    extracted = _extract_path_vertices_for_link(paths, 0, 0)

    assert extracted.shape == (1, 2, 2, 3, 3)
    np.testing.assert_allclose(extracted[0, 1, 1, 0], [12.0, 3.0, 1.0])
    assert np.isnan(extracted[0, 0, 0]).all()


class _FakePlanarArray:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

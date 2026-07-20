import numpy as np
import pytest

from isac_6d_sampler.core.model import (
    BaseStation,
    DynamicObject,
    RadiomapConfig,
    SceneDesign,
    TrajectorySpec,
    UserEquipment,
)
from isac_6d_sampler.sim.planner import build_simulation_plan, orientation_for, position_for


def test_plan_links_include_monostatic_and_ordered_ue_bs_links():
    scene = SceneDesign(
        base_stations=[BaseStation(id="bs0"), BaseStation(id="bs1")],
        user_equipments=[UserEquipment(id="ue0"), UserEquipment(id="ue1")],
    )

    plan = build_simulation_plan(scene)
    names = {link.dataset_name for link in plan.timeframes[0].links}

    assert [device.id for device in plan.devices] == ["ue0", "ue1", "bs0", "bs1"]
    assert {"rx0_tx0", "rx1_tx1"}.issubset(names)
    assert {"rx2_tx2", "rx3_tx3"}.issubset(names)
    assert "rx0_tx2" in names
    assert "rx2_tx0" in names
    assert "rx1_tx3" in names
    assert "rx3_tx1" in names
    assert "rx0_tx1" not in names
    assert "rx2_tx3" not in names
    bs_mono = next(link for link in plan.timeframes[0].links if link.dataset_name == "rx2_tx2")
    assert bs_mono.rx_panel.element_count == 100
    assert bs_mono.tx_panel.element_count == 1


def test_plan_timeframes_follow_ue_and_object_trajectories():
    ue = UserEquipment(
        id="ue0",
        position=(0.0, 0.0, 1.5),
        trajectory=TrajectorySpec.linear((0.0, 0.0, 1.5), (2.0, 0.0, 1.5), 3),
    )
    obj = DynamicObject(
        id="car0",
        position=(0.0, 0.0, 0.0),
        orientation_rad=(1.0, 0.2, 0.3),
        trajectory=TrajectorySpec.linear((0.0, 0.0, 0.0), (0.0, 2.0, 0.0), 3),
    )
    scene = SceneDesign(
        base_stations=[BaseStation(id="bs0", orientation_rad=(0.1, 0.0, 0.0))],
        user_equipments=[ue],
        objects=[obj],
    )

    plan = build_simulation_plan(scene)

    assert [frame.name for frame in plan.timeframes] == ["tf000", "tf001", "tf002"]
    assert position_for(ue, plan.timeframes[-1]) == (2.0, 0.0, 1.5)
    assert plan.timeframes[-1].object_positions["car0"] == (0.0, 2.0, 0.0)
    assert plan.timeframes[-1].device_orientations["bs0"] == (0.1, 0.0, 0.0)
    assert plan.timeframes[-1].object_orientations["car0"] == (1.0, 0.2, 0.3)


def test_plan_timeframes_follow_trajectory_orientation_points():
    ue = UserEquipment(
        id="ue0",
        position=(0.0, 0.0, 1.5),
        orientation_rad=(0.0, 0.0, 0.0),
        trajectory=TrajectorySpec.linear((0.0, 0.0, 1.5), (2.0, 0.0, 1.5), 3),
    )
    ue.trajectory.orientation_rad_points = [(0.0, 0.0, 0.0), (1.0, 0.5, 0.25)]
    obj = DynamicObject(
        id="car0",
        position=(0.0, 0.0, 0.0),
        orientation_rad=(0.0, 0.0, 0.0),
        trajectory=TrajectorySpec.linear((0.0, 0.0, 0.0), (0.0, 2.0, 0.0), 3),
    )
    obj.trajectory.orientation_rad_points = [(0.0, 0.0, 0.0), (0.2, 0.3, 0.4)]
    scene = SceneDesign(
        base_stations=[BaseStation(id="bs0")],
        user_equipments=[ue],
        objects=[obj],
    )

    plan = build_simulation_plan(scene)

    assert orientation_for(ue, plan.timeframes[-1]) == (1.0, 0.5, 0.25)
    assert plan.timeframes[-1].object_orientations["car0"] == (0.2, 0.3, 0.4)


def test_curve_plan_orientation_follows_tangent_direction():
    ue = UserEquipment(
        id="ue0",
        trajectory=TrajectorySpec(
            kind="curve",
            points=[(0.0, 0.0, 1.5), (0.0, 10.0, 1.5)],
            bezier_handles=[
                ((0.0, 0.0, 1.5), (0.0, 4.0, 1.5)),
                ((0.0, 6.0, 1.5), (0.0, 10.0, 1.5)),
            ],
            samples=3,
        ),
    )
    scene = SceneDesign(
        user_equipments=[ue],
        base_stations=[BaseStation(id="bs0")],
    )

    plan = build_simulation_plan(scene)

    assert orientation_for(ue, plan.timeframes[0]) == pytest.approx((np.pi / 2.0, 0.0, 0.0))
    assert orientation_for(ue, plan.timeframes[-1]) == pytest.approx((np.pi / 2.0, 0.0, 0.0))


def test_radiomap_plan_materializes_xy_grid_as_timeframes():
    scene = SceneDesign(
        base_stations=[BaseStation(id="bs0")],
        radiomap=RadiomapConfig(
            enabled=True,
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
            x_spacing=1.0,
            y_spacing=1.0,
            height=1.25,
        ),
    )

    plan = build_simulation_plan(scene)

    assert len(plan.timeframes) == 4
    assert scene.user_equipments == []
    assert [device.id for device in plan.devices] == ["ue_radiomap", "bs0"]
    assert set(plan.timeframes[0].device_positions) == {"ue_radiomap", "bs0"}
    assert plan.timeframes[0].device_positions["bs0"] == (0.0, 0.0, 2.0)
    assert {frame.device_positions["ue_radiomap"][2] for frame in plan.timeframes} == {1.25}
    assert set(plan.timeframes[0].device_orientations) == {"ue_radiomap", "bs0"}
    assert plan.timeframes[0].metadata["frame_kind"] == "radiomap"
    assert plan.timeframes[3].metadata["radiomap_grid_index"] == 3
    assert plan.timeframes[3].metadata["radiomap_x_index"] == 1
    assert plan.timeframes[3].metadata["radiomap_y_index"] == 1


def test_radiomap_plan_does_not_replace_existing_scene_ues():
    scene = SceneDesign(
        base_stations=[BaseStation(id="bs0")],
        user_equipments=[UserEquipment(id="ue0")],
        radiomap=RadiomapConfig(
            enabled=True,
            x_min=0.0,
            x_max=0.0,
            y_min=0.0,
            y_max=0.0,
            x_spacing=1.0,
            y_spacing=1.0,
            height=1.25,
        ),
    )

    plan = build_simulation_plan(scene)

    assert [ue.id for ue in scene.user_equipments] == ["ue0"]
    assert [device.id for device in plan.devices] == ["ue_radiomap", "bs0"]
    assert set(plan.timeframes[0].device_positions) == {"ue_radiomap", "bs0"}


def test_radiomap_plan_repeats_grid_for_each_object_trajectory_state():
    obj = DynamicObject(
        id="car0",
        position=(0.0, 0.0, 0.0),
        orientation_rad=(0.0, 0.0, 0.5),
        trajectory=TrajectorySpec.linear((0.0, 0.0, 0.0), (0.0, 2.0, 0.0), 2),
    )
    scene = SceneDesign(
        base_stations=[BaseStation(id="bs0")],
        objects=[obj],
        radiomap=RadiomapConfig(
            enabled=True,
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=0.0,
            x_spacing=1.0,
            y_spacing=1.0,
            height=1.25,
        ),
    )

    plan = build_simulation_plan(scene)

    assert [frame.name for frame in plan.timeframes] == ["tf000", "tf001", "tf002", "tf003"]
    assert [frame.metadata["radiomap_grid_index"] for frame in plan.timeframes] == [0, 1, 0, 1]
    assert [frame.metadata["radiomap_object_state_index"] for frame in plan.timeframes] == [0, 0, 1, 1]
    assert [frame.device_positions["ue_radiomap"] for frame in plan.timeframes] == [
        (0.0, 0.0, 1.25),
        (1.0, 0.0, 1.25),
        (0.0, 0.0, 1.25),
        (1.0, 0.0, 1.25),
    ]
    assert [frame.object_positions["car0"] for frame in plan.timeframes] == [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, 2.0, 0.0),
    ]
    assert {frame.object_orientations["car0"] for frame in plan.timeframes} == {(0.0, 0.0, 0.5)}

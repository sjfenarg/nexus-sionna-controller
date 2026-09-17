import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace

pytest.importorskip("PySide6", reason="PySide6 is required for GUI tests")
pytest.importorskip("pyqtgraph", reason="pyqtgraph is required for the 3D GUI")
from PySide6.QtGui import QMatrix4x4

from isac_6d_sampler.core.model import DynamicObject, RadiomapConfig, SceneDesign, TrajectorySpec, UserEquipment
from isac_6d_sampler.core.scenarios import MeshAsset
from isac_6d_sampler.gui.scene_3d import (
    _ANTENNA_DIAGRAM_RADIUS_M,
    _arrowhead_segments,
    _antenna_diagram_mesh,
    _antenna_normalized_gain_db,
    _axis_view_angles,
    _closest_point_on_axis_to_ray,
    _curve_tangent_arrow_segments,
    _intersect_ray_plane,
    _local_axes,
    _plane_handle_points,
    _point_to_ray_distance,
    _pointing_vector,
    _select_important_bs_to_ue_paths,
    _radiomap_corner_world_per_pixel,
    _screen_angle,
    _screen_pixels_to_world,
    _screen_distance_to_polyline,
    _screen_distance_to_segment,
    _screen_plane_components,
    _screen_point_in_polygon,
    _screen_unit_vector,
    _signed_angle,
    _transform_mesh_vertices,
    _trajectory_preview_positions,
    _trajectory_world_per_pixel,
    _viewport_rect,
    _wrap_angle,
)


def test_path_range_filter_uses_matching_geometry_and_rejects_false_los():
    speed_of_light = 299_792_458.0
    delays = np.asarray([[[[[[78.0 / speed_of_light]]]]]], dtype=np.float64)
    coefficients = np.ones_like(delays, dtype=np.complex64)
    bounce_y = np.sqrt(39.0**2 - 20.0**2)
    vertices = np.full((1, 1, 1, 3, 3), np.nan, dtype=np.float32)
    vertices[0, 0, 0, 0] = [20.0, bounce_y, 0.0]
    link = SimpleNamespace(
        tx_id="bs0",
        rx_id="ue0",
        metadata={
            "path_delays_s": delays,
            "path_coefficients": coefficients,
            "path_vertices": vertices,
        },
    )

    selected, stats = _select_important_bs_to_ue_paths(
        [link],
        {"bs0": (0.0, 0.0, 0.0), "ue0": (40.0, 0.0, 0.0)},
        max_paths=10,
        min_delay_separation_s=2e-9,
        min_range_m=77.0,
        max_range_m=80.0,
    )

    assert len(selected) == 1
    assert stats["paths_rejected_geometry_mismatch"] == 0
    assert np.linalg.norm(np.diff(selected[0]["points"], axis=0), axis=1).sum() == pytest.approx(78.0)

    link.metadata["path_vertices"] = np.full_like(vertices, np.nan)
    selected, stats = _select_important_bs_to_ue_paths(
        [link],
        {"bs0": (0.0, 0.0, 0.0), "ue0": (40.0, 0.0, 0.0)},
        max_paths=10,
        min_delay_separation_s=2e-9,
        min_range_m=77.0,
        max_range_m=80.0,
    )

    assert selected == []
    assert stats["paths_rejected_geometry_mismatch"] == 1


def test_intersect_ray_plane_returns_world_point():
    point = _intersect_ray_plane(
        np.asarray([1.0, 2.0, 10.0]),
        np.asarray([0.0, 0.0, -1.0]),
        np.asarray([0.0, 0.0, 3.0]),
        np.asarray([0.0, 0.0, 1.0]),
    )

    assert np.allclose(point, [1.0, 2.0, 3.0])


def test_closest_point_on_axis_to_ray_tracks_axis_coordinate():
    point = _closest_point_on_axis_to_ray(
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([4.0, 2.0, 0.0]),
        np.asarray([0.0, -1.0, 0.0]),
    )

    assert np.allclose(point, [4.0, 0.0, 0.0])


def test_point_to_ray_distance_uses_perpendicular_distance():
    distance = _point_to_ray_distance(
        np.asarray([2.0, 3.0, 0.0]),
        np.asarray([2.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
    )

    assert distance == 0.0


def test_local_axes_follow_yaw_rotation():
    axes = _local_axes((np.pi / 2.0, 0.0, 0.0))

    assert np.allclose(axes["x"], [0.0, 1.0, 0.0])
    assert np.allclose(axes["y"], [-1.0, 0.0, 0.0])
    assert np.allclose(axes["z"], [0.0, 0.0, 1.0])


def test_pointing_vector_is_local_positive_x():
    pointing = _pointing_vector((np.pi / 2.0, 0.0, 0.0))

    assert np.allclose(pointing, [0.0, 1.0, 0.0])


def test_signed_angle_around_axis():
    angle = _signed_angle(
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
    )

    assert np.isclose(angle, np.pi / 2.0)


def test_viewport_tuple_is_converted_to_qrect():
    rect = _viewport_rect((1, 2, 300, 400))

    assert (rect.x(), rect.y(), rect.width(), rect.height()) == (1, 2, 300, 400)


def test_mouse_ray_keeps_tuple_for_pyqtgraph_projection(monkeypatch):
    observed = {}

    class DummyView:
        _mouse_ray = __import__(
            "isac_6d_sampler.gui.scene_3d",
            fromlist=["Scene3DView"],
        ).Scene3DView._mouse_ray

        def getViewport(self):
            return (0, 0, 640, 480)

        def width(self):
            return 640

        def height(self):
            return 480

        def viewMatrix(self):
            return QMatrix4x4()

    view = DummyView()

    def fake_projection_matrix(region, viewport):
        observed["projection_viewport"] = viewport
        return QMatrix4x4()

    view.projectionMatrix = fake_projection_matrix

    ray = view._mouse_ray(320, 240)

    assert observed["projection_viewport"] == (0, 0, 640, 480)
    assert ray is not None


def test_screen_segment_distance_uses_pixels():
    distance = _screen_distance_to_segment((5.0, 4.0), (0.0, 0.0), (10.0, 0.0))

    assert distance == 4.0


def test_screen_polyline_distance_uses_closest_segment():
    distance = _screen_distance_to_polyline(
        (10.0, 8.0),
        [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0)],
    )

    assert distance == 2.0


def test_screen_point_in_polygon():
    polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

    assert _screen_point_in_polygon((5.0, 5.0), polygon)
    assert not _screen_point_in_polygon((15.0, 5.0), polygon)


def test_plane_handle_points_are_offset_from_origin():
    points = _plane_handle_points(
        np.zeros(3),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        10.0,
    )

    assert points.shape == (5, 3)
    assert np.all(points[:, 0] >= 2.4)
    assert np.all(points[:, 1] >= 2.4)


def test_screen_pixels_to_world_scales_with_camera_distance():
    near = _screen_pixels_to_world(distance=10.0, fov_degrees=60.0, viewport_height=500.0, pixels=100.0)
    far = _screen_pixels_to_world(distance=20.0, fov_degrees=60.0, viewport_height=500.0, pixels=100.0)

    assert far == 2.0 * near


def test_screen_unit_vector_normalizes_screen_axis():
    assert np.allclose(_screen_unit_vector((0.0, 0.0), (0.0, 4.0)), [0.0, 1.0])
    assert _screen_unit_vector((1.0, 1.0), (1.0, 1.0)) is None


def test_screen_plane_components_solve_screen_basis():
    first, second = _screen_plane_components(
        np.asarray([10.0, 5.0]),
        np.asarray([1.0, 0.0]),
        np.asarray([0.0, 1.0]),
    )

    assert first == 10.0
    assert second == 5.0


def test_screen_angle_is_defined_by_projected_center():
    assert np.isclose(_screen_angle((0.0, 0.0), (0.0, 1.0)), np.pi / 2.0)
    assert _screen_angle((1.0, 1.0), (1.0, 1.0)) is None


def test_wrap_angle_keeps_short_rotation_delta():
    assert np.isclose(_wrap_angle(3.0 * np.pi), -np.pi)
    assert np.isclose(_wrap_angle(-3.0 * np.pi), -np.pi)


def test_arrowhead_segments_build_four_line_segments():
    segments = _arrowhead_segments(
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        2.0,
    )

    assert segments.shape == (8, 3)
    assert np.allclose(segments[0], [1.0, 0.0, 0.0])
    assert np.allclose(segments[2], [1.0, 0.0, 0.0])


def test_axis_view_angles_cover_standard_planes():
    assert _axis_view_angles("z", 1) == (90.0, -90.0)
    assert _axis_view_angles("z", -1) == (-90.0, -90.0)
    assert _axis_view_angles("x", 1) == (0.0, 0.0)
    assert _axis_view_angles("x", -1) == (0.0, 180.0)
    assert _axis_view_angles("y", 1) == (0.0, 90.0)
    assert _axis_view_angles("y", -1) == (0.0, -90.0)


def test_antenna_gain_is_normalized_to_local_pointing_direction():
    directions = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    gain = _antenna_normalized_gain_db("tr38901", directions)

    assert gain[0] == 0.0
    assert gain[1] < gain[0]
    assert gain[2] < gain[1]


def test_antenna_diagram_mesh_scales_max_radius_to_two_meters():
    vertices, faces, colors = _antenna_diagram_mesh(
        origin=np.zeros(3),
        orientation_rad=(0.0, 0.0, 0.0),
        pattern_name="tr38901",
        alpha_samples=7,
        beta_samples=9,
    )

    radius = np.linalg.norm(vertices, axis=1)

    assert vertices.shape == (63, 3)
    assert faces.shape == (96, 3)
    assert colors.shape == (96, 4)
    assert np.isclose(np.max(radius), _ANTENNA_DIAGRAM_RADIUS_M)


def test_radiomap_corner_drag_uses_camera_independent_scale():
    config = RadiomapConfig(enabled=True, x_min=-10.0, x_max=10.0, y_min=-5.0, y_max=5.0)
    scale = _radiomap_corner_world_per_pixel(config)
    state = {
        "target": "radiomap_corner",
        "corner": "min",
        "mouse_start": np.asarray([100.0, 100.0]),
        "start_x": -10.0,
        "start_y": -5.0,
        "x_min": -10.0,
        "x_max": 10.0,
        "y_min": -5.0,
        "y_max": 5.0,
        "world_per_pixel": scale,
        "screen_x_axis": np.asarray([1.0, 0.0]),
        "screen_y_axis": np.asarray([0.0, -1.0]),
    }

    bounds = __import__(
        "isac_6d_sampler.gui.scene_3d",
        fromlist=["Scene3DView"],
    ).Scene3DView._drag_radiomap_corner(None, 150.0, 125.0, state)

    assert bounds == (-10.0 + 50.0 * scale, 10.0, -5.0 - 25.0 * scale, 5.0)


def test_curve_preview_positions_are_sampled_and_drag_scale_is_distance_independent():
    trajectory = TrajectorySpec(
        kind="curve",
        points=[(0.0, 0.0, 1.5), (10.0, 0.0, 1.5)],
        bezier_handles=[
            ((0.0, 0.0, 1.5), (0.0, 5.0, 1.5)),
            ((10.0, 5.0, 1.5), (10.0, 0.0, 1.5)),
        ],
        samples=4,
    )

    preview = _trajectory_preview_positions(trajectory)

    assert preview.shape == (96, 3)
    assert np.max(preview[:, 1]) > 2.0
    assert _trajectory_world_per_pixel(trajectory) == pytest.approx(10.0 / 420.0)


def test_curve_trajectory_suppresses_endpoint_gizmo_pose():
    module = __import__("isac_6d_sampler.gui.scene_3d", fromlist=["Scene3DView"])
    ue = UserEquipment(id="ue0")
    ue.trajectory = TrajectorySpec(
        kind="curve",
        points=[(0.0, 0.0, 1.5), (10.0, 0.0, 1.5)],
        samples=4,
    )

    class DummyView:
        _trajectory_endpoint_pose = module.Scene3DView._trajectory_endpoint_pose

    assert DummyView()._trajectory_endpoint_pose(ue) is None


def test_curve_tangent_arrow_segments_follow_path_direction():
    positions = np.asarray(
        [
            [0.0, 0.0, 1.5],
            [1.0, 0.0, 1.5],
            [2.0, 0.0, 1.5],
        ],
        dtype=np.float64,
    )

    segments = _curve_tangent_arrow_segments(positions, length=0.5, max_arrows=1)

    assert segments.shape[0] >= 2
    np.testing.assert_allclose(segments[1] - segments[0], [0.5, 0.0, 0.0])


def test_dynamic_object_mesh_vertices_are_centered_on_object_position():
    vertices = np.asarray(
        [
            [-1.0, -1.0, 0.0],
            [1.0, 1.0, 1.5],
        ],
        dtype=np.float64,
    )
    anchor = np.asarray([0.0, 0.0, 0.75], dtype=np.float64)
    position = np.asarray([0.0, 0.0, 0.75], dtype=np.float64)

    transformed = _transform_mesh_vertices(vertices, anchor, position, (0.0, 0.0, 0.0))

    np.testing.assert_allclose(np.min(transformed, axis=0), [-1.0, -1.0, 0.0])
    np.testing.assert_allclose(np.max(transformed, axis=0), [1.0, 1.0, 1.5])


def test_configured_car_mesh_is_dynamic_not_static():
    module = __import__("isac_6d_sampler.gui.scene_3d", fromlist=["Scene3DView"])
    car = MeshAsset(name="CAR_obj", path=Path("car.ply"))
    wall = MeshAsset(name="WALL_obj", path=Path("wall.ply"))

    class DummyView:
        _static_meshes = module.Scene3DView._static_meshes
        _dynamic_meshes = module.Scene3DView._dynamic_meshes
        _asset = type("Asset", (), {"meshes": (car, wall)})()
        _design = SceneDesign(objects=[DynamicObject(id="car0", object_name="CAR_obj")])

    view = DummyView()

    assert view._static_meshes() == (wall,)
    assert view._dynamic_meshes() == (car,)


def test_configured_external_obj_mesh_is_dynamic():
    module = __import__("isac_6d_sampler.gui.scene_3d", fromlist=["Scene3DView"])
    car = MeshAsset(name="CAR_obj", path=Path("car.ply"))
    drone = MeshAsset(name="DRONE_obj", path=Path("drone.obj"))
    wall = MeshAsset(name="WALL_obj", path=Path("wall.ply"))

    class DummyView:
        _static_meshes = module.Scene3DView._static_meshes
        _dynamic_meshes = module.Scene3DView._dynamic_meshes
        _asset = type("Asset", (), {"meshes": (car, wall), "object_meshes": (drone,)})()
        _design = SceneDesign(objects=[DynamicObject(id="drone0", object_name="DRONE_obj")])

    view = DummyView()

    assert view._static_meshes() == (wall,)
    assert view._dynamic_meshes() == (drone,)

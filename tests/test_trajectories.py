import numpy as np
import pytest

from isac_6d_sampler.core.model import RadiomapConfig, TrajectorySpec
from isac_6d_sampler.core.trajectories import (
    default_bezier_handles,
    radiomap_grid_shape,
    sample_orientations,
    sample_radiomap_grid,
    sample_radiomap_preview_grid,
    sample_trajectory,
)


def test_linear_trajectory_endpoints():
    trajectory = sample_trajectory(TrajectorySpec.linear((0, 0, 1), (10, 0, 1), 5))
    assert trajectory.shape == (5, 3)
    np.testing.assert_allclose(trajectory[0], [0, 0, 1])
    np.testing.assert_allclose(trajectory[-1], [10, 0, 1])


def test_static_trajectory_ignores_sample_count_for_planning():
    trajectory = sample_trajectory(TrajectorySpec(kind="static", points=[(1.0, 2.0, 3.0)], samples=8))

    assert trajectory.shape == (1, 3)
    np.testing.assert_allclose(trajectory[0], [1.0, 2.0, 3.0])


def test_linear_orientation_control_points_are_sampled():
    spec = TrajectorySpec.linear((0, 0, 1), (10, 0, 1), 3)
    spec.orientation_rad_points = [(0.0, 0.0, 0.0), (1.0, 0.5, 0.25)]

    orientations = sample_orientations(spec, (0.0, 0.0, 0.0))

    assert orientations.shape == (3, 3)
    np.testing.assert_allclose(orientations[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(orientations[-1], [1.0, 0.5, 0.25])


def test_curve_trajectory_samples_cubic_bezier_in_xy_plane():
    spec = TrajectorySpec(
        kind="curve",
        points=[(0.0, 0.0, 1.5), (10.0, 0.0, 3.0)],
        bezier_handles=[
            ((0.0, 0.0, 1.5), (0.0, 10.0, 1.5)),
            ((10.0, 10.0, 1.5), (10.0, 0.0, 1.5)),
        ],
        samples=5,
    )

    trajectory = sample_trajectory(spec)

    assert trajectory.shape == (5, 3)
    np.testing.assert_allclose(trajectory[0], [0.0, 0.0, 1.5])
    np.testing.assert_allclose(trajectory[-1], [10.0, 0.0, 1.5])
    assert np.max(trajectory[:, 1]) > 4.0
    np.testing.assert_allclose(trajectory[:, 2], 1.5)


def test_curve_orientation_follows_xy_tangent_yaw():
    spec = TrajectorySpec(
        kind="curve",
        points=[(0.0, 0.0, 1.5), (0.0, 10.0, 1.5)],
        bezier_handles=[
            ((0.0, 0.0, 1.5), (0.0, 4.0, 1.5)),
            ((0.0, 6.0, 1.5), (0.0, 10.0, 1.5)),
        ],
        orientation_rad_points=[(0.0, 0.0, 9.0), (0.0, 0.0, 9.0)],
        samples=5,
    )

    orientations = sample_orientations(spec, (0.0, 0.0, 0.0))

    np.testing.assert_allclose(orientations[:, 0], np.pi / 2.0)
    np.testing.assert_allclose(orientations[:, 1:], 0.0)


def test_default_bezier_handles_create_one_pair_per_anchor():
    handles = default_bezier_handles([(0.0, 0.0, 2.0), (6.0, 0.0, 2.0), (6.0, 6.0, 2.0)])

    assert len(handles) == 3
    assert handles[0][0] == (0.0, 0.0, 2.0)
    assert handles[-1][1] == (6.0, 6.0, 2.0)


def test_radiomap_grid_parallel_to_xy():
    grid = sample_radiomap_grid(RadiomapConfig(enabled=True, x_min=0, x_max=1, y_min=0, y_max=1, x_spacing=1, y_spacing=1, height=1.25))
    assert grid.shape == (4, 3)
    np.testing.assert_allclose(grid[:, 2], 1.25)


def test_radiomap_grid_shape_matches_materialized_grid():
    config = RadiomapConfig(enabled=True, x_min=0, x_max=2, y_min=0, y_max=1, x_spacing=1, y_spacing=1)

    assert radiomap_grid_shape(config) == (3, 2)
    assert sample_radiomap_grid(config).shape == (6, 3)


def test_radiomap_preview_grid_is_capped_but_reports_total_points():
    config = RadiomapConfig(enabled=True, x_min=0, x_max=9, y_min=0, y_max=9, x_spacing=1, y_spacing=1)

    preview, total_points = sample_radiomap_preview_grid(config, max_points=16)

    assert total_points == 100
    assert preview.shape[0] <= 16
    np.testing.assert_allclose(preview[:, 2], config.height)


def test_radiomap_preview_grid_cap_handles_rectangular_grid():
    config = RadiomapConfig(enabled=True, x_min=0, x_max=0, y_min=0, y_max=999, x_spacing=1, y_spacing=1)

    preview, total_points = sample_radiomap_preview_grid(config, max_points=16)

    assert total_points == 1000
    assert preview.shape[0] <= 16
    assert set(preview[:, 0]) == {0.0}


def test_radiomap_grid_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="x_max"):
        sample_radiomap_grid(RadiomapConfig(enabled=True, x_min=2, x_max=1))

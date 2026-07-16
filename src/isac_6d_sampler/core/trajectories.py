from __future__ import annotations

import numpy as np

from .model import RadiomapConfig, TrajectorySpec, Vector3


def sample_trajectory(spec: TrajectorySpec) -> np.ndarray:
    """Return positions with shape ``[samples, 3]``."""
    return _sample_control_points(spec, np.asarray(spec.points, dtype=np.float64))


def sample_orientations(spec: TrajectorySpec, default_orientation: Vector3) -> np.ndarray:
    """Return orientation samples with shape ``[samples, 3]``."""
    if spec.orientation_rad_points:
        points = np.unwrap(np.asarray(spec.orientation_rad_points, dtype=np.float64), axis=0)
    else:
        points = np.asarray([default_orientation], dtype=np.float64)
    return _sample_control_points(spec, points)


def _sample_control_points(spec: TrajectorySpec, points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Trajectory control points must be a list of 3D coordinates")

    if spec.kind == "static" or len(points) == 1:
        return points[:1].copy()

    samples = max(1, int(spec.samples))

    if len(points) < 2:
        raise ValueError(f"{spec.kind} trajectories require at least two points")

    moving_t = _motion_profile(samples, spec.start_static_fraction, spec.end_static_fraction, spec.easing)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = float(np.sum(segment_lengths))
    if total_length == 0.0:
        return np.repeat(points[:1], samples, axis=0)

    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)]) / total_length
    out = np.empty((samples, 3), dtype=np.float64)
    for idx, t in enumerate(moving_t):
        segment = np.searchsorted(cumulative, t, side="right") - 1
        segment = min(max(segment, 0), len(segment_lengths) - 1)
        denom = cumulative[segment + 1] - cumulative[segment]
        local_t = 0.0 if denom == 0.0 else (t - cumulative[segment]) / denom
        out[idx] = points[segment] + local_t * (points[segment + 1] - points[segment])
    return out


def sample_radiomap_grid(config: RadiomapConfig) -> np.ndarray:
    _validate_radiomap_grid(config)
    xs = np.arange(config.x_min, config.x_max + config.x_spacing * 0.5, config.x_spacing)
    ys = np.arange(config.y_min, config.y_max + config.y_spacing * 0.5, config.y_spacing)
    return _radiomap_positions(xs, ys, config.height)


def radiomap_grid_shape(config: RadiomapConfig) -> tuple[int, int]:
    _validate_radiomap_grid(config)
    x_points = _axis_count(config.x_min, config.x_max, config.x_spacing)
    y_points = _axis_count(config.y_min, config.y_max, config.y_spacing)
    return x_points, y_points


def sample_radiomap_preview_grid(config: RadiomapConfig, max_points: int = 2500) -> tuple[np.ndarray, int]:
    x_points, y_points = radiomap_grid_shape(config)
    total_points = x_points * y_points
    if total_points <= max_points:
        return sample_radiomap_grid(config), total_points

    stride = _preview_stride(x_points, y_points, max_points)
    xs = config.x_min + np.arange(0, x_points, stride, dtype=np.float64) * config.x_spacing
    ys = config.y_min + np.arange(0, y_points, stride, dtype=np.float64) * config.y_spacing
    return _radiomap_positions(xs, ys, config.height), total_points


def _validate_radiomap_grid(config: RadiomapConfig) -> None:
    if config.x_spacing <= 0 or config.y_spacing <= 0:
        raise ValueError("Radiomap spacing must be positive")
    if config.x_max < config.x_min:
        raise ValueError("Radiomap x_max must be greater than or equal to x_min")
    if config.y_max < config.y_min:
        raise ValueError("Radiomap y_max must be greater than or equal to y_min")


def _axis_count(start: float, stop: float, spacing: float) -> int:
    return int(np.floor((stop - start + spacing * 0.5) / spacing)) + 1


def _preview_stride(x_points: int, y_points: int, max_points: int) -> int:
    max_points = max(1, int(max_points))
    total_points = x_points * y_points
    stride = max(1, int(np.ceil(np.sqrt(total_points / max_points))))
    while _strided_count(x_points, stride) * _strided_count(y_points, stride) > max_points:
        stride += 1
    return stride


def _strided_count(points: int, stride: int) -> int:
    return int(np.ceil(points / stride))


def _radiomap_positions(xs: np.ndarray, ys: np.ndarray, height: float) -> np.ndarray:
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    zz = np.full_like(xx, height, dtype=np.float64)
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _motion_profile(
    samples: int,
    start_static_fraction: float,
    end_static_fraction: float,
    easing: str,
) -> np.ndarray:
    start_static_fraction = float(np.clip(start_static_fraction, 0.0, 1.0))
    end_static_fraction = float(np.clip(end_static_fraction, 0.0, 1.0))
    if start_static_fraction + end_static_fraction >= 1.0:
        raise ValueError("Static fractions must sum to less than 1")

    t = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    moving_start = start_static_fraction
    moving_stop = 1.0 - end_static_fraction
    u = np.clip((t - moving_start) / (moving_stop - moving_start), 0.0, 1.0)
    if easing == "smoothstep":
        return u * u * (3.0 - 2.0 * u)
    if easing != "linear":
        raise ValueError("Trajectory easing must be 'linear' or 'smoothstep'")
    return u

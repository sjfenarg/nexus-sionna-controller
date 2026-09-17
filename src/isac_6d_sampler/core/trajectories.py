from __future__ import annotations

import numpy as np

from .model import RadiomapConfig, TrajectorySpec, Vector3


def sample_trajectory(spec: TrajectorySpec) -> np.ndarray:
    """Return positions with shape ``[samples, 3]``."""
    points = np.asarray(spec.points, dtype=np.float64)
    if spec.kind == "curve":
        return _sample_bezier_curve(spec, points)
    return _sample_control_points(spec, points)


def sample_orientations(spec: TrajectorySpec, default_orientation: Vector3) -> np.ndarray:
    """Return orientation samples with shape ``[samples, 3]``."""
    if spec.kind == "curve":
        return sample_curve_tangent_orientations(spec, default_orientation)
    if spec.orientation_rad_points:
        points = np.unwrap(np.asarray(spec.orientation_rad_points, dtype=np.float64), axis=0)
    else:
        points = np.asarray([default_orientation], dtype=np.float64)
    orientation_spec = TrajectorySpec(
        kind="polyline" if len(points) > 1 else "static",
        points=[tuple(float(value) for value in point) for point in points],
        samples=spec.samples,
        start_static_fraction=spec.start_static_fraction,
        end_static_fraction=spec.end_static_fraction,
        easing=spec.easing,
    )
    return _sample_control_points(orientation_spec, points)


def sample_curve_tangent_orientations(spec: TrajectorySpec, default_orientation: Vector3) -> np.ndarray:
    """Return orientations whose yaw follows the curve tangent and pitch/roll stay configured."""
    positions = sample_trajectory(spec)
    if positions.shape[0] <= 1:
        return np.asarray([default_orientation], dtype=np.float64)
    tangents = _trajectory_tangents_xy(positions)
    out = np.repeat(np.asarray([default_orientation], dtype=np.float64), positions.shape[0], axis=0)
    out[:, 0] = np.arctan2(tangents[:, 1], tangents[:, 0])
    return out


def normalized_bezier_handles(spec: TrajectorySpec) -> list[tuple[Vector3, Vector3]]:
    """Return one absolute incoming/outgoing Bezier handle pair per anchor point."""
    points = [tuple(float(value) for value in point) for point in spec.points]
    if not points:
        return []
    plane_z = float(points[0][2])
    existing = list(spec.bezier_handles)
    if len(existing) == len(points):
        return [
            (_project_to_curve_plane(handle_in, plane_z), _project_to_curve_plane(handle_out, plane_z))
            for handle_in, handle_out in existing
        ]
    return default_bezier_handles(points)


def default_bezier_handles(points: list[Vector3]) -> list[tuple[Vector3, Vector3]]:
    points = [tuple(float(value) for value in point) for point in points]
    if not points:
        return []
    if len(points) == 1:
        return [(points[0], points[0])]
    plane_z = float(points[0][2])
    handles: list[tuple[Vector3, Vector3]] = []
    for index, point in enumerate(points):
        point = _project_to_curve_plane(point, plane_z)
        previous_point = np.asarray(points[max(index - 1, 0)], dtype=np.float64)
        next_point = np.asarray(points[min(index + 1, len(points) - 1)], dtype=np.float64)
        tangent = next_point - previous_point
        tangent[2] = 0.0
        handle_in = np.asarray(point, dtype=np.float64) - tangent / 6.0
        handle_out = np.asarray(point, dtype=np.float64) + tangent / 6.0
        handle_in[2] = plane_z
        handle_out[2] = plane_z
        if index == 0:
            handle_in = np.asarray(point, dtype=np.float64)
        if index == len(points) - 1:
            handle_out = np.asarray(point, dtype=np.float64)
        handles.append((
            tuple(float(value) for value in handle_in),
            tuple(float(value) for value in handle_out),
        ))
    return handles


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


def _sample_bezier_curve(spec: TrajectorySpec, points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Trajectory control points must be a list of 3D coordinates")
    if len(points) == 1:
        return points[:1].copy()
    if len(points) < 2:
        raise ValueError("Curve trajectories require at least two points")
    samples = max(1, int(spec.samples))
    plane_z = float(points[0, 2])
    anchors = points.copy()
    anchors[:, 2] = plane_z
    if samples == 1:
        return anchors[:1].copy()
    handles = np.asarray(normalized_bezier_handles(spec), dtype=np.float64)
    handles[:, :, 2] = plane_z
    moving_t = _motion_profile(samples, spec.start_static_fraction, spec.end_static_fraction, spec.easing)

    segment_samples = max(24, int(np.ceil(192 / max(len(anchors) - 1, 1))))
    dense_segments = []
    for segment_index in range(len(anchors) - 1):
        t_values = np.linspace(0.0, 1.0, segment_samples + 1, dtype=np.float64)
        if segment_index:
            t_values = t_values[1:]
        dense_segments.append(
            _cubic_bezier(
                anchors[segment_index],
                handles[segment_index, 1],
                handles[segment_index + 1, 0],
                anchors[segment_index + 1],
                t_values,
            )
        )
    dense = np.vstack(dense_segments)
    segment_lengths = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    total_length = float(np.sum(segment_lengths))
    if total_length <= 1e-12:
        return np.repeat(anchors[:1], samples, axis=0)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)]) / total_length
    out = np.empty((samples, 3), dtype=np.float64)
    for idx, t in enumerate(moving_t):
        dense_index = np.searchsorted(cumulative, t, side="right") - 1
        dense_index = min(max(dense_index, 0), len(segment_lengths) - 1)
        denom = cumulative[dense_index + 1] - cumulative[dense_index]
        local_t = 0.0 if denom == 0.0 else (t - cumulative[dense_index]) / denom
        out[idx] = dense[dense_index] + local_t * (dense[dense_index + 1] - dense[dense_index])
    out[0] = anchors[0]
    out[-1] = anchors[-1]
    out[:, 2] = plane_z
    return out


def _cubic_bezier(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    t_values: np.ndarray,
) -> np.ndarray:
    t = t_values.reshape(-1, 1)
    omt = 1.0 - t
    return omt**3 * p0 + 3.0 * omt**2 * t * p1 + 3.0 * omt * t**2 * p2 + t**3 * p3


def _project_to_curve_plane(point: Vector3, plane_z: float) -> Vector3:
    return (float(point[0]), float(point[1]), float(plane_z))


def _trajectory_tangents_xy(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    tangents = np.zeros_like(positions)
    if len(positions) < 2:
        tangents[:, 0] = 1.0
        return tangents
    tangents[0] = positions[1] - positions[0]
    tangents[-1] = positions[-1] - positions[-2]
    if len(positions) > 2:
        tangents[1:-1] = positions[2:] - positions[:-2]
    tangents[:, 2] = 0.0
    norms = np.linalg.norm(tangents[:, :2], axis=1)
    valid = norms > 1e-12
    if not np.any(valid):
        tangents[:, 0] = 1.0
        return tangents
    valid_indices = np.flatnonzero(valid)
    for index in range(len(tangents)):
        if valid[index]:
            tangents[index, :2] /= norms[index]
            continue
        nearest = valid_indices[int(np.argmin(np.abs(valid_indices - index)))]
        tangents[index, :2] = tangents[nearest, :2] / norms[nearest]
    return tangents


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

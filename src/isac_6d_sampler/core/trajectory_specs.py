from __future__ import annotations

from .model import TrajectorySpec, Vector3


def parse_point_list(raw: str) -> list[Vector3]:
    """Parse ``x,y,z; x,y,z`` point lists used by the GUI."""
    points: list[Vector3] = []
    for chunk in raw.replace("|", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [part.strip() for part in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError("Trajectory points must use x,y,z triples separated by semicolons")
        try:
            points.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError as exc:
            raise ValueError(f"Invalid trajectory point: {chunk}") from exc
    if not points:
        raise ValueError("At least one trajectory point is required")
    return points


def format_point_list(points: list[Vector3]) -> str:
    return "; ".join(f"{x:g},{y:g},{z:g}" for x, y, z in points)


def translate_trajectory(spec: TrajectorySpec, delta: Vector3) -> TrajectorySpec:
    dx, dy, dz = delta
    return TrajectorySpec(
        kind=spec.kind,
        points=[(x + dx, y + dy, z + dz) for x, y, z in spec.points],
        orientation_rad_points=list(spec.orientation_rad_points),
        samples=spec.samples,
        start_static_fraction=spec.start_static_fraction,
        end_static_fraction=spec.end_static_fraction,
        easing=spec.easing,
    )


def anchor_trajectory(spec: TrajectorySpec, anchor: Vector3) -> TrajectorySpec:
    """Translate a trajectory so its first control point starts at ``anchor``."""
    if not spec.points:
        return TrajectorySpec.static(anchor)
    first = spec.points[0]
    return translate_trajectory(
        spec,
        (
            anchor[0] - first[0],
            anchor[1] - first[1],
            anchor[2] - first[2],
        ),
    )


def build_trajectory_spec(
    kind: str,
    points_text: str,
    samples: int,
    easing: str = "linear",
    start_static_fraction: float = 0.0,
    end_static_fraction: float = 0.0,
) -> TrajectorySpec:
    points = parse_point_list(points_text)
    if kind == "static":
        points = points[:1]
        samples = max(1, int(samples))
    elif kind == "linear":
        if len(points) < 2:
            raise ValueError("Linear trajectories require start and end points")
        points = points[:2]
    elif kind == "polyline":
        if len(points) < 2:
            raise ValueError("Polyline trajectories require at least two points")
    else:
        raise ValueError("Trajectory kind must be static, linear, or polyline")

    return TrajectorySpec(
        kind=kind,
        points=points,
        orientation_rad_points=[],
        samples=max(1, int(samples)),
        start_static_fraction=float(start_static_fraction),
        end_static_fraction=float(end_static_fraction),
        easing=easing,
    )

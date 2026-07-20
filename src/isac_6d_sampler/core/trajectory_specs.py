from __future__ import annotations

from .model import TrajectorySpec, Vector3
from .trajectories import default_bezier_handles, normalized_bezier_handles


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


def parse_bezier_handles(raw: str) -> list[tuple[Vector3, Vector3]]:
    handles: list[tuple[Vector3, Vector3]] = []
    for chunk in raw.replace("|", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [part.strip() for part in chunk.split(">")]
        if len(parts) != 2:
            raise ValueError("Bezier handles must use in_x,in_y,in_z>out_x,out_y,out_z pairs")
        handles.append((parse_point_list(parts[0])[0], parse_point_list(parts[1])[0]))
    return handles


def format_bezier_handles(handles: list[tuple[Vector3, Vector3]]) -> str:
    return "; ".join(f"{hin[0]:g},{hin[1]:g},{hin[2]:g}>{hout[0]:g},{hout[1]:g},{hout[2]:g}" for hin, hout in handles)


def translate_trajectory(spec: TrajectorySpec, delta: Vector3) -> TrajectorySpec:
    dx, dy, dz = delta
    return TrajectorySpec(
        kind=spec.kind,
        points=[(x + dx, y + dy, z + dz) for x, y, z in spec.points],
        bezier_handles=[
            (
                (hin[0] + dx, hin[1] + dy, hin[2] + dz),
                (hout[0] + dx, hout[1] + dy, hout[2] + dz),
            )
            for hin, hout in normalized_bezier_handles(spec)
        ] if spec.kind == "curve" else list(spec.bezier_handles),
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
    bezier_handles_text: str = "",
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
    elif kind == "curve":
        if len(points) < 2:
            raise ValueError("Curve trajectories require at least start and end points")
    else:
        raise ValueError("Trajectory kind must be static, linear, polyline, or curve")

    bezier_handles = []
    if kind == "curve":
        bezier_handles = parse_bezier_handles(bezier_handles_text) if bezier_handles_text.strip() else default_bezier_handles(points)
        if len(bezier_handles) != len(points):
            raise ValueError("Curve Bezier handles must contain one in/out pair per trajectory point")

    return TrajectorySpec(
        kind=kind,
        points=points,
        bezier_handles=bezier_handles,
        orientation_rad_points=[],
        samples=max(1, int(samples)),
        start_static_fraction=float(start_static_fraction),
        end_static_fraction=float(end_static_fraction),
        easing=easing,
    )

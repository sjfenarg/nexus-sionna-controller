from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_INPUT = Path("scenes/4ue_1bs.json")
DEFAULT_OUTPUT_DIR = Path("scenes/channel_charting_dataset_v2")
DEFAULT_DATASET_OUTPUT_DIR = Path("output/channel_charting_dataset_v2")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-output-dir", type=Path, default=DEFAULT_DATASET_OUTPUT_DIR)
    parser.add_argument("--scenes", type=int, default=12)
    parser.add_argument("--ues", type=int, default=6)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2026072318)
    args = parser.parse_args()

    request_template = json.loads(args.input.read_text(encoding="utf-8"))
    bounds = _trajectory_bounds(request_template)
    height = float(args.height if args.height is not None else request_template["scene"]["radiomap"]["height"])
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    used_starts: list[np.ndarray] = []
    used_paths: list[np.ndarray] = []

    for scene_index in range(args.scenes):
        request = copy.deepcopy(request_template)
        request["scene"]["description"] = (
            "Channel charting v2: non-repeating Bezier trajectories inside the "
            "validated building-free rectangle, with measurement horn orientation"
        )
        request["scene"]["radiomap"]["enabled"] = False
        request["output_dir"] = str(args.dataset_output_dir)
        request["sample_id"] = f"cc_v2_scene_{scene_index:03d}"
        request["scene"]["user_equipments"] = generate_scene_ues(
            request_template,
            rng,
            scene_index=scene_index,
            ue_count=args.ues,
            samples=args.samples,
            height=height,
            bounds=bounds,
            used_starts=used_starts,
            used_paths=used_paths,
        )
        path = args.output_dir / f"cc_v2_scene_{scene_index:03d}.json"
        path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
        print(path)

    return 0


def _trajectory_bounds(request: dict) -> tuple[float, float, float, float]:
    radiomap = request["scene"]["radiomap"]
    return (
        float(radiomap["x_min"]),
        float(radiomap["x_max"]),
        float(radiomap["y_min"]),
        float(radiomap["y_max"]),
    )


def generate_scene_ues(
    request_template: dict,
    rng: np.random.Generator,
    *,
    scene_index: int,
    ue_count: int,
    samples: int,
    height: float,
    bounds: tuple[float, float, float, float],
    used_starts: list[np.ndarray],
    used_paths: list[np.ndarray],
) -> list[dict]:
    ue_template = copy.deepcopy(request_template["scene"]["user_equipments"][0])
    pitch = float(ue_template["orientation_rad"][1])
    roll = float(ue_template["orientation_rad"][2])
    ues = []
    for ue_index in range(ue_count):
        anchors_xy, handles_xy, sampled_xy = sample_bezier_path(
            rng,
            bounds,
            bs_xy=np.asarray(request_template["scene"]["base_stations"][0]["position"][:2], dtype=np.float64),
            scene_index=scene_index,
            ue_index=ue_index,
            used_starts=used_starts,
            used_paths=used_paths,
            samples=samples,
        )
        used_starts.append(anchors_xy[0])
        used_paths.append(sampled_xy)
        points = [[float(x), float(y), height] for x, y in anchors_xy]
        handles = [
            [
                [float(pair[0][0]), float(pair[0][1]), height],
                [float(pair[1][0]), float(pair[1][1]), height],
            ]
            for pair in handles_xy
        ]
        yaw0 = yaw_from_tangent(anchors_xy, handles_xy, 0)
        ue = copy.deepcopy(ue_template)
        ue["id"] = f"ue{ue_index}"
        ue["position"] = points[0]
        ue["orientation_rad"] = [yaw0, pitch, roll]
        ue["panel"]["rows"] = 1
        ue["panel"]["cols"] = 1
        ue["panel"]["pattern"] = "isac_horn_77_81"
        ue["panel"]["element_diagram"] = "isac_horn_77_81"
        ue["panel"]["orientation_rad"] = [yaw0, pitch, roll]
        ue["trajectory"] = {
            "kind": "curve",
            "points": points,
            "bezier_handles": handles,
            "orientation_rad_points": [[yaw0, pitch, roll] for _ in points],
            "samples": samples,
            "start_static_fraction": 0.0,
            "end_static_fraction": 0.0,
            "easing": "smoothstep",
        }
        ues.append(ue)
    return ues


def sample_bezier_path(
    rng: np.random.Generator,
    bounds: tuple[float, float, float, float],
    *,
    bs_xy: np.ndarray,
    scene_index: int,
    ue_index: int,
    used_starts: list[np.ndarray],
    used_paths: list[np.ndarray],
    samples: int,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    x_min, x_max, y_min, y_max = bounds
    margin = 2.0
    inner = (x_min + margin, x_max - margin, y_min + margin, y_max - margin)
    for _ in range(10000):
        anchors = random_anchors(rng, inner, bs_xy=bs_xy, scene_index=scene_index, ue_index=ue_index)
        if used_starts and min(float(np.linalg.norm(anchors[0] - start)) for start in used_starts) < 3.0:
            continue
        segment_lengths = np.linalg.norm(np.diff(anchors, axis=0), axis=1)
        if np.min(segment_lengths) < 8.0 or np.sum(segment_lengths) < 45.0 or np.max(segment_lengths) > 58.0:
            continue
        handles = catmull_rom_bezier_handles(anchors, tension=float(rng.uniform(0.65, 1.05)), bounds=inner)
        sampled = sample_bezier(anchors, handles, samples=samples)
        if not inside_bounds(sampled, inner):
            continue
        if max_heading_error_to_bs(sampled, bs_xy) > math.radians(105.0):
            continue
        if used_paths:
            closest = min(symmetric_mean_nearest_distance(sampled, other) for other in used_paths)
            if closest < 1.75:
                continue
        return anchors, handles, sampled
    raise RuntimeError("Could not sample a diverse Bezier trajectory inside the configured rectangle")


def random_anchors(
    rng: np.random.Generator,
    bounds: tuple[float, float, float, float],
    *,
    bs_xy: np.ndarray,
    scene_index: int,
    ue_index: int,
) -> np.ndarray:
    x_min, x_max, y_min, y_max = bounds
    x_span = x_max - x_min

    start_x = rng.uniform(x_min, x_min + 0.42 * x_span)
    end_low = max(start_x + 45.0, x_min + 0.62 * x_span)
    end_x = rng.uniform(end_low, x_max)
    xs = np.sort(rng.uniform(start_x, end_x, 4))
    xs[0] = start_x
    xs[-1] = end_x

    lane_count = 6
    lane_index = (scene_index * 2 + ue_index) % lane_count
    lane_center = y_min + (lane_index + 0.5) * (y_max - y_min) / lane_count
    lane_center += rng.normal(0.0, 1.4)
    lateral_span = rng.uniform(5.0, 14.0)
    phase = rng.uniform(-math.pi, math.pi)
    ys = lane_center + np.sin(np.linspace(0.0, math.pi, 4) + phase) * lateral_span

    # Keep the overall motion pointing toward the BS side of the rectangle while
    # leaving enough lateral curvature for non-trivial channel-charting geometry.
    if bs_xy[1] < (y_min + y_max) * 0.5:
        ys += np.linspace(rng.uniform(1.0, 4.0), rng.uniform(-5.0, -1.0), 4)
    else:
        ys += np.linspace(rng.uniform(-4.0, -1.0), rng.uniform(1.0, 5.0), 4)

    anchors = np.stack([xs, ys], axis=1)
    anchors[:, 0] = np.clip(anchors[:, 0], x_min, x_max)
    anchors[:, 1] = np.clip(anchors[:, 1], y_min, y_max)
    anchors += rng.normal(0.0, [1.4, 1.0], anchors.shape)
    anchors[:, 0] = np.clip(anchors[:, 0], x_min, x_max)
    anchors[:, 1] = np.clip(anchors[:, 1], y_min, y_max)
    return anchors.astype(np.float64)


def max_heading_error_to_bs(points: np.ndarray, bs_xy: np.ndarray) -> float:
    tangents = np.zeros_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    tangents[1:-1] = points[2:] - points[:-2]
    yaw = np.arctan2(tangents[:, 1], tangents[:, 0])
    to_bs = bs_xy.reshape(1, 2) - points
    yaw_to_bs = np.arctan2(to_bs[:, 1], to_bs[:, 0])
    diff = np.arctan2(np.sin(yaw - yaw_to_bs), np.cos(yaw - yaw_to_bs))
    return float(np.max(np.abs(diff)))


def catmull_rom_bezier_handles(
    anchors: np.ndarray,
    *,
    tension: float,
    bounds: tuple[float, float, float, float],
) -> list[tuple[np.ndarray, np.ndarray]]:
    x_min, x_max, y_min, y_max = bounds
    handles = []
    for index, point in enumerate(anchors):
        prev_point = anchors[max(0, index - 1)]
        next_point = anchors[min(len(anchors) - 1, index + 1)]
        tangent = (next_point - prev_point) * tension
        handle_in = point - tangent / 6.0
        handle_out = point + tangent / 6.0
        if index == 0:
            handle_in = point.copy()
        if index == len(anchors) - 1:
            handle_out = point.copy()
        for handle in (handle_in, handle_out):
            handle[0] = np.clip(handle[0], x_min, x_max)
            handle[1] = np.clip(handle[1], y_min, y_max)
        handles.append((handle_in, handle_out))
    return handles


def sample_bezier(anchors: np.ndarray, handles: list[tuple[np.ndarray, np.ndarray]], samples: int) -> np.ndarray:
    dense = []
    per_segment = max(32, samples // max(1, len(anchors) - 1))
    for index in range(len(anchors) - 1):
        t = np.linspace(0.0, 1.0, per_segment, dtype=np.float64)
        if index:
            t = t[1:]
        p0 = anchors[index]
        p1 = handles[index][1]
        p2 = handles[index + 1][0]
        p3 = anchors[index + 1]
        dense.append(cubic(p0, p1, p2, p3, t))
    dense_xy = np.vstack(dense)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense_xy, axis=0), axis=1))])
    if arc[-1] <= 1e-12:
        return np.repeat(anchors[:1], samples, axis=0)
    target = np.linspace(0.0, arc[-1], samples)
    out = np.empty((samples, 2), dtype=np.float64)
    out[:, 0] = np.interp(target, arc, dense_xy[:, 0])
    out[:, 1] = np.interp(target, arc, dense_xy[:, 1])
    return out


def cubic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: np.ndarray) -> np.ndarray:
    tt = t.reshape(-1, 1)
    omt = 1.0 - tt
    return omt**3 * p0 + 3.0 * omt**2 * tt * p1 + 3.0 * omt * tt**2 * p2 + tt**3 * p3


def inside_bounds(points: np.ndarray, bounds: tuple[float, float, float, float]) -> bool:
    x_min, x_max, y_min, y_max = bounds
    return bool(
        np.all(points[:, 0] >= x_min)
        and np.all(points[:, 0] <= x_max)
        and np.all(points[:, 1] >= y_min)
        and np.all(points[:, 1] <= y_max)
    )


def mean_nearest_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = a[:, None, :] - b[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    return float(np.mean(np.min(distances, axis=1)))


def symmetric_mean_nearest_distance(a: np.ndarray, b: np.ndarray) -> float:
    return min(mean_nearest_distance(a, b), mean_nearest_distance(b, a))


def yaw_from_tangent(anchors: np.ndarray, handles: list[tuple[np.ndarray, np.ndarray]], index: int) -> float:
    tangent = handles[index][1] - anchors[index]
    if float(np.linalg.norm(tangent)) < 1e-9:
        tangent = anchors[min(index + 1, len(anchors) - 1)] - anchors[max(index - 1, 0)]
    return math.atan2(float(tangent[1]), float(tangent[0]))


if __name__ == "__main__":
    raise SystemExit(main())

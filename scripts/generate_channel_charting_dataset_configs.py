from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path


DEFAULT_INPUT = Path("scenes/4ue_1bs.json")
DEFAULT_OUTPUT_DIR = Path("scenes/channel_charting_dataset")
DEFAULT_DATASET_OUTPUT_DIR = Path("output/channel_charting_dataset")

# Bounds copied from the computed radiomap stored in output/radiomap_20260719_185512.h5
# and mirrored by scenes/4ue_1bs.json.
X_MIN = -48.347
X_MAX = 46.433
Y_MIN = -16.986
Y_MAX = 23.44


BASE_ROUTES = [
    [(-45.0, -14.5), (-28.0, -13.5), (-8.0, -14.0), (18.0, -13.5), (43.0, -12.5)],
    [(-44.0, 20.5), (-26.0, 18.0), (-3.0, 15.5), (22.0, 17.0), (43.0, 21.0)],
    [(-34.0, 7.0), (-18.0, 3.0), (0.0, 3.5), (19.0, 0.5), (38.0, -3.5)],
    [(-46.0, -5.5), (-25.0, -2.0), (2.0, -2.5), (29.0, 0.5), (44.0, 8.5)],
    [(-10.0, 22.0), (-2.0, 15.5), (6.0, 9.5), (14.0, 4.0), (23.0, 2.0), (32.0, 6.5)],
    [(5.0, -15.0), (12.0, -8.0), (21.0, 0.5), (31.0, 10.0), (43.0, 19.0)],
    [(-38.0, 13.0), (-21.0, 10.0), (-4.0, 8.0), (16.0, 9.0), (36.0, 12.0)],
    [(-42.0, -10.0), (-26.0, -7.5), (-7.0, -6.0), (12.0, -5.5), (34.0, -7.0)],
    [(-30.0, 21.0), (-20.0, 15.0), (-9.0, 9.0), (3.0, 4.0), (18.0, 3.0), (35.0, 6.0)],
    [(-47.0, 1.0), (-31.0, 4.0), (-12.0, 5.0), (8.0, 2.0), (28.0, -1.0), (45.0, 2.5)],
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-output-dir", type=Path, default=DEFAULT_DATASET_OUTPUT_DIR)
    parser.add_argument("--scenes", type=int, default=8)
    parser.add_argument("--ues", type=int, default=6)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    base_request = json.loads(args.input.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for scene_index in range(args.scenes):
        rng = random.Random(args.seed + scene_index)
        request = copy.deepcopy(base_request)
        request["scene"]["description"] = (
            "Channel charting dataset scene: horn UEs on smooth trajectories "
            "inside the validated radiomap region"
        )
        request["scene"]["radiomap"]["enabled"] = False
        request["output_dir"] = str(args.dataset_output_dir)
        request["sample_id"] = f"cc_scene_{scene_index:03d}"
        request["scene"]["user_equipments"] = _scene_ues(base_request, rng, scene_index, args.ues, args.samples)

        path = args.output_dir / f"cc_scene_{scene_index:03d}.json"
        path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
        print(path)

    return 0


def _scene_ues(base_request: dict, rng: random.Random, scene_index: int, count: int, samples: int) -> list[dict]:
    template = copy.deepcopy(base_request["scene"]["user_equipments"][0])
    roll = float(template["orientation_rad"][2])
    routes = []
    offset = scene_index % len(BASE_ROUTES)
    for index in range(count):
        route = BASE_ROUTES[(offset + index) % len(BASE_ROUTES)]
        routes.append(_jitter_route(route, rng, scene_index, index))

    ues = []
    for index, xy_points in enumerate(routes):
        ue = copy.deepcopy(template)
        ue["id"] = f"ue{index}"
        points = [[float(x), float(y), 0.5] for x, y in xy_points]
        orientations = [[_yaw_at(xy_points, point_index), 0.0, roll] for point_index in range(len(xy_points))]

        ue["position"] = points[0]
        ue["orientation_rad"] = orientations[0]
        ue["panel"]["rows"] = 1
        ue["panel"]["cols"] = 1
        ue["panel"]["pattern"] = "isac_horn_77_81"
        ue["panel"]["element_diagram"] = "isac_horn_77_81"
        ue["panel"]["orientation_rad"] = orientations[0]
        ue["trajectory"] = {
            "kind": "polyline",
            "points": points,
            "bezier_handles": [],
            "orientation_rad_points": orientations,
            "samples": samples,
            "start_static_fraction": 0.0,
            "end_static_fraction": 0.0,
            "easing": "smoothstep",
        }
        ues.append(ue)
    return ues


def _jitter_route(route: list[tuple[float, float]], rng: random.Random, scene_index: int, ue_index: int) -> list[tuple[float, float]]:
    dx = rng.uniform(-1.75, 1.75) + 0.35 * ((scene_index % 3) - 1)
    dy = rng.uniform(-1.25, 1.25) + 0.25 * ((ue_index % 3) - 1)
    scale_y = 1.0 + rng.uniform(-0.035, 0.035)
    out = []
    for x, y in route:
        local_x = x + dx + rng.uniform(-0.45, 0.45)
        local_y = y * scale_y + dy + rng.uniform(-0.35, 0.35)
        out.append((_clip(local_x, X_MIN + 1.0, X_MAX - 1.0), _clip(local_y, Y_MIN + 1.0, Y_MAX - 1.0)))
    return out


def _yaw_at(points: list[tuple[float, float]], index: int) -> float:
    if index == 0:
        dx = points[1][0] - points[0][0]
        dy = points[1][1] - points[0][1]
    elif index == len(points) - 1:
        dx = points[-1][0] - points[-2][0]
        dy = points[-1][1] - points[-2][1]
    else:
        dx = points[index + 1][0] - points[index - 1][0]
        dy = points[index + 1][1] - points[index - 1][1]
    return math.atan2(dy, dx)


def _clip(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), lo), hi)


if __name__ == "__main__":
    raise SystemExit(main())

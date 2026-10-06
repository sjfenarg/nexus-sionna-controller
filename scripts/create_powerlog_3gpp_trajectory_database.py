"""Build a reproducible PowerLog / 3GPP trajectory request and visual summaries."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import distance_transform_edt

from isac_6d_sampler.core.config_io import read_request
from isac_6d_sampler.core.model import TrajectorySpec
from isac_6d_sampler.core.radiomap_occupancy import radiomap_inside_building_mask
from isac_6d_sampler.core.trajectories import radiomap_grid_shape, rotate_radiomap_xy, sample_trajectory


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs" / "radiomap_db.json"
CONFIG_OUT = ROOT / "configs" / "powerlog_3gpp_200_trajectories.json"
PREVIEW_OUT = ROOT / "configs" / "powerlog_3gpp_200_trajectories_preview.json"
PLOTS = ROOT / "output" / "powerlog_3gpp_200_trajectories_final"
SEED = 20261006
FAMILIES = ("line", "arc", "s_curve", "zigzag", "loop")
COLORS = {
    "line": "#2563eb", "arc": "#f97316", "s_curve": "#9333ea",
    "zigzag": "#0891b2", "loop": "#16a34a",
}


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    request = read_request(SOURCE)
    radiomap = request.scene.radiomap
    nx, ny = radiomap_grid_shape(radiomap)
    inside = radiomap_inside_building_mask(request.scene.scenario_path, radiomap).reshape(ny, nx)
    clearance = distance_transform_edt(~inside, sampling=(radiomap.y_spacing, radiomap.x_spacing))
    rng = np.random.default_rng(SEED)
    tracks = build_tracks(rng, radiomap, clearance)
    objects = build_objects(rng, radiomap, clearance)

    output = deepcopy(source)
    output["scene"]["description"] = (
        "PowerLog BS and UE antenna configuration with 200 distinct outdoor UE trajectories "
        "inside the saved rotated radiomap bounds, plus human, AGV and car 3GPP sensing targets. "
        "Radiomap mode is disabled so the UE trajectories are simulated."
    )
    output["scene"]["radiomap"]["enabled"] = False
    output["scene"]["user_equipments"] = [track["ue"] for track in tracks]
    output["scene"]["objects"] = objects
    output["scene"]["timeframe_interval_s"] = 0.2
    output["sample_id"] = "powerlog_3gpp_200_trajectories"
    output["output_dir"] = "output/powerlog_3gpp_200_trajectories/database"
    output["sionna"]["sensing_channel"] = "combined"
    output["sionna"]["seed"] = SEED
    output["sionna"]["batch_timeframes"] = 1
    CONFIG_OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    preview = deepcopy(output)
    preview["sample_id"] = "powerlog_3gpp_200_trajectories_preview"
    preview["output_dir"] = "output/powerlog_3gpp_200_trajectories/preview"
    preview["bands"] = [deepcopy(output["bands"][0])]
    preview["bands"][0]["points"] = 64
    preview["sionna"]["samples_per_src"] = 20_000
    preview["sionna"]["max_depth"] = 2
    preview["sionna"]["rcs_samples_per_sp"] = 10_000
    preview["sionna"]["rcs_buffer_size_per_sp"] = 10_000
    PREVIEW_OUT.write_text(json.dumps(preview, indent=2) + "\n", encoding="utf-8")

    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_overview(tracks, objects, radiomap, inside, source["scene"]["base_stations"][0]["position"])
    plot_families(tracks, radiomap, inside)
    plot_spacing(tracks)
    plot_objects(objects, tracks, radiomap, inside, source["scene"]["base_stations"][0]["position"])
    summary = {
        "source": str(SOURCE.relative_to(ROOT)),
        "gui_request": str(CONFIG_OUT.relative_to(ROOT)),
        "gui_preview_request": str(PREVIEW_OUT.relative_to(ROOT)),
        "seed": SEED,
        "trajectories": len(tracks),
        "families": dict(Counter(track["family"] for track in tracks)),
        "samples_per_trajectory": dict(Counter(track["samples"] for track in tracks)),
        "targets": [dict(id=obj["id"], type=obj["object_name"]) for obj in objects],
        "radiomap_bounds": {key: source["scene"]["radiomap"][key] for key in (
            "x_min", "x_max", "y_min", "y_max", "height", "rotation_deg",
        )},
        "outside_grid_points": int((~inside).sum()),
        "inside_grid_points": int(inside.sum()),
    }
    (PLOTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_tracks(rng, rm, clearance) -> list[dict]:
    tracks = []
    starts = []
    ue_template = json.loads(SOURCE.read_text(encoding="utf-8"))["scene"]["user_equipments"][0]
    for family in FAMILIES:
        for family_index in range(40):
            for _attempt in range(30000):
                local_points = candidate_path(rng, rm, family)
                if any(np.linalg.norm(local_points[0] - other) < 1.8 for other in starts):
                    continue
                samples = (12, 16, 20, 24)[(family_index + FAMILIES.index(family)) % 4]
                easing = "smoothstep" if family_index % 3 == 0 else "linear"
                static_fraction = 0.12 if family_index % 7 == 0 else 0.0
                trajectory = make_trajectory(local_points, rm, family, samples, easing, static_fraction)
                dense = deepcopy(trajectory)
                dense.samples = 161
                dense.start_static_fraction = dense.end_static_fraction = 0.0
                dense.easing = "linear"
                sampled = sample_trajectory(dense)
                if not safe_world_points(sampled[:, :2], rm, clearance, 2.2):
                    continue
                if np.linalg.norm(sampled[-1, :2] - sampled[0, :2]) < 2.0 and family != "loop":
                    continue
                starts.append(local_points[0])
                ue = deepcopy(ue_template)
                ue["id"] = f"ue_{len(tracks):03d}_{family}"
                ue["position"] = list(trajectory.points[0])
                ue["trajectory"] = trajectory_to_dict(trajectory)
                ue["orientation_rad"] = [float(np.arctan2(sampled[1, 1]-sampled[0, 1], sampled[1, 0]-sampled[0, 0])), 0.0, 0.0]
                ue["panel"]["orientation_rad"] = ue["orientation_rad"]
                tracks.append({
                    "family": family, "ue": ue, "samples": samples, "easing": easing,
                    "static_fraction": static_fraction, "sampled": sampled,
                    "length_m": float(np.sum(np.linalg.norm(np.diff(sampled[:, :2], axis=0), axis=1))),
                })
                break
            else:
                raise RuntimeError(f"Could not place trajectory {family} {family_index}")
    return tracks


def candidate_path(rng, rm, family) -> np.ndarray:
    margin = 4.5
    start = np.asarray([
        rng.uniform(rm.x_min + margin, rm.x_max - margin),
        rng.uniform(rm.y_min + margin, rm.y_max - margin),
    ])
    angle = rng.uniform(-np.pi, np.pi)
    direction = np.asarray([np.cos(angle), np.sin(angle)])
    lateral = np.asarray([-direction[1], direction[0]])
    length = rng.uniform(10.0, 32.0)
    if family == "line":
        return np.asarray([start, start + length * direction])
    if family == "arc":
        bend = rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 8.0)
        return np.asarray([start, start + .5 * length * direction + bend * lateral,
                           start + length * direction + 1.4 * bend * lateral])
    if family == "s_curve":
        bend = rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 6.0)
        return np.asarray([start, start + .33 * length * direction + bend * lateral,
                           start + .67 * length * direction - bend * lateral,
                           start + length * direction])
    if family == "zigzag":
        bend = rng.choice([-1.0, 1.0]) * rng.uniform(2.5, 5.0)
        return np.asarray([start, start + .25 * length * direction + bend * lateral,
                           start + .5 * length * direction - bend * lateral,
                           start + .75 * length * direction + bend * lateral,
                           start + length * direction])
    width = rng.uniform(4.0, 9.0)
    depth = rng.uniform(4.0, 9.0)
    return np.asarray([start, start + width * direction,
                       start + width * direction + depth * lateral,
                       start + depth * lateral, start])


def make_trajectory(local_points, rm, family, samples, easing="linear", static_fraction=0.0) -> TrajectorySpec:
    world_xy = rotate_radiomap_xy(local_points, rm)
    points = [(float(x), float(y), float(rm.height)) for x, y in world_xy]
    kind = "linear" if family == "line" else "curve" if family in ("arc", "s_curve") else "polyline"
    orientation_points = []
    if kind != "curve":
        for index in range(len(world_xy)):
            previous = world_xy[max(0, index - 1)]
            following = world_xy[min(len(world_xy) - 1, index + 1)]
            tangent = following - previous
            orientation_points.append((float(np.arctan2(tangent[1], tangent[0])), 0.0, 0.0))
    return TrajectorySpec(
        kind=kind, points=points, samples=samples, easing=easing,
        start_static_fraction=static_fraction, end_static_fraction=static_fraction,
        orientation_rad_points=orientation_points,
    )


def trajectory_to_dict(spec):
    return {
        "kind": spec.kind, "points": [list(point) for point in spec.points],
        "bezier_handles": [[list(a), list(b)] for a, b in spec.bezier_handles],
        "orientation_rad_points": [list(point) for point in spec.orientation_rad_points],
        "samples": spec.samples, "start_static_fraction": spec.start_static_fraction,
        "end_static_fraction": spec.end_static_fraction, "easing": spec.easing,
    }


def safe_world_points(world_xy, rm, clearance, required_m):
    angle = np.deg2rad(rm.rotation_deg)
    center = np.asarray([(rm.x_min + rm.x_max) / 2, (rm.y_min + rm.y_max) / 2])
    inverse = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    local = center + (world_xy - center) @ inverse
    ix = np.rint((local[:, 0] - rm.x_min) / rm.x_spacing).astype(int)
    iy = np.rint((local[:, 1] - rm.y_min) / rm.y_spacing).astype(int)
    margin = required_m + 1.0
    within = (
        (local[:, 0] >= rm.x_min + margin) & (local[:, 0] <= rm.x_max - margin)
        & (local[:, 1] >= rm.y_min + margin) & (local[:, 1] <= rm.y_max - margin)
        & (ix >= 0) & (ix < clearance.shape[1]) & (iy >= 0) & (iy < clearance.shape[0])
    )
    if not np.all(within):
        return False
    return bool(np.all(clearance[iy, ix] >= required_m))


def build_objects(rng, rm, clearance):
    types = ["HUMAN_3GPP", "HUMAN_3GPP", "AGV_3GPP", "AGV_3GPP", "CAR_3GPP"]
    height = {"HUMAN_3GPP": .875, "AGV_3GPP": .25, "CAR_3GPP": .8}
    objects = []
    for index, target_type in enumerate(types):
        for _attempt in range(30000):
            family = ("line", "arc", "zigzag", "s_curve", "loop")[index]
            local = candidate_path(rng, rm, family)
            spec = make_trajectory(local, rm, family, 24)
            dense = deepcopy(spec)
            dense.samples = 161
            sampled = sample_trajectory(dense)
            if not safe_world_points(sampled[:, :2], rm, clearance, 3.5 if target_type == "CAR_3GPP" else 2.5):
                continue
            z = height[target_type]
            spec.points = [(x, y, z) for x, y, _ in spec.points]
            objects.append({
                "id": f"target_{index:02d}_{target_type.lower()}",
                "object_name": target_type,
                "position": list(spec.points[0]),
                "orientation_rad": [float(np.arctan2(sampled[1, 1]-sampled[0, 1], sampled[1, 0]-sampled[0, 0])), 0.0, 0.0],
                "trajectory": trajectory_to_dict(spec),
                "sensing": {"model_type": 2, "dimensions": None, "mesh": None,
                            "random_sigma_s": False, "random_phases": False, "random_xpr": False},
            })
            break
        else:
            raise RuntimeError(f"Could not place target {target_type}")
    return objects


def map_background(ax, rm, inside):
    iy, ix = np.nonzero(inside)
    local = np.column_stack([rm.x_min + ix * rm.x_spacing, rm.y_min + iy * rm.y_spacing])
    world = rotate_radiomap_xy(local, rm)
    ax.scatter(world[:, 0], world[:, 1], s=1.8, c="#94a3b8", alpha=.38, rasterized=True, zorder=1)
    bounds = np.asarray([[rm.x_min, rm.y_min], [rm.x_max, rm.y_min],
                         [rm.x_max, rm.y_max], [rm.x_min, rm.y_max], [rm.x_min, rm.y_min]])
    border = rotate_radiomap_xy(bounds, rm)
    ax.plot(border[:, 0], border[:, 1], c="#334155", lw=1.5, zorder=2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Scene X (m)")
    ax.set_ylabel("Scene Y (m)")
    ax.grid(alpha=.2)


def plot_overview(tracks, objects, rm, inside, bs):
    fig, ax = plt.subplots(figsize=(12, 10), dpi=180)
    map_background(ax, rm, inside)
    for family in FAMILIES:
        lines = [track["sampled"][:, :2] for track in tracks if track["family"] == family]
        ax.add_collection(LineCollection(lines, colors=COLORS[family], linewidths=.8, alpha=.55, zorder=3))
    for obj in objects:
        samples = sample_trajectory(TrajectorySpec(**obj["trajectory"]))
        ax.plot(samples[:, 0], samples[:, 1], c="#dc2626", lw=2.8, zorder=5)
        ax.scatter(samples[0, 0], samples[0, 1], s=65, c="#dc2626", edgecolors="white", zorder=6)
    ax.scatter(bs[0], bs[1], marker="*", s=280, c="#facc15", edgecolors="#0f172a", zorder=7)
    ax.set_title("PowerLog 3GPP trajectory database · 200 UE paths and 5 moving targets", loc="left", pad=15)
    ax.legend(handles=[Line2D([], [], c=COLORS[name], lw=2, label=name.replace("_", " ")) for name in FAMILIES]
              + [Line2D([], [], c="#dc2626", lw=2.8, label="3GPP targets"),
                 Line2D([], [], marker="*", color="none", markerfacecolor="#facc15", markersize=12, label="PowerLog BS")],
              loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "01_trajectory_overview.png")
    plt.close(fig)


def plot_families(tracks, rm, inside):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=170)
    for ax, family in zip(axes.flat, FAMILIES):
        map_background(ax, rm, inside)
        family_tracks = [track for track in tracks if track["family"] == family]
        ax.add_collection(LineCollection([track["sampled"][:, :2] for track in family_tracks],
                                         colors=COLORS[family], linewidths=1.4, alpha=.8, zorder=3))
        ax.set_title(f"{family.replace('_', ' ').title()} · {len(family_tracks)} paths")
    axes.flat[-1].axis("off")
    fig.suptitle("Trajectory families inside the rotated radiomap boundary", fontsize=16, y=.995)
    fig.tight_layout()
    fig.savefig(PLOTS / "02_trajectory_families.png")
    plt.close(fig)


def plot_spacing(tracks):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=180)
    lengths = np.asarray([track["length_m"] for track in tracks])
    axes[0].hist(lengths, bins=20, color="#2563eb", edgecolor="white")
    axes[0].set(xlabel="Path length (m)", ylabel="Trajectories", title="Movement distance")
    counts = Counter(track["samples"] for track in tracks)
    axes[1].bar([str(key) for key in sorted(counts)], [counts[key] for key in sorted(counts)], color="#0891b2")
    axes[1].set(xlabel="Samples per trajectory", ylabel="Trajectories", title="Temporal density")
    spacing = [track["length_m"] / max(track["samples"] - 1, 1) for track in tracks]
    axes[2].hist(spacing, bins=20, color="#f97316", edgecolor="white")
    axes[2].set(xlabel="Mean sample spacing (m)", ylabel="Trajectories", title="Spatial sampling")
    fig.suptitle("Trajectory length and spacing variation", fontsize=15)
    fig.tight_layout()
    fig.savefig(PLOTS / "03_length_and_spacing.png")
    plt.close(fig)


def plot_objects(objects, tracks, rm, inside, bs):
    fig, ax = plt.subplots(figsize=(12, 9), dpi=180)
    map_background(ax, rm, inside)
    colors = ["#dc2626", "#f97316", "#7c3aed", "#a855f7", "#0f766e"]
    for obj, color in zip(objects, colors, strict=True):
        samples = sample_trajectory(TrajectorySpec(**obj["trajectory"]))
        ax.plot(samples[:, 0], samples[:, 1], c=color, lw=2.8, label=f"{obj['id']} ({obj['object_name']})", zorder=4)
        ax.scatter(samples[0, 0], samples[0, 1], c=color, s=65, marker="o", zorder=5)
        ax.scatter(samples[-1, 0], samples[-1, 1], c=color, s=65, marker="x", zorder=5)
    ax.scatter(bs[0], bs[1], marker="*", s=280, c="#facc15", edgecolors="#0f172a", zorder=6)
    ax.set_title("3GPP sensing targets · starts (dots), ends (crosses)", loc="left", pad=15)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "04_3gpp_target_paths.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

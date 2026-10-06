"""Generate 200 distinct GUI-ready PowerLog / 3GPP trajectory scenes.

Run from the repository root with ``python scripts/create_powerlog_3gpp_ensemble.py``.
The input radiomap supplies the rotated area, BS and antenna/band settings.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from xml.etree import ElementTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from scipy.ndimage import distance_transform_edt

from isac_6d_sampler.core.config_io import read_request
from isac_6d_sampler.core.model import TrajectorySpec
from isac_6d_sampler.core.radiomap_occupancy import _read_ply_mesh, radiomap_inside_building_mask
from isac_6d_sampler.core.trajectories import radiomap_grid_shape, rotate_radiomap_xy, sample_trajectory
from isac_6d_sampler.core.validation import validate_request

from create_powerlog_3gpp_trajectory_database import map_background, safe_world_points, trajectory_to_dict


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs" / "radiomap_db.json"
OUT = ROOT / "configs" / "powerlog_3gpp_ensemble"
PLOTS = ROOT / "output" / "powerlog_3gpp_ensemble_200"
FAMILIES = ("line", "arc", "s_curve", "zigzag", "circle", "ellipse")
COLORS = dict(zip(FAMILIES, ("#2563eb", "#f97316", "#9333ea", "#0891b2", "#16a34a", "#65a30d")))
PROFILES = ("short", "medium", "long", "bimodal", "broad")
HEIGHTS = {"HUMAN_3GPP": 0.875, "AGV_3GPP": 0.25, "CAR_3GPP": 0.8}
KAPPA = 4.0 / 3.0 * np.tan(np.pi / 8.0)


def desired_length(rng, profile: str) -> float:
    if profile == "short":
        return float(rng.uniform(6, 12))
    if profile == "medium":
        return float(rng.uniform(12, 20))
    if profile == "long":
        return float(rng.uniform(20, 30))
    if profile == "bimodal":
        return float(rng.uniform(6, 11) if rng.random() < 0.5 else rng.uniform(22, 30))
    return float(rng.uniform(6, 30))


def local_path(rng, rm, family: str, length: float):
    margin = 4.0
    origin = np.array([rng.uniform(rm.x_min + margin, rm.x_max - margin),
                       rng.uniform(rm.y_min + margin, rm.y_max - margin)])
    angle = rng.uniform(-np.pi, np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)])
    side = np.array([-direction[1], direction[0]])
    if family == "line":
        return np.array([origin, origin + length * direction]), None
    if family == "arc":
        bend = rng.choice([-1.0, 1.0]) * length * rng.uniform(.12, .25)
        return np.array([origin, origin + .5 * length * direction + bend * side,
                         origin + length * direction + 1.4 * bend * side]), None
    if family == "s_curve":
        bend = rng.choice([-1.0, 1.0]) * length * rng.uniform(.12, .22)
        return np.array([origin, origin + .33 * length * direction + bend * side,
                         origin + .67 * length * direction - bend * side,
                         origin + length * direction]), None
    if family == "zigzag":
        bend = rng.choice([-1.0, 1.0]) * length * rng.uniform(.06, .13)
        return np.array([origin, origin + .25 * length * direction + bend * side,
                         origin + .5 * length * direction - bend * side,
                         origin + .75 * length * direction + bend * side,
                         origin + length * direction]), None

    # Four cubic Bezier quarters give an actual closed circle/ellipse, with
    # tangent-continuous joins; an axis-aligned rectangle is never used.
    ratio = 1.0 if family == "circle" else float(rng.uniform(.48, .78))
    radius_x = length / (2 * np.pi * np.sqrt((1 + ratio * ratio) / 2))
    radius_y = ratio * radius_x
    directions = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.], [1., 0.]])
    tangents = np.array([[0., 1.], [-1., 0.], [0., -1.], [1., 0.], [0., 1.]])
    radii = np.array([radius_x, radius_y])
    anchors = origin + (directions * radii) @ np.column_stack([direction, side]).T
    offsets = (KAPPA * tangents * radii) @ np.column_stack([direction, side]).T
    return anchors, offsets


def make_spec(local, offsets, rm, family, samples, z, easing, pause):
    world = rotate_radiomap_xy(local, rm)
    points = [(float(x), float(y), float(z)) for x, y in world]
    kind = "linear" if family == "line" else "polyline" if family == "zigzag" else "curve"
    handles = []
    if offsets is not None:
        # Rotation affects vectors only, so subtract the transformed origin.
        origin = rotate_radiomap_xy(local[:1], rm)[0]
        transformed = rotate_radiomap_xy(local[:1] + offsets, rm) - origin
        for point, tangent in zip(points, transformed):
            handles.append(((point[0] - float(tangent[0]), point[1] - float(tangent[1]), z),
                            (point[0] + float(tangent[0]), point[1] + float(tangent[1]), z)))
    orientations = []
    if kind != "curve":
        for i in range(len(world)):
            tangent = world[min(i + 1, len(world) - 1)] - world[max(i - 1, 0)]
            orientations.append((float(np.arctan2(tangent[1], tangent[0])), 0., 0.))
    return TrajectorySpec(kind=kind, points=points, bezier_handles=handles,
                          orientation_rad_points=orientations, samples=samples,
                          easing=easing, start_static_fraction=pause, end_static_fraction=pause)


def quick_clearance(local, rm, clearance, required_m):
    """Reject obviously blocked paths using local line subdivisions."""
    fractions = np.linspace(0., 1., 9)
    probes = local[:-1, None, :] + fractions[None, :, None] * np.diff(local, axis=0)[:, None, :]
    probes = probes.reshape(-1, 2)
    margin = required_m + 1.
    if np.any(probes[:, 0] < rm.x_min + margin) or np.any(probes[:, 0] > rm.x_max - margin):
        return False
    if np.any(probes[:, 1] < rm.y_min + margin) or np.any(probes[:, 1] > rm.y_max - margin):
        return False
    ix = np.rint((probes[:, 0] - rm.x_min) / rm.x_spacing).astype(int)
    iy = np.rint((probes[:, 1] - rm.y_min) / rm.y_spacing).astype(int)
    return bool(np.all(clearance[iy, ix] >= required_m))


def surface_footprint(mesh_path, rm, spacing=.5):
    """Rasterize upward-facing terrain triangles into radiomap-local XY."""
    vertices, faces = _read_ply_mesh(str(mesh_path.resolve()), mesh_path.stat().st_mtime_ns)
    xs = np.arange(rm.x_min, rm.x_max + spacing * .5, spacing)
    ys = np.arange(rm.y_min, rm.y_max + spacing * .5, spacing)
    mask = np.zeros((len(ys), len(xs)), dtype=bool)
    center = np.array([(rm.x_min + rm.x_max) / 2, (rm.y_min + rm.y_max) / 2])
    angle = np.deg2rad(rm.rotation_deg)
    inverse = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    for triangle in vertices[faces]:
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        if abs(normal[2]) / max(np.linalg.norm(normal), 1e-12) < .7:
            continue
        tri = center + (triangle[:, :2] - center) @ inverse
        ix = np.flatnonzero((xs >= tri[:, 0].min()) & (xs <= tri[:, 0].max()))
        iy = np.flatnonzero((ys >= tri[:, 1].min()) & (ys <= tri[:, 1].max()))
        if not len(ix) or not len(iy):
            continue
        a, b, c = tri
        u, v = b - a, c - a
        determinant = u[0] * v[1] - u[1] * v[0]
        if abs(determinant) < 1e-12:
            continue
        dx = xs[ix][None, :] - a[0]
        dy = ys[iy][:, None] - a[1]
        first = (dx * v[1] - dy * v[0]) / determinant
        second = (u[0] * dy - u[1] * dx) / determinant
        mask[np.ix_(iy, ix)] |= ((first >= -1e-7) & (second >= -1e-7)
                                  & (first + second <= 1 + 1e-7))
    return mask


def driving_clearance(rm, scenario_path):
    mesh_dir = scenario_path.parent / "meshes"
    ground = surface_footprint(mesh_dir / "GROUND_Obj.ply", rm)
    grass = surface_footprint(mesh_dir / "GRASS_Obj.ply", rm)
    xs = np.arange(rm.x_min, rm.x_max + .25, .5)
    ys = np.arange(rm.y_min, rm.y_max + .25, .5)
    xx, yy = np.meshgrid(xs, ys)
    world = rotate_radiomap_xy(np.column_stack([xx.ravel(), yy.ravel()]), rm).reshape(*xx.shape, 2)
    trees = np.zeros_like(ground)
    for lo, hi in tree_trunk_bounds(scenario_path):
        trees |= ((world[:, :, 0] >= lo[0] - .5) & (world[:, :, 0] <= hi[0] + .5)
                  & (world[:, :, 1] >= lo[1] - .5) & (world[:, :, 1] <= hi[1] + .5))
    driveable = ground & ~grass & ~trees
    # Pad the boundary to exclude paths close to the edge of the surveyed map.
    driveable[[0, -1], :] = False
    driveable[:, [0, -1]] = False
    return distance_transform_edt(driveable, sampling=(.5, .5)), ground, grass, trees


def require_tree_instances(scenario_path):
    shapes = ElementTree.parse(scenario_path).getroot().findall(".//shape")
    tree_ids = {shape.get("id") for shape in shapes if shape.get("type") == "ply"}
    required = {"TREE_V10_Final_00", "TREE_V10_Final_01"}
    if not required.issubset(tree_ids):
        raise ValueError(f"The configured 6D scene is missing tree instances: {required - tree_ids}")


def tree_trunk_bounds(scenario_path):
    """Read tree vertex sections only; ignore large face sections."""
    root = ElementTree.parse(scenario_path).getroot()
    bounds = []
    for shape in root.findall(".//shape"):
        if not (shape.get("id") or "").startswith("TREE_V10_Final_"):
            continue
        filename = shape.find("./string[@name='filename']")
        path = scenario_path.parent / filename.get("value")
        with path.open("rb") as stream:
            header = []
            while True:
                line = stream.readline().decode("ascii").strip()
                header.append(line)
                if line == "end_header":
                    break
            count = int(next(line.split()[2] for line in header if line.startswith("element vertex ")))
            vertices = np.fromfile(stream, dtype="<f4", count=count * 3).reshape(-1, 3)
        low = vertices[vertices[:, 2] < 1.6, :2]
        if len(low):
            bounds.append((low.min(axis=0), low.max(axis=0)))
    return bounds


def safe_car_path(world_xy, rm, drive_clearance, required_m=1.5):
    center = np.array([(rm.x_min + rm.x_max) / 2, (rm.y_min + rm.y_max) / 2])
    angle = np.deg2rad(rm.rotation_deg)
    inverse = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    local = center + (world_xy - center) @ inverse
    ix = np.rint((local[:, 0] - rm.x_min) / .5).astype(int)
    iy = np.rint((local[:, 1] - rm.y_min) / .5).astype(int)
    if np.any(ix < 0) or np.any(iy < 0) or np.any(ix >= drive_clearance.shape[1]) or np.any(iy >= drive_clearance.shape[0]):
        return False
    return bool(np.all(drive_clearance[iy, ix] >= required_m))


def car_local_path(rng, rm):
    family = str(rng.choice(("line", "arc")))
    length = float(rng.uniform(7, 25))
    if family == "line":
        local, _ = local_path(rng, rm, "line", length)
        return family, local
    margin = 4.
    start = np.array([rng.uniform(rm.x_min + margin, rm.x_max - margin),
                      rng.uniform(rm.y_min + margin, rm.y_max - margin)])
    angle = rng.uniform(-np.pi, np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)])
    side = np.array([-direction[1], direction[0]])
    bend = rng.choice([-1., 1.]) * length * rng.uniform(.035, .10)
    return family, np.array([start, start + .5 * length * direction + bend * side,
                             start + length * direction + 1.4 * bend * side])


def make_tracks(rng, rm, clearance, ue_template, profile):
    tracks = []
    # 40 lines, 40 arcs, 40 S-curves, 40 zigzags, 20 circles, 20 ellipses.
    families = ["line"] * 40 + ["arc"] * 40 + ["s_curve"] * 40 + ["zigzag"] * 40 + ["circle"] * 20 + ["ellipse"] * 20
    rng.shuffle(families)
    for index, family in enumerate(families):
        for attempt in range(6000):
            length = desired_length(rng, profile)
            local, offsets = local_path(rng, rm, family, length)
            if not quick_clearance(local, rm, clearance, 2.2):
                continue
            samples = (12, 16, 20, 24, 32)[(index + int(rng.integers(5))) % 5]
            easing = "smoothstep" if index % 4 == 0 else "linear"
            pause = .10 if index % 9 == 0 else 0.
            spec = make_spec(local, offsets, rm, family, samples, rm.height, easing, pause)
            dense = deepcopy(spec)
            dense.samples = 97
            dense.start_static_fraction = dense.end_static_fraction = 0.
            dense.easing = "linear"
            sampled = sample_trajectory(dense)
            if not safe_world_points(sampled[:, :2], rm, clearance, 2.2):
                continue
            # Avoid identical start positions within a scene, while allowing
            # paths to cross: the database describes independent trials.
            if any(np.linalg.norm(sampled[0, :2] - t["sampled"][0, :2]) < 1.0 for t in tracks):
                continue
            ue = deepcopy(ue_template)
            ue["id"] = f"ue_{index:03d}_{family}"
            ue["position"] = list(spec.points[0])
            ue["trajectory"] = trajectory_to_dict(spec)
            yaw = float(np.arctan2(sampled[1, 1] - sampled[0, 1], sampled[1, 0] - sampled[0, 0]))
            ue["orientation_rad"] = [yaw, 0., 0.]
            ue["panel"]["orientation_rad"] = [yaw, 0., 0.]
            tracks.append({"family": family, "ue": ue, "sampled": sampled,
                           "length_m": float(np.linalg.norm(np.diff(sampled[:, :2], axis=0), axis=1).sum()),
                           "samples": samples})
            break
        else:
            raise RuntimeError(f"Unable to place {family} path {index} for profile {profile}")
    return tracks


def make_targets(rng, rm, clearance, count, drive_clearance, types_override=None):
    targets = []
    available = list(HEIGHTS)
    rng.shuffle(available)
    types = (list(types_override) if types_override is not None else
             available[:min(count, 3)] + list(rng.choice(available, size=max(0, count - 3))))
    if types_override is None:
        rng.shuffle(types)
    for index, target_type in enumerate(types):
        for attempt in range(6000):
            if target_type == "CAR_3GPP":
                family, local = car_local_path(rng, rm)
                offsets = None
            else:
                family = str(rng.choice(FAMILIES))
                length = float(rng.uniform(7, 24))
                local, offsets = local_path(rng, rm, family, length)
            z = HEIGHTS[target_type]
            clearance_m = 3.5 if target_type == "CAR_3GPP" else 2.5
            if not quick_clearance(local, rm, clearance, clearance_m):
                continue
            spec = make_spec(local, offsets, rm, family, 24, z, "linear", 0.)
            dense = deepcopy(spec)
            dense.samples = 97
            sampled = sample_trajectory(dense)
            if not safe_world_points(sampled[:, :2], rm, clearance, clearance_m):
                continue
            if target_type == "CAR_3GPP":
                # Check at finer than 0.5 m spacing between stored timeframes.
                fine = deepcopy(spec)
                fine.samples = 257
                if not safe_car_path(sample_trajectory(fine)[:, :2], rm, drive_clearance):
                    continue
            if any(np.linalg.norm(sampled[0, :2] - np.array(obj["position"][:2])) < 2. for obj in targets):
                continue
            yaw = float(np.arctan2(sampled[1, 1] - sampled[0, 1], sampled[1, 0] - sampled[0, 0]))
            targets.append({
                "id": f"target_{index:02d}_{target_type.lower()}",
                "object_name": target_type,
                "position": list(spec.points[0]),
                "orientation_rad": [yaw, 0., 0.],
                "trajectory": trajectory_to_dict(spec),
                "sensing": {"model_type": 2, "dimensions": None, "mesh": None,
                            "random_sigma_s": False, "random_phases": False, "random_xpr": False},
            })
            break
        else:
            raise RuntimeError(f"Unable to place target {index}")
    return targets


def plot_examples(examples, rm, inside, bs):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=160)
    for ax, (number, tracks, targets, profile) in zip(axes.flat, examples):
        map_background(ax, rm, inside)
        for family in FAMILIES:
            lines = [track["sampled"][:, :2] for track in tracks if track["family"] == family]
            ax.add_collection(LineCollection(lines, colors=COLORS[family], linewidths=.8, alpha=.65, zorder=3))
        for target in targets:
            sampled = sample_trajectory(TrajectorySpec(**target["trajectory"]))
            ax.plot(sampled[:, 0], sampled[:, 1], color="#dc2626", lw=2.5, zorder=5)
            ax.scatter(sampled[0, 0], sampled[0, 1], color="#dc2626", s=25, zorder=6)
        ax.scatter(bs[0], bs[1], marker="*", s=140, color="#facc15", edgecolors="black", zorder=7)
        ax.set_title(f"Scene {number:03d} | {profile} | {len(targets)} target(s)")
    axes.flat[-1].axis("off")
    fig.suptitle("200 paths per scene; gray points are building interiors", fontsize=16)
    fig.tight_layout()
    fig.savefig(PLOTS / "01_scene_examples.png")
    plt.close(fig)


def plot_distributions(manifest):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=170)
    for profile in PROFILES:
        values = [length for entry in manifest["scenes"] if entry["length_profile"] == profile
                  for length in entry["path_lengths_m"]]
        axes[0].hist(values, bins=np.arange(0, 55, 2), alpha=.48, label=profile, density=True)
    axes[0].set(xlabel="UE path length (m)", ylabel="Density", title="Five different length distributions")
    axes[0].legend()
    counts = Counter(entry["target_count"] for entry in manifest["scenes"])
    axes[1].bar(list(counts), list(counts.values()), color="#dc2626")
    axes[1].set(xlabel="3GPP targets in JSON", ylabel="JSON files", title="Target count distribution", xticks=range(1, 6))
    fig.tight_layout()
    fig.savefig(PLOTS / "02_distributions.png")
    plt.close(fig)


def plot_loops(examples):
    _, tracks, _, _ = examples[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=180)
    for ax, family in zip(axes, ("circle", "ellipse")):
        paths = [track for track in tracks if track["family"] == family]
        for path in paths[:4]:
            sampled = path["sampled"][:, :2]
            ax.plot(sampled[:, 0] - sampled[:, 0].mean(), sampled[:, 1] - sampled[:, 1].mean(), lw=1.7)
        ax.set(aspect="equal", xlabel="Relative X (m)", ylabel="Relative Y (m)", title=f"{family.title()} examples (four of 20)")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(PLOTS / "03_true_circle_ellipse_loops.png")
    plt.close(fig)


def plot_target_diversity(rm, inside, bs):
    """Compare one through five independently positioned moving targets."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), dpi=160)
    target_colors = {"HUMAN_3GPP": "#dc2626", "AGV_3GPP": "#7c3aed", "CAR_3GPP": "#0f766e"}
    for count, ax in enumerate(axes.flat[:5], start=1):
        number = count - 1
        data = json.loads((OUT / f"scene_{number:03d}.json").read_text(encoding="utf-8"))
        map_background(ax, rm, inside)
        for ue in data["scene"]["user_equipments"]:
            path = sample_trajectory(TrajectorySpec(**ue["trajectory"]))
            ax.plot(path[:, 0], path[:, 1], color="#60a5fa", alpha=.23, lw=.55, zorder=2)
        for target in data["scene"]["objects"]:
            path = sample_trajectory(TrajectorySpec(**target["trajectory"]))
            color = target_colors[target["object_name"]]
            ax.plot(path[:, 0], path[:, 1], color=color, lw=3, zorder=4)
            ax.scatter(path[0, 0], path[0, 1], color=color, marker="o", s=35, zorder=5)
            ax.scatter(path[-1, 0], path[-1, 1], color=color, marker="x", s=35, zorder=5)
        ax.scatter(bs[0], bs[1], marker="*", s=150, color="#facc15", edgecolors="black", zorder=6)
        ax.set_title(f"Scene {number:03d} | {count} target(s)")
    axes.flat[-1].axis("off")
    fig.suptitle("Target layouts (red human, purple AGV, green car); dots = starts", fontsize=16)
    fig.tight_layout()
    fig.savefig(PLOTS / "04_target_layouts.png")
    plt.close(fig)


def plot_car_ground(rm, ground, grass, trees, bs):
    xs = np.arange(rm.x_min, rm.x_max + .25, .5)
    ys = np.arange(rm.y_min, rm.y_max + .25, .5)
    xx, yy = np.meshgrid(xs, ys)
    world = rotate_radiomap_xy(np.column_stack([xx.ravel(), yy.ravel()]), rm).reshape(*xx.shape, 2)
    region = np.where(trees, 3, np.where(grass, 2, np.where(ground, 1, 0)))
    colors = ListedColormap(["#f8fafc", "#c8b79d", "#64a45b", "#205b35"])
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], colors.N)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=160)
    for ax, number in zip(axes.flat, (2, 4, 42, 84)):
        data = json.loads((OUT / f"scene_{number:03d}.json").read_text(encoding="utf-8"))
        ax.pcolormesh(world[:, :, 0], world[:, :, 1], region, cmap=colors, norm=norm,
                      shading="nearest", rasterized=True)
        cars = [obj for obj in data["scene"]["objects"] if obj["object_name"] == "CAR_3GPP"]
        for car in cars:
            dense = TrajectorySpec(**car["trajectory"])
            dense.samples = 257
            path = sample_trajectory(dense)
            ax.plot(path[:, 0], path[:, 1], color="#dc2626", lw=3.5, zorder=4)
            ax.scatter(path[0, 0], path[0, 1], color="#dc2626", edgecolors="white", s=50, zorder=5)
            ax.scatter(path[-1, 0], path[-1, 1], color="#dc2626", marker="x", s=55, zorder=5)
        ax.scatter(bs[0], bs[1], marker="*", s=120, color="#facc15", edgecolors="black", zorder=5)
        ax.set(aspect="equal", xlabel="Scene X (m)", ylabel="Scene Y (m)",
               title=f"Scene {number:03d} | {len(cars)} car(s)")
    fig.suptitle("Cars on GROUND (tan); GRASS is green; tree trunks dark green", fontsize=15)
    fig.tight_layout()
    fig.savefig(PLOTS / "05_car_paths_ground_not_grass.png")
    plt.close(fig)


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    request = read_request(SOURCE)
    rm = request.scene.radiomap
    require_tree_instances(request.scene.scenario_path)
    nx, ny = radiomap_grid_shape(rm)
    inside = radiomap_inside_building_mask(request.scene.scenario_path, rm).reshape(ny, nx)
    clearance = distance_transform_edt(~inside, sampling=(rm.y_spacing, rm.x_spacing))
    drive_clearance, ground, grass, trees = driving_clearance(rm, request.scene.scenario_path)
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    prior_manifest_path = OUT / "manifest.json"
    prior_entries = ({entry["file"]: entry for entry in json.loads(prior_manifest_path.read_text(encoding="utf-8"))["scenes"]}
                     if prior_manifest_path.exists() else {})
    manifest = {"source": str(SOURCE.relative_to(ROOT)), "seed_base": 20261006,
                "ue_count_per_scene": 200, "scene_count": 200,
                "notes": "Each JSON is independently loadable in the GUI. All 200 UEs move in each simulation.",
                "scenes": []}
    examples = []
    for number in range(200):
        profile = PROFILES[number // 40]
        target_count = number % 5 + 1
        seed = 20261006 + number
        path = OUT / f"scene_{number:03d}.json"
        if path.exists():
            # A resumed run reuses configs already parsed and validated by the
            # prior run, and rebuilds their summary directly from stored paths.
            existing = json.loads(path.read_text(encoding="utf-8"))
            repaired = False
            for target_index, target in enumerate(existing["scene"]["objects"]):
                if target["object_name"] != "CAR_3GPP":
                    continue
                old_spec = TrajectorySpec(**target["trajectory"])
                old_spec.samples = 257
                valid_shape = (old_spec.kind == "linear" and len(old_spec.points) == 2) or (
                    old_spec.kind == "curve" and len(old_spec.points) == 3)
                if valid_shape and safe_car_path(sample_trajectory(old_spec)[:, :2], rm, drive_clearance):
                    continue
                fresh = make_targets(np.random.default_rng(seed + 100000 + target_index),
                                     rm, clearance, 1, drive_clearance,
                                     types_override=["CAR_3GPP"])[0]
                fresh["id"] = target["id"]
                existing["scene"]["objects"][target_index] = fresh
                repaired = True
            if repaired:
                path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
                validate_request(read_request(path))
            if path.name in prior_entries:
                lengths = prior_entries[path.name]["path_lengths_m"]
            else:
                lengths = []
                for ue in existing["scene"]["user_equipments"]:
                    spec = TrajectorySpec(**ue["trajectory"])
                    spec.samples = 97
                    spec.start_static_fraction = spec.end_static_fraction = 0.
                    spec.easing = "linear"
                    sampled = sample_trajectory(spec)
                    lengths.append(round(float(np.linalg.norm(np.diff(sampled[:, :2], axis=0), axis=1).sum()), 3))
            targets = existing["scene"]["objects"]
            families = [ue["id"].split("_", 2)[2] for ue in existing["scene"]["user_equipments"]]
            manifest["scenes"].append({"file": path.name, "seed": seed, "length_profile": profile,
                                        "target_count": len(targets),
                                        "target_types": [target["object_name"] for target in targets],
                                        "family_counts": dict(Counter(families)),
                                        "path_lengths_m": lengths,
                                        "length_mean_m": round(float(np.mean(lengths)), 3),
                                        "length_median_m": round(float(np.median(lengths)), 3)})
            if number in (0, 40, 80, 120, 160):
                tracks = []
                for ue, family, length in zip(existing["scene"]["user_equipments"], families, lengths):
                    spec = TrajectorySpec(**ue["trajectory"])
                    spec.samples = 97
                    tracks.append({"ue": ue, "family": family, "sampled": sample_trajectory(spec),
                                   "length_m": length})
                examples.append((number, tracks, targets, profile))
            if number % 20 == 19:
                print(f"Checked car paths in {number + 1}/200 scenes", flush=True)
            continue
        rng = np.random.default_rng(seed)
        tracks = make_tracks(rng, rm, clearance, source["scene"]["user_equipments"][0], profile)
        targets = make_targets(rng, rm, clearance, target_count, drive_clearance)
        output = deepcopy(source)
        output["scene"]["description"] = (
            f"PowerLog 3GPP ensemble scene {number:03d}: 200 distinct outdoor UE paths, "
            f"{target_count} moving sensing target(s), {profile} path length distribution. "
            "UE and target trajectories lie in the configured rotated radiomap area."
        )
        output["scene"]["radiomap"]["enabled"] = False
        output["scene"]["user_equipments"] = [track["ue"] for track in tracks]
        output["scene"]["objects"] = targets
        output["scene"]["timeframe_interval_s"] = .2
        output["sample_id"] = f"powerlog_3gpp_ensemble_{number:03d}"
        output["output_dir"] = f"output/powerlog_3gpp_ensemble_200/simulated/scene_{number:03d}"
        output["sionna"]["sensing_channel"] = "combined"
        output["sionna"]["seed"] = seed
        output["sionna"]["batch_timeframes"] = 1
        path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        validate_request(read_request(path))
        lengths = [round(track["length_m"], 3) for track in tracks]
        manifest["scenes"].append({"file": path.name, "seed": seed, "length_profile": profile,
                                    "target_count": target_count,
                                    "target_types": [target["object_name"] for target in targets],
                                    "family_counts": dict(Counter(track["family"] for track in tracks)),
                                    "path_lengths_m": lengths,
                                    "length_mean_m": round(float(np.mean(lengths)), 3),
                                    "length_median_m": round(float(np.median(lengths)), 3)})
        if number in (0, 40, 80, 120, 160):
            examples.append((number, tracks, targets, profile))
        if number % 10 == 9:
            print(f"Generated and validated {number + 1}/200 scenes", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    plot_examples(examples, rm, inside, source["scene"]["base_stations"][0]["position"])
    plot_distributions(manifest)
    plot_loops(examples)
    plot_target_diversity(rm, inside, source["scene"]["base_stations"][0]["position"])
    plot_car_ground(rm, ground, grass, trees, source["scene"]["base_stations"][0]["position"])
    print(f"Done: {len(manifest['scenes'])} GUI-ready configs in {OUT}", flush=True)


if __name__ == "__main__":
    main()

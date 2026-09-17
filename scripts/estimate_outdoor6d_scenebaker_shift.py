from __future__ import annotations

import csv
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
OUTDOOR_XML = ROOT / "scenarios" / "Outdoor6D_w_car.xml"
SCENEBAKER_XML = (
    ROOT
    / "scenarios"
    / "scenebaker_upv_vera_catastro_subset"
    / "upv_vera_catastro_subset_no_trees_no_lidar.xml"
)
OUT_DIR = ROOT / "scenarios" / "outdoor6d_scenebaker_alignment"

OUTDOOR_STRUCTURAL = {
    "BRICKS_Obj.ply",
    "CONCRETE_Obj.ply",
    "GLASS_Obj.ply",
    "METAL_Obj.ply",
    "PLASTIC_Obj.ply",
}
SCENEBAKER_STRUCTURAL = {
    "catastro_buildingparts-wall.ply",
    "catastro_buildingparts-roof.ply",
    "osm_fallback_buildings-wall.ply",
    "osm_fallback_buildings-roof.ply",
}


def parse_scene_meshes(xml_path: Path) -> list[Path]:
    root = ET.parse(xml_path).getroot()
    meshes = []
    for shape in root.findall("shape"):
        string = shape.find("string[@name='filename']")
        if string is None:
            continue
        meshes.append(xml_path.parent / string.attrib["value"])
    return meshes


def read_ply(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path} has no end_header")
            text = line.decode("ascii").strip()
            header.append(text)
            if text == "end_header":
                break

        if "format binary_little_endian 1.0" not in header:
            raise ValueError(f"{path} is not binary_little_endian PLY")

        vertex_count = int(next(line.split()[-1] for line in header if line.startswith("element vertex ")))
        face_count = int(next(line.split()[-1] for line in header if line.startswith("element face ")))
        vertex_props = 0
        in_vertex = False
        for line in header:
            if line.startswith("element vertex "):
                in_vertex = True
                continue
            if line.startswith("element face "):
                in_vertex = False
                continue
            if in_vertex and line.startswith("property "):
                vertex_props += 1

        vertex_raw = np.frombuffer(f.read(vertex_count * vertex_props * 4), dtype="<f4")
        vertices = vertex_raw.reshape(vertex_count, vertex_props)[:, :3].copy()
        faces: list[list[int]] = []
        for _ in range(face_count):
            count = struct.unpack("<B", f.read(1))[0]
            faces.append(list(struct.unpack("<" + "i" * count, f.read(4 * count))))
    return vertices, faces


def mesh_centroids(vertices: np.ndarray, faces: list[list[int]]) -> np.ndarray:
    if not faces:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray([vertices[face].mean(axis=0) for face in faces], dtype=np.float64)


def load_points(mesh_paths: list[Path], include_names: set[str] | None) -> tuple[np.ndarray, list[dict]]:
    clouds = []
    summary = []
    for path in mesh_paths:
        if include_names is not None and path.name not in include_names:
            continue
        vertices, faces = read_ply(path)
        centroids = mesh_centroids(vertices, faces)
        points = centroids if len(centroids) else vertices
        if len(points):
            clouds.append(points)
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        summary.append(
            {
                "mesh": path.name,
                "vertices": len(vertices),
                "faces": len(faces),
                "x_min": mins[0],
                "y_min": mins[1],
                "z_min": mins[2],
                "x_max": maxs[0],
                "y_max": maxs[1],
                "z_max": maxs[2],
            }
        )
    if not clouds:
        raise RuntimeError("No point clouds loaded")
    return np.vstack(clouds), summary


def percentile_center(points: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    lo = np.percentile(points[:, :2], low, axis=0)
    hi = np.percentile(points[:, :2], high, axis=0)
    return (lo + hi) / 2.0


def estimate_translation(source_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    # Use robust bounding-box centers. The detailed model has facade/interior detail and cars;
    # percentile bounds keep those from dominating the shift estimate.
    return percentile_center(target_xy) - percentile_center(source_xy)


def sample_points(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(points) <= n:
        return points[:, :2].astype(np.float64)
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), n, replace=False), :2].astype(np.float64)


def principal_angle(points: np.ndarray) -> float:
    centered = points[:, :2] - np.mean(points[:, :2], axis=0)
    cov = centered.T @ centered / max(len(centered), 1)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    return float(np.arctan2(axis[1], axis[0]))


def rotate(points: np.ndarray, angle_rad: float, center: np.ndarray) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (points - center) @ rot.T + center


def chamfer_score(source_xy: np.ndarray, target_xy: np.ndarray) -> float:
    target_tree = cKDTree(target_xy)
    source_tree = cKDTree(source_xy)
    d_st, _ = target_tree.query(source_xy, k=1, workers=-1)
    d_ts, _ = source_tree.query(target_xy, k=1, workers=-1)
    return float(np.median(d_st) + np.median(d_ts))


def estimate_similarity(source_points: np.ndarray, target_points: np.ndarray) -> dict:
    source = sample_points(source_points, 12_000, 11)
    target = sample_points(target_points, 12_000, 13)
    source_center = percentile_center(source)
    target_center = percentile_center(target)

    base_angle = principal_angle(target) - principal_angle(source)
    # PCA axes have a 180 degree ambiguity; also test quarter-turn-like alternatives because
    # campus blocks and exported CAD axes can be ambiguous.
    coarse_angles = []
    for offset in [0, np.pi, np.pi / 2, -np.pi / 2]:
        coarse_angles.extend(base_angle + offset + np.deg2rad(np.linspace(-35, 35, 71)))

    best = None
    for angle in coarse_angles:
        shifted = rotate(source, angle, source_center)
        translation = target_center - percentile_center(shifted)
        transformed = shifted + translation
        score = chamfer_score(transformed, target)
        if best is None or score < best["score"]:
            best = {"angle_rad": float(angle), "translation": translation, "score": score}

    # Refine around the best angle.
    for step_deg in [1.0, 0.25, 0.05]:
        angles = best["angle_rad"] + np.deg2rad(np.linspace(-5 * step_deg, 5 * step_deg, 41))
        for angle in angles:
            shifted = rotate(source, angle, source_center)
            translation = target_center - percentile_center(shifted)
            transformed = shifted + translation
            score = chamfer_score(transformed, target)
            if score < best["score"]:
                best = {"angle_rad": float(angle), "translation": translation, "score": score}

    return {
        "angle_rad": best["angle_rad"],
        "angle_deg": float(np.rad2deg(best["angle_rad"])),
        "rotation_center_x": float(source_center[0]),
        "rotation_center_y": float(source_center[1]),
        "dx_m": float(best["translation"][0]),
        "dy_m": float(best["translation"][1]),
        "score_m": float(best["score"]),
    }


def save_summary(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def transform_xy(points: np.ndarray, transform: dict | None = None, shift: np.ndarray | None = None) -> np.ndarray:
    xy = points[:, :2].astype(np.float64)
    if transform is not None:
        center = np.array([transform["rotation_center_x"], transform["rotation_center_y"]], dtype=np.float64)
        xy = rotate(xy, transform["angle_rad"], center)
        xy = xy + np.array([transform["dx_m"], transform["dy_m"]], dtype=np.float64)
        return xy
    if shift is not None:
        return xy + shift
    return xy


def make_plot(outdoor: np.ndarray, scenebaker: np.ndarray, path: Path, *, shift=None, transform=None) -> None:
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    rng = np.random.default_rng(7)
    max_points = 35_000
    outdoor_sample = outdoor[rng.choice(len(outdoor), min(len(outdoor), max_points), replace=False)]
    scene_sample = scenebaker[rng.choice(len(scenebaker), min(len(scenebaker), max_points), replace=False)]
    ax.scatter(scene_sample[:, 0], scene_sample[:, 1], s=1, c="#d62728", alpha=0.35, label="SceneBaker/Catastro")
    outdoor_xy = transform_xy(outdoor_sample, transform=transform, shift=shift)
    ax.scatter(
        outdoor_xy[:, 0],
        outdoor_xy[:, 1],
        s=1,
        c="#1f77b4",
        alpha=0.35,
        label="Outdoor6D shifted",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("SceneBaker X [m]")
    ax.set_ylabel("SceneBaker Y [m]")
    if transform is not None:
        ax.set_title(
            f"Outdoor6D mesh registration: dx={transform['dx_m']:.3f} m, "
            f"dy={transform['dy_m']:.3f} m, rot={transform['angle_deg']:.3f} deg"
        )
    else:
        ax.set_title(f"Outdoor6D shifted by dx={shift[0]:.3f} m, dy={shift[1]:.3f} m")
    ax.legend(loc="upper right")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_bounds_plot(outdoor: np.ndarray, scenebaker: np.ndarray, shift: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    for points, label, color in [
        (scenebaker[:, :2], "SceneBaker/Catastro structural", "#d62728"),
        (outdoor[:, :2] + shift, "Outdoor6D structural shifted", "#1f77b4"),
    ]:
        lo = np.percentile(points, 2, axis=0)
        hi = np.percentile(points, 98, axis=0)
        rect = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]], [lo[0], lo[1]]])
        ax.plot(rect[:, 0], rect[:, 1], color=color, linewidth=2.0, label=label)
        ax.scatter(points[:, 0].mean(), points[:, 1].mean(), color=color, s=55)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("SceneBaker X [m]")
    ax.set_ylabel("SceneBaker Y [m]")
    ax.set_title("Robust 2-98% bounds after shift")
    ax.legend(loc="upper right")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outdoor_meshes = parse_scene_meshes(OUTDOOR_XML)
    scenebaker_meshes = parse_scene_meshes(SCENEBAKER_XML)
    outdoor_points, outdoor_summary = load_points(outdoor_meshes, OUTDOOR_STRUCTURAL)
    scenebaker_points, scenebaker_summary = load_points(scenebaker_meshes, SCENEBAKER_STRUCTURAL)
    shift = estimate_translation(outdoor_points[:, :2], scenebaker_points[:, :2])
    similarity = estimate_similarity(outdoor_points, scenebaker_points)

    save_summary(OUT_DIR / "outdoor6d_mesh_summary.csv", outdoor_summary)
    save_summary(OUT_DIR / "scenebaker_structural_mesh_summary.csv", scenebaker_summary)

    with (OUT_DIR / "estimated_shift.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dx_m", "dy_m", "dz_m", "method"])
        writer.writerow([f"{shift[0]:.6f}", f"{shift[1]:.6f}", "0.000000", "align structural mesh 2-98 percentile XY centers"])

    with (OUT_DIR / "estimated_mesh_similarity_transform.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dx_m",
                "dy_m",
                "dz_m",
                "rotation_deg",
                "rotation_center_x",
                "rotation_center_y",
                "score_m",
                "method",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dx_m": f"{similarity['dx_m']:.6f}",
                "dy_m": f"{similarity['dy_m']:.6f}",
                "dz_m": "0.000000",
                "rotation_deg": f"{similarity['angle_deg']:.6f}",
                "rotation_center_x": f"{similarity['rotation_center_x']:.6f}",
                "rotation_center_y": f"{similarity['rotation_center_y']:.6f}",
                "score_m": f"{similarity['score_m']:.6f}",
                "method": "mesh-only 2D rotation+translation registration, median symmetric nearest-neighbor score",
            }
        )

    make_plot(outdoor_points[:, :2], scenebaker_points[:, :2], OUT_DIR / "outdoor6d_shifted_overlay.png", shift=shift)
    make_plot(
        outdoor_points[:, :2],
        scenebaker_points[:, :2],
        OUT_DIR / "outdoor6d_mesh_registered_overlay.png",
        transform=similarity,
    )
    make_bounds_plot(outdoor_points, scenebaker_points, shift, OUT_DIR / "outdoor6d_shifted_bounds.png")


if __name__ == "__main__":
    main()

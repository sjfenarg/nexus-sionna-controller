from __future__ import annotations

import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_XML = ROOT / "scenarios" / "Outdoor6D_w_car.xml"
ALIGN_DIR = ROOT / "scenarios" / "outdoor6d_scenebaker_alignment"
TRANSFORM_PATH = ALIGN_DIR / "manual_surface_outdoor6d_to_scenebaker_transform.json"
H5_PATH = ROOT / "meas" / "nexus_trials.h5"
OUT_DIR = ALIGN_DIR / "plots"
BS_HEIGHT_M = 15.0
UE_EVEN_HEIGHT_M = 1.16
UE_ODD_HEIGHT_M = 1.0

OSM_BOUNDS = {
    "minlat": 39.4774000,
    "minlon": -0.3442000,
    "maxlat": 39.4842000,
    "maxlon": -0.3348000,
}

MATERIAL_COLORS = {
    "BRICKS_Obj.ply": "#a64b2a",
    "CONCRETE_Obj.ply": "#b8b8b8",
    "GLASS_Obj.ply": "#4fc3f7",
    "GRASS_Obj.ply": "#4daf4a",
    "GRAVEL_Obj.ply": "#8c7b6b",
    "GROUND_Obj.ply": "#d9c8a9",
    "METAL_Obj.ply": "#6f7f8f",
    "PLASTIC_Obj.ply": "#f2c94c",
    "CAR_obj.ply": "#d62728",
}


def latlon_to_scene_xy(lat: float, lon: float) -> tuple[float, float]:
    lat0 = (OSM_BOUNDS["minlat"] + OSM_BOUNDS["maxlat"]) / 2.0
    lon0 = (OSM_BOUNDS["minlon"] + OSM_BOUNDS["maxlon"]) / 2.0
    meters_per_lat = 111_320.0
    meters_per_lon = meters_per_lat * np.cos(np.deg2rad(lat0))
    return (lon - lon0) * meters_per_lon, (lat - lat0) * meters_per_lat


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
        vertices = np.frombuffer(f.read(vertex_count * vertex_props * 4), dtype="<f4").reshape(vertex_count, vertex_props)[:, :3].copy()
        faces = []
        for _ in range(face_count):
            count = struct.unpack("<B", f.read(1))[0]
            faces.append(list(struct.unpack("<" + "i" * count, f.read(4 * count))))
    return vertices, faces


def scenario_meshes() -> list[tuple[str, Path]]:
    root = ET.parse(SCENARIO_XML).getroot()
    meshes = []
    for shape in root.findall("shape"):
        string = shape.find("string[@name='filename']")
        if string is None:
            continue
        rel = string.attrib["value"]
        meshes.append((Path(rel).name, SCENARIO_XML.parent / rel))
    return meshes


def transform_xy(xy: np.ndarray, transform: dict) -> np.ndarray:
    center = np.array([transform["rotation_center_x"], transform["rotation_center_y"]], dtype=np.float64)
    scaled = center + (xy - center) * float(transform["scale_xy"])
    angle = np.deg2rad(float(transform["rotation_deg"]))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    moved = (scaled - center) @ rot.T + center
    moved[:, 0] += float(transform["dx_m"])
    moved[:, 1] += float(transform["dy_m"])
    return moved


def load_surface_polygons(transform: dict, max_faces_per_mesh: int = 18_000) -> tuple[list[tuple[str, list[np.ndarray]]], np.ndarray]:
    rng = np.random.default_rng(9)
    result = []
    all_xy = []
    for name, path in scenario_meshes():
        vertices, faces = read_ply(path)
        face_indices = np.arange(len(faces))
        if len(face_indices) > max_faces_per_mesh:
            face_indices = rng.choice(face_indices, max_faces_per_mesh, replace=False)
        polys = []
        for idx in face_indices:
            face = faces[int(idx)]
            if len(face) < 3:
                continue
            xy = transform_xy(vertices[face, :2].astype(np.float64), transform)
            polys.append(xy)
            all_xy.append(xy)
        result.append((name, polys))
    return result, np.vstack(all_xy)


def load_hdf5_positions() -> list[dict]:
    positions = []
    with h5py.File(H5_PATH, "r") as f:
        group = f["scenarios/nexus_trials/s001/parameters/antenna_params"]
        for name in sorted(group.keys()):
            if "ant_coord" not in group[name]:
                continue
            lat, lon = group[name]["ant_coord"][...]
            x, y = latlon_to_scene_xy(float(lat), float(lon))
            role = "tx" if name.startswith("bs") else "rx"
            if role == "tx":
                z = BS_HEIGHT_M
            else:
                ue_index = int(name[2:])
                z = UE_EVEN_HEIGHT_M if ue_index % 2 == 0 else UE_ODD_HEIGHT_M
            positions.append({"name": name, "role": role, "x": x, "y": y, "z": z})
    return positions


def save_positions_csv(positions: list[dict]) -> None:
    path = ALIGN_DIR / "hdf5_antenna_positions_scene_xyz.csv"
    with path.open("w", encoding="utf-8") as f:
        f.write("name,role,x_scene_m,y_scene_m,z_scene_m\n")
        for p in positions:
            f.write(f"{p['name']},{p['role']},{p['x']:.6f},{p['y']:.6f},{p['z']:.6f}\n")


def add_positions(ax, positions: list[dict]) -> None:
    for role, marker, color, label in [
        ("rx", "o", "#1f77b4", "RX UE"),
        ("tx", "^", "#ff7f0e", "TX BS"),
    ]:
        pts = [p for p in positions if p["role"] == role]
        ax.scatter([p["x"] for p in pts], [p["y"] for p in pts], s=70, marker=marker, c=color, edgecolor="black", linewidth=0.7, label=label, zorder=20)
        for p in pts:
            ax.annotate(p["name"], (p["x"], p["y"]), xytext=(5, 5), textcoords="offset points", fontsize=8, color=color, zorder=21)


def save_plan(polys_by_mesh: list[tuple[str, list[np.ndarray]]], all_xy: np.ndarray, positions: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    for name, polys in polys_by_mesh:
        coll = PolyCollection(
            polys,
            facecolors=MATERIAL_COLORS.get(name, "#999999"),
            edgecolors="black",
            linewidths=0.05,
            alpha=0.55 if name != "GLASS_Obj.ply" else 0.38,
            label=name.replace("_Obj.ply", "").replace("_obj.ply", ""),
        )
        ax.add_collection(coll)
    add_positions(ax, positions)
    ax.set_aspect("equal", adjustable="box")
    pad = 15.0
    pts = np.vstack([all_xy, np.array([[p["x"], p["y"]] for p in positions])])
    ax.set_xlim(float(pts[:, 0].min() - pad), float(pts[:, 0].max() + pad))
    ax.set_ylim(float(pts[:, 1].min() - pad), float(pts[:, 1].max() + pad))
    ax.set_xlabel("SceneBaker global X [m]")
    ax.set_ylabel("SceneBaker global Y [m]")
    ax.set_title("Outdoor6D_w_car transformed to SceneBaker coordinates with HDF5 antenna positions")
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=8, ncols=2)
    fig.savefig(OUT_DIR / "outdoor6d_global_materials_with_hdf5_positions.png", dpi=240)
    plt.close(fig)


def save_zoom(polys_by_mesh: list[tuple[str, list[np.ndarray]]], positions: list[dict]) -> None:
    pos_xy = np.array([[p["x"], p["y"]] for p in positions])
    center = pos_xy.mean(axis=0)
    half = 55.0
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    for name, polys in polys_by_mesh:
        coll = PolyCollection(
            polys,
            facecolors=MATERIAL_COLORS.get(name, "#999999"),
            edgecolors="black",
            linewidths=0.06,
            alpha=0.60 if name != "GLASS_Obj.ply" else 0.42,
            label=name.replace("_Obj.ply", "").replace("_obj.ply", ""),
        )
        ax.add_collection(coll)
    add_positions(ax, positions)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(center[0] - half), float(center[0] + half))
    ax.set_ylim(float(center[1] - half), float(center[1] + half))
    ax.set_xlabel("SceneBaker global X [m]")
    ax.set_ylabel("SceneBaker global Y [m]")
    ax.set_title("Outdoor6D transformed, zoom around HDF5 antenna positions")
    ax.legend(loc="upper right", fontsize=8, ncols=2)
    fig.savefig(OUT_DIR / "outdoor6d_global_materials_with_hdf5_positions_zoom.png", dpi=240)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    transform = json.loads(TRANSFORM_PATH.read_text(encoding="utf-8"))
    polys_by_mesh, all_xy = load_surface_polygons(transform)
    positions = load_hdf5_positions()
    save_positions_csv(positions)
    save_plan(polys_by_mesh, all_xy, positions)
    save_zoom(polys_by_mesh, positions)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
OUTDOOR_XML = ROOT / "scenarios" / "Outdoor6D_w_car.xml"
SCENEBAKER_XML = (
    ROOT
    / "scenarios"
    / "scenebaker_upv_vera_catastro_subset"
    / "upv_vera_catastro_subset_no_trees_no_lidar.xml"
)
ALIGN_DIR = ROOT / "scenarios" / "outdoor6d_scenebaker_alignment"
TRANSFORM_PATH = ALIGN_DIR / "manual_surface_outdoor6d_to_scenebaker_transform.json"
H5_PATH = ROOT / "meas" / "nexus_trials.h5"
OUT_DIR = ALIGN_DIR / "plots"

OSM_BOUNDS = {
    "minlat": 39.4774000,
    "minlon": -0.3442000,
    "maxlat": 39.4842000,
    "maxlon": -0.3348000,
}


def scene_x_to_lon(x: float) -> float:
    lat0 = (OSM_BOUNDS["minlat"] + OSM_BOUNDS["maxlat"]) / 2.0
    lon0 = (OSM_BOUNDS["minlon"] + OSM_BOUNDS["maxlon"]) / 2.0
    meters_per_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
    return lon0 + x / meters_per_lon


def scene_y_to_lat(y: float) -> float:
    lat0 = (OSM_BOUNDS["minlat"] + OSM_BOUNDS["maxlat"]) / 2.0
    return lat0 + y / 111_320.0

DETAILED_COLORS = {
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

SIMPLIFIED_COLORS = {
    "Plane.ply": "#d9c8a9",
    "catastro_buildingparts-wall.ply": "#8f8f8f",
    "catastro_buildingparts-roof.ply": "#b8b8b8",
    "osm_fallback_buildings-wall.ply": "#8f8f8f",
    "osm_fallback_buildings-roof.ply": "#b8b8b8",
    "map_osm_roads_pedestrian.ply": "#6f7f8f",
    "map_osm_roads_service.ply": "#6f7f8f",
    "map_osm_roads_primary.ply": "#6f7f8f",
    "map_osm_roads_residential.ply": "#6f7f8f",
    "map_osm_roads_secondary.ply": "#6f7f8f",
    "map_osm_roads_tertiary.ply": "#6f7f8f",
    "map_osm_roads_track.ply": "#6f7f8f",
    "map_osm_roads_unclassified.ply": "#6f7f8f",
    "map_osm_paths_cycleway.ply": "#9aa7b2",
    "map_osm_paths_footway.ply": "#9aa7b2",
    "map_osm_paths_steps.ply": "#9aa7b2",
    "map_osm_areas_park.ply": "#4daf4a",
    "map_osm_areas_pedestrian.ply": "#8c7b6b",
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


def scenario_meshes(xml_path: Path) -> list[tuple[str, Path]]:
    root = ET.parse(xml_path).getroot()
    meshes = []
    for shape in root.findall("shape"):
        string = shape.find("string[@name='filename']")
        if string is None:
            continue
        rel = string.attrib["value"]
        meshes.append((Path(rel).name, xml_path.parent / rel))
    return meshes


def filtered_scenario_meshes(xml_path: Path) -> list[tuple[str, Path]]:
    meshes = scenario_meshes(xml_path)
    if xml_path == OUTDOOR_XML:
        return [(name, path) for name, path in meshes if name.lower() != "car_obj.ply"]
    return meshes


def transform_outdoor_xy(xy: np.ndarray, transform: dict) -> np.ndarray:
    center = np.array([transform["rotation_center_x"], transform["rotation_center_y"]], dtype=np.float64)
    scaled = center + (xy - center) * float(transform["scale_xy"])
    angle = np.deg2rad(float(transform["rotation_deg"]))
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float64)
    moved = (scaled - center) @ rot.T + center
    moved[:, 0] += float(transform["dx_m"])
    moved[:, 1] += float(transform["dy_m"])
    return moved


def load_polygons(
    xml_path: Path,
    colors: dict[str, str],
    transform: dict | None,
    max_faces_per_mesh: int,
    seed: int,
) -> tuple[list[tuple[str, str, list[np.ndarray]]], np.ndarray]:
    rng = np.random.default_rng(seed)
    result = []
    all_xy = []
    for name, path in filtered_scenario_meshes(xml_path):
        vertices, faces = read_ply(path)
        face_indices = np.arange(len(faces))
        if len(face_indices) > max_faces_per_mesh:
            face_indices = rng.choice(face_indices, max_faces_per_mesh, replace=False)
        polys = []
        for idx in face_indices:
            face = faces[int(idx)]
            if len(face) < 3:
                continue
            xy = vertices[face, :2].astype(np.float64)
            if transform is not None:
                xy = transform_outdoor_xy(xy, transform)
            polys.append(xy)
            all_xy.append(xy)
        result.append((name, colors.get(name, "#999999"), polys))
    return result, np.vstack(all_xy)


def load_positions() -> list[dict]:
    positions = []
    with h5py.File(H5_PATH, "r") as f:
        group = f["scenarios/nexus_trials/s001/parameters/antenna_params"]
        for name in sorted(group.keys()):
            if "ant_coord" not in group[name]:
                continue
            lat, lon = group[name]["ant_coord"][...]
            x, y = latlon_to_scene_xy(float(lat), float(lon))
            positions.append({"name": name, "role": "tx" if name.startswith("bs") else "rx", "x": x, "y": y})
    return positions


def add_map(ax, polygons, title: str) -> None:
    for name, color, polys in polygons:
        if not polys:
            continue
        alpha = 0.48
        if "GLASS" in name:
            alpha = 0.35
        if name == "Plane.ply":
            alpha = 0.25
        coll = PolyCollection(
            polys,
            facecolors=color,
            edgecolors="black",
            linewidths=0.045,
            alpha=alpha,
        )
        ax.add_collection(coll)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")


def add_positions(ax, positions: list[dict]) -> None:
    for role, marker, color, label in [
        ("rx", "o", "#1f77b4", "RX UE"),
        ("tx", "^", "#ff7f0e", "TX BS"),
    ]:
        pts = [p for p in positions if p["role"] == role]
        ax.scatter(
            [p["x"] for p in pts],
            [p["y"] for p in pts],
            s=64,
            marker=marker,
            c=color,
            edgecolor="black",
            linewidth=0.7,
            zorder=20,
            label=label,
        )
        for p in pts:
            ax.annotate(p["name"], (p["x"], p["y"]), xytext=(4, 4), textcoords="offset points", fontsize=8, color=color, zorder=21)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    transform = json.loads(TRANSFORM_PATH.read_text(encoding="utf-8"))
    detailed, detailed_xy = load_polygons(OUTDOOR_XML, DETAILED_COLORS, transform, 18_000, 31)
    simplified, simplified_xy = load_polygons(SCENEBAKER_XML, SIMPLIFIED_COLORS, None, 30_000, 32)
    positions = load_positions()
    pos_xy = np.array([[p["x"], p["y"]] for p in positions])

    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5), constrained_layout=True)
    add_map(axes[0], detailed, "Detailed Map: Outdoor6D_w_car")
    add_map(axes[1], simplified, "Simplified Map: SceneBaker + Catastro")
    for ax in axes:
        add_positions(ax, positions)
        pts = np.vstack([detailed_xy, simplified_xy, pos_xy])
        pad = 12.0
        ax.set_xlim(float(pts[:, 0].min() - pad), float(pts[:, 0].max() + pad))
        ax.set_ylim(float(pts[:, 1].min() - pad), float(pts[:, 1].max() + pad))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{scene_x_to_lon(x):.6f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{scene_y_to_lat(y):.6f}"))
        ax.tick_params(axis="x", labelrotation=25)

    legend_items = [
        Patch(facecolor="#d9c8a9", edgecolor="black", alpha=0.5, label="Ground"),
        Patch(facecolor="#b8b8b8", edgecolor="black", alpha=0.5, label="Concrete / Catastro roof"),
        Patch(facecolor="#8f8f8f", edgecolor="black", alpha=0.5, label="Catastro wall"),
        Patch(facecolor="#6f7f8f", edgecolor="black", alpha=0.5, label="Metal / roads"),
        Patch(facecolor="#4fc3f7", edgecolor="black", alpha=0.5, label="Glass"),
        Patch(facecolor="#4daf4a", edgecolor="black", alpha=0.5, label="Grass / park"),
        Patch(facecolor="#f2c94c", edgecolor="black", alpha=0.5, label="Plastic / details"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f77b4", markeredgecolor="black", markersize=9, label="RX UE"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#ff7f0e", markeredgecolor="black", markersize=9, label="TX BS"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncols=5, bbox_to_anchor=(0.5, -0.035))
    axes[0].set_title("Detailed 6D Map")
    axes[1].set_title("Simplified 6D Map")
    fig.suptitle("Antenna positions in aligned UPV Vera 6D maps", fontsize=16)
    fig.savefig(OUT_DIR / "detailed_vs_scenebaker_two_panel_with_antennas.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

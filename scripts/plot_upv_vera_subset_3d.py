from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "scenebaker_upv_vera_catastro_subset"
XML = SCENARIO / "upv_vera_catastro_subset_no_trees_no_lidar.xml"
PLOTS = SCENARIO / "plots"
MEAS_H5 = ROOT / "meas" / "nexus_trials.h5"
BS_HEIGHT_M = 15.0
UE_EVEN_HEIGHT_M = 1.16
UE_ODD_HEIGHT_M = 1.0

OSM_BOUNDS = {
    "minlat": 39.4774000,
    "minlon": -0.3442000,
    "maxlat": 39.4842000,
    "maxlon": -0.3348000,
}

REQUESTED_POLYGON_LATLON = [
    (39.479673, -0.339566),
    (39.479160, -0.337873),
    (39.478200, -0.338270),
    (39.478740, -0.339993),
]

COLORS = {
    "mat-ground": "#d8d2c4",
    "mat-road": "#595959",
    "mat-park": "#2ca02c",
    "mat-building_wall": "#9c6b53",
    "mat-building_roof": "#cf4c4c",
}


def latlon_to_scene_xy(lat: float, lon: float) -> tuple[float, float]:
    lat0 = (OSM_BOUNDS["minlat"] + OSM_BOUNDS["maxlat"]) / 2.0
    lon0 = (OSM_BOUNDS["minlon"] + OSM_BOUNDS["maxlon"]) / 2.0
    meters_per_lat = 111_320.0
    meters_per_lon = meters_per_lat * np.cos(np.deg2rad(lat0))
    return (lon - lon0) * meters_per_lon, (lat - lat0) * meters_per_lat


def load_antenna_positions() -> list[dict]:
    positions = []
    with h5py.File(MEAS_H5, "r") as f:
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
            positions.append({"name": name, "role": role, "lat": lat, "lon": lon, "x": x, "y": y, "z": z})
    return positions


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
        vertices = np.frombuffer(f.read(vertex_count * 12), dtype="<f4").reshape(vertex_count, 3).copy()
        faces = []
        for _ in range(face_count):
            count = struct.unpack("<B", f.read(1))[0]
            faces.append(list(struct.unpack("<" + "i" * count, f.read(4 * count))))
    return vertices, faces


def load_scene_meshes() -> list[dict]:
    root = ET.parse(XML).getroot()
    meshes = []
    for shape in root.findall("shape"):
        filename = shape.find("string[@name='filename']").attrib["value"]
        material = shape.find("ref[@name='bsdf']").attrib["id"]
        vertices, faces = read_ply(SCENARIO / filename)
        meshes.append(
            {
                "id": shape.attrib["id"],
                "filename": filename,
                "material": material,
                "vertices": vertices,
                "faces": faces,
            }
        )
    return meshes


def set_equal_axes(ax, vertices: np.ndarray) -> None:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    centers = (mins + maxs) / 2
    radius = max(maxs - mins) / 2
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def add_mesh_surface(ax, mesh: dict, alpha: float, edge: bool = False) -> None:
    vertices = mesh["vertices"]
    polys = [vertices[face] for face in mesh["faces"]]
    collection = Poly3DCollection(
        polys,
        facecolor=COLORS.get(mesh["material"], "#999999"),
        edgecolor="#ffffff" if edge else "none",
        linewidth=0.15 if edge else 0,
        alpha=alpha,
    )
    ax.add_collection3d(collection)


def add_antennas_3d(ax, antennas: list[dict]) -> None:
    for role, color, marker, label in [
        ("rx", "#1f77b4", "o", "RX UE"),
        ("tx", "#ff7f0e", "^", "TX BS"),
    ]:
        pts = [p for p in antennas if p["role"] == role]
        if not pts:
            continue
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        zs = [p["z"] for p in pts]
        ax.scatter(xs, ys, zs, s=68, c=color, marker=marker, edgecolor="black", linewidth=0.6, label=label)
        for p in pts:
            ax.plot([p["x"], p["x"]], [p["y"], p["y"]], [-0.8, p["z"]], color=color, linewidth=1.2, alpha=0.85)
            ax.text(p["x"], p["y"], p["z"] + 0.9, p["name"], fontsize=7, color=color)
    ax.legend(loc="upper left")


def add_antennas_2d(ax, antennas: list[dict]) -> None:
    for role, color, marker, label in [
        ("rx", "#1f77b4", "o", "RX UE"),
        ("tx", "#ff7f0e", "^", "TX BS"),
    ]:
        pts = [p for p in antennas if p["role"] == role]
        if not pts:
            continue
        ax.scatter(
            [p["x"] for p in pts],
            [p["y"] for p in pts],
            s=58,
            c=color,
            marker=marker,
            edgecolor="black",
            linewidth=0.6,
            label=label,
            zorder=5,
        )
        for p in pts:
            ax.annotate(p["name"], (p["x"], p["y"]), xytext=(4, 4), textcoords="offset points", fontsize=7, color=color)


def save_surface_view(
    meshes: list[dict],
    antennas: list[dict],
    name: str,
    elev: float,
    azim: float,
    include_ground: bool = True,
) -> None:
    fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for mesh in meshes:
        if mesh["material"] == "mat-ground" and not include_ground:
            continue
        alpha = 0.28 if mesh["material"] == "mat-ground" else 0.78
        add_mesh_surface(ax, mesh, alpha=alpha, edge=mesh["material"].startswith("mat-building"))
    all_vertices = np.vstack([mesh["vertices"] for mesh in meshes if include_ground or mesh["material"] != "mat-ground"])
    set_equal_axes(ax, all_vertices)
    add_antennas_3d(ax, antennas)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("Scene X [m]")
    ax.set_ylabel("Scene Y [m]")
    ax.set_zlabel("Scene Z [m]")
    ax.set_title(name.replace("_", " "))
    fig.savefig(PLOTS / f"{name}.png", dpi=220)
    plt.close(fig)


def save_height_profile(meshes: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for mesh in meshes:
        vertices = mesh["vertices"]
        faces = mesh["faces"]
        if mesh["material"] == "mat-ground":
            continue
        centroids = np.array([vertices[face].mean(axis=0) for face in faces])
        ax.scatter(
            centroids[:, 0],
            centroids[:, 2],
            s=7 if "building" in mesh["material"] else 3,
            color=COLORS.get(mesh["material"], "#999999"),
            alpha=0.65,
            label=mesh["material"],
        )
    ground = next(mesh for mesh in meshes if mesh["material"] == "mat-ground")
    ax.axhline(float(ground["vertices"][0, 2]), color=COLORS["mat-ground"], linewidth=4, label="ground plane")
    ax.set_xlabel("Scene X [m]")
    ax.set_ylabel("Scene Z [m]")
    ax.set_title("UPV Vera Catastro subset, X-Z height profile")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right")
    fig.savefig(PLOTS / "upv_vera_catastro_subset_height_profile_xz.png", dpi=220)
    plt.close(fig)


def save_ground_extent(meshes: list[dict], antennas: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ground = next(mesh for mesh in meshes if mesh["material"] == "mat-ground")
    for mesh in meshes:
        if mesh["material"] != "mat-ground":
            vertices = mesh["vertices"]
            for face in mesh["faces"]:
                xy = vertices[face, :2]
                ax.fill(xy[:, 0], xy[:, 1], color=COLORS.get(mesh["material"], "#999999"), alpha=0.45, linewidth=0)
    gv = ground["vertices"]
    for face in ground["faces"]:
        xy = gv[face, :2]
        ax.fill(xy[:, 0], xy[:, 1], color=COLORS["mat-ground"], alpha=0.25, linewidth=0)
    closed = np.vstack([gv[[0, 1, 2, 3], :2], gv[0, :2]])
    ax.plot(closed[:, 0], closed[:, 1], color="#8a8172", linewidth=2.2, label="Sionna ground plane")
    add_antennas_2d(ax, antennas)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Scene X [m]")
    ax.set_ylabel("Scene Y [m]")
    ax.set_title("UPV Vera Catastro subset, ground plane extent")
    ax.legend(loc="upper right")
    fig.savefig(PLOTS / "upv_vera_catastro_subset_ground_extent.png", dpi=220)
    plt.close(fig)


def save_plan_view(meshes: list[dict], antennas: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for mesh in meshes:
        if mesh["material"] == "mat-ground":
            continue
        vertices = mesh["vertices"]
        for face in mesh["faces"]:
            xy = vertices[face, :2]
            ax.fill(xy[:, 0], xy[:, 1], color=COLORS.get(mesh["material"], "#999999"), alpha=0.55, linewidth=0)
    polygon_xy = np.array([latlon_to_scene_xy(lat, lon) for lat, lon in REQUESTED_POLYGON_LATLON])
    closed = np.vstack([polygon_xy, polygon_xy[0]])
    ax.plot(closed[:, 0], closed[:, 1], color="#2b6cb0", linewidth=2.0, label="requested crop")
    add_antennas_2d(ax, antennas)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Scene X [m]")
    ax.set_ylabel("Scene Y [m]")
    ax.set_title("UPV Vera Catastro subset, plan view with antennas")
    ax.legend(loc="upper right")
    fig.savefig(PLOTS / "upv_vera_catastro_subset_plan.png", dpi=220)
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    meshes = load_scene_meshes()
    antennas = load_antenna_positions()
    with (SCENARIO / "meas_antenna_positions_scene_xy.csv").open("w", encoding="utf-8") as f:
        f.write("name,role,lat,lon,x_scene_m,y_scene_m,plot_z_m\n")
        for p in antennas:
            f.write(
                f"{p['name']},{p['role']},{p['lat']:.9f},{p['lon']:.9f},"
                f"{p['x']:.3f},{p['y']:.3f},{p['z']:.3f}\n"
            )
    save_surface_view(meshes, antennas, "upv_vera_catastro_subset_3d_surface_ground", elev=28, azim=-58)
    save_surface_view(meshes, antennas, "upv_vera_catastro_subset_3d_surface_no_ground", elev=32, azim=-48, include_ground=False)
    save_surface_view(meshes, antennas, "upv_vera_catastro_subset_3d_low_angle", elev=12, azim=-68)
    save_surface_view(meshes, antennas, "upv_vera_catastro_subset_3d_top_oblique", elev=72, azim=-62)
    save_height_profile(meshes)
    save_ground_extent(meshes, antennas)
    save_plan_view(meshes, antennas)


if __name__ == "__main__":
    main()

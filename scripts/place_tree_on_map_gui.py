from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.widgets import Button, Slider

from estimate_outdoor6d_scenebaker_shift import read_ply

ROOT = Path(__file__).resolve().parents[1]
MAP_XML = ROOT / "scenarios" / "Detailed 6D Map" / "Detailed 6D Map.xml"
TREE_OBJ = ROOT / "docs" / "Tree_V10_Final.obj"
OUT_DIR = ROOT / "scenarios" / "tree_placement"
OUT_JSON = OUT_DIR / "tree_v10_final_to_detailed_6d_map_transform.json"
OUT_CSV = OUT_DIR / "tree_v10_final_to_detailed_6d_map_transform.csv"

STRUCTURAL_MATERIAL_COLORS = {
    "BRICKS": "#b35806",
    "CONCRETE": "#9e9e9e",
    "GLASS": "#4dbbd5",
    "METAL": "#525252",
    "PLASTIC": "#fee08b",
    "GROUND": "#b8a77a",
    "GRASS": "#66a61e",
    "GRAVEL": "#969696",
}


def parse_scene_meshes(xml_path: Path) -> list[tuple[Path, str]]:
    root = ET.parse(xml_path).getroot()
    meshes: list[tuple[Path, str]] = []
    for shape in root.findall("shape"):
        string = shape.find("string[@name='filename']")
        if string is None:
            continue
        filename = string.attrib["value"]
        material = "UNKNOWN"
        ref = shape.find("ref[@name='bsdf']")
        if ref is not None:
            material = ref.attrib.get("id", "UNKNOWN").upper()
        meshes.append((xml_path.parent / filename, material))
    return meshes


def load_map_polygons(xml_path: Path, max_faces_per_mesh: int = 18_000, seed: int = 5):
    rng = np.random.default_rng(seed)
    polys: list[np.ndarray] = []
    colors: list[str] = []
    all_xy: list[np.ndarray] = []
    for mesh_path, material in parse_scene_meshes(xml_path):
        if not mesh_path.exists() or mesh_path.suffix.lower() != ".ply":
            continue
        try:
            vertices, faces = read_ply(mesh_path)
        except Exception as exc:
            print(f"Skipping {mesh_path}: {exc}")
            continue
        idx = np.arange(len(faces))
        if len(idx) > max_faces_per_mesh:
            idx = rng.choice(idx, max_faces_per_mesh, replace=False)
        color = color_for_material(mesh_path, material)
        for face_idx in idx:
            face = faces[int(face_idx)]
            if len(face) < 3:
                continue
            xy = vertices[face, :2].astype(np.float64)
            polys.append(xy)
            colors.append(color)
            all_xy.append(xy)
    if not polys:
        raise RuntimeError(f"No PLY polygons loaded from {xml_path}")
    return polys, colors, np.vstack(all_xy)


def color_for_material(mesh_path: Path, material: str) -> str:
    name = f"{mesh_path.stem}_{material}".upper()
    for key, color in STRUCTURAL_MATERIAL_COLORS.items():
        if key in name:
            return color
    return "#cccccc"


def load_obj_vertices_and_sampled_faces(path: Path, max_faces: int = 30_000, seed: int = 7):
    verts: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    rng = np.random.default_rng(seed)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                parts = line.split()[1:]
                face = []
                for part in parts:
                    token = part.split("/")[0]
                    if not token:
                        continue
                    idx = int(token)
                    face.append(idx - 1 if idx > 0 else len(verts) + idx)
                if len(face) >= 3:
                    faces.append(face[:3])
    vertices = np.asarray(verts, dtype=np.float32)
    face_array = np.asarray(faces, dtype=np.int64)
    if len(face_array) > max_faces:
        face_array = face_array[rng.choice(len(face_array), max_faces, replace=False)]
    return vertices, face_array


def obj_to_local_scene_xyz(vertices: np.ndarray) -> np.ndarray:
    # Tree OBJ is modeled as X/Z horizontal and Y vertical. Place the trunk base
    # at local (0, 0, 0), with local Z being scene height.
    xyz = np.column_stack([vertices[:, 0], vertices[:, 2], vertices[:, 1]]).astype(np.float64)
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    base_center = np.array([(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, lo[2]], dtype=np.float64)
    return xyz - base_center


def yaw_rotate_xy(xy: np.ndarray, yaw_deg: float) -> np.ndarray:
    angle = np.deg2rad(yaw_deg)
    c = np.cos(angle)
    s = np.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return xy @ rot.T


def transform_tree_polys(local_xyz: np.ndarray, faces: np.ndarray, dx: float, dy: float, dz: float, yaw_deg: float, scale: float):
    xy = yaw_rotate_xy(local_xyz[:, :2] * scale, yaw_deg) + np.array([dx, dy], dtype=np.float64)
    z = local_xyz[:, 2] * scale + dz
    polys = [xy[face] for face in faces]
    heights = np.array([np.mean(z[face]) for face in faces], dtype=np.float64)
    return polys, heights


def save_transform(dx: float, dy: float, dz: float, yaw_deg: float, scale: float, tree_local_bounds: dict[str, list[float]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "anchored_scenario": str(MAP_XML.relative_to(ROOT)),
        "tree_obj": str(TREE_OBJ.relative_to(ROOT)),
        "dx_m": float(dx),
        "dy_m": float(dy),
        "dz_m": float(dz),
        "yaw_deg": float(yaw_deg),
        "scale": float(scale),
        "obj_axis_mapping": "OBJ x -> scene x, OBJ z -> scene y, OBJ y -> scene z",
        "tree_local_origin": "horizontal bounding-box center at trunk base/min height",
        "transform_order": "recenter OBJ to local base origin, scale, yaw around local vertical axis, then translate by dx/dy/dz in map coordinates",
        "tree_local_bounds_m": tree_local_bounds,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)
    print(f"Saved {OUT_JSON}")
    print(f"Saved {OUT_CSV}")


def main() -> None:
    map_polys, map_colors, map_xy = load_map_polygons(MAP_XML)
    vertices, faces = load_obj_vertices_and_sampled_faces(TREE_OBJ)
    local_xyz = obj_to_local_scene_xyz(vertices)
    bounds = {"min": local_xyz.min(axis=0).round(6).tolist(), "max": local_xyz.max(axis=0).round(6).tolist()}

    initial_dx = float(np.mean(map_xy[:, 0]))
    initial_dy = float(np.mean(map_xy[:, 1]))
    initial_dz = 0.0
    initial_yaw = 0.0
    initial_scale = 7.0 / 19.753479000000002

    fig, ax = plt.subplots(figsize=(12, 8.5))
    plt.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.29)

    map_collection = PolyCollection(map_polys, facecolors=map_colors, edgecolors="none", alpha=0.42, label="Detailed 6D Map fixed")
    tree_polys, tree_heights = transform_tree_polys(local_xyz, faces, initial_dx, initial_dy, initial_dz, initial_yaw, initial_scale)
    tree_collection = PolyCollection(tree_polys, array=tree_heights, cmap="Greens", edgecolors="#145a32", linewidths=0.03, alpha=0.72, label="Tree movable")
    tree_collection.set_clim(float(np.min(tree_heights)), float(np.max(tree_heights)))
    ax.add_collection(map_collection)
    ax.add_collection(tree_collection)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Map X [m]")
    ax.set_ylabel("Map Y [m]")
    ax.legend(loc="upper right")

    pad = 35.0
    ax.set_xlim(float(np.min(map_xy[:, 0]) - pad), float(np.max(map_xy[:, 0]) + pad))
    ax.set_ylim(float(np.min(map_xy[:, 1]) - pad), float(np.max(map_xy[:, 1]) + pad))

    dx_ax = fig.add_axes([0.14, 0.215, 0.62, 0.022])
    dy_ax = fig.add_axes([0.14, 0.175, 0.62, 0.022])
    dz_ax = fig.add_axes([0.14, 0.135, 0.62, 0.022])
    yaw_ax = fig.add_axes([0.14, 0.095, 0.62, 0.022])
    scale_ax = fig.add_axes([0.14, 0.055, 0.62, 0.022])
    save_ax = fig.add_axes([0.80, 0.126, 0.08, 0.065])
    reset_ax = fig.add_axes([0.90, 0.126, 0.08, 0.065])

    dx_slider = Slider(dx_ax, "x [m]", initial_dx - 250.0, initial_dx + 250.0, valinit=initial_dx, valstep=0.05)
    dy_slider = Slider(dy_ax, "y [m]", initial_dy - 250.0, initial_dy + 250.0, valinit=initial_dy, valstep=0.05)
    dz_slider = Slider(dz_ax, "z [m]", -10.0, 30.0, valinit=initial_dz, valstep=0.05)
    yaw_slider = Slider(yaw_ax, "yaw [deg]", -180.0, 180.0, valinit=initial_yaw, valstep=0.1)
    scale_slider = Slider(scale_ax, "scale", 0.05, 3.00, valinit=initial_scale, valstep=0.005)
    save_button = Button(save_ax, "Save")
    reset_button = Button(reset_ax, "Reset")

    state = {"dragging": False, "last": None}

    def redraw(_=None) -> None:
        polys, heights = transform_tree_polys(local_xyz, faces, dx_slider.val, dy_slider.val, dz_slider.val, yaw_slider.val, scale_slider.val)
        tree_collection.set_verts(polys)
        tree_collection.set_array(heights)
        tree_collection.set_clim(float(np.min(heights)), float(np.max(heights)))
        ax.set_title(
            "Tree placement on fixed Detailed 6D Map. Drag tree or use sliders. "
            f"x={dx_slider.val:.2f}, y={dy_slider.val:.2f}, z={dz_slider.val:.2f}, "
            f"yaw={yaw_slider.val:.1f} deg, scale={scale_slider.val:.3f}"
        )
        fig.canvas.draw_idle()

    def on_press(event) -> None:
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        state["dragging"] = True
        state["last"] = np.array([event.xdata, event.ydata], dtype=np.float64)

    def on_motion(event) -> None:
        if not state["dragging"] or event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        current = np.array([event.xdata, event.ydata], dtype=np.float64)
        delta = current - state["last"]
        state["last"] = current
        dx_slider.set_val(dx_slider.val + float(delta[0]))
        dy_slider.set_val(dy_slider.val + float(delta[1]))

    def on_release(_event) -> None:
        state["dragging"] = False
        state["last"] = None

    def on_key(event) -> None:
        step = 0.25 if event.key not in {"shift+left", "shift+right", "shift+up", "shift+down"} else 2.5
        if event.key in {"left", "shift+left"}:
            dx_slider.set_val(dx_slider.val - step)
        elif event.key in {"right", "shift+right"}:
            dx_slider.set_val(dx_slider.val + step)
        elif event.key in {"down", "shift+down"}:
            dy_slider.set_val(dy_slider.val - step)
        elif event.key in {"up", "shift+up"}:
            dy_slider.set_val(dy_slider.val + step)
        elif event.key == "s":
            save_transform(dx_slider.val, dy_slider.val, dz_slider.val, yaw_slider.val, scale_slider.val, bounds)

    def on_save(_event) -> None:
        save_transform(dx_slider.val, dy_slider.val, dz_slider.val, yaw_slider.val, scale_slider.val, bounds)

    def on_reset(_event) -> None:
        dx_slider.set_val(initial_dx)
        dy_slider.set_val(initial_dy)
        dz_slider.set_val(initial_dz)
        yaw_slider.set_val(initial_yaw)
        scale_slider.set_val(initial_scale)

    dx_slider.on_changed(redraw)
    dy_slider.on_changed(redraw)
    dz_slider.on_changed(redraw)
    yaw_slider.on_changed(redraw)
    scale_slider.on_changed(redraw)
    save_button.on_clicked(on_save)
    reset_button.on_clicked(on_reset)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw()
    plt.show()


if __name__ == "__main__":
    main()

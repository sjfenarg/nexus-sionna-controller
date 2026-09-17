from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.widgets import Button, Slider

from estimate_outdoor6d_scenebaker_shift import (
    OUTDOOR_STRUCTURAL,
    OUTDOOR_XML,
    SCENEBAKER_STRUCTURAL,
    SCENEBAKER_XML,
    parse_scene_meshes,
    read_ply,
    rotate,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "scenarios" / "outdoor6d_scenebaker_alignment"


def load_polygons(xml_path: Path, include_names: set[str], max_faces_per_mesh: int, seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    polys: list[np.ndarray] = []
    all_xy = []
    for mesh_path in parse_scene_meshes(xml_path):
        if mesh_path.name not in include_names:
            continue
        vertices, faces = read_ply(mesh_path)
        face_indices = np.arange(len(faces))
        if len(face_indices) > max_faces_per_mesh:
            face_indices = rng.choice(face_indices, max_faces_per_mesh, replace=False)
        for idx in face_indices:
            face = faces[int(idx)]
            if len(face) < 3:
                continue
            xy = vertices[face, :2].astype(np.float64)
            polys.append(xy)
            all_xy.append(xy)
    if not polys:
        raise RuntimeError(f"No polygons loaded from {xml_path}")
    return polys, np.vstack(all_xy)


def model_center(xy: np.ndarray) -> np.ndarray:
    lo = np.percentile(xy, 2, axis=0)
    hi = np.percentile(xy, 98, axis=0)
    return (lo + hi) / 2.0


def transform_polys(
    polys: list[np.ndarray], center: np.ndarray, dx: float, dy: float, rot_deg: float, scale: float
) -> list[np.ndarray]:
    moved = []
    delta = np.array([dx, dy], dtype=np.float64)
    for poly in polys:
        scaled = center + (poly - center) * scale
        moved.append(rotate(scaled, np.deg2rad(rot_deg), center) + delta)
    return moved


def save_transform(dx: float, dy: float, rot_deg: float, scale: float, center: np.ndarray) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "anchored_scenario": str(SCENEBAKER_XML.relative_to(ROOT)),
        "moved_scenario": str(OUTDOOR_XML.relative_to(ROOT)),
        "dx_m": float(dx),
        "dy_m": float(dy),
        "dz_m": 0.0,
        "rotation_deg": float(rot_deg),
        "scale_xy": float(scale),
        "rotation_center_x": float(center[0]),
        "rotation_center_y": float(center[1]),
        "transform_order": "scale Outdoor6D about rotation_center, rotate about rotation_center, then add dx/dy",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "manual_surface_outdoor6d_to_scenebaker_transform.json"
    csv_path = OUT_DIR / "manual_surface_outdoor6d_to_scenebaker_transform.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


def main() -> None:
    scenebaker_polys, scenebaker_xy = load_polygons(SCENEBAKER_XML, SCENEBAKER_STRUCTURAL, 30_000, 1)
    outdoor_polys, outdoor_xy = load_polygons(OUTDOOR_XML, OUTDOOR_STRUCTURAL, 9_000, 2)
    center = model_center(outdoor_xy)

    initial_dx = 0.0
    initial_dy = 0.0
    initial_rot = 0.0
    initial_scale = 1.0

    fig, ax = plt.subplots(figsize=(11, 8))
    plt.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.25)

    scene_collection = PolyCollection(
        scenebaker_polys,
        facecolors="#d95f02",
        edgecolors="#7f2704",
        linewidths=0.15,
        alpha=0.42,
        label="SceneBaker fixed",
    )
    moved_collection = PolyCollection(
        transform_polys(outdoor_polys, center, initial_dx, initial_dy, initial_rot, initial_scale),
        facecolors="#1f77b4",
        edgecolors="#08306b",
        linewidths=0.10,
        alpha=0.38,
        label="Outdoor6D movable",
    )
    ax.add_collection(scene_collection)
    ax.add_collection(moved_collection)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("SceneBaker X [m]")
    ax.set_ylabel("SceneBaker Y [m]")
    ax.legend(loc="upper right")

    all_xy = np.vstack([scenebaker_xy, outdoor_xy])
    pad = 30.0
    ax.set_xlim(float(np.min(all_xy[:, 0]) - pad), float(np.max(all_xy[:, 0]) + pad))
    ax.set_ylim(float(np.min(all_xy[:, 1]) - pad), float(np.max(all_xy[:, 1]) + pad))

    dx_ax = fig.add_axes([0.14, 0.175, 0.62, 0.022])
    dy_ax = fig.add_axes([0.14, 0.135, 0.62, 0.022])
    rot_ax = fig.add_axes([0.14, 0.095, 0.62, 0.022])
    scale_ax = fig.add_axes([0.14, 0.055, 0.62, 0.022])
    save_ax = fig.add_axes([0.80, 0.105, 0.08, 0.065])
    reset_ax = fig.add_axes([0.90, 0.105, 0.08, 0.065])

    dx_slider = Slider(dx_ax, "dx [m]", -400.0, 400.0, valinit=initial_dx, valstep=0.05)
    dy_slider = Slider(dy_ax, "dy [m]", -400.0, 400.0, valinit=initial_dy, valstep=0.05)
    rot_slider = Slider(rot_ax, "rot [deg]", -180.0, 180.0, valinit=initial_rot, valstep=0.01)
    scale_slider = Slider(scale_ax, "scale", 0.50, 1.50, valinit=initial_scale, valstep=0.001)
    save_button = Button(save_ax, "Save")
    reset_button = Button(reset_ax, "Reset")

    state = {"dragging": False, "last": None}

    def redraw(_=None) -> None:
        moved_collection.set_verts(
            transform_polys(outdoor_polys, center, dx_slider.val, dy_slider.val, rot_slider.val, scale_slider.val)
        )
        ax.set_title(
            "SceneBaker fixed. Drag blue surfaces or use sliders. "
            f"dx={dx_slider.val:.2f} m, dy={dy_slider.val:.2f} m, "
            f"rot={rot_slider.val:.2f} deg, scale={scale_slider.val:.3f}"
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
            save_transform(dx_slider.val, dy_slider.val, rot_slider.val, scale_slider.val, center)

    def on_save(_event) -> None:
        save_transform(dx_slider.val, dy_slider.val, rot_slider.val, scale_slider.val, center)

    def on_reset(_event) -> None:
        dx_slider.set_val(initial_dx)
        dy_slider.set_val(initial_dy)
        rot_slider.set_val(initial_rot)
        scale_slider.set_val(initial_scale)

    dx_slider.on_changed(redraw)
    dy_slider.on_changed(redraw)
    rot_slider.on_changed(redraw)
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

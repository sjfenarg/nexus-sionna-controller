from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from estimate_outdoor6d_scenebaker_shift import (
    OUTDOOR_STRUCTURAL,
    OUTDOOR_XML,
    SCENEBAKER_STRUCTURAL,
    SCENEBAKER_XML,
    load_points,
    parse_scene_meshes,
    rotate,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "scenarios" / "outdoor6d_scenebaker_alignment"


def sample_xy(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    xy = points[:, :2].astype(np.float64)
    if len(xy) <= n:
        return xy
    rng = np.random.default_rng(seed)
    return xy[rng.choice(len(xy), n, replace=False)]


def model_center(xy: np.ndarray) -> np.ndarray:
    lo = np.percentile(xy, 2, axis=0)
    hi = np.percentile(xy, 98, axis=0)
    return (lo + hi) / 2.0


def transform_xy(xy: np.ndarray, center: np.ndarray, dx: float, dy: float, rot_deg: float) -> np.ndarray:
    moved = rotate(xy, np.deg2rad(rot_deg), center)
    moved[:, 0] += dx
    moved[:, 1] += dy
    return moved


def save_transform(dx: float, dy: float, rot_deg: float, center: np.ndarray) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "anchored_scenario": str(SCENEBAKER_XML.relative_to(ROOT)),
        "moved_scenario": str(OUTDOOR_XML.relative_to(ROOT)),
        "dx_m": float(dx),
        "dy_m": float(dy),
        "dz_m": 0.0,
        "rotation_deg": float(rot_deg),
        "rotation_center_x": float(center[0]),
        "rotation_center_y": float(center[1]),
        "transform_order": "rotate Outdoor6D about rotation_center in its original XY frame, then add dx/dy",
    }
    json_path = OUT_DIR / "manual_simple_outdoor6d_to_scenebaker_transform.json"
    csv_path = OUT_DIR / "manual_simple_outdoor6d_to_scenebaker_transform.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


def main() -> None:
    outdoor_points, _ = load_points(parse_scene_meshes(OUTDOOR_XML), OUTDOOR_STRUCTURAL)
    scenebaker_points, _ = load_points(parse_scene_meshes(SCENEBAKER_XML), SCENEBAKER_STRUCTURAL)

    outdoor = sample_xy(outdoor_points, 60_000, 4)
    scenebaker = sample_xy(scenebaker_points, 60_000, 5)
    center = model_center(outdoor)

    initial_dx = 0.0
    initial_dy = 0.0
    initial_rot = 0.0

    fig, ax = plt.subplots(figsize=(11, 8))
    plt.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.25)

    ax.scatter(scenebaker[:, 0], scenebaker[:, 1], s=2, c="#d62728", alpha=0.42, label="SceneBaker fixed")
    moved0 = transform_xy(outdoor, center, initial_dx, initial_dy, initial_rot)
    moved_artist = ax.scatter(moved0[:, 0], moved0[:, 1], s=2, c="#1f77b4", alpha=0.42, label="Outdoor6D movable")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("SceneBaker X [m]")
    ax.set_ylabel("SceneBaker Y [m]")
    ax.legend(loc="upper right")

    all_xy = np.vstack([scenebaker, moved0])
    pad = 20.0
    ax.set_xlim(float(np.min(all_xy[:, 0]) - pad), float(np.max(all_xy[:, 0]) + pad))
    ax.set_ylim(float(np.min(all_xy[:, 1]) - pad), float(np.max(all_xy[:, 1]) + pad))

    dx_ax = fig.add_axes([0.14, 0.155, 0.62, 0.025])
    dy_ax = fig.add_axes([0.14, 0.11, 0.62, 0.025])
    rot_ax = fig.add_axes([0.14, 0.065, 0.62, 0.025])
    save_ax = fig.add_axes([0.80, 0.105, 0.08, 0.065])
    reset_ax = fig.add_axes([0.90, 0.105, 0.08, 0.065])

    dx_slider = Slider(dx_ax, "dx [m]", -400.0, 400.0, valinit=initial_dx, valstep=0.05)
    dy_slider = Slider(dy_ax, "dy [m]", -400.0, 400.0, valinit=initial_dy, valstep=0.05)
    rot_slider = Slider(rot_ax, "rot [deg]", -180.0, 180.0, valinit=initial_rot, valstep=0.01)
    save_button = Button(save_ax, "Save")
    reset_button = Button(reset_ax, "Reset")

    state = {"dragging": False, "last": None}

    def redraw(_=None) -> None:
        moved = transform_xy(outdoor, center, dx_slider.val, dy_slider.val, rot_slider.val)
        moved_artist.set_offsets(moved)
        ax.set_title(
            "SceneBaker fixed. Drag blue points or use sliders. "
            f"dx={dx_slider.val:.2f} m, dy={dy_slider.val:.2f} m, rot={rot_slider.val:.2f} deg"
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
            save_transform(dx_slider.val, dy_slider.val, rot_slider.val, center)

    def on_save(_event) -> None:
        save_transform(dx_slider.val, dy_slider.val, rot_slider.val, center)

    def on_reset(_event) -> None:
        dx_slider.set_val(initial_dx)
        dy_slider.set_val(initial_dy)
        rot_slider.set_val(initial_rot)

    dx_slider.on_changed(redraw)
    dy_slider.on_changed(redraw)
    rot_slider.on_changed(redraw)
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

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from estimate_outdoor6d_scenebaker_shift import (
    OUT_DIR,
    OUTDOOR_STRUCTURAL,
    OUTDOOR_XML,
    SCENEBAKER_STRUCTURAL,
    SCENEBAKER_XML,
    load_points,
    parse_scene_meshes,
    rotate,
)


def load_initial_transform() -> dict:
    path = OUT_DIR / "estimated_mesh_similarity_transform.csv"
    if not path.exists():
        return {
            "dx_m": 29.484666,
            "dy_m": -206.939652,
            "rotation_deg": 180.182727,
            "rotation_center_x": 11.681282,
            "rotation_center_y": -6.442456,
        }
    with path.open("r", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    return {
        "dx_m": float(row["dx_m"]),
        "dy_m": float(row["dy_m"]),
        "rotation_deg": float(row["rotation_deg"]),
        "rotation_center_x": float(row["rotation_center_x"]),
        "rotation_center_y": float(row["rotation_center_y"]),
    }


def sample(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(points) <= n:
        return points[:, :2]
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), n, replace=False), :2]


def transformed(points: np.ndarray, dx: float, dy: float, angle_deg: float, center: np.ndarray) -> np.ndarray:
    xy = rotate(points[:, :2], np.deg2rad(angle_deg), center)
    xy[:, 0] += dx
    xy[:, 1] += dy
    return xy


def save_transform(dx: float, dy: float, angle_deg: float, center: np.ndarray) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "dx_m": float(dx),
        "dy_m": float(dy),
        "dz_m": 0.0,
        "rotation_deg": float(angle_deg),
        "rotation_center_x": float(center[0]),
        "rotation_center_y": float(center[1]),
        "note": "Manual Outdoor6D to SceneBaker XY transform. Apply rotation about Outdoor6D center, then translation.",
    }
    (OUT_DIR / "manual_outdoor6d_to_scenebaker_transform.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (OUT_DIR / "manual_outdoor6d_to_scenebaker_transform.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)
    print(f"Saved manual transform to {OUT_DIR}")


def main() -> None:
    outdoor_points, _ = load_points(parse_scene_meshes(OUTDOOR_XML), OUTDOOR_STRUCTURAL)
    scenebaker_points, _ = load_points(parse_scene_meshes(SCENEBAKER_XML), SCENEBAKER_STRUCTURAL)
    outdoor = sample(outdoor_points, 45_000, 21)
    scenebaker = sample(scenebaker_points, 45_000, 22)
    init = load_initial_transform()
    center = np.array([init["rotation_center_x"], init["rotation_center_y"]], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.24)
    ax.scatter(scenebaker[:, 0], scenebaker[:, 1], s=2, c="#d62728", alpha=0.35, label="SceneBaker/Catastro")
    shifted = transformed(outdoor, init["dx_m"], init["dy_m"], init["rotation_deg"], center)
    outdoor_artist = ax.scatter(shifted[:, 0], shifted[:, 1], s=2, c="#1f77b4", alpha=0.35, label="Outdoor6D")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("SceneBaker X [m]")
    ax.set_ylabel("SceneBaker Y [m]")
    ax.legend(loc="upper right")
    ax.set_title("Manual Outdoor6D to SceneBaker alignment")

    dx_ax = fig.add_axes([0.16, 0.15, 0.68, 0.025])
    dy_ax = fig.add_axes([0.16, 0.105, 0.68, 0.025])
    rot_ax = fig.add_axes([0.16, 0.06, 0.68, 0.025])
    save_ax = fig.add_axes([0.86, 0.055, 0.10, 0.075])

    dx_slider = Slider(dx_ax, "dx [m]", init["dx_m"] - 80, init["dx_m"] + 80, valinit=init["dx_m"])
    dy_slider = Slider(dy_ax, "dy [m]", init["dy_m"] - 80, init["dy_m"] + 80, valinit=init["dy_m"])
    rot_slider = Slider(rot_ax, "rot [deg]", init["rotation_deg"] - 30, init["rotation_deg"] + 30, valinit=init["rotation_deg"])
    save_button = Button(save_ax, "Save")

    def redraw(_=None) -> None:
        xy = transformed(outdoor, dx_slider.val, dy_slider.val, rot_slider.val, center)
        outdoor_artist.set_offsets(xy)
        ax.set_title(
            f"Manual Outdoor6D to SceneBaker alignment: dx={dx_slider.val:.3f}, "
            f"dy={dy_slider.val:.3f}, rot={rot_slider.val:.3f} deg"
        )
        fig.canvas.draw_idle()

    def on_save(_event) -> None:
        save_transform(dx_slider.val, dy_slider.val, rot_slider.val, center)

    dx_slider.on_changed(redraw)
    dy_slider.on_changed(redraw)
    rot_slider.on_changed(redraw)
    save_button.on_clicked(on_save)
    redraw()
    plt.show()


if __name__ == "__main__":
    main()

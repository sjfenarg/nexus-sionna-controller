from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from isac_6d_sampler.core.antenna_patterns import (
    POWERLOG_FREQUENCY_LABELS_HZ,
    QOM_OMNI_FREQUENCY_LABELS_HZ,
    _powerlog_numpy_cuts,
    powerlog_normalized_gain_db_from_local_dirs,
    qom_omni_normalized_gain_db_from_local_dirs,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "antenna_diagrams_3d_by_frequency"
DB_FLOOR = -45.0
ALPHA_SAMPLES = 73
BETA_SAMPLES = 145


def sphere_dirs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = np.linspace(0.0, np.pi, ALPHA_SAMPLES, dtype=np.float64)
    beta = np.linspace(0.0, 2.0 * np.pi, BETA_SAMPLES, dtype=np.float64)
    aa, bb = np.meshgrid(alpha, beta, indexing="ij")
    x = np.cos(aa)
    y = np.sin(aa) * np.cos(bb)
    z = np.sin(aa) * np.sin(bb)
    return x, y, z


def normalized_radius(gain_db: np.ndarray) -> np.ndarray:
    clipped = np.clip(gain_db, DB_FLOOR, 0.0)
    return np.maximum((clipped - DB_FLOOR) / abs(DB_FLOOR), 0.04)


def plot_pattern(
    *,
    antenna_name: str,
    frequency_label: str,
    gain_db: np.ndarray,
    dirs: tuple[np.ndarray, np.ndarray, np.ndarray],
    path: Path,
) -> None:
    x_dir, y_dir, z_dir = dirs
    radius = normalized_radius(gain_db)
    x = radius * x_dir
    y = radius * y_dir
    z = radius * z_dir

    fig = plt.figure(figsize=(8.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.get_cmap("viridis")((np.clip(gain_db, DB_FLOOR, 0.0) - DB_FLOOR) / abs(DB_FLOOR))
    ax.plot_surface(x, y, z, facecolors=colors, linewidth=0.0, antialiased=True, shade=False)
    ax.plot([0.0, 1.15], [0.0, 0.0], [0.0, 0.0], color="black", linewidth=1.5)
    ax.text(1.22, 0.0, 0.0, "+x boresight", fontsize=8)
    ax.set_title(f"{antenna_name} antenna diagram, {frequency_label}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.view_init(elev=24.0, azim=-42.0)
    mappable = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=DB_FLOOR, vmax=0.0))
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.72, pad=0.08, label="Gain [dB rel. max]")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_contact_sheet(kind: str, paths: list[Path], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, image_path in zip(axes.flat, paths):
        image = plt.imread(image_path)
        ax.imshow(image)
        ax.set_title(image_path.stem.replace(f"{kind}_", "").replace("_3d", ""))
        ax.axis("off")
    for ax in axes.flat[len(paths):]:
        ax.axis("off")
    fig.suptitle(f"{kind} 3D antenna diagrams by frequency", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def safe_label(label: str) -> str:
    return label.replace(".", "p")


def powerlog_original_normalized_gain_db(local_dirs: np.ndarray, frequency_label: str) -> np.ndarray:
    x = np.clip(local_dirs[..., 0], -1.0, 1.0)
    y = local_dirs[..., 1]
    z = local_dirs[..., 2]
    horizontal_angle = np.arctan2(y, x)
    vertical_angle = np.arctan2(z, np.maximum(np.sqrt(x * x + y * y), 1e-12))
    h_db, v_db, max_gain_db = _powerlog_numpy_cuts(frequency_label)
    return h_db(horizontal_angle) + v_db(vertical_angle) - 2.0 * max_gain_db


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dirs = sphere_dirs()
    local_dirs = np.stack(dirs, axis=-1)

    powerlog_paths = []
    for label in POWERLOG_FREQUENCY_LABELS_HZ:
        gain = powerlog_normalized_gain_db_from_local_dirs(local_dirs, label)
        path = OUT_DIR / f"powerLog_{safe_label(label)}_3d.png"
        plot_pattern(antenna_name="powerLog", frequency_label=label, gain_db=gain, dirs=dirs, path=path)
        powerlog_paths.append(path)
        print(path)

    omni_paths = []
    for label in QOM_OMNI_FREQUENCY_LABELS_HZ:
        gain = qom_omni_normalized_gain_db_from_local_dirs(local_dirs, label)
        path = OUT_DIR / f"qom_st_2_18_omni_{safe_label(label)}_3d.png"
        plot_pattern(antenna_name="QOM-ST-2-18 omni", frequency_label=label, gain_db=gain, dirs=dirs, path=path)
        omni_paths.append(path)
        print(path)

    plot_contact_sheet("powerLog", powerlog_paths, OUT_DIR / "powerLog_all_frequencies_3d.png")
    plot_contact_sheet("qom_st_2_18_omni", omni_paths, OUT_DIR / "qom_st_2_18_omni_all_frequencies_3d.png")

    powerlog_original_paths = []
    for label in POWERLOG_FREQUENCY_LABELS_HZ:
        gain = powerlog_original_normalized_gain_db(local_dirs, label)
        path = OUT_DIR / f"powerLog_original_no_back_lobe_{safe_label(label)}_3d.png"
        plot_pattern(antenna_name="powerLog original, no back-lobe boost", frequency_label=label, gain_db=gain, dirs=dirs, path=path)
        powerlog_original_paths.append(path)
        print(path)

    plot_contact_sheet(
        "powerLog_original_no_back_lobe",
        powerlog_original_paths,
        OUT_DIR / "powerLog_original_no_back_lobe_all_frequencies_3d.png",
    )


if __name__ == "__main__":
    main()

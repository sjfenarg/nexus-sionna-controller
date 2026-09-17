from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

C0 = 299_792_458.0
ROOT = Path(__file__).resolve().parents[1]
H5_PATH = ROOT / "output" / "hdf5_antennas_powerlog" / "scene_20260917_175259.h5"
OUT_DIR = H5_PATH.parent / "pdp_heatmaps"
TIMEFRAME = "tf000"
TX_INDEX = 10
FIBER_DISTANCE_M = 0.0
MAX_RANGE_M = 150.0


def find_sample_base(h5: h5py.File) -> str:
    scenario = next(iter(h5["scenarios"].keys()))
    sample = next(k for k in h5[f"scenarios/{scenario}"].keys() if k != "metadata")
    return f"scenarios/{scenario}/{sample}"


def uniform_frequency_grid(f_vector: np.ndarray) -> tuple[np.ndarray, int, float]:
    f_vector = np.asarray(f_vector, dtype=np.float64).reshape(-1)
    diffs = np.diff(np.sort(f_vector))
    df = float(np.min(diffs[diffs > 0]))
    n_grid = int(round((float(f_vector[-1]) - float(f_vector[0])) / df)) + 1
    indices = np.rint((f_vector - f_vector[0]) / df).astype(np.int64)
    return indices, n_grid, df


def cfr_to_pdp_rows(h: np.ndarray, indices: np.ndarray, n_grid: int) -> np.ndarray:
    values = np.asarray(h, dtype=np.complex64)
    frequency_count = values.shape[-1]
    rows = values.reshape((-1, frequency_count))
    window = np.hamming(frequency_count).astype(np.float32)
    pdp_rows = []
    for row in rows:
        grid = np.zeros(n_grid, dtype=np.complex64)
        grid[indices] = row * window
        cir = np.fft.ifft(grid)
        pdp_rows.append(np.abs(cir) ** 2)
    return np.asarray(pdp_rows)


def to_db(power: np.ndarray, floor_db: float = -80.0) -> np.ndarray:
    ref = np.nanmax(power)
    if not np.isfinite(ref) or ref <= 0.0:
        ref = 1.0
    db = 10.0 * np.log10(np.maximum(power / ref, 10 ** (floor_db / 10.0)))
    return np.maximum(db, floor_db)


def save_heatmap(data_db: np.ndarray, distance: np.ndarray, row_labels: list[str], path: Path, title: str) -> None:
    range_m = distance - FIBER_DISTANCE_M
    keep = (range_m >= 0.0) & (range_m <= MAX_RANGE_M)
    if not np.any(keep):
        raise RuntimeError("No range samples remain after fiber shift/range crop")
    fig_height = max(6.0, min(18.0, 0.15 * len(row_labels) + 3.0))
    fig, ax = plt.subplots(figsize=(12.5, fig_height), constrained_layout=True)
    im = ax.imshow(
        data_db[:, keep],
        aspect="auto",
        origin="lower",
        extent=[range_m[keep][0], range_m[keep][-1], -0.5, len(row_labels) - 0.5],
        cmap="inferno",
        vmin=-60.0,
        vmax=0.0,
        interpolation="nearest",
    )
    ax.set_xlabel(f"Simulated range [m]")
    ax.set_ylabel("BS antenna element")
    ax.set_title(title)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Power [dB rel. max]")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with h5py.File(H5_PATH, "r") as f:
        base = find_sample_base(f)
        f_vector = f[f"{base}/parameters/f_vector"][...].reshape(-1)
        indices, n_grid, df = uniform_frequency_grid(f_vector)
        distance = np.arange(n_grid) * C0 / (n_grid * df)
        h_group = f[f"{base}/timeframes/{TIMEFRAME}/h"]
        ue_links = []
        for key in sorted(h_group.keys()):
            if not key.startswith("rx") or f"_tx{TX_INDEX}" not in key:
                continue
            rx_idx = int(key.split("_tx", 1)[0][2:])
            if rx_idx >= TX_INDEX:
                continue
            ue_links.append((rx_idx, key))
        if not ue_links:
            raise RuntimeError(f"No BS->UE links tx{TX_INDEX} found in {H5_PATH}")
        for rx_idx, key in ue_links:
            ue = f"ue{rx_idx}"
            h = h_group[key][...]
            pdp = cfr_to_pdp_rows(h, indices, n_grid)
            row_labels = [f"bs_el{idx:03d}" for idx in range(pdp.shape[0])]
            pdp_db = to_db(pdp)
            save_heatmap(
                pdp_db,
                distance,
                row_labels,
                OUT_DIR / f"simulation_pdp_{ue}_bs_elements.png",
                f"Simulation PDP, bs0 BS elements to {ue}",
            )
            range_m = distance - FIBER_DISTANCE_M
            keep = (range_m >= 0.0) & (range_m <= MAX_RANGE_M)
            np.savetxt(
                OUT_DIR / f"simulation_pdp_{ue}_bs_elements_db.csv",
                np.c_[range_m[keep], pdp_db[:, keep].T],
                delimiter=",",
                header="range_m," + ",".join(row_labels),
                comments="",
            )
            print(OUT_DIR / f"simulation_pdp_{ue}_bs_elements.png")


if __name__ == "__main__":
    main()

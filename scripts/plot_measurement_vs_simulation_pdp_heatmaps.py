from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

C0 = 299_792_458.0
ROOT = Path(__file__).resolve().parents[1]
MEAS_H5 = ROOT / "meas" / "nexus_trials.h5"
SIM_TAG = "20260918_003534"
SIM_H5 = ROOT / "output" / "hdf5_antennas_powerlog" / f"scene_{SIM_TAG}.h5"
OUT_DIR = ROOT / "output" / "hdf5_antennas_powerlog" / f"pdp_heatmaps_comparison_{SIM_TAG}"
FIBER_SHIFT_MEAS_M = 170.0
MAX_RANGE_M = 150.0
TX_BS_INDEX = 10
TARGET_UES: set[str] | None = None
COLORMAP_MIN_DB = -60.0


def find_base(h5: h5py.File) -> str:
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
    middle = values.shape[4] if values.ndim >= 6 else 1
    rows = values.reshape((-1, middle, frequency_count))
    window = np.hamming(frequency_count).astype(np.float32)
    pdp_rows = []
    for row in rows:
        grid = np.zeros((row.shape[0], n_grid), dtype=np.complex64)
        grid[:, indices] = row * window[None, :]
        cir = np.fft.ifft(grid, axis=-1)
        pdp_rows.append(np.mean(np.abs(cir) ** 2, axis=0))
    return np.asarray(pdp_rows)


def to_db(power: np.ndarray, floor_db: float = -80.0) -> np.ndarray:
    ref = np.nanmax(power)
    if not np.isfinite(ref) or ref <= 0.0:
        ref = 1.0
    db = 10.0 * np.log10(np.maximum(power / ref, 10 ** (floor_db / 10.0)))
    return np.maximum(db, floor_db)


def load_bs_to_ue_pdps(path: Path, *, measurement: bool) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with h5py.File(path, "r") as f:
        base = find_base(f)
        f_vector = f[f"{base}/parameters/f_vector"][...].reshape(-1)
        indices, n_grid, df = uniform_frequency_grid(f_vector)
        distance = np.arange(n_grid) * C0 / (n_grid * df)
        h_group = f[f"{base}/timeframes/tf000/h"]
        out: dict[str, np.ndarray] = {}
        for ue_idx in range(10):
            key = f"rx{ue_idx}_tx{TX_BS_INDEX}"
            if key not in h_group:
                continue
            out[f"ue{ue_idx}"] = cfr_to_pdp_rows(h_group[key][...], indices, n_grid)
        return out, distance


def add_panel(ax, data_db: np.ndarray, range_m: np.ndarray, title: str) -> None:
    keep = (range_m >= 0.0) & (range_m <= MAX_RANGE_M)
    im = ax.imshow(
        data_db[:, keep],
        aspect="auto",
        origin="lower",
        extent=[range_m[keep][0], range_m[keep][-1], -0.5, data_db.shape[0] - 0.5],
        cmap="inferno",
        vmin=COLORMAP_MIN_DB,
        vmax=0.0,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("BS antenna element")
    ax.set_yticks(np.arange(data_db.shape[0]))
    ax.set_yticklabels([f"bs_el{idx:03d}" for idx in range(data_db.shape[0])])
    return im


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meas_pdps, meas_distance = load_bs_to_ue_pdps(MEAS_H5, measurement=True)
    sim_pdps, sim_distance = load_bs_to_ue_pdps(SIM_H5, measurement=False)
    common = sorted(set(meas_pdps) & set(sim_pdps), key=lambda name: int(name[2:]))
    if not common:
        raise RuntimeError("No common BS->UE links found")
    for ue in common:
        if TARGET_UES is not None and ue not in TARGET_UES:
            continue
        meas_db = to_db(meas_pdps[ue])
        sim_db = to_db(sim_pdps[ue])
        fig, axes = plt.subplots(1, 2, figsize=(19, 7.2), constrained_layout=True, sharey=True)
        im = add_panel(
            axes[0],
            meas_db,
            meas_distance - FIBER_SHIFT_MEAS_M,
            f"Measurement {ue}: 170 m fiber-shifted range",
        )
        add_panel(
            axes[1],
            sim_db,
            sim_distance,
            f"Simulation {SIM_TAG} {ue}: true range",
        )
        axes[1].set_ylabel("")
        fig.colorbar(im, ax=axes, location="right", shrink=0.92, label="Power [dB rel. panel max]")
        fig.suptitle(f"PDP comparison, bs0 BS elements to {ue}", fontsize=16)
        path = OUT_DIR / f"pdp_comparison_{ue}_measurement_vs_simulation_{SIM_TAG}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()

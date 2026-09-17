from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

C0 = 299_792_458.0
ROOT = Path(__file__).resolve().parents[1]
MEAS_H5 = ROOT / "meas" / "nexus_trials.h5"
SIM_TAG = "20260917_202918"
SIM_H5 = ROOT / "output" / "hdf5_antennas_powerlog" / f"scene_{SIM_TAG}.h5"
BISTATIC_OUT_DIR = ROOT / "output" / "hdf5_antennas_powerlog" / f"pdp_heatmaps_ueue_bistatic_comparison_{SIM_TAG}"
MONOSTATIC_OUT_DIR = ROOT / "output" / "hdf5_antennas_powerlog" / f"pdp_heatmaps_ue_monostatic_comparison_{SIM_TAG}"
MAX_RANGE_M = 150.0
COLORMAP_MIN_DB = -35.0
UE_COUNT = 10


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


def load_ueue_pdps(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with h5py.File(path, "r") as f:
        base = find_base(f)
        f_vector = f[f"{base}/parameters/f_vector"][...].reshape(-1)
        indices, n_grid, df = uniform_frequency_grid(f_vector)
        distance = np.arange(n_grid) * C0 / (n_grid * df)
        h_group = f[f"{base}/timeframes/tf000/h"]
        out: dict[str, np.ndarray] = {}
        for rx_idx in range(UE_COUNT):
            for tx_idx in range(UE_COUNT):
                key = f"rx{rx_idx}_tx{tx_idx}"
                if key in h_group:
                    out[f"ue{rx_idx}_from_ue{tx_idx}"] = cfr_to_pdp_rows(h_group[key][...], indices, n_grid)
        return out, distance


def add_panel(ax, data_db: np.ndarray, range_m: np.ndarray, title: str, link_label: str):
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
    ax.set_ylabel("UE-UE link")
    ax.set_yticks(np.arange(data_db.shape[0]))
    ax.set_yticklabels([link_label] if data_db.shape[0] == 1 else [f"path{idx}" for idx in range(data_db.shape[0])])
    return im


def link_sort_key(name: str) -> tuple[int, int]:
    rx, tx = name.split("_from_")
    return int(rx.removeprefix("ue")), int(tx.removeprefix("ue"))


def main() -> None:
    BISTATIC_OUT_DIR.mkdir(parents=True, exist_ok=True)
    MONOSTATIC_OUT_DIR.mkdir(parents=True, exist_ok=True)
    meas_pdps, meas_distance = load_ueue_pdps(MEAS_H5)
    sim_pdps, sim_distance = load_ueue_pdps(SIM_H5)
    common = sorted(set(meas_pdps) & set(sim_pdps), key=link_sort_key)
    if not common:
        raise RuntimeError("No common UE-UE links found")
    for link in common:
        meas_db = to_db(meas_pdps[link])
        sim_db = to_db(sim_pdps[link])
        rx_label, tx_label = link.split("_from_")
        is_monostatic = rx_label == tx_label
        pretty = f"{rx_label} <- {tx_label}"
        fig, axes = plt.subplots(1, 2, figsize=(19, 4.8), constrained_layout=True, sharey=True)
        im = add_panel(
            axes[0],
            meas_db,
            meas_distance,
            f"Measurement {pretty}: true range",
            pretty,
        )
        add_panel(
            axes[1],
            sim_db,
            sim_distance,
            f"Simulation {SIM_TAG} {pretty}: true range",
            pretty,
        )
        axes[1].set_ylabel("")
        fig.colorbar(im, ax=axes, location="right", shrink=0.9, label="Power [dB rel. panel max]")
        if is_monostatic:
            fig.suptitle(f"UE monostatic PDP comparison, {pretty}", fontsize=16)
            path = MONOSTATIC_OUT_DIR / f"pdp_comparison_{link}_measurement_vs_simulation_{SIM_TAG}.png"
        else:
            fig.suptitle(f"UE-UE bistatic PDP comparison, {pretty}", fontsize=16)
            path = BISTATIC_OUT_DIR / f"pdp_comparison_{link}_measurement_vs_simulation_{SIM_TAG}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_INPUT_DIR = Path("output/channel_charting_dataset")
DEFAULT_OUTPUT = Path("output/channel_charting_dataset_ml/channel_charting_cfr_features.h5")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--drop-empty", action="store_true", default=True)
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.h5"))
    if not files:
        raise SystemExit(f"No HDF5 files found in {args.input_dir}")

    rows = []
    f_vector = None
    dropped = []
    for scene_index, path in enumerate(files):
        with h5py.File(path, "r") as h5:
            scenario_name = next(key for key in h5["scenarios"].keys() if key != "metadata")
            scenario = h5["scenarios"][scenario_name]
            sample_id = [key for key in scenario.keys() if key != "metadata"][0]
            sample = scenario[sample_id]
            request = json.loads(sample["metadata"].attrs["request_json"])
            n_ue = len(request["scene"]["user_equipments"])
            bs_index = n_ue
            if f_vector is None:
                f_vector = np.asarray(sample["parameters/f_vector"][0], dtype=np.float64)
            timeframes = sorted(sample["timeframes"].keys(), key=lambda key: int(key[2:]))
            for timeframe_index, timeframe_name in enumerate(timeframes):
                timeframe = sample["timeframes"][timeframe_name]
                positions = timeframe["positions/devices"]
                orientations = timeframe["orientations/devices"]
                for ue_index in range(n_ue):
                    ue_id = f"ue{ue_index}"
                    link_key = f"rx{bs_index}_tx{ue_index}"
                    cfr = np.asarray(timeframe["h"][link_key][...], dtype=np.complex64).reshape(-1, f_vector.size)
                    if np.count_nonzero(cfr) == 0:
                        dropped.append((sample_id, timeframe_name, ue_id, link_key))
                        if args.drop_empty:
                            continue
                    power = np.mean(np.abs(cfr) ** 2, axis=0, dtype=np.float64)
                    power_db = (10.0 * np.log10(power + 1e-30)).astype(np.float32)
                    rows.append(
                        {
                            "scene_index": scene_index,
                            "sample_id": sample_id,
                            "scenario_name": scenario_name,
                            "source_file": path.name,
                            "timeframe_index": timeframe_index,
                            "timeframe_name": timeframe_name,
                            "ue_index": ue_index,
                            "ue_id": ue_id,
                            "link_key": link_key,
                            "position": np.asarray(positions[ue_id][:], dtype=np.float32),
                            "orientation": np.asarray(orientations[ue_id][:], dtype=np.float32),
                            "rss_db": np.float32(10.0 * np.log10(np.mean(power) + 1e-30)),
                            "cfr_power_db": power_db,
                            "cfr_power_db_norm": (power_db - np.mean(power_db)).astype(np.float32),
                        }
                    )

    if not rows:
        raise SystemExit("No valid channel samples found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(args.output, "w") as out:
        out.attrs["description"] = "Channel charting features extracted from trajectory-scene CFR HDF5 files"
        out.attrs["input_dir"] = str(args.input_dir)
        out.attrs["n_samples"] = len(rows)
        out.attrs["n_dropped_empty_links"] = len(dropped)
        out.attrs["feature_note"] = "cfr_power_db is panel-averaged UE->BS CFR power over frequency at the BS array"
        out.create_dataset("frequency_hz", data=f_vector, compression="gzip")
        out.create_dataset("position_xyz", data=np.vstack([row["position"] for row in rows]), compression="gzip")
        out.create_dataset("orientation_ypr_rad", data=np.vstack([row["orientation"] for row in rows]), compression="gzip")
        out.create_dataset("rss_db", data=np.asarray([row["rss_db"] for row in rows], dtype=np.float32), compression="gzip")
        out.create_dataset(
            "cfr_power_db",
            data=np.vstack([row["cfr_power_db"] for row in rows]).astype(np.float32),
            compression="gzip",
            chunks=(min(128, len(rows)), f_vector.size),
        )
        out.create_dataset(
            "cfr_power_db_norm",
            data=np.vstack([row["cfr_power_db_norm"] for row in rows]).astype(np.float32),
            compression="gzip",
            chunks=(min(128, len(rows)), f_vector.size),
        )
        out.create_dataset("scene_index", data=np.asarray([row["scene_index"] for row in rows], dtype=np.int32))
        out.create_dataset("timeframe_index", data=np.asarray([row["timeframe_index"] for row in rows], dtype=np.int32))
        out.create_dataset("ue_index", data=np.asarray([row["ue_index"] for row in rows], dtype=np.int32))
        for key in ("sample_id", "scenario_name", "source_file", "timeframe_name", "ue_id", "link_key"):
            out.create_dataset(key, data=np.asarray([row[key] for row in rows], dtype=object), dtype=string_dtype)

        split = np.full(len(rows), "train", dtype=object)
        scene_indices = np.asarray([row["scene_index"] for row in rows], dtype=np.int32)
        if np.max(scene_indices) >= 6:
            split[scene_indices == 6] = "val"
        if np.max(scene_indices) >= 7:
            split[scene_indices == 7] = "test"
        out.create_dataset("split", data=split, dtype=string_dtype)

        dropped_group = out.create_group("dropped_empty_links")
        dropped_group.attrs["count"] = len(dropped)
        if dropped:
            dropped_group.create_dataset("sample_id", data=np.asarray([item[0] for item in dropped], dtype=object), dtype=string_dtype)
            dropped_group.create_dataset("timeframe_name", data=np.asarray([item[1] for item in dropped], dtype=object), dtype=string_dtype)
            dropped_group.create_dataset("ue_id", data=np.asarray([item[2] for item in dropped], dtype=object), dtype=string_dtype)
            dropped_group.create_dataset("link_key", data=np.asarray([item[3] for item in dropped], dtype=object), dtype=string_dtype)

    print(args.output)
    print(f"samples={len(rows)} dropped_empty_links={len(dropped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

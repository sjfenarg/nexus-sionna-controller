from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py


DEFAULT_CONFIG_DIR = Path("scenes/channel_charting_dataset")
DEFAULT_OUTPUT_DIR = Path("output/channel_charting_dataset")
DEFAULT_LOG_DIR = Path("output/channel_charting_dataset_logs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--pattern", default="cc_scene_*.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Run configs even if their sample_id already exists")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    configs = sorted(args.config_dir.glob(args.pattern))
    if args.limit is not None:
        configs = configs[: max(0, args.limit)]
    if not configs:
        raise SystemExit(f"No configs matched {args.config_dir / args.pattern}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    completed = set() if args.force else _completed_sample_ids(args.output_dir)
    print(f"configs={len(configs)} output_dir={args.output_dir} completed={len(completed)}")

    for config in configs:
        sample_id = _sample_id(config)
        if sample_id in completed:
            print(f"SKIP {config.name}: sample_id {sample_id} already exists")
            continue
        command = [
            sys.executable,
            "-m",
            "isac_6d_sampler.cli",
            "--config",
            str(config),
            "--output-dir",
            str(args.output_dir),
        ]
        if args.dry_run:
            command.append("--dry-run")
        log_path = args.log_dir / f"{sample_id}.log"
        print(f"RUN {config.name}: sample_id={sample_id} log={log_path}")
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        if process.returncode != 0:
            print(f"FAIL {config.name}: returncode={process.returncode}; see {log_path}")
            return process.returncode
        completed.add(sample_id)
        print(f"DONE {config.name}")
    return 0


def _sample_id(config: Path) -> str:
    with config.open("r", encoding="utf-8") as fh:
        return str(json.load(fh).get("sample_id", config.stem))


def _completed_sample_ids(output_dir: Path) -> set[str]:
    sample_ids: set[str] = set()
    for path in sorted(output_dir.glob("*.h5")):
        try:
            with h5py.File(path, "r") as h5:
                scenarios = h5.get("scenarios")
                if scenarios is None:
                    continue
                for scenario in scenarios.values():
                    for key in scenario.keys():
                        if key != "metadata":
                            sample_ids.add(str(key))
        except OSError:
            continue
    return sample_ids


if __name__ == "__main__":
    raise SystemExit(main())

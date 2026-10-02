"""Run the controller with the parent workspace's optional local LLVM runtime."""

import argparse
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interface", choices=("cli", "gui"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    scratch = repo.parent / "tmp" / "msc-implementation"
    scratch.mkdir(parents=True, exist_ok=True)
    llvm = scratch / "runtime" / "llvm22" / "bin" / "LLVM-C.dll"
    if llvm.exists():
        os.environ.setdefault("DRJIT_LIBLLVM_PATH", str(llvm))
    os.environ.setdefault("DRJIT_CACHE_DIR", str(scratch / "drjit-cache-llvm22"))
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo / "src"))
    module = "isac_6d_sampler.gui.app" if args.interface == "gui" else "isac_6d_sampler.cli"
    sys.argv = [module, *args.arguments]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()

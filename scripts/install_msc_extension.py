"""Install the versioned MSC sources into the selected Sionna RT environment.

Run with the controller's .venv Python. Backups go to the parent tmp/ directory.
The operation is idempotent and does not modify upstream TR38901 code.
"""

import argparse
from datetime import datetime
from importlib.metadata import distribution, version
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[1]
EXPORT = "\n# MSC extension (managed by install_msc_extension.py)\nfrom .msc import (MSCRCS, MSCCPM, MSCScatteringModel, MSCSensingTarget,\n                  MSCCenterParameters, MSCPolarizationParameters)\n"
ANCHOR = "        # Generates sources and targets positions and orientations."
HOOK = """        # MSC extension: bind frequency-dependent models to the scene carrier.
        for target in scene.sensing_targets.values():
            bind_frequency = getattr(target.scattering_model, "set_frequency", None)
            if bind_frequency is not None:
                bind_frequency(float(scene.frequency[0]))

"""


def install():
    if version("sionna-rt") != "2.2.0":
        raise RuntimeError("MSC installer currently supports Sionna RT 2.2.0 only")
    rcs = Path(distribution("sionna-rt").locate_file("sionna/rt/rcs")).resolve()
    init_path, solver_path = rcs / "__init__.py", rcs / "solver.py"
    absorber_path = rcs.parent / "radio_materials" / "absorber_material.py"
    scene_path = rcs.parent / "scene.py"
    init_text, solver_text = init_path.read_text(encoding="utf-8"), solver_path.read_text(encoding="utf-8")
    absorber_text = absorber_path.read_text(encoding="utf-8")
    scene_text = scene_path.read_text(encoding="utf-8")
    if solver_text.count(ANCHOR) != 1:
        raise RuntimeError("Unexpected RCSSolver layout; refusing to patch")
    absorber_anchor = "    def traverse(self, callback: mi.TraversalCallback):"
    carrier_anchor = "        self._frequency = mi.Float(f)\n"
    if "    def frequency_update(" not in absorber_text and absorber_text.count(absorber_anchor) != 1:
        raise RuntimeError("Unexpected absorber layout; refusing to patch")
    if ("# MSC extension: keep carrier sweeps out of JIT constants" not in scene_text
            and scene_text.count(carrier_anchor) != 1):
        raise RuntimeError("Unexpected scene carrier setter; refusing to patch")
    # Validate all supported upstream layouts before writing any installed file.
    backup = REPO.parent / "tmp" / "msc-implementation" / "sionna-backups" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup.mkdir(parents=True)
    shutil.copy2(init_path, backup / "rcs-init.py")
    shutil.copy2(solver_path, backup / "solver.py")
    shutil.copy2(absorber_path, backup / "absorber_material.py")
    shutil.copy2(scene_path, backup / "scene.py")
    if (rcs / "msc").exists():
        shutil.copytree(rcs / "msc", backup / "msc")
    shutil.copytree(REPO / "sionna_extensions" / "msc", rcs / "msc", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if "# MSC extension (managed" not in init_text:
        init_path.write_text(init_text.rstrip() + "\n" + EXPORT, encoding="utf-8")
    marker = "        # MSC extension: bind frequency"
    if marker in solver_text:
        start, end = solver_text.index(marker), solver_text.index(ANCHOR)
        solver_text = solver_text[:start] + HOOK + solver_text[end:]
    else:
        solver_text = solver_text.replace(ANCHOR, HOOK + ANCHOR)
    solver_path.write_text(solver_text, encoding="utf-8")
    # Sionna 2.2's scene.frequency setter calls this method on every material;
    # the frequency-independent sensing-target absorber lacks it upstream.
    if "    def frequency_update(" not in absorber_text:
        method = ('    def frequency_update(self):\n'
                  '        """Frequency-independent absorber; compatible with carrier sweeps."""\n'
                  '        pass\n\n')
        absorber_path.write_text(absorber_text.replace(absorber_anchor, method + absorber_anchor), encoding="utf-8")
    if "# MSC extension: keep carrier sweeps out of JIT constants" not in scene_text:
        scene_path.write_text(scene_text.replace(carrier_anchor, carrier_anchor +
                             "        # MSC extension: keep carrier sweeps out of JIT constants.\n"
                             "        dr.make_opaque(self._frequency)\n"), encoding="utf-8")
    print(f"Installed MSC in {rcs}; original files backed up to {backup}")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    install()

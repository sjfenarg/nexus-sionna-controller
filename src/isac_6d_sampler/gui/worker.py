from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from isac_6d_sampler.core.model import SimulationRequest
from isac_6d_sampler.io.reference_h5 import ReferenceH5Writer, default_output_path
from isac_6d_sampler.sim.dry_run import DryRunSimulator
from isac_6d_sampler.sim.planner import build_simulation_plan
from isac_6d_sampler.sim.sionna_backend import SionnaSimulator


class SimulationWorker(QThread):
    """Run a simulation off the GUI thread and write the reference HDF5 output."""

    progress = Signal(int, int, str)
    timeframe_pose = Signal(int, object, object, object, object)
    result_ready = Signal(object, object)
    finished_path = Signal(str)
    failed = Signal(str)

    def __init__(self, request: SimulationRequest):
        super().__init__()
        self.request = request

    def run(self) -> None:
        try:
            plan = build_simulation_plan(self.request.scene)
            if self.request.scene.radiomap.enabled:
                grid_points = int(plan.timeframes[0].metadata["radiomap_total_grid_points"])
                skipped = sum(
                    bool(frame.metadata.get("radiomap_inside_building", False))
                    for frame in plan.timeframes[:grid_points]
                )
                self.progress.emit(
                    0, len(plan.timeframes),
                    f"Building check at {self.request.scene.radiomap.height:g} m: "
                    f"{skipped} of {grid_points} grid points skipped",
                )

            def emit_progress(i, n, msg):
                if 0 <= int(i) < len(plan.timeframes):
                    timeframe = plan.timeframes[int(i)]
                    self.timeframe_pose.emit(
                        int(i),
                        dict(timeframe.device_positions),
                        dict(timeframe.device_orientations),
                        dict(timeframe.object_positions),
                        dict(timeframe.object_orientations),
                    )
                self.progress.emit(i + 1, n, msg)

            simulator = DryRunSimulator() if self.request.dry_run else SionnaSimulator()
            result = simulator.simulate(
                self.request,
                progress=emit_progress,
            )
            self.result_ready.emit(self.request, result)
            output_path = default_output_path(self.request)
            self.progress.emit(1, 1, "Writing and validating HDF5 output")
            ReferenceH5Writer().write(output_path, self.request, result)
            self.finished_path.emit(str(output_path))
        except Exception as exc:  # noqa: BLE001 - surfaced to GUI log
            self.failed.emit(str(exc))

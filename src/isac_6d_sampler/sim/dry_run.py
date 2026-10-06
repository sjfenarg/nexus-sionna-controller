from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from isac_6d_sampler.core.model import SimulationRequest
from isac_6d_sampler.core.validation import validate_request
from isac_6d_sampler.sim.channel import render_channel_samples
from isac_6d_sampler.sim.power import dbm_to_watt, field_amplitude_from_dbm

from .planner import build_simulation_plan, position_for
from .results import LinkResult, SimulationResult, TimeframeResult
from .skipped import skipped_radiomap_timeframe


class DryRunSimulator:
    """Fast deterministic simulator used for GUI/schema validation."""

    def simulate(self, request: SimulationRequest, progress=None) -> SimulationResult:
        scene = request.scene
        scene.ensure_defaults()
        validate_request(request)
        subbands = [band.vector() for band in request.bands]
        f_vector = np.concatenate(subbands)
        plan = build_simulation_plan(scene)
        timeframes: list[TimeframeResult] = []
        tx_amplitude = field_amplitude_from_dbm(request.sionna.tx_power_dbm)

        for tf_idx, timeframe in enumerate(plan.timeframes):
            if progress:
                progress(tf_idx, len(plan.timeframes), f"Generating dry-run timeframe {tf_idx}")
            if timeframe.metadata.get("radiomap_inside_building", False):
                timeframes.append(skipped_radiomap_timeframe(timeframe, len(f_vector), request.channel_mode))
                continue
            links: list[LinkResult] = []
            for link in timeframe.links:
                distance = np.linalg.norm(
                    np.asarray(position_for(link.rx, timeframe))
                    - np.asarray(position_for(link.tx, timeframe))
                )
                rx_panel = link.rx_panel
                tx_panel = link.tx_panel
                rx_ant = rx_panel.element_count
                tx_ant = tx_panel.element_count
                delays = np.full((rx_ant, tx_ant, 1), distance / 299_792_458.0, dtype=np.float64)
                coeffs = np.full(
                    (rx_ant, tx_ant, 1),
                    tx_amplitude / max(distance, 1.0),
                    dtype=np.complex64,
                )
                h = render_channel_samples(delays, coeffs, f_vector, request.channel_mode).reshape(
                    rx_panel.rows,
                    rx_panel.cols,
                    tx_panel.rows,
                    tx_panel.cols,
                    1,
                    -1,
                )
                path_delays = delays.reshape(
                    rx_panel.rows,
                    rx_panel.cols,
                    tx_panel.rows,
                    tx_panel.cols,
                    1,
                    -1,
                )
                links.append(
                    LinkResult(
                        rx_index=link.rx_index,
                        tx_index=link.tx_index,
                        rx_id=link.rx.id,
                        tx_id=link.tx.id,
                        h=h.astype(np.complex64),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        metadata={
                            "path_delays_s": path_delays.astype(np.float64),
                            "path_coefficients": coeffs.reshape(
                                rx_panel.rows,
                                rx_panel.cols,
                                tx_panel.rows,
                                tx_panel.cols,
                                1,
                                -1,
                            ).astype(np.complex64),
                        },
                    )
                )
            timeframes.append(
                TimeframeResult(
                    name=timeframe.name,
                    links=links,
                    metadata={
                        **timeframe.metadata,
                        "device_positions": timeframe.device_positions,
                        "object_positions": timeframe.object_positions,
                        "device_orientations": timeframe.device_orientations,
                        "object_orientations": timeframe.object_orientations,
                    },
                )
            )

        return SimulationResult(
            scenario_name=scene.name,
            sample_id=request.sample_id,
            frequency_vector_hz=f_vector[None, :],
            subband_vectors_hz=[v[None, :] for v in subbands],
            timeframes=timeframes,
            metadata={
                "backend": "dry_run",
                "channel_mode": request.channel_mode.value,
                "tx_power_dbm": float(request.sionna.tx_power_dbm),
                "tx_power_w": dbm_to_watt(request.sionna.tx_power_dbm),
            },
        )

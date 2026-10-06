"""Placeholder results for radiomap positions excluded before simulation."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from isac_6d_sampler.core.model import ChannelMode

from .planner import TimeframePlan
from .results import LinkResult, TimeframeResult


def skipped_radiomap_timeframe(
    timeframe: TimeframePlan,
    frequency_count: int,
    channel_mode: ChannelMode,
) -> TimeframeResult:
    sample_count = 1 if channel_mode == ChannelMode.CIR_PATHS else frequency_count
    links = []
    for link in timeframe.links:
        shape = (
            link.rx_panel.rows, link.rx_panel.cols,
            link.tx_panel.rows, link.tx_panel.cols, 1,
        )
        links.append(LinkResult(
            rx_index=link.rx_index,
            tx_index=link.tx_index,
            rx_id=link.rx.id,
            tx_id=link.tx.id,
            h=np.full((*shape, sample_count), np.nan + 1j * np.nan, dtype=np.complex64),
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "path_delays_s": np.full((*shape, 1), np.nan, dtype=np.float64),
                "path_coefficients": np.full((*shape, 1), np.nan + 1j * np.nan, dtype=np.complex64),
            },
        ))
    metadata = {
        **timeframe.metadata,
        "device_positions": timeframe.device_positions,
        "object_positions": timeframe.object_positions,
        "device_orientations": timeframe.device_orientations,
        "object_orientations": timeframe.object_orientations,
    }
    if timeframe.object_velocities:
        metadata["object_velocities"] = timeframe.object_velocities
    return TimeframeResult(name=timeframe.name, links=links, metadata=metadata)

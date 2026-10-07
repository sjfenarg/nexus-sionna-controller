"""Placeholder results for radiomap positions excluded before simulation."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache

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
            h=_shared_nan_channel((*shape, sample_count)),
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=(
                {}
                if channel_mode == ChannelMode.FREQUENCY_DOMAIN
                else {
                    "path_delays_s": _shared_nan_delays((*shape, 1)),
                    "path_coefficients": _shared_nan_channel((*shape, 1)),
                }
            ),
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


@lru_cache(maxsize=32)
def _shared_nan_channel(shape: tuple[int, ...]) -> np.ndarray:
    values = np.full(shape, np.nan + 1j * np.nan, dtype=np.complex64)
    values.setflags(write=False)
    return values


@lru_cache(maxsize=32)
def _shared_nan_delays(shape: tuple[int, ...]) -> np.ndarray:
    values = np.full(shape, np.nan, dtype=np.float64)
    values.setflags(write=False)
    return values

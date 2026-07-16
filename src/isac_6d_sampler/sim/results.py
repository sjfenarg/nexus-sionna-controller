from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class LinkResult:
    rx_index: int
    tx_index: int
    rx_id: str
    tx_id: str
    h: np.ndarray
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TimeframeResult:
    name: str
    links: list[LinkResult]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SimulationResult:
    scenario_name: str
    sample_id: str
    frequency_vector_hz: np.ndarray
    subband_vectors_hz: list[np.ndarray]
    timeframes: list[TimeframeResult]
    metadata: dict[str, Any] = field(default_factory=dict)

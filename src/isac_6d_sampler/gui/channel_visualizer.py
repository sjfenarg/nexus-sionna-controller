from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QWidget,
)

from isac_6d_sampler.core.model import ChannelMode, SPEED_OF_LIGHT_M_PER_S, SimulationRequest
from isac_6d_sampler.core.config_io import request_from_dict
from isac_6d_sampler.sim.channel import paths_to_frequency_response
from isac_6d_sampler.sim.results import LinkResult, SimulationResult, TimeframeResult

_POWER_FLOOR = 1e-30


@dataclass(frozen=True, slots=True)
class ChannelSelection:
    timeframe_index: int
    link_index: int


class ChannelVisualizerWindow(QMainWindow):
    """Single-plot debug viewer for computed channel datasets."""

    def __init__(self, request: SimulationRequest | None = None, result: SimulationResult | None = None):
        super().__init__()
        self.setWindowTitle("Realtime Channel Visualizer")
        self.setObjectName("channel_visualizer")
        self._request: SimulationRequest | None = None
        self._result: SimulationResult | None = None
        self._updating_controls = False

        self.status = QLabel("Waiting for simulation result")
        self.status.setObjectName("channel_visualizer_status")
        self.domain = QComboBox()
        self.domain.setObjectName("channel_visualizer_domain")
        self.domain.addItem("Frequency domain", "frequency")
        self.domain.addItem("Delay domain", "delay")
        self.x_axis = QComboBox()
        self.x_axis.setObjectName("channel_visualizer_x_axis")
        self.x_axis.addItem("Delay", "delay")
        self.x_axis.addItem("Distance", "distance")
        self.link_kind = QComboBox()
        self.link_kind.setObjectName("channel_visualizer_link_kind")
        self.link_kind.addItem("All channels", "all")
        self.link_kind.addItem("Monostatic only", "mono")
        self.link_kind.addItem("Bistatic only", "bi")
        self.timeframe = QComboBox()
        self.timeframe.setObjectName("channel_visualizer_timeframe")
        self.rx = QComboBox()
        self.rx.setObjectName("channel_visualizer_rx")
        self.tx = QComboBox()
        self.tx.setObjectName("channel_visualizer_tx")
        self.channel = QComboBox()
        self.channel.setObjectName("channel_visualizer_channel")

        self.plot = pg.PlotWidget()
        self.plot.setObjectName("channel_visualizer_plot")
        self.plot.setBackground("w")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("left", "Power", units="dB")
        self.curve = self.plot.plot([], [], pen=pg.mkPen("#1f77b4", width=2), symbol=None)
        self.los_marker = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen("#d62728", width=2, style=Qt.PenStyle.DashLine),
            label="LOS",
            labelOpts={"position": 0.92, "color": "#d62728"},
        )
        self.los_marker.setVisible(False)
        self.plot.addItem(self.los_marker)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.addRow("View", self.domain)
        form.addRow("X axis", self.x_axis)
        form.addRow("Link type", self.link_kind)
        form.addRow("Timeframe", self.timeframe)
        form.addRow("RX", self.rx)
        form.addRow("TX", self.tx)
        form.addRow("Channel", self.channel)

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.addWidget(form_widget, 0)
        right = QWidget()
        right_layout = QFormLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addRow(self.status)
        right_layout.addRow(self.plot)
        layout.addWidget(right, 1)
        self.setCentralWidget(root)
        self.resize(1100, 650)

        self.domain.currentIndexChanged.connect(self._update_plot)
        self.x_axis.currentIndexChanged.connect(self._update_plot)
        self.link_kind.currentIndexChanged.connect(self._rebuild_link_controls)
        self.timeframe.currentIndexChanged.connect(self._rebuild_link_controls)
        self.rx.currentIndexChanged.connect(self._rebuild_link_controls)
        self.tx.currentIndexChanged.connect(self._rebuild_link_controls)
        self.channel.currentIndexChanged.connect(self._update_plot)

        if request is not None and result is not None:
            self.set_result(request, result)

    def set_result(self, request: SimulationRequest, result: SimulationResult) -> None:
        self._request = request
        self._result = result
        self._updating_controls = True
        try:
            self.domain.setCurrentIndex(1)
            self.x_axis.setCurrentIndex(0)
            self.timeframe.clear()
            for index, timeframe in enumerate(result.timeframes):
                self.timeframe.addItem(timeframe.name, index)
            self._populate_entity_filters()
        finally:
            self._updating_controls = False
        self._rebuild_link_controls()

    def set_h5_path(self, path: str | Path) -> None:
        request, result = load_result_from_reference_h5(path)
        self.set_result(request, result)

    def _populate_entity_filters(self) -> None:
        self.rx.clear()
        self.tx.clear()
        self.rx.addItem("Any RX", "all")
        self.tx.addItem("Any TX", "all")
        for rx_id in _ordered_unique(link.rx_id for link in self._all_links()):
            self.rx.addItem(rx_id, rx_id)
        for tx_id in _ordered_unique(link.tx_id for link in self._all_links()):
            self.tx.addItem(tx_id, tx_id)

    def _all_links(self) -> list[LinkResult]:
        if self._result is None:
            return []
        links: list[LinkResult] = []
        for timeframe in self._result.timeframes:
            links.extend(timeframe.links)
        return links

    def _rebuild_link_controls(self, *_):
        if self._updating_controls or self._result is None:
            return
        previous = self.channel.currentData()
        self._updating_controls = True
        try:
            self.channel.clear()
            timeframe = self._selected_timeframe()
            if timeframe is not None:
                for link_index, link in enumerate(timeframe.links):
                    if not self._link_matches_filters(link):
                        continue
                    self.channel.addItem(_link_label(link), link_index)
            if previous is not None:
                index = self.channel.findData(previous)
                if index >= 0:
                    self.channel.setCurrentIndex(index)
        finally:
            self._updating_controls = False
        self._update_plot()

    def _selected_timeframe(self) -> TimeframeResult | None:
        if self._result is None:
            return None
        index = self.timeframe.currentData()
        if index is None or not (0 <= int(index) < len(self._result.timeframes)):
            return None
        return self._result.timeframes[int(index)]

    def _selected_link(self) -> LinkResult | None:
        timeframe = self._selected_timeframe()
        link_index = self.channel.currentData()
        if timeframe is None or link_index is None:
            return None
        if not (0 <= int(link_index) < len(timeframe.links)):
            return None
        return timeframe.links[int(link_index)]

    def _link_matches_filters(self, link: LinkResult) -> bool:
        kind = self.link_kind.currentData()
        if kind == "mono" and link.rx_index != link.tx_index:
            return False
        if kind == "bi" and link.rx_index == link.tx_index:
            return False
        rx_filter = self.rx.currentData()
        tx_filter = self.tx.currentData()
        if rx_filter not in (None, "all") and link.rx_id != rx_filter:
            return False
        if tx_filter not in (None, "all") and link.tx_id != tx_filter:
            return False
        return True

    def _update_plot(self, *_):
        if self._updating_controls:
            return
        link = self._selected_link()
        if self._request is None or self._result is None or link is None:
            self.curve.setData([], [])
            self.status.setText("No channel selected")
            return
        domain = str(self.domain.currentData())
        x_mode = str(self.x_axis.currentData())
        x, y_db, x_label, x_units, note = channel_power_trace(
            self._request,
            self._result,
            link,
            domain,
            x_mode,
        )
        self.curve.setData(np.asarray(x, dtype=np.float64), np.asarray(y_db, dtype=np.float64))
        self.plot.setLabel("bottom", x_label, units=x_units)
        self._update_los_marker(link, domain, x_mode)
        self.plot.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self.status.setText(
            f"backend={self._result.metadata.get('backend', 'unknown')} | "
            f"{_link_label(link)} | {domain}/{x_mode} | {link.h.shape} | {note}"
        )

    def _update_los_marker(self, link: LinkResult, domain: str, x_mode: str) -> None:
        self.los_marker.setVisible(False)
        if domain != "delay" or link.rx_id == link.tx_id:
            return
        timeframe = self._selected_timeframe()
        if timeframe is None:
            return
        expected = expected_los_x_value(timeframe, link, x_mode)
        if expected is None:
            return
        self.los_marker.setValue(expected)
        self.los_marker.setVisible(True)


def load_result_from_reference_h5(path: str | Path) -> tuple[SimulationRequest, SimulationResult]:
    """Load the stored channel datasets used by the debug visualizer."""
    path = Path(path)
    with h5py.File(path, "r") as h5:
        scenario_name = next(iter(h5["scenarios"].keys()))
        scenario = h5["scenarios"][scenario_name]
        sample_names = [name for name in scenario.keys() if name != "metadata"]
        if not sample_names:
            raise ValueError(f"No samples found in {path}")
        sample_id = sample_names[0]
        sample = scenario[sample_id]
        request = _request_from_sample(sample)
        frequency_vector = np.asarray(sample["parameters/f_vector"], dtype=np.float64)
        subbands = [
            np.asarray(group[name], dtype=np.float64)
            for group in [sample["parameters/subband_f_vectors"]]
            for name in sorted(group.keys())
        ]
        timeframes: list[TimeframeResult] = []
        for tf_name in sorted(sample["timeframes"].keys()):
            tf_group = sample["timeframes"][tf_name]
            links: list[LinkResult] = []
            for name in sorted(tf_group["h"].keys()):
                dataset = tf_group["h"][name]
                metadata = {}
                if "tau" in tf_group and name in tf_group["tau"]:
                    metadata["path_delays_s"] = np.asarray(tf_group["tau"][name], dtype=np.float64)
                if "a" in tf_group and name in tf_group["a"]:
                    metadata["path_coefficients"] = np.asarray(tf_group["a"][name], dtype=np.complex64)
                links.append(
                    LinkResult(
                        rx_index=int(dataset.attrs["rx_index"]),
                        tx_index=int(dataset.attrs["tx_index"]),
                        rx_id=str(dataset.attrs["rx_id"]),
                        tx_id=str(dataset.attrs["tx_id"]),
                        h=np.asarray(dataset, dtype=np.complex64),
                        metadata=metadata,
                    )
                )
            metadata = dict(tf_group.attrs)
            metadata["device_positions"] = _read_position_group(tf_group, "devices")
            metadata["object_positions"] = _read_position_group(tf_group, "objects")
            timeframes.append(TimeframeResult(name=tf_name, links=links, metadata=metadata))
        metadata = {
            key: value
            for key, value in sample["metadata"].attrs.items()
            if key != "request_json"
        }
        metadata.setdefault("channel_mode", request.channel_mode.value)
        metadata["source_h5"] = str(path)
        return request, SimulationResult(
            scenario_name=scenario_name,
            sample_id=sample_id,
            frequency_vector_hz=frequency_vector,
            subband_vectors_hz=subbands,
            timeframes=timeframes,
            metadata=metadata,
        )


def _request_from_sample(sample) -> SimulationRequest:
    request_json = sample["metadata"].attrs.get("request_json") if "metadata" in sample else None
    if request_json:
        request = request_from_dict(json.loads(str(request_json)))
    else:
        request = SimulationRequest()
    mode = sample["parameters"].attrs.get("channel_mode", request.channel_mode.value)
    request.channel_mode = ChannelMode(str(mode))
    return request


def channel_power_trace(
    request: SimulationRequest,
    result: SimulationResult,
    link: LinkResult,
    domain: str,
    x_axis: str = "delay",
) -> tuple[np.ndarray, np.ndarray, str, str, str]:
    """Return x-axis and dB power for one stored channel dataset."""
    mode = request.channel_mode
    domain = "frequency" if domain == "frequency" else "delay"
    h = np.asarray(link.h)
    freqs_hz = np.asarray(result.frequency_vector_hz, dtype=np.float64).reshape(-1)

    if domain == "frequency":
        values, note = _frequency_values(mode, h, link, freqs_hz)
        return _frequency_axis(freqs_hz), _power_db(values, mode, "frequency"), "Frequency", "GHz", note

    values, x_s, note = _delay_values(mode, h, link, freqs_hz)
    x, label, units = _delay_or_distance_axis(x_s, x_axis)
    return x, _power_db(values, mode, "delay"), label, units, note


def expected_los_x_value(timeframe: TimeframeResult, link: LinkResult, x_axis: str = "delay") -> float | None:
    positions = timeframe.metadata.get("device_positions", {})
    if not isinstance(positions, dict):
        return None
    if link.rx_id not in positions or link.tx_id not in positions:
        return None
    rx_position = np.asarray(positions[link.rx_id], dtype=np.float64)
    tx_position = np.asarray(positions[link.tx_id], dtype=np.float64)
    distance_m = float(np.linalg.norm(rx_position - tx_position))
    if x_axis == "distance":
        return distance_m
    return distance_m / SPEED_OF_LIGHT_M_PER_S * 1e9


def _frequency_values(
    mode: ChannelMode,
    h: np.ndarray,
    link: LinkResult,
    freqs_hz: np.ndarray,
) -> tuple[np.ndarray, str]:
    if mode == ChannelMode.FREQUENCY_DOMAIN:
        return h, "stored H(f)"
    if mode == ChannelMode.CIR_PATHS:
        tau = np.asarray(link.metadata.get("path_delays_s", np.zeros_like(h.real)), dtype=np.float64)
        try:
            values = paths_to_frequency_response(tau, h, freqs_hz)
            return values, "H(f) rendered from stored CIR paths"
        except ValueError:
            return h, "stored CIR coefficients; frequency rendering unavailable"
    if mode == ChannelMode.PDP_BINNED:
        return h, "stored linear PDP; phase-free frequency view unavailable"
    return np.fft.fft(h, axis=-1), "FFT of stored delay response"


def _delay_values(
    mode: ChannelMode,
    h: np.ndarray,
    link: LinkResult,
    freqs_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    if mode == ChannelMode.FREQUENCY_DOMAIN:
        if "path_delays_s" in link.metadata and "path_coefficients" in link.metadata:
            values, x_s = _physical_path_pdp(
                np.asarray(link.metadata["path_delays_s"], dtype=np.float64),
                np.asarray(link.metadata["path_coefficients"], dtype=np.complex64),
                freqs_hz,
                h.shape[-1],
            )
            return values, x_s, "physical CIR path PDP from stored tau/a"
        values = np.fft.ifft(h, axis=-1)
        return values, _delay_axis(freqs_hz, h.shape[-1]), "circular IFFT of stored H(f)"
    if mode == ChannelMode.CIR_PATHS:
        tau = np.asarray(link.metadata.get("path_delays_s", np.zeros_like(h.real)), dtype=np.float64)
        values, x_s = _physical_path_pdp(tau, h, freqs_hz, h.shape[-1])
        return values, x_s, "physical CIR path PDP from stored tau/a"
    return h, _delay_axis(freqs_hz, h.shape[-1]), "stored delay-domain samples"


def _power_db(values: np.ndarray, mode: ChannelMode, domain: str) -> np.ndarray:
    values = np.asarray(values)
    if values.size == 0:
        return np.zeros(0, dtype=np.float64)
    if mode == ChannelMode.PDP_BINNED and domain == "delay":
        power = np.maximum(np.asarray(values.real, dtype=np.float64), 0.0)
    else:
        power = np.abs(values) ** 2
    if power.ndim > 1:
        power = np.nanmean(power.reshape(-1, power.shape[-1]), axis=0)
    return 10.0 * np.log10(np.maximum(np.asarray(power, dtype=np.float64), _POWER_FLOOR))


def _frequency_axis(freqs_hz: np.ndarray) -> np.ndarray:
    return np.asarray(freqs_hz, dtype=np.float64).reshape(-1) / 1e9


def _delay_or_distance_axis(delays_s: np.ndarray, x_axis: str) -> tuple[np.ndarray, str, str]:
    delays = np.asarray(delays_s, dtype=np.float64)
    if x_axis == "distance":
        return delays * SPEED_OF_LIGHT_M_PER_S, "Path length", "m"
    return delays * 1e9, "Delay", "ns"


def _delay_axis(freqs_hz: np.ndarray, sample_count: int) -> np.ndarray:
    if sample_count <= 0:
        return np.zeros(0, dtype=np.float64)
    return np.arange(sample_count, dtype=np.float64) * _delay_bin_width(freqs_hz, sample_count)


def _mean_sample_axis(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.where(values >= 0.0, values, np.nan)
    if values.ndim <= 1:
        return values.reshape(-1)
    rows = values.reshape(-1, values.shape[-1])
    valid = np.isfinite(rows)
    counts = np.sum(valid, axis=0)
    sums = np.nansum(rows, axis=0)
    return np.divide(sums, counts, out=np.full(rows.shape[-1], np.nan, dtype=np.float64), where=counts > 0)


def _physical_path_pdp(
    delays_s: np.ndarray,
    coefficients: np.ndarray,
    freqs_hz: np.ndarray,
    fallback_sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    delays = np.asarray(delays_s, dtype=np.float64)
    coeffs = np.asarray(coefficients, dtype=np.complex128)
    if delays.shape != coeffs.shape:
        coeffs = np.reshape(coeffs, delays.shape)
    valid = (
        np.isfinite(delays)
        & (delays >= 0.0)
        & np.isfinite(coeffs.real)
        & np.isfinite(coeffs.imag)
    )
    if not np.any(valid):
        return np.zeros(0, dtype=np.complex64), np.zeros(0, dtype=np.float64)

    bin_width_s = _delay_bin_width(freqs_hz, fallback_sample_count)
    bins = np.floor(delays[valid] / bin_width_s).astype(np.int64)
    bin_count = int(np.max(bins)) + 1
    power = np.zeros(bin_count, dtype=np.float64)
    np.add.at(power, bins, np.abs(coeffs[valid]) ** 2)
    antenna_pair_count = max(1, int(np.prod(delays.shape[:-1])))
    power /= antenna_pair_count
    return np.sqrt(power).astype(np.complex64), np.arange(bin_count, dtype=np.float64) * bin_width_s


def _delay_bin_width(freqs_hz: np.ndarray, sample_count: int) -> float:
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64).reshape(-1)
    if freqs_hz.size > 1:
        spacing_hz = float(np.median(np.diff(freqs_hz)))
    else:
        spacing_hz = 1.0
    return 1.0 / max(max(1, int(sample_count)) * spacing_hz, 1.0)


def _stored_domain(mode: ChannelMode) -> str:
    return "frequency" if mode == ChannelMode.FREQUENCY_DOMAIN else "delay"


def _read_position_group(tf_group, group_name: str) -> dict[str, tuple[float, float, float]]:
    path = f"positions/{group_name}"
    if path not in tf_group:
        return {}
    group = tf_group[path]
    return {
        name: tuple(float(value) for value in np.asarray(dataset, dtype=np.float64).reshape(3))
        for name, dataset in group.items()
    }


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _link_label(link: LinkResult) -> str:
    kind = "monostatic" if link.rx_index == link.tx_index else "bistatic"
    return f"rx{link.rx_index}_tx{link.tx_index} ({link.rx_id} <- {link.tx_id}, {kind})"

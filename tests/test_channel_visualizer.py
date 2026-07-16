import numpy as np
import pytest

from isac_6d_sampler.core.model import ChannelMode, FrequencyBand, SPEED_OF_LIGHT_M_PER_S, SimulationRequest
from isac_6d_sampler.gui.channel_visualizer import (
    channel_power_trace,
    expected_los_x_value,
    load_result_from_reference_h5,
)
from isac_6d_sampler.io.reference_h5 import ReferenceH5Writer
from isac_6d_sampler.sim.results import LinkResult, SimulationResult, TimeframeResult
from isac_6d_sampler.sim.dry_run import DryRunSimulator


def test_channel_visualizer_plots_frequency_power_from_stored_channel():
    request = SimulationRequest(dry_run=True)
    request.scene.ensure_defaults()
    request.bands = [FrequencyBand(start_hz=77e9, stop_hz=78e9, points=4)]
    result = DryRunSimulator().simulate(request)
    link = result.timeframes[0].links[0]

    x, y_db, x_label, x_units, note = channel_power_trace(request, result, link, "frequency")

    np.testing.assert_allclose(x, request.bands[0].vector() / 1e9)
    assert y_db.shape == x.shape
    assert np.all(np.isfinite(y_db))
    assert x_label == "Frequency"
    assert x_units == "GHz"
    assert note == "stored H(f)"


def test_channel_visualizer_converts_frequency_channel_to_delay_power():
    request = SimulationRequest(dry_run=True)
    request.scene.ensure_defaults()
    request.bands = [FrequencyBand(start_hz=77e9, stop_hz=78e9, points=4)]
    result = DryRunSimulator().simulate(request)
    link = next(link for link in result.timeframes[0].links if link.rx_id == "ue0" and link.tx_id == "bs0")

    x, y_db, x_label, x_units, note = channel_power_trace(request, result, link, "delay")

    los_delay_ns = float(np.asarray(link.metadata["path_delays_s"]).reshape(-1)[0] * 1e9)
    assert x[np.argmax(y_db)] == pytest.approx(los_delay_ns, abs=0.75)
    assert y_db.shape == x.shape
    assert np.all(np.isfinite(y_db))
    assert x_label == "Delay"
    assert x_units == "ns"
    assert note == "physical CIR path PDP from stored tau/a"


def test_channel_visualizer_can_use_distance_axis_for_delay_power():
    request = SimulationRequest(dry_run=True)
    request.scene.ensure_defaults()
    request.bands = [FrequencyBand(start_hz=77e9, stop_hz=78e9, points=4)]
    result = DryRunSimulator().simulate(request)
    link = next(link for link in result.timeframes[0].links if link.rx_id == "ue0" and link.tx_id == "bs0")

    x, y_db, x_label, x_units, note = channel_power_trace(request, result, link, "delay", "distance")

    los_distance = float(np.asarray(link.metadata["path_delays_s"]).reshape(-1)[0] * SPEED_OF_LIGHT_M_PER_S)
    assert x[np.argmax(y_db)] == pytest.approx(los_distance, abs=0.25)
    assert y_db.shape == x.shape
    assert x_label == "Path length"
    assert x_units == "m"
    assert note == "physical CIR path PDP from stored tau/a"


def test_channel_visualizer_expected_los_uses_timeframe_device_positions():
    request = SimulationRequest(dry_run=True)
    request.scene.ensure_defaults()
    result = DryRunSimulator().simulate(request)
    timeframe = result.timeframes[0]
    link = next(link for link in timeframe.links if link.rx_id == "ue0" and link.tx_id == "bs0")

    distance = expected_los_x_value(timeframe, link, "distance")
    delay_ns = expected_los_x_value(timeframe, link, "delay")

    expected_distance = np.linalg.norm(
        np.asarray(timeframe.metadata["device_positions"]["ue0"])
        - np.asarray(timeframe.metadata["device_positions"]["bs0"])
    )
    assert distance == pytest.approx(expected_distance)
    assert delay_ns == pytest.approx(expected_distance / SPEED_OF_LIGHT_M_PER_S * 1e9)


def test_channel_visualizer_uses_linear_power_for_pdp_binned_delay_values():
    request = SimulationRequest(dry_run=True)
    request.scene.ensure_defaults()
    request.channel_mode = ChannelMode.PDP_BINNED
    request.bands = [FrequencyBand(start_hz=77e9, stop_hz=78e9, points=4)]
    result = DryRunSimulator().simulate(request)
    link = result.timeframes[0].links[0]

    _, y_db, _, _, note = channel_power_trace(request, result, link, "delay")

    expected_power = np.nanmean(np.asarray(link.h.real).reshape(-1, link.h.shape[-1]), axis=0)
    np.testing.assert_allclose(y_db, 10.0 * np.log10(np.maximum(expected_power, 1e-30)))
    assert note == "stored delay-domain samples"


def test_channel_visualizer_filters_negative_sionna_path_delay_sentinels():
    request = SimulationRequest(dry_run=True)
    request.scene.ensure_defaults()
    request.channel_mode = ChannelMode.CIR_PATHS
    result = SimulationResult(
        scenario_name="s",
        sample_id="s000",
        frequency_vector_hz=np.asarray([[77e9, 78e9]]),
        subband_vectors_hz=[],
        timeframes=[
            TimeframeResult(
                name="tf000",
                links=[
                    LinkResult(
                        rx_index=0,
                        tx_index=1,
                        rx_id="ue0",
                        tx_id="bs0",
                        h=np.ones((1, 1, 1, 1, 1, 3), dtype=np.complex64),
                        metadata={"path_delays_s": np.asarray([[[[[[-1.0, 2e-9, 4e-9]]]]]])},
                    )
                ],
            )
        ],
    )

    x, _, _, _, note = channel_power_trace(request, result, result.timeframes[0].links[0], "delay")

    assert np.min(x[np.isfinite(x)]) >= 0.0
    assert np.any(np.isclose(x, 2.0, atol=0.5))
    assert np.any(np.isclose(x, 4.0, atol=0.5))
    assert note == "physical CIR path PDP from stored tau/a"


def test_channel_visualizer_window_filters_computed_links(monkeypatch):
    pytest.importorskip("PySide6", reason="PySide6 is required for GUI tests")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication
    from isac_6d_sampler.gui.channel_visualizer import ChannelVisualizerWindow

    app = QApplication.instance() or QApplication([])
    request = SimulationRequest(dry_run=True)
    request.scene.ensure_defaults()
    result = DryRunSimulator().simulate(request)

    window = ChannelVisualizerWindow(request, result)
    try:
        assert window.domain.currentData() == "delay"
        assert window.x_axis.currentData() == "delay"

        window.link_kind.setCurrentIndex(window.link_kind.findData("mono"))
        mono_labels = [window.channel.itemText(index) for index in range(window.channel.count())]
        assert mono_labels
        assert all("monostatic" in label for label in mono_labels)

        window.link_kind.setCurrentIndex(window.link_kind.findData("bi"))
        bi_labels = [window.channel.itemText(index) for index in range(window.channel.count())]
        assert bi_labels
        assert all("bistatic" in label for label in bi_labels)
        assert window.los_marker.isVisible()

        window.rx.setCurrentIndex(window.rx.findData("ue0"))
        filtered_labels = [window.channel.itemText(index) for index in range(window.channel.count())]
        assert filtered_labels
        assert all("ue0 <-" in label for label in filtered_labels)
    finally:
        window.close()
        app.processEvents()


def test_channel_visualizer_loads_exact_stored_h5_datasets(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    request.bands = [FrequencyBand(start_hz=77e9, stop_hz=78e9, points=4)]
    result = DryRunSimulator().simulate(request)
    path = tmp_path / "scene.h5"
    ReferenceH5Writer().write(path, request, result)

    loaded_request, loaded_result = load_result_from_reference_h5(path)
    link = loaded_result.timeframes[0].links[1]
    x, y_db, _, _, note = channel_power_trace(loaded_request, loaded_result, link, "frequency")

    assert loaded_request.channel_mode == ChannelMode.FREQUENCY_DOMAIN
    assert loaded_result.metadata["backend"] == "dry_run"
    assert loaded_result.timeframes[0].links[1].rx_id == result.timeframes[0].links[1].rx_id
    assert loaded_result.timeframes[0].links[1].tx_id == result.timeframes[0].links[1].tx_id
    assert "path_coefficients" in loaded_result.timeframes[0].links[1].metadata
    np.testing.assert_allclose(x, request.bands[0].vector() / 1e9)
    assert y_db.shape == (4,)
    assert np.any(np.isfinite(y_db))
    assert note == "stored H(f)"

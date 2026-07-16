import numpy as np

from isac_6d_sampler.core.model import ChannelMode
from isac_6d_sampler.sim.channel import (
    IFFT_GRIDDED_DELAY_OVERSAMPLING,
    coherent_delay_bin_aggregate,
    paths_to_frequency_response,
)
from isac_6d_sampler.sim.channel import render_channel_samples


def test_paths_to_frequency_response_single_path():
    h = paths_to_frequency_response(np.array([0.0]), np.array([2.0 + 0j]), np.array([1.0, 2.0]))
    np.testing.assert_allclose(h, np.array([2.0 + 0j, 2.0 + 0j], dtype=np.complex64))


def test_paths_to_frequency_response_empty_paths_returns_zero_channel():
    h = paths_to_frequency_response(
        np.empty((1, 1, 0)),
        np.empty((1, 1, 0), dtype=np.complex64),
        np.array([1.0, 2.0]),
    )
    assert h.shape == (1, 1, 2)
    np.testing.assert_allclose(h, 0.0)


def test_coherent_delay_bin_aggregate_sums_same_bin():
    out = coherent_delay_bin_aggregate(
        delays_s=np.array([0.01, 0.011, 0.20]),
        coefficients=np.array([1 + 1j, 2 + 0j, 3 + 0j]),
        bandwidth_hz=100.0,
        num_bins=4,
    )
    np.testing.assert_allclose(out[1], 3 + 1j)


def test_render_channel_samples_coherent_per_bin_preserves_phase_sum():
    rendered = render_channel_samples(
        delays_s=np.array([[0.005, 0.0051, 0.01]]),
        coefficients=np.array([[1 + 0j, -1 + 0j, 2 + 0j]]),
        frequencies_hz=np.array([77e9, 77e9 + 100.0, 77e9 + 200.0]),
        mode=ChannelMode.COHERENT_PER_BIN,
    )

    assert rendered.shape == (1, 3)
    np.testing.assert_allclose(rendered[0, 1], 0.0 + 0.0j)
    np.testing.assert_allclose(rendered[0, 2], 2.0 + 0.0j)


def test_render_channel_samples_pdp_binned_accumulates_linear_power():
    rendered = render_channel_samples(
        delays_s=np.array([[0.005, 0.0051, 0.0101]]),
        coefficients=np.array([[3 + 4j, 1 + 0j, 2 + 0j]]),
        frequencies_hz=np.array([77e9, 77e9 + 100.0, 77e9 + 200.0]),
        mode=ChannelMode.PDP_BINNED,
    )

    assert rendered.shape == (1, 3)
    np.testing.assert_allclose(rendered[0, 1], 26.0 + 0.0j)
    np.testing.assert_allclose(rendered[0, 2], 4.0 + 0.0j)


def test_render_channel_samples_ifft_mode_keeps_frequency_count():
    rendered = render_channel_samples(
        delays_s=np.array([0.0]),
        coefficients=np.array([1.0 + 0j]),
        frequencies_hz=np.array([1.0, 2.0, 3.0, 4.0]),
        mode=ChannelMode.PDP_IFFT_EXACT,
    )

    assert rendered.shape == (4,)
    np.testing.assert_allclose(rendered[0], 1.0 + 0.0j)


def test_render_channel_samples_gridded_ifft_uses_oversampled_delay_grid():
    frequencies = np.array([0.0, 1.0, 2.0, 3.0])
    delay = 0.2
    rendered = render_channel_samples(
        delays_s=np.array([delay]),
        coefficients=np.array([1.0 + 0j]),
        frequencies_hz=frequencies,
        mode=ChannelMode.PDP_IFFT_GRIDDED,
    )

    grid_size = frequencies.size * IFFT_GRIDDED_DELAY_OVERSAMPLING
    delay_grid = np.zeros(grid_size, dtype=np.complex128)
    grid_idx = round(delay * (frequencies[-1] - frequencies[0]) * IFFT_GRIDDED_DELAY_OVERSAMPLING)
    delay_grid[grid_idx] = 1.0 + 0j
    expected = np.fft.ifft(np.fft.fft(delay_grid)[: frequencies.size])
    np.testing.assert_allclose(rendered, expected.astype(np.complex64), rtol=1e-6, atol=1e-6)


def test_render_channel_samples_exact_and_gridded_ifft_are_distinct_for_off_grid_delay():
    frequencies = np.array([0.0, 1.0, 2.0, 3.0])
    exact = render_channel_samples(
        delays_s=np.array([0.2]),
        coefficients=np.array([1.0 + 0j]),
        frequencies_hz=frequencies,
        mode=ChannelMode.PDP_IFFT_EXACT,
    )
    gridded = render_channel_samples(
        delays_s=np.array([0.2]),
        coefficients=np.array([1.0 + 0j]),
        frequencies_hz=frequencies,
        mode=ChannelMode.PDP_IFFT_GRIDDED,
    )

    assert not np.allclose(exact, gridded)

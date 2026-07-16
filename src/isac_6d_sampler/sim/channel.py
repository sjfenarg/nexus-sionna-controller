from __future__ import annotations

import numpy as np

from isac_6d_sampler.core.model import ChannelMode

IFFT_GRIDDED_DELAY_OVERSAMPLING = 32


def paths_to_frequency_response(
    delays_s: np.ndarray,
    coefficients: np.ndarray,
    frequencies_hz: np.ndarray,
    chunk_size: int = 512,
    max_phase_elements: int = 8_000_000,
) -> np.ndarray:
    """Render complex paths into ``H(f)`` using vectorized chunked exponentials.

    ``delays_s`` and ``coefficients`` may have any common leading shape ending in
    ``num_paths``. The returned array has leading shape plus ``num_frequencies``.
    """
    delays = np.asarray(delays_s, dtype=np.float64)
    coeffs = np.asarray(coefficients, dtype=np.complex128)
    freqs = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    if delays.shape != coeffs.shape:
        raise ValueError(f"delays and coefficients shape mismatch: {delays.shape} != {coeffs.shape}")
    if delays.ndim == 0:
        delays = delays.reshape(1)
        coeffs = coeffs.reshape(1)

    leading = delays.shape[:-1]
    path_count = delays.shape[-1]
    if path_count == 0:
        return np.zeros((*leading, freqs.size), dtype=np.complex64)
    flat_delays = delays.reshape(-1, path_count)
    flat_coeffs = coeffs.reshape(-1, path_count)
    out = np.zeros((flat_delays.shape[0], freqs.size), dtype=np.complex128)

    chunk_size = max(1, int(chunk_size))
    max_phase_elements = max(1, int(max_phase_elements))
    freq_count = max(1, int(freqs.size))
    for path_start in range(0, path_count, chunk_size):
        path_stop = min(path_start + chunk_size, path_count)
        path_chunk = path_stop - path_start
        row_chunk = max(1, max_phase_elements // max(1, path_chunk * freq_count))
        for row_start in range(0, flat_delays.shape[0], row_chunk):
            row_stop = min(row_start + row_chunk, flat_delays.shape[0])
            phase = (
                -2j
                * np.pi
                * flat_delays[row_start:row_stop, path_start:path_stop, None]
                * freqs[None, None, :]
            )
            out[row_start:row_stop] += np.sum(
                flat_coeffs[row_start:row_stop, path_start:path_stop, None] * np.exp(phase),
                axis=1,
            )
    return out.reshape(*leading, freqs.size).astype(np.complex64)


def render_channel_samples(
    delays_s: np.ndarray,
    coefficients: np.ndarray,
    frequencies_hz: np.ndarray,
    mode: ChannelMode,
) -> np.ndarray:
    """Render path-domain data into the selected channel sample representation."""
    if mode == ChannelMode.FREQUENCY_DOMAIN:
        return paths_to_frequency_response(delays_s, coefficients, frequencies_hz)
    if mode == ChannelMode.COHERENT_PER_BIN:
        return _render_delay_bins(delays_s, coefficients, frequencies_hz, coherent=True)
    if mode == ChannelMode.PDP_BINNED:
        return _render_delay_bins(delays_s, coefficients, frequencies_hz, coherent=False)
    if mode == ChannelMode.PDP_IFFT_EXACT:
        h_f = paths_to_frequency_response(delays_s, coefficients, frequencies_hz)
        return np.fft.ifft(h_f, axis=-1).astype(np.complex64)
    if mode == ChannelMode.PDP_IFFT_GRIDDED:
        return _render_ifft_gridded(delays_s, coefficients, frequencies_hz)
    if mode == ChannelMode.CIR_PATHS:
        return np.asarray(coefficients, dtype=np.complex64)
    raise ValueError(f"Unsupported channel mode: {mode}")


def coherent_delay_bin_aggregate(
    delays_s: np.ndarray,
    coefficients: np.ndarray,
    bandwidth_hz: float,
    num_bins: int,
) -> np.ndarray:
    """Coherently sum path coefficients into delay bins."""
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")
    delays = np.asarray(delays_s, dtype=np.float64).reshape(-1)
    coeffs = np.asarray(coefficients, dtype=np.complex128).reshape(-1)
    n = min(delays.size, coeffs.size)
    delays = delays[:n]
    coeffs = coeffs[:n]
    valid = np.isfinite(delays) & np.isfinite(coeffs.real) & np.isfinite(coeffs.imag) & (delays >= 0)
    bins = np.floor(delays[valid] * bandwidth_hz).astype(np.intp)
    valid_bins = (bins >= 0) & (bins < num_bins)
    out = np.zeros(num_bins, dtype=np.complex128)
    np.add.at(out, bins[valid_bins], coeffs[valid][valid_bins])
    return out.astype(np.complex64)


def _render_delay_bins(
    delays_s: np.ndarray,
    coefficients: np.ndarray,
    frequencies_hz: np.ndarray,
    coherent: bool,
) -> np.ndarray:
    delays = np.asarray(delays_s, dtype=np.float64)
    coeffs = np.asarray(coefficients, dtype=np.complex128)
    if delays.shape != coeffs.shape:
        raise ValueError(f"delays and coefficients shape mismatch: {delays.shape} != {coeffs.shape}")
    freqs = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    bandwidth_hz = float(freqs[-1] - freqs[0]) if freqs.size > 1 else 1.0
    bandwidth_hz = max(bandwidth_hz, 1.0)
    num_bins = freqs.size

    leading = delays.shape[:-1]
    if delays.shape[-1] == 0:
        return np.zeros((*leading, num_bins), dtype=np.complex64)
    flat_delays = delays.reshape(-1, delays.shape[-1])
    flat_coeffs = coeffs.reshape(-1, coeffs.shape[-1])
    out = np.zeros((flat_delays.shape[0], num_bins), dtype=np.complex128)
    for idx, (delay_row, coeff_row) in enumerate(zip(flat_delays, flat_coeffs, strict=True)):
        if coherent:
            out[idx] = coherent_delay_bin_aggregate(delay_row, coeff_row, bandwidth_hz, num_bins)
        else:
            out[idx] = _power_delay_bin_aggregate(delay_row, coeff_row, bandwidth_hz, num_bins)
    return out.reshape(*leading, num_bins).astype(np.complex64)


def _render_ifft_gridded(
    delays_s: np.ndarray,
    coefficients: np.ndarray,
    frequencies_hz: np.ndarray,
    delay_oversampling: int = IFFT_GRIDDED_DELAY_OVERSAMPLING,
) -> np.ndarray:
    """Approximate the exact IFFT renderer through an oversampled delay grid."""
    delays = np.asarray(delays_s, dtype=np.float64)
    coeffs = np.asarray(coefficients, dtype=np.complex128)
    if delays.shape != coeffs.shape:
        raise ValueError(f"delays and coefficients shape mismatch: {delays.shape} != {coeffs.shape}")
    if delays.ndim == 0:
        delays = delays.reshape(1)
        coeffs = coeffs.reshape(1)

    freqs = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    num_bins = freqs.size
    if num_bins == 0:
        return np.zeros((*delays.shape[:-1], 0), dtype=np.complex64)

    bandwidth_hz = float(freqs[-1] - freqs[0]) if num_bins > 1 else 1.0
    bandwidth_hz = max(bandwidth_hz, 1.0)
    oversampling = max(1, int(delay_oversampling))
    grid_size = num_bins * oversampling
    leading = delays.shape[:-1]
    path_count = delays.shape[-1]
    if path_count == 0:
        return np.zeros((*leading, num_bins), dtype=np.complex64)

    flat_delays = delays.reshape(-1, path_count)
    flat_coeffs = coeffs.reshape(-1, path_count)
    out = np.zeros((flat_delays.shape[0], num_bins), dtype=np.complex128)
    for idx, (delay_row, coeff_row) in enumerate(zip(flat_delays, flat_coeffs, strict=True)):
        out[idx] = _ifft_gridded_row(
            delay_row,
            coeff_row,
            first_frequency_hz=float(freqs[0]),
            bandwidth_hz=bandwidth_hz,
            num_bins=num_bins,
            grid_size=grid_size,
            oversampling=oversampling,
        )
    return out.reshape(*leading, num_bins).astype(np.complex64)


def _ifft_gridded_row(
    delays_s: np.ndarray,
    coefficients: np.ndarray,
    first_frequency_hz: float,
    bandwidth_hz: float,
    num_bins: int,
    grid_size: int,
    oversampling: int,
) -> np.ndarray:
    valid = (
        np.isfinite(delays_s)
        & np.isfinite(coefficients.real)
        & np.isfinite(coefficients.imag)
        & (delays_s >= 0)
    )
    if not np.any(valid):
        return np.zeros(num_bins, dtype=np.complex128)

    delays = delays_s[valid]
    coeffs = coefficients[valid]
    grid_idx = np.rint(delays * bandwidth_hz * oversampling).astype(np.intp)
    in_window = (grid_idx >= 0) & (grid_idx < grid_size)
    if not np.any(in_window):
        return np.zeros(num_bins, dtype=np.complex128)

    grid_idx = grid_idx[in_window]
    coeffs = coeffs[in_window]
    quantized_delays = grid_idx.astype(np.float64) / (bandwidth_hz * oversampling)
    carrier_phase = np.exp(-2j * np.pi * first_frequency_hz * quantized_delays)
    delay_grid = np.zeros(grid_size, dtype=np.complex128)
    np.add.at(delay_grid, grid_idx, coeffs * carrier_phase)
    frequency_response = np.fft.fft(delay_grid)[:num_bins]
    return np.fft.ifft(frequency_response)


def _power_delay_bin_aggregate(
    delays_s: np.ndarray,
    coefficients: np.ndarray,
    bandwidth_hz: float,
    num_bins: int,
) -> np.ndarray:
    delays = np.asarray(delays_s, dtype=np.float64).reshape(-1)
    coeffs = np.asarray(coefficients, dtype=np.complex128).reshape(-1)
    n = min(delays.size, coeffs.size)
    delays = delays[:n]
    coeffs = coeffs[:n]
    valid = np.isfinite(delays) & np.isfinite(coeffs.real) & np.isfinite(coeffs.imag) & (delays >= 0)
    bins = np.floor(delays[valid] * bandwidth_hz).astype(np.intp)
    valid_bins = (bins >= 0) & (bins < num_bins)
    out = np.zeros(num_bins, dtype=np.float64)
    np.add.at(out, bins[valid_bins], np.abs(coeffs[valid][valid_bins]) ** 2)
    return out.astype(np.complex64)


def frequency_vector_from_subbands(subbands: list[np.ndarray]) -> np.ndarray:
    if not subbands:
        raise ValueError("At least one frequency subband is required")
    return np.concatenate([np.asarray(v, dtype=np.float64).reshape(-1) for v in subbands])[None, :]

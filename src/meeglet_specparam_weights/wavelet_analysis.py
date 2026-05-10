"""Wavelet analysis layer: wrap meeglet to produce complex wavelet coefficients Z(f,t)."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

import meeglet


@dataclass
class WaveletDecomposition:
    coefficients: np.ndarray  # complex, (n_freqs, n_times) or (n_channels, n_freqs, n_times)
    foi: np.ndarray  # center frequencies, shape (n_freqs,)
    sigma_time: np.ndarray  # temporal std per freq in seconds, shape (n_freqs,)
    sigma_freq: np.ndarray  # spectral std per freq in Hz, shape (n_freqs,)
    times: np.ndarray  # time points in seconds, shape (n_times,)
    sfreq: float
    bw_oct: float
    delta_oct: float
    kernel_width: int = 5
    density: str = "oct"
    n_channels: int = 1


def wavelet_decompose(
    signal: np.ndarray,
    sfreq: float,
    foi_start: float = 2.0,
    foi_end: float = 32.0,
    bw_oct: float = 0.5,
    delta_oct: float | None = None,
    kernel_width: int = 5,
    density: str = "oct",
) -> WaveletDecomposition:
    """Decompose a signal into complex wavelet coefficients Z(f,t).

    Accepts 1D (n_samples,) or 2D (n_channels, n_samples) input.
    For 1D input, coefficients shape is (n_freqs, n_times).
    For 2D input, coefficients shape is (n_channels, n_freqs, n_times).

    Uses meeglet's log-frequency Morlet wavelets. Convolution is performed at
    every sample to produce coefficient matrices with a common time grid across
    all frequencies.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        multichannel = False
        signals = signal[np.newaxis, :]
    elif signal.ndim == 2:
        multichannel = True
        signals = signal
    else:
        raise ValueError(f"signal must be 1D or 2D, got {signal.ndim}D shape {signal.shape}")

    n_channels, n_samples = signals.shape
    times = np.arange(n_samples) / sfreq

    foi, sigma_time, sigma_freq, bw_oct_out, _qt = meeglet.define_frequencies(
        foi_start=foi_start, foi_end=foi_end, delta_oct=delta_oct, bw_oct=bw_oct
    )
    if delta_oct is None:
        delta_oct = bw_oct_out / 4.0

    wavelets = meeglet.define_wavelets(
        foi=foi,
        sigma_time=sigma_time,
        sfreq=sfreq,
        kernel_width=kernel_width,
        window_shift=1.0,
        density=density,
    )

    n_freqs = len(foi)
    all_coefficients = np.empty((n_channels, n_freqs, n_samples), dtype=np.complex128)

    for i_ch in range(n_channels):
        ch_signal = signals[i_ch]
        nan_mask = np.isnan(ch_signal)
        has_nans = np.any(nan_mask)
        ch_clean = ch_signal.copy()
        if has_nans:
            ch_clean[nan_mask] = 0.0

        for i_freq, (kernel, scaling, n_samp_eff, _n_shift) in enumerate(wavelets):
            k = kernel[:, 0]
            conv = fftconvolve(ch_clean, k, mode="same") * scaling
            all_coefficients[i_ch, i_freq] = conv

        if has_nans:
            _propagate_nans(all_coefficients[i_ch], nan_mask, sigma_time, sfreq, kernel_width)

    if not multichannel:
        all_coefficients = all_coefficients[0]

    return WaveletDecomposition(
        coefficients=all_coefficients,
        foi=foi,
        sigma_time=sigma_time,
        sigma_freq=sigma_freq,
        times=times,
        sfreq=sfreq,
        bw_oct=bw_oct_out,
        delta_oct=delta_oct,
        kernel_width=kernel_width,
        density=density,
        n_channels=n_channels if multichannel else 1,
    )


def _propagate_nans(
    coefficients: np.ndarray,
    nan_mask: np.ndarray,
    sigma_time: np.ndarray,
    sfreq: float,
    kernel_width: int,
) -> None:
    """Set coefficients to NaN where input NaNs fall within the wavelet's support."""
    from scipy.ndimage import binary_dilation

    n_samples = coefficients.shape[1]
    for i_freq, st in enumerate(sigma_time):
        half_support = int(np.ceil(kernel_width * st * sfreq / 2.0))
        struct = np.ones(2 * half_support + 1, dtype=bool)
        affected = binary_dilation(nan_mask, structure=struct)[:n_samples]
        coefficients[i_freq, affected] = np.nan + 1j * np.nan

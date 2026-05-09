"""Wavelet analysis layer: wrap meeglet to produce complex wavelet coefficients Z(f,t)."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

import meeglet


@dataclass
class WaveletDecomposition:
    coefficients: np.ndarray  # complex, shape (n_freqs, n_times)
    foi: np.ndarray  # center frequencies, shape (n_freqs,)
    sigma_time: np.ndarray  # temporal std per freq in seconds, shape (n_freqs,)
    sigma_freq: np.ndarray  # spectral std per freq in Hz, shape (n_freqs,)
    times: np.ndarray  # time points in seconds, shape (n_times,)
    sfreq: float
    bw_oct: float
    delta_oct: float
    kernel_width: int = 5
    density: str = "oct"


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
    """Decompose a 1D signal into complex wavelet coefficients Z(f,t).

    Uses meeglet's log-frequency Morlet wavelets. Convolution is performed at
    every sample to produce a (n_freqs, n_samples) coefficient matrix with a
    common time grid across all frequencies.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got shape {signal.shape}")

    n_samples = len(signal)
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

    nan_mask = np.isnan(signal)
    has_nans = np.any(nan_mask)
    signal_clean = signal.copy()
    if has_nans:
        signal_clean[nan_mask] = 0.0

    n_freqs = len(foi)
    coefficients = np.empty((n_freqs, n_samples), dtype=np.complex128)

    for i_freq, (kernel, scaling, n_samp_eff, _n_shift) in enumerate(wavelets):
        k = kernel[:, 0]
        conv = fftconvolve(signal_clean, k, mode="same") * scaling
        coefficients[i_freq] = conv

    if has_nans:
        _propagate_nans(coefficients, nan_mask, sigma_time, sfreq, kernel_width)

    return WaveletDecomposition(
        coefficients=coefficients,
        foi=foi,
        sigma_time=sigma_time,
        sigma_freq=sigma_freq,
        times=times,
        sfreq=sfreq,
        bw_oct=bw_oct_out,
        delta_oct=delta_oct,
        kernel_width=kernel_width,
        density=density,
    )


def _propagate_nans(
    coefficients: np.ndarray,
    nan_mask: np.ndarray,
    sigma_time: np.ndarray,
    sfreq: float,
    kernel_width: int,
) -> None:
    """Set coefficients to NaN where input NaNs fall within the wavelet's support."""
    n_samples = coefficients.shape[1]
    nan_indices = np.where(nan_mask)[0]

    for i_freq, st in enumerate(sigma_time):
        half_support = int(np.ceil(kernel_width * st * sfreq / 2.0))
        for idx in nan_indices:
            lo = max(0, idx - half_support)
            hi = min(n_samples, idx + half_support + 1)
            coefficients[i_freq, lo:hi] = np.nan + 1j * np.nan

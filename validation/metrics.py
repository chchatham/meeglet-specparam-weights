"""Validation metrics and signal generators."""

import numpy as np
from scipy.signal import welch


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two signals."""
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if denom < 1e-30:
        return 0.0
    return float(np.sum(a * b) / denom)


def rms(x: np.ndarray) -> float:
    """Root mean square of a signal."""
    return float(np.sqrt(np.mean(x ** 2)))


def rms_relative_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Relative RMS difference: |rms(a) - rms(b)| / max(rms(a), rms(b))."""
    ra, rb = rms(a), rms(b)
    denom = max(ra, rb)
    if denom < 1e-30:
        return 0.0
    return abs(ra - rb) / denom


def energy_ratio(reconstruction: np.ndarray, original: np.ndarray) -> float:
    """Energy ratio: ||reconstruction||² / ||original||²."""
    e_orig = np.sum(original ** 2)
    if e_orig < 1e-30:
        return 0.0
    return float(np.sum(reconstruction ** 2) / e_orig)


def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Signal-to-noise ratio in dB."""
    p_signal = np.mean(signal ** 2)
    p_noise = np.mean(noise ** 2)
    if p_noise < 1e-30:
        return float('inf')
    return float(10 * np.log10(p_signal / p_noise))


def generate_pink_noise(
    sfreq: float, n_samples: int, exponent: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate unit-variance pink noise with given spectral exponent."""
    white = rng.standard_normal(n_samples)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    freqs[0] = 1.0
    fft *= 1.0 / np.power(freqs, exponent / 2.0)
    pink = np.fft.irfft(fft, n=n_samples)
    return pink / np.std(pink)


def alpha_power_ratio(
    reconstruction: np.ndarray,
    ground_truth: np.ndarray,
    sfreq: float,
    peak_freq: float = 10.0,
    nperseg: int = 512,
    bandwidth: float = 2.0,
) -> float:
    """Ratio of reconstructed to true PSD at the peak frequency.

    Averages PSD over [peak_freq - bandwidth/2, peak_freq + bandwidth/2].
    Returns ratio; 1.0 is perfect.
    """
    f_r, psd_r = welch(reconstruction, fs=sfreq, nperseg=nperseg)
    f_t, psd_t = welch(ground_truth, fs=sfreq, nperseg=nperseg)
    mask = (f_r >= peak_freq - bandwidth / 2) & (f_r <= peak_freq + bandwidth / 2)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(psd_r[mask]) / max(np.mean(psd_t[mask]), 1e-30))


def spectral_shape_error(
    reconstruction: np.ndarray,
    ground_truth: np.ndarray,
    sfreq: float,
    nperseg: int = 512,
    freq_range: tuple[float, float] = (2.0, 50.0),
) -> float:
    """RMS error of log10(PSD) between reconstruction and ground truth.

    Computed over the specified frequency range. Lower is better.
    """
    f_r, psd_r = welch(reconstruction, fs=sfreq, nperseg=nperseg)
    f_t, psd_t = welch(ground_truth, fs=sfreq, nperseg=nperseg)
    mask = (f_r >= freq_range[0]) & (f_r <= freq_range[1])
    psd_r_m = np.maximum(psd_r[mask], 1e-30)
    psd_t_m = np.maximum(psd_t[mask], 1e-30)
    log_diff = np.log10(psd_r_m) - np.log10(psd_t_m)
    return float(np.sqrt(np.mean(log_diff ** 2)))

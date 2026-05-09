"""Validation metrics and signal generators."""

import numpy as np


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

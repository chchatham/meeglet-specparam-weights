"""Shared fixtures for meeglet-specparam-weights tests."""

import numpy as np
import pytest


@pytest.fixture
def sfreq():
    return 256.0


@pytest.fixture
def pink_noise(sfreq):
    """4 seconds of pink noise (1/f, exponent=1)."""
    rng = np.random.default_rng(42)
    n_samples = int(4 * sfreq)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    freqs[0] = 1.0  # avoid division by zero
    amplitudes = 1.0 / np.sqrt(freqs)
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    spectrum = amplitudes * np.exp(1j * phases)
    spectrum[0] = 0.0
    signal = np.fft.irfft(spectrum, n=n_samples)
    return signal


@pytest.fixture
def alpha_signal(sfreq):
    """4 seconds of 10 Hz sine wave."""
    n_samples = int(4 * sfreq)
    t = np.arange(n_samples) / sfreq
    return np.sin(2 * np.pi * 10 * t)


@pytest.fixture
def pink_plus_alpha(pink_noise, alpha_signal):
    """Pink noise + 10 Hz oscillation."""
    alpha_scaled = alpha_signal * 2.0  # make alpha clearly visible
    return pink_noise + alpha_scaled

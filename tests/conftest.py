"""Shared fixtures for meeglet-specparam-weights tests."""

import numpy as np
import pytest


def make_pink_noise(n_samples, sfreq, exponent_half=0.5, seed=42):
    """Generate pink noise with spectral amplitude ~ 1/f^exponent_half.

    Power spectrum scales as 1/f^(2*exponent_half), so exponent_half=0.5
    gives 1/f noise, exponent_half=0.75 gives 1/f^1.5 noise, etc.
    """
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    freqs[0] = 1.0
    amplitudes = 1.0 / freqs ** exponent_half
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    spectrum = amplitudes * np.exp(1j * phases)
    spectrum[0] = 0.0
    return np.fft.irfft(spectrum, n=n_samples)


@pytest.fixture
def sfreq():
    return 256.0


@pytest.fixture
def pink_noise(sfreq):
    """4 seconds of pink noise (1/f, exponent=1)."""
    return make_pink_noise(int(4 * sfreq), sfreq, exponent_half=0.5)


@pytest.fixture
def alpha_signal(sfreq):
    """4 seconds of 10 Hz sine wave."""
    n_samples = int(4 * sfreq)
    t = np.arange(n_samples) / sfreq
    return np.sin(2 * np.pi * 10 * t)


@pytest.fixture
def pink_plus_alpha(pink_noise, alpha_signal):
    """Pink noise + 10 Hz oscillation."""
    alpha_scaled = alpha_signal * 2.0
    return pink_noise + alpha_scaled

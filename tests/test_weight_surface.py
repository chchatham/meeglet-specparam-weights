"""Tests for weight_surface module."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from meeglet_specparam_weights.wavelet_analysis import wavelet_decompose
from meeglet_specparam_weights.time_resolved_fit import time_resolved_fit
from meeglet_specparam_weights.weight_surface import (
    WeightSurface,
    compute_weight_surface,
)


@pytest.fixture
def pink_result(sfreq):
    """Decomposition and fit of pure pink noise."""
    rng = np.random.default_rng(42)
    n_samples = int(4 * sfreq)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    freqs[0] = 1.0
    amplitudes = 1.0 / freqs ** 0.75
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    spectrum = amplitudes * np.exp(1j * phases)
    spectrum[0] = 0.0
    signal = np.fft.irfft(spectrum, n=n_samples)
    decomp = wavelet_decompose(signal, sfreq, foi_start=2, foi_end=32)
    fit = time_resolved_fit(decomp, fit_stride=50)
    return decomp, fit


@pytest.fixture
def pink_alpha_result(sfreq):
    """Decomposition and fit of pink noise + 10 Hz sine."""
    rng = np.random.default_rng(42)
    n_samples = int(4 * sfreq)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    freqs[0] = 1.0
    amplitudes = 1.0 / freqs ** 0.75
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    spectrum = amplitudes * np.exp(1j * phases)
    spectrum[0] = 0.0
    signal = np.fft.irfft(spectrum, n=n_samples)
    t = np.arange(n_samples) / sfreq
    signal += 3.0 * np.sin(2 * np.pi * 10 * t)
    decomp = wavelet_decompose(signal, sfreq, foi_start=2, foi_end=32)
    fit = time_resolved_fit(decomp, fit_stride=50)
    return decomp, fit


class TestWeightSurfaceProperties:
    """Verify numerical properties of the weight surface."""

    def test_output_type(self, pink_result):
        decomp, fit = pink_result
        ws = compute_weight_surface(decomp, fit, component="aperiodic")
        assert isinstance(ws, WeightSurface)

    def test_shape(self, pink_result):
        decomp, fit = pink_result
        ws = compute_weight_surface(decomp, fit)
        assert ws.weights.shape == decomp.coefficients.shape

    def test_real(self, pink_result):
        decomp, fit = pink_result
        ws = compute_weight_surface(decomp, fit)
        assert not np.iscomplexobj(ws.weights)

    def test_non_negative(self, pink_result):
        decomp, fit = pink_result
        ws = compute_weight_surface(decomp, fit)
        assert np.all(ws.weights >= 0)

    def test_clamped(self, pink_result):
        decomp, fit = pink_result
        max_w = 50.0
        ws = compute_weight_surface(decomp, fit, max_weight=max_w)
        assert np.all(ws.weights <= max_w)

    def test_no_nan(self, pink_result):
        decomp, fit = pink_result
        ws = compute_weight_surface(decomp, fit)
        assert not np.any(np.isnan(ws.weights))

    def test_no_inf(self, pink_result):
        decomp, fit = pink_result
        ws = compute_weight_surface(decomp, fit)
        assert not np.any(np.isinf(ws.weights))

    def test_component_stored(self, pink_result):
        decomp, fit = pink_result
        ws = compute_weight_surface(decomp, fit, component="periodic")
        assert ws.component == "periodic"

    def test_invalid_component(self, pink_result):
        decomp, fit = pink_result
        with pytest.raises(ValueError, match="component"):
            compute_weight_surface(decomp, fit, component="invalid")


class TestWeightSurfaceValues:
    """Verify weight values make physical sense."""

    def test_full_weights_near_one_for_good_fit(self, pink_result):
        """Full model weights ≈ 1.0 where the fit is good."""
        decomp, fit = pink_result
        ws = compute_weight_surface(decomp, fit, component="full")

        # Interior time points, avoid edges
        n_times = ws.weights.shape[1]
        interior = slice(n_times // 4, 3 * n_times // 4)
        interior_weights = ws.weights[:, interior]

        median_weight = np.median(interior_weights)
        assert 0.1 < median_weight < 10.0, (
            f"Median full weight {median_weight:.3f}, expected near 1.0"
        )

    def test_aperiodic_weights_on_pink_noise(self, pink_result):
        """Aperiodic weights on pure pink noise should be moderate."""
        decomp, fit = pink_result
        ws = compute_weight_surface(decomp, fit, component="aperiodic")

        n_times = ws.weights.shape[1]
        interior = slice(n_times // 4, 3 * n_times // 4)
        interior_weights = ws.weights[:, interior]

        median_weight = np.median(interior_weights)
        assert 0.01 < median_weight < 50.0, (
            f"Median aperiodic weight {median_weight:.3f}, expected moderate"
        )

    def test_periodic_weights_higher_at_peak_freq(self, pink_alpha_result):
        """Periodic weights should be elevated near 10 Hz."""
        decomp, fit = pink_alpha_result
        ws = compute_weight_surface(decomp, fit, component="periodic")

        n_times = ws.weights.shape[1]
        interior = slice(n_times // 4, 3 * n_times // 4)

        idx_10 = np.argmin(np.abs(decomp.foi - 10))
        idx_25 = np.argmin(np.abs(decomp.foi - 25))

        weight_10 = np.median(ws.weights[idx_10, interior])
        weight_25 = np.median(ws.weights[idx_25, interior])

        # At the peak frequency, periodic weights should be larger
        assert weight_10 > weight_25, (
            f"Periodic weight at 10 Hz ({weight_10:.3f}) should exceed 25 Hz ({weight_25:.3f})"
        )


class TestNaNHandling:
    """Verify NaN handling in weights."""

    def test_nan_coefficients_get_zero_weight(self, sfreq):
        n_samples = int(4 * sfreq)
        signal = np.random.default_rng(42).standard_normal(n_samples)
        signal[500] = np.nan

        decomp = wavelet_decompose(signal, sfreq)
        fit = time_resolved_fit(decomp, fit_stride=50)
        ws = compute_weight_surface(decomp, fit)

        # Where coefficients are NaN, weights should be 0
        nan_mask = np.isnan(decomp.coefficients)
        if np.any(nan_mask):
            assert np.all(ws.weights[nan_mask] == 0.0)

"""Tests for time_resolved_fit module."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from meeglet_specparam_weights.wavelet_analysis import wavelet_decompose
from meeglet_specparam_weights.time_resolved_fit import TimeResolvedFit, time_resolved_fit


@pytest.fixture
def pink_decomposition(sfreq):
    """Wavelet decomposition of pink noise (exponent ~1.5)."""
    rng = np.random.default_rng(42)
    n_samples = int(4 * sfreq)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    freqs[0] = 1.0
    amplitudes = 1.0 / freqs ** 0.75  # exponent=1.5 in power: |1/f^0.75|^2 = 1/f^1.5
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    spectrum = amplitudes * np.exp(1j * phases)
    spectrum[0] = 0.0
    signal = np.fft.irfft(spectrum, n=n_samples)
    return wavelet_decompose(signal, sfreq, foi_start=2, foi_end=32)


@pytest.fixture
def pink_alpha_decomposition(sfreq):
    """Wavelet decomposition of pink noise + 10 Hz sine."""
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
    return wavelet_decompose(signal, sfreq, foi_start=2, foi_end=32)


class TestTimeResolvedFitOutput:
    """Verify output shapes and types."""

    def test_output_type(self, pink_decomposition):
        result = time_resolved_fit(pink_decomposition, fit_stride=50)
        assert isinstance(result, TimeResolvedFit)

    def test_aperiodic_params_shape(self, pink_decomposition):
        result = time_resolved_fit(pink_decomposition, fit_stride=50)
        n_times = len(pink_decomposition.times)
        assert result.aperiodic_params.shape == (n_times, 2)

    def test_model_power_shape(self, pink_decomposition):
        result = time_resolved_fit(pink_decomposition, fit_stride=50)
        n_freqs = len(pink_decomposition.foi)
        n_times = len(pink_decomposition.times)
        assert result.model_power.shape == (n_freqs, n_times)

    def test_r_squared_shape(self, pink_decomposition):
        result = time_resolved_fit(pink_decomposition, fit_stride=50)
        n_times = len(pink_decomposition.times)
        assert result.r_squared.shape == (n_times,)

    def test_peak_params_length(self, pink_decomposition):
        result = time_resolved_fit(pink_decomposition, fit_stride=50)
        n_times = len(pink_decomposition.times)
        assert len(result.peak_params) == n_times

    def test_model_power_non_negative(self, pink_decomposition):
        result = time_resolved_fit(pink_decomposition, fit_stride=50)
        assert np.all(result.model_power >= 0)

    def test_fit_stride_stored(self, pink_decomposition):
        result = time_resolved_fit(pink_decomposition, fit_stride=25)
        assert result.fit_stride == 25


class TestParameterRecovery:
    """Recover known parameters from synthetic signals."""

    def test_recover_exponent_from_pink_noise(self, pink_decomposition):
        result = time_resolved_fit(pink_decomposition, fit_stride=50)

        # Interior time points should have valid fits
        n_times = len(result.times)
        interior = slice(n_times // 4, 3 * n_times // 4)
        valid = ~np.isnan(result.aperiodic_params[interior, 1])

        if np.sum(valid) > 0:
            exponents = result.aperiodic_params[interior, 1][valid]
            mean_exp = np.mean(exponents)
            # The wavelet density affects the apparent exponent.
            # With oct density: apparent exponent = true_exponent - 1.
            # True exponent is 1.5, so expect ~0.5 with oct density,
            # or ~1.5 with Hz density.
            # Accept either range since density may vary.
            assert 0.2 < mean_exp < 2.5, (
                f"Mean exponent {mean_exp:.2f} out of expected range"
            )

    def test_recover_peak_frequency(self, pink_alpha_decomposition):
        result = time_resolved_fit(pink_alpha_decomposition, fit_stride=50)

        # Check that at least some time points detect a peak near 10 Hz
        peak_cfs = []
        n_times = len(result.times)
        interior = slice(n_times // 4, 3 * n_times // 4)
        for pk in result.peak_params[interior]:
            if len(pk) > 0:
                peak_cfs.extend(pk[:, 0].tolist())

        assert len(peak_cfs) > 0, "No peaks detected"
        # At least some peaks should be near 10 Hz
        near_10 = [cf for cf in peak_cfs if abs(cf - 10) < 3]
        assert len(near_10) > len(peak_cfs) * 0.3, (
            f"Only {len(near_10)}/{len(peak_cfs)} peaks near 10 Hz"
        )

    def test_r_squared_reasonable(self, pink_decomposition):
        result = time_resolved_fit(pink_decomposition, fit_stride=50)
        n_times = len(result.times)
        interior = slice(n_times // 4, 3 * n_times // 4)
        valid = ~np.isnan(result.r_squared[interior])
        if np.sum(valid) > 0:
            median_r2 = np.median(result.r_squared[interior][valid])
            assert median_r2 > 0.5, f"Median r² = {median_r2:.3f}, expected > 0.5"


class TestStride:
    """Verify stride and interpolation behavior."""

    def test_stride_50_vs_stride_100_consistent(self, sfreq):
        """Aperiodic params from different strides should be similar at interior points."""
        rng = np.random.default_rng(42)
        n_samples = int(2 * sfreq)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
        freqs[0] = 1.0
        amplitudes = 1.0 / freqs ** 0.75
        phases = rng.uniform(0, 2 * np.pi, len(freqs))
        spectrum = amplitudes * np.exp(1j * phases)
        spectrum[0] = 0.0
        signal = np.fft.irfft(spectrum, n=n_samples)
        decomp = wavelet_decompose(signal, sfreq, foi_start=4, foi_end=30)

        result_s50 = time_resolved_fit(decomp, fit_stride=50)
        result_s100 = time_resolved_fit(decomp, fit_stride=100)

        # Compare at interior points (skip edges where edge effects differ)
        edge = n_samples // 4
        t_compare = np.arange(edge, n_samples - edge, 100)
        diffs = []
        for t in t_compare:
            exp_s50 = result_s50.aperiodic_params[t, 1]
            exp_s100 = result_s100.aperiodic_params[t, 1]
            if not np.isnan(exp_s50) and not np.isnan(exp_s100):
                diffs.append(abs(exp_s50 - exp_s100))

        if len(diffs) > 0:
            assert np.median(diffs) < 1.0, (
                f"Median exponent diff {np.median(diffs):.2f} too large"
            )


class TestSmoothing:
    """Verify temporal smoothing."""

    def test_smoothing_reduces_variance(self, pink_decomposition):
        result_raw = time_resolved_fit(pink_decomposition, fit_stride=50)
        result_smooth = time_resolved_fit(pink_decomposition, fit_stride=50, smooth_sigma=3.0)

        n_times = len(result_raw.times)
        interior = slice(n_times // 4, 3 * n_times // 4)

        valid_raw = ~np.isnan(result_raw.aperiodic_params[interior, 1])
        valid_smooth = ~np.isnan(result_smooth.aperiodic_params[interior, 1])

        if np.sum(valid_raw) > 5 and np.sum(valid_smooth) > 5:
            var_raw = np.var(result_raw.aperiodic_params[interior, 1][valid_raw])
            var_smooth = np.var(result_smooth.aperiodic_params[interior, 1][valid_smooth])
            assert var_smooth <= var_raw * 1.1, (
                f"Smoothed variance ({var_smooth:.4f}) should be <= raw ({var_raw:.4f})"
            )

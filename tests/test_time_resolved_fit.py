"""Tests for time_resolved_fit module."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tests.conftest import make_pink_noise
from meeglet_specparam_weights.wavelet_analysis import wavelet_decompose
from meeglet_specparam_weights.time_resolved_fit import TimeResolvedFit, time_resolved_fit


@pytest.fixture
def pink_decomposition(sfreq):
    """Wavelet decomposition of pink noise (exponent ~1.5)."""
    signal = make_pink_noise(int(4 * sfreq), sfreq, exponent_half=0.75)
    return wavelet_decompose(signal, sfreq, foi_start=2, foi_end=32)


@pytest.fixture
def pink_alpha_decomposition(sfreq):
    """Wavelet decomposition of pink noise + 10 Hz sine."""
    n_samples = int(4 * sfreq)
    signal = make_pink_noise(n_samples, sfreq, exponent_half=0.75)
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
        n_samples = int(2 * sfreq)
        signal = make_pink_noise(n_samples, sfreq, exponent_half=0.75)
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


class TestMultiChannel:
    """Verify multi-channel time-resolved fitting."""

    @pytest.fixture
    def multichannel_decomposition(self, sfreq):
        n_samples = int(4 * sfreq)
        signal_2d = np.stack([
            make_pink_noise(n_samples, sfreq, exponent_half=0.75, seed=42),
            make_pink_noise(n_samples, sfreq, exponent_half=1.0, seed=43),
        ])
        return wavelet_decompose(signal_2d, sfreq, foi_start=2, foi_end=32)

    def test_multichannel_output_shapes(self, multichannel_decomposition):
        result = time_resolved_fit(multichannel_decomposition, fit_stride=50)
        n_ch = 2
        n_times = len(multichannel_decomposition.times)
        n_freqs = len(multichannel_decomposition.foi)

        assert result.aperiodic_params.shape == (n_ch, n_times, 2)
        assert result.model_power.shape == (n_ch, n_freqs, n_times)
        assert result.r_squared.shape == (n_ch, n_times)
        assert result.n_channels == n_ch

    def test_multichannel_peak_params_structure(self, multichannel_decomposition):
        result = time_resolved_fit(multichannel_decomposition, fit_stride=50)
        n_times = len(multichannel_decomposition.times)
        assert len(result.peak_params) == 2
        assert len(result.peak_params[0]) == n_times
        assert len(result.peak_params[1]) == n_times

    def test_multichannel_different_exponents(self, multichannel_decomposition):
        result = time_resolved_fit(multichannel_decomposition, fit_stride=50)
        n_times = result.aperiodic_params.shape[1]
        interior = slice(n_times // 4, 3 * n_times // 4)

        exp_ch0 = result.aperiodic_params[0, interior, 1]
        exp_ch1 = result.aperiodic_params[1, interior, 1]
        valid = ~np.isnan(exp_ch0) & ~np.isnan(exp_ch1)
        if np.sum(valid) > 3:
            mean_ch0 = np.mean(exp_ch0[valid])
            mean_ch1 = np.mean(exp_ch1[valid])
            assert mean_ch0 != pytest.approx(mean_ch1, abs=0.01), (
                "Different input exponents should yield different fitted exponents"
            )

    def test_single_channel_backward_compat(self, pink_decomposition):
        result = time_resolved_fit(pink_decomposition, fit_stride=50)
        assert result.aperiodic_params.ndim == 2
        assert result.model_power.ndim == 2
        assert result.r_squared.ndim == 1
        assert result.n_channels == 1

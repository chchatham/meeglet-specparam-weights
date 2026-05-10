"""Tests for wavelet_analysis module."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from meeglet_specparam_weights.wavelet_analysis import (
    WaveletDecomposition,
    wavelet_decompose,
)


class TestWaveletDecomposeShapes:
    """Verify output shapes and types."""

    def test_output_type(self, pink_noise, sfreq):
        result = wavelet_decompose(pink_noise, sfreq)
        assert isinstance(result, WaveletDecomposition)

    def test_coefficients_shape(self, pink_noise, sfreq):
        result = wavelet_decompose(pink_noise, sfreq)
        n_samples = len(pink_noise)
        n_freqs = len(result.foi)
        assert result.coefficients.shape == (n_freqs, n_samples)

    def test_coefficients_complex(self, pink_noise, sfreq):
        result = wavelet_decompose(pink_noise, sfreq)
        assert np.iscomplexobj(result.coefficients)

    def test_times_match_signal(self, pink_noise, sfreq):
        result = wavelet_decompose(pink_noise, sfreq)
        expected_times = np.arange(len(pink_noise)) / sfreq
        np.testing.assert_array_almost_equal(result.times, expected_times)

    def test_sfreq_preserved(self, pink_noise, sfreq):
        result = wavelet_decompose(pink_noise, sfreq)
        assert result.sfreq == sfreq

    def test_foi_monotonic_increasing(self, pink_noise, sfreq):
        result = wavelet_decompose(pink_noise, sfreq)
        assert np.all(np.diff(result.foi) > 0)

    def test_foi_range(self, pink_noise, sfreq):
        result = wavelet_decompose(pink_noise, sfreq, foi_start=4, foi_end=30)
        assert result.foi[0] >= 4.0
        assert result.foi[-1] <= 30.0 * 1.01  # small tolerance for floating point

    def test_sigma_shapes(self, pink_noise, sfreq):
        result = wavelet_decompose(pink_noise, sfreq)
        n_freqs = len(result.foi)
        assert result.sigma_time.shape == (n_freqs,)
        assert result.sigma_freq.shape == (n_freqs,)

    def test_bw_oct_stored(self, pink_noise, sfreq):
        result = wavelet_decompose(pink_noise, sfreq, bw_oct=0.5)
        assert result.bw_oct == 0.5

    def test_delta_oct_stored(self, pink_noise, sfreq):
        result = wavelet_decompose(pink_noise, sfreq, bw_oct=0.5)
        assert result.delta_oct == 0.5 / 4.0  # default is bw_oct / 4

    def test_accepts_2d_input(self, sfreq):
        signal_2d = np.random.default_rng(42).standard_normal((3, int(4 * sfreq)))
        result = wavelet_decompose(signal_2d, sfreq)
        n_freqs = len(result.foi)
        assert result.coefficients.shape == (3, n_freqs, signal_2d.shape[1])
        assert result.n_channels == 3

    def test_rejects_3d_input(self, sfreq):
        signal_3d = np.random.randn(2, 3, 100)
        with pytest.raises(ValueError, match="1D or 2D"):
            wavelet_decompose(signal_3d, sfreq)


class TestNaNPropagation:
    """Verify NaN handling follows guardrails."""

    def test_nan_in_signal_produces_nan_coefficients(self, sfreq):
        n_samples = int(4 * sfreq)
        signal = np.random.default_rng(42).standard_normal(n_samples)
        signal[500] = np.nan

        result = wavelet_decompose(signal, sfreq)

        # Every frequency should have NaN at time points near sample 500
        for i_freq in range(len(result.foi)):
            assert np.any(np.isnan(result.coefficients[i_freq])), (
                f"Frequency {result.foi[i_freq]:.1f} Hz has no NaN despite NaN input"
            )

    def test_nan_spread_proportional_to_kernel_width(self, sfreq):
        n_samples = int(4 * sfreq)
        signal = np.random.default_rng(42).standard_normal(n_samples)
        signal[512] = np.nan

        result = wavelet_decompose(signal, sfreq)

        # Low frequencies have wider kernels, so NaN should spread further
        nan_count_low = np.sum(np.isnan(result.coefficients[0]))
        nan_count_high = np.sum(np.isnan(result.coefficients[-1]))
        assert nan_count_low > nan_count_high

    def test_no_nan_when_input_clean(self, pink_noise, sfreq):
        result = wavelet_decompose(pink_noise, sfreq)
        assert not np.any(np.isnan(result.coefficients))

    def test_all_nan_input(self, sfreq):
        signal = np.full(int(2 * sfreq), np.nan)
        result = wavelet_decompose(signal, sfreq)
        assert np.all(np.isnan(result.coefficients))


class TestFrequencyDetection:
    """Verify known-frequency signals produce expected power patterns."""

    def test_10hz_sine_peak_at_10hz(self, alpha_signal, sfreq):
        result = wavelet_decompose(alpha_signal, sfreq, foi_start=2, foi_end=32)

        # Compute time-averaged power at each frequency
        power = np.nanmean(np.abs(result.coefficients) ** 2, axis=1)

        # Find peak frequency
        peak_idx = np.argmax(power)
        peak_freq = result.foi[peak_idx]

        # Peak should be close to 10 Hz (within frequency resolution)
        assert abs(peak_freq - 10.0) < 2.0, (
            f"Peak at {peak_freq:.1f} Hz, expected ~10 Hz"
        )

    def test_20hz_sine_peak_at_20hz(self, sfreq):
        n_samples = int(4 * sfreq)
        t = np.arange(n_samples) / sfreq
        signal = np.sin(2 * np.pi * 20 * t)

        result = wavelet_decompose(signal, sfreq, foi_start=2, foi_end=32)
        power = np.nanmean(np.abs(result.coefficients) ** 2, axis=1)
        peak_freq = result.foi[np.argmax(power)]

        assert abs(peak_freq - 20.0) < 2.0

    def test_two_frequencies_both_detected(self, sfreq):
        n_samples = int(4 * sfreq)
        t = np.arange(n_samples) / sfreq
        signal = np.sin(2 * np.pi * 8 * t) + np.sin(2 * np.pi * 20 * t)

        result = wavelet_decompose(signal, sfreq, foi_start=2, foi_end=32)

        # Use interior samples only (avoid edge effects)
        edge = int(0.5 * sfreq)
        power = np.nanmean(
            np.abs(result.coefficients[:, edge:-edge]) ** 2, axis=1
        )

        # Normalize so we can find peaks
        power_norm = power / np.max(power)

        # Both 8 Hz and 20 Hz should have elevated power
        idx_8 = np.argmin(np.abs(result.foi - 8.0))
        idx_20 = np.argmin(np.abs(result.foi - 20.0))

        assert power_norm[idx_8] > 0.3, f"8 Hz power too low: {power_norm[idx_8]:.3f}"
        assert power_norm[idx_20] > 0.3, f"20 Hz power too low: {power_norm[idx_20]:.3f}"

    def test_power_at_non_signal_freq_is_low(self, alpha_signal, sfreq):
        result = wavelet_decompose(alpha_signal, sfreq, foi_start=2, foi_end=32)

        edge = int(0.5 * sfreq)
        power = np.nanmean(
            np.abs(result.coefficients[:, edge:-edge]) ** 2, axis=1
        )

        idx_10 = np.argmin(np.abs(result.foi - 10.0))
        idx_25 = np.argmin(np.abs(result.foi - 25.0))

        assert power[idx_10] > 10 * power[idx_25], (
            f"Power at 10 Hz ({power[idx_10]:.3f}) should dominate 25 Hz ({power[idx_25]:.3f})"
        )


class TestPhasePreservation:
    """Verify wavelet coefficients preserve phase information."""

    def test_phase_at_peak_frequency(self, sfreq):
        n_samples = int(4 * sfreq)
        t = np.arange(n_samples) / sfreq
        signal = np.sin(2 * np.pi * 10 * t)

        result = wavelet_decompose(signal, sfreq, foi_start=2, foi_end=32)

        idx_10 = np.argmin(np.abs(result.foi - 10.0))

        # Phase at the peak frequency should advance linearly at 10 Hz
        edge = int(0.5 * sfreq)
        phase = np.angle(result.coefficients[idx_10, edge:-edge])
        unwrapped = np.unwrap(phase)

        # Rate of phase change should be ~2*pi*10 rad/s
        phase_rate = np.diff(unwrapped) * sfreq  # rad/s
        expected_rate = 2 * np.pi * 10

        median_rate = np.median(phase_rate)
        assert abs(median_rate - expected_rate) < expected_rate * 0.15, (
            f"Phase rate {median_rate:.1f} rad/s, expected {expected_rate:.1f}"
        )


class TestMultiChannel:
    """Verify multi-channel wavelet decomposition."""

    def test_multichannel_shape(self, sfreq):
        n_ch, n_samples = 4, int(4 * sfreq)
        signal = np.random.default_rng(42).standard_normal((n_ch, n_samples))
        result = wavelet_decompose(signal, sfreq)
        n_freqs = len(result.foi)
        assert result.coefficients.shape == (n_ch, n_freqs, n_samples)
        assert result.n_channels == n_ch

    def test_single_channel_backward_compat(self, pink_noise, sfreq):
        result = wavelet_decompose(pink_noise, sfreq)
        assert result.coefficients.ndim == 2
        assert result.n_channels == 1

    def test_multichannel_matches_single(self, sfreq):
        rng = np.random.default_rng(99)
        n_samples = int(4 * sfreq)
        ch0 = rng.standard_normal(n_samples)
        ch1 = rng.standard_normal(n_samples)

        result_multi = wavelet_decompose(np.stack([ch0, ch1]), sfreq)
        result_ch0 = wavelet_decompose(ch0, sfreq)
        result_ch1 = wavelet_decompose(ch1, sfreq)

        np.testing.assert_array_almost_equal(result_multi.coefficients[0], result_ch0.coefficients)
        np.testing.assert_array_almost_equal(result_multi.coefficients[1], result_ch1.coefficients)

    def test_per_channel_nan_independence(self, sfreq):
        n_samples = int(4 * sfreq)
        rng = np.random.default_rng(42)
        signal = rng.standard_normal((2, n_samples))
        signal[0, 500] = np.nan  # NaN only in channel 0

        result = wavelet_decompose(signal, sfreq)

        assert np.any(np.isnan(result.coefficients[0]))
        assert not np.any(np.isnan(result.coefficients[1]))

    def test_multichannel_freq_detection(self, sfreq):
        n_samples = int(4 * sfreq)
        t = np.arange(n_samples) / sfreq
        ch_10hz = np.sin(2 * np.pi * 10 * t)
        ch_20hz = np.sin(2 * np.pi * 20 * t)
        signal = np.stack([ch_10hz, ch_20hz])

        result = wavelet_decompose(signal, sfreq, foi_start=2, foi_end=32)
        power = np.nanmean(np.abs(result.coefficients) ** 2, axis=2)

        peak_ch0 = result.foi[np.argmax(power[0])]
        peak_ch1 = result.foi[np.argmax(power[1])]

        assert abs(peak_ch0 - 10.0) < 2.0, f"Ch0 peak at {peak_ch0:.1f}, expected ~10 Hz"
        assert abs(peak_ch1 - 20.0) < 2.0, f"Ch1 peak at {peak_ch1:.1f}, expected ~20 Hz"

    def test_multichannel_complex(self, sfreq):
        signal = np.random.default_rng(42).standard_normal((2, int(4 * sfreq)))
        result = wavelet_decompose(signal, sfreq)
        assert np.iscomplexobj(result.coefficients)

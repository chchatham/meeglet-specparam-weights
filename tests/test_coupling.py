"""Tests for coupling module."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tests.conftest import make_pink_noise
from meeglet_specparam_weights.wavelet_analysis import wavelet_decompose
from meeglet_specparam_weights.time_resolved_fit import time_resolved_fit
from meeglet_specparam_weights.coupling import (
    AperiodicCouplingResult,
    aperiodic_virtual_channels,
    compute_aperiodic_csd,
    aperiodic_amplitude_correlation,
    effective_dof,
    wavelet_effective_dof,
)


@pytest.fixture
def pink_coupled(sfreq):
    """Single-channel pink noise with wavelet decomposition and fit."""
    signal = make_pink_noise(int(4 * sfreq), sfreq, exponent_half=0.75)
    decomp = wavelet_decompose(signal, sfreq, foi_start=2, foi_end=32)
    fit = time_resolved_fit(decomp, fit_stride=50)
    return decomp, fit


@pytest.fixture
def multichannel_coupled(sfreq):
    """2-channel pink noise with wavelet decomposition and fit."""
    n_samples = int(4 * sfreq)
    signal_2d = np.stack([
        make_pink_noise(n_samples, sfreq, exponent_half=0.75, seed=42),
        make_pink_noise(n_samples, sfreq, exponent_half=1.0, seed=43),
    ])
    decomp = wavelet_decompose(signal_2d, sfreq, foi_start=2, foi_end=32)
    fit = time_resolved_fit(decomp, fit_stride=50)
    return decomp, fit


class TestCSDShape:
    """Verify CSD shape and Hermitian symmetry."""

    def test_single_channel_csd_shape(self, pink_coupled):
        decomp, fit = pink_coupled
        result = compute_aperiodic_csd(decomp, fit)
        n_freqs = len(decomp.foi)
        assert result.csd.shape == (3, 3, n_freqs)

    def test_multichannel_csd_shape(self, multichannel_coupled):
        decomp, fit = multichannel_coupled
        result = compute_aperiodic_csd(decomp, fit)
        n_freqs = len(decomp.foi)
        assert result.csd.shape == (4, 4, n_freqs)

    def test_csd_hermitian(self, pink_coupled):
        decomp, fit = pink_coupled
        result = compute_aperiodic_csd(decomp, fit)
        for f_idx in range(len(decomp.foi)):
            csd_f = result.csd[:, :, f_idx]
            np.testing.assert_array_almost_equal(
                csd_f, csd_f.conj().T,
                err_msg=f"CSD not Hermitian at freq index {f_idx}",
            )

    def test_output_type(self, pink_coupled):
        decomp, fit = pink_coupled
        result = compute_aperiodic_csd(decomp, fit)
        assert isinstance(result, AperiodicCouplingResult)

    def test_channel_labels(self, multichannel_coupled):
        decomp, fit = multichannel_coupled
        result = compute_aperiodic_csd(decomp, fit)
        assert result.channel_labels == ["ch0", "ch1", "exponent", "offset"]


class TestNyquistEnforcement:
    """Verify virtual channel coefficients are zeroed above effective Nyquist."""

    def test_virtual_coefficients_zeroed_above_nyquist(self, pink_coupled):
        decomp, fit = pink_coupled
        virtual, effective_nyq = aperiodic_virtual_channels(fit, decomp)
        foi = decomp.foi

        for f_idx in range(len(foi)):
            if foi[f_idx] > effective_nyq:
                assert np.all(virtual[:, f_idx, :] == 0.0), (
                    f"Virtual coefficients not zeroed at {foi[f_idx]:.1f} Hz "
                    f"(Nyquist = {effective_nyq:.1f} Hz)"
                )

    def test_effective_nyquist_value(self, sfreq):
        rng = np.random.default_rng(42)
        n_samples = int(4 * sfreq)
        signal = rng.standard_normal(n_samples)
        decomp = wavelet_decompose(signal, sfreq, foi_start=2, foi_end=32)
        fit = time_resolved_fit(decomp, fit_stride=50)
        _, effective_nyq = aperiodic_virtual_channels(fit, decomp)
        expected = sfreq / (2 * 50)
        assert effective_nyq == pytest.approx(expected)

    def test_virtual_coefficients_shape(self, pink_coupled):
        decomp, fit = pink_coupled
        virtual, _ = aperiodic_virtual_channels(fit, decomp)
        n_freqs = len(decomp.foi)
        n_times = len(decomp.times)
        assert virtual.shape == (2, n_freqs, n_times)


class TestKnownCoupling:
    """Test coupling recovery on synthetic signals with known amplitude modulation."""

    def test_amplitude_modulated_signal_has_coupling(self, sfreq):
        """Signal with amplitude-modulated alpha should show correlation at 10 Hz."""
        n_samples = int(6 * sfreq)
        t = np.arange(n_samples) / sfreq

        pink = make_pink_noise(n_samples, sfreq, exponent_half=0.75)

        # Alpha with amplitude that varies strongly over time (on/off blocks)
        alpha_envelope = np.zeros(n_samples)
        block_len = int(1.5 * sfreq)
        for i in range(0, n_samples, 2 * block_len):
            alpha_envelope[i:i + block_len] = 5.0
        alpha = alpha_envelope * np.sin(2 * np.pi * 10 * t)

        signal = pink + alpha
        decomp = wavelet_decompose(signal, sfreq, foi_start=2, foi_end=32)
        fit = time_resolved_fit(decomp, fit_stride=25)

        amp_corr = aperiodic_amplitude_correlation(decomp, fit)
        idx_10 = np.argmin(np.abs(decomp.foi - 10))

        assert amp_corr.shape == (1, len(decomp.foi))
        assert np.abs(amp_corr[0, idx_10]) > 0.05, (
            f"Expected non-trivial correlation at 10 Hz, got {amp_corr[0, idx_10]:.4f}"
        )


class TestNullCoupling:
    """Test that independent signals show minimal coupling."""

    def test_stationary_pink_has_low_coupling(self, pink_coupled):
        """Stationary pink noise should show weak amplitude correlation."""
        decomp, fit = pink_coupled
        amp_corr = aperiodic_amplitude_correlation(decomp, fit)

        assert amp_corr.shape == (1, len(decomp.foi))
        median_abs_corr = np.median(np.abs(amp_corr))
        assert median_abs_corr < 0.5, (
            f"Median |correlation| = {median_abs_corr:.3f}, expected low for stationary signal"
        )


class TestEffectiveDOF:
    """Test effective degrees of freedom estimation."""

    def test_white_noise_full_dof(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(1000)
        y = rng.standard_normal(1000)
        n_eff = effective_dof(x, y)
        assert n_eff > 500, f"White noise should retain most DOF, got {n_eff:.0f}"

    def test_autocorrelated_reduces_dof(self):
        rng = np.random.default_rng(42)
        n = 1000
        x = np.cumsum(rng.standard_normal(n))
        y = np.cumsum(rng.standard_normal(n))
        n_eff = effective_dof(x, y)
        assert n_eff < n * 0.5, (
            f"Autocorrelated signal should have reduced DOF, got {n_eff:.0f}"
        )

    def test_minimum_dof(self):
        x = np.ones(100)
        y = np.ones(100)
        n_eff = effective_dof(x, y)
        assert n_eff >= 2.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            effective_dof(np.ones(10), np.ones(20))


class TestWaveletEffectiveDOF:
    """Test wavelet-aware effective DOF estimation."""

    def test_decreases_with_sigma_time(self):
        sigma_time = np.array([0.1, 0.5, 1.0])
        n_eff = wavelet_effective_dof(sigma_time, sfreq=256.0, n_samples=2560)
        assert n_eff[0] > n_eff[1] > n_eff[2]

    def test_clamped_to_valid_range(self):
        sigma_time = np.array([0.001, 100.0])
        n_eff = wavelet_effective_dof(sigma_time, sfreq=256.0, n_samples=1000)
        assert np.all(n_eff >= 2.0)
        assert np.all(n_eff <= 1000)

    def test_matches_signal_duration(self):
        sigma_time = np.array([0.5])
        n_eff = wavelet_effective_dof(sigma_time, sfreq=256.0, n_samples=2560)
        T = 2560 / 256.0
        expected = T / (2 * 0.5)
        assert n_eff[0] == pytest.approx(expected)

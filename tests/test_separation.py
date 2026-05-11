"""Tests for separation module."""

import numpy as np
import pytest
import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tests.conftest import make_pink_noise
from meeglet_specparam_weights.separation import (
    SeparationResult,
    subtraction_separate,
    wiener_separate,
    decomposition_bias_estimate,
)
from meeglet_specparam_weights.wavelet_analysis import wavelet_decompose
from meeglet_specparam_weights.time_resolved_fit import time_resolved_fit
from meeglet_specparam_weights.pipeline import meeglet_specparam_reconstruct


@pytest.fixture
def signal_and_fit(sfreq):
    """Pink noise + alpha signal with wavelet decomposition and fit."""
    n_samples = int(4 * sfreq)
    pink = make_pink_noise(n_samples, sfreq, exponent_half=0.75)
    t = np.arange(n_samples) / sfreq
    alpha = 3.0 * np.sin(2 * np.pi * 10 * t)
    signal = pink + alpha
    decomposition = wavelet_decompose(signal, sfreq)
    fit = time_resolved_fit(decomposition, fit_stride=50)
    return signal, decomposition, fit


class TestSeparationResult:
    """Verify SeparationResult structure."""

    def test_subtraction_result_type(self, signal_and_fit, sfreq):
        signal, decomposition, fit = signal_and_fit
        result = subtraction_separate(signal, decomposition, fit)
        assert isinstance(result, SeparationResult)

    def test_subtraction_method_field(self, signal_and_fit, sfreq):
        signal, decomposition, fit = signal_and_fit
        result = subtraction_separate(signal, decomposition, fit)
        assert result.method == "subtraction"

    def test_wiener_method_field(self, signal_and_fit, sfreq):
        signal, decomposition, fit = signal_and_fit
        result = wiener_separate(signal, decomposition, fit)
        assert result.method == "wiener"

    def test_shapes_match_signal(self, signal_and_fit, sfreq):
        signal, decomposition, fit = signal_and_fit
        result = subtraction_separate(signal, decomposition, fit)
        assert result.aperiodic.shape == signal.shape
        assert result.periodic.shape == signal.shape

    def test_bias_estimate_shape(self, signal_and_fit, sfreq):
        signal, decomposition, fit = signal_and_fit
        result = subtraction_separate(signal, decomposition, fit)
        assert result.bias_estimate.shape == (len(decomposition.foi),)


class TestSubtractionSeparation:
    """Verify subtraction separation matches pipeline output."""

    def test_matches_pipeline_aperiodic(self, signal_and_fit, sfreq):
        signal, decomposition, fit = signal_and_fit
        sep_result = subtraction_separate(signal, decomposition, fit)
        pipe_result = meeglet_specparam_reconstruct(
            signal, sfreq, component="aperiodic", fit_stride=50,
        )
        np.testing.assert_allclose(
            sep_result.aperiodic, pipe_result.reconstruction, atol=1e-10,
        )

    def test_decomposition_sums_to_original(self, signal_and_fit, sfreq):
        signal, decomposition, fit = signal_and_fit
        result = subtraction_separate(signal, decomposition, fit)
        np.testing.assert_allclose(
            result.aperiodic + result.periodic, signal, atol=1e-10,
        )

    def test_excess_weights_bounded(self, signal_and_fit, sfreq):
        signal, decomposition, fit = signal_and_fit
        result = subtraction_separate(signal, decomposition, fit)
        assert result.weights is not None
        assert result.weights.weights.min() >= 0.0
        assert result.weights.weights.max() <= 1.0 + 1e-10


class TestWienerSeparation:
    """Verify Wiener separation properties."""

    def test_power_preserved_at_alpha(self, sfreq):
        """Wiener aperiodic should preserve correct power at alpha."""
        n_samples = int(10 * sfreq)
        pink = make_pink_noise(n_samples, sfreq, exponent_half=0.75)
        t = np.arange(n_samples) / sfreq
        alpha = 5.0 * np.sin(2 * np.pi * 10 * t)
        signal = pink + alpha

        decomposition = wavelet_decompose(signal, sfreq)
        fit = time_resolved_fit(decomposition, fit_stride=50)
        result = wiener_separate(signal, decomposition, fit)

        edge = int(1.0 * sfreq)
        ap_trimmed = result.aperiodic[edge:-edge]
        pink_trimmed = pink[edge:-edge]

        freqs_ap = np.fft.rfftfreq(len(ap_trimmed), d=1.0 / sfreq)
        psd_ap = np.abs(np.fft.rfft(ap_trimmed)) ** 2
        psd_pink = np.abs(np.fft.rfft(pink_trimmed)) ** 2

        idx_10 = np.argmin(np.abs(freqs_ap - 10))
        window = slice(max(0, idx_10 - 3), idx_10 + 4)
        ratio = np.mean(psd_ap[window]) / max(np.mean(psd_pink[window]), 1e-30)
        assert ratio > 0.1, (
            f"Wiener aperiodic power at alpha should be within ~10x of true, got ratio={ratio:.4f}"
        )

    def test_bias_estimate_is_one(self, signal_and_fit, sfreq):
        signal, decomposition, fit = signal_and_fit
        result = wiener_separate(signal, decomposition, fit)
        np.testing.assert_allclose(result.bias_estimate, 1.0)


class TestBiasEstimate:
    """Verify decomposition_bias_estimate calculations."""

    def test_wiener_bias_all_ones(self, signal_and_fit, sfreq):
        _, decomposition, fit = signal_and_fit
        bias = decomposition_bias_estimate(decomposition, fit, method="wiener")
        np.testing.assert_allclose(bias, 1.0)

    def test_state_space_bias_all_ones(self, signal_and_fit, sfreq):
        _, decomposition, fit = signal_and_fit
        bias = decomposition_bias_estimate(decomposition, fit, method="state_space")
        np.testing.assert_allclose(bias, 1.0)

    def test_subtraction_bias_below_one_at_peak(self, signal_and_fit, sfreq):
        """At alpha peak, subtraction bias should be < 1 (suppressed)."""
        _, decomposition, fit = signal_and_fit
        bias = decomposition_bias_estimate(decomposition, fit, method="subtraction")
        foi = decomposition.foi
        idx_10 = np.argmin(np.abs(foi - 10))
        assert bias[idx_10] < 0.5, (
            f"Subtraction bias at 10 Hz should be < 0.5, got {bias[idx_10]:.3f}"
        )

    def test_subtraction_bias_near_one_on_pure_noise(self, sfreq):
        """On pure pink noise (no peaks), bias should be near 1 everywhere."""
        n_samples = int(4 * sfreq)
        signal = make_pink_noise(n_samples, sfreq, exponent_half=0.75)
        decomposition = wavelet_decompose(signal, sfreq)
        fit = time_resolved_fit(decomposition, fit_stride=50, min_peak_height=1.0)
        bias = decomposition_bias_estimate(decomposition, fit, method="subtraction")
        assert np.mean(bias) > 0.5, (
            f"Bias on pure noise should be mostly near 1, got mean={np.mean(bias):.3f}"
        )

    def test_bias_formula_matches_theory(self):
        """Test the bias formula directly: bias = (1-sqrt(r))^2 / (1-r)."""
        r_values = [0.3, 0.5, 0.7, 0.9, 0.95]
        for r in r_values:
            expected = (1.0 - np.sqrt(r)) ** 2 / (1.0 - r)
            assert expected < 1.0, f"Bias should be < 1 for r={r}"
            assert expected > 0.0, f"Bias should be > 0 for r={r}"
        r_09 = 0.9
        bias_09 = (1.0 - np.sqrt(r_09)) ** 2 / (1.0 - r_09)
        assert bias_09 < 0.05, f"At r=0.9, bias should be < 0.05, got {bias_09:.4f}"


class TestPipelineSeparationParam:
    """Verify the new 'separation' parameter on the pipeline."""

    def test_separation_subtraction(self, sfreq):
        n_samples = int(4 * sfreq)
        signal = make_pink_noise(n_samples, sfreq, exponent_half=0.75)
        result = meeglet_specparam_reconstruct(
            signal, sfreq, component="aperiodic", fit_stride=50,
            separation="subtraction",
        )
        assert result.method == "subtraction"
        assert result.bias_estimate is not None

    def test_separation_wiener(self, sfreq):
        n_samples = int(4 * sfreq)
        signal = make_pink_noise(n_samples, sfreq, exponent_half=0.75)
        result = meeglet_specparam_reconstruct(
            signal, sfreq, component="aperiodic", fit_stride=50,
            separation="wiener",
        )
        assert result.method == "wiener"
        assert result.bias_estimate is not None
        np.testing.assert_allclose(result.bias_estimate, 1.0)

    def test_deprecated_aperiodic_method_warns(self, sfreq):
        n_samples = int(4 * sfreq)
        signal = make_pink_noise(n_samples, sfreq, exponent_half=0.75)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            meeglet_specparam_reconstruct(
                signal, sfreq, component="aperiodic", fit_stride=50,
                aperiodic_method="subtraction",
            )
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "aperiodic_method" in str(deprecation_warnings[0].message)

    def test_both_params_raises(self, sfreq):
        n_samples = int(4 * sfreq)
        signal = make_pink_noise(n_samples, sfreq, exponent_half=0.75)
        with pytest.raises(ValueError, match="Cannot specify both"):
            meeglet_specparam_reconstruct(
                signal, sfreq, component="aperiodic", fit_stride=50,
                separation="subtraction",
                aperiodic_method="wiener",
            )

    def test_invalid_separation_raises(self, sfreq):
        n_samples = int(4 * sfreq)
        signal = make_pink_noise(n_samples, sfreq, exponent_half=0.75)
        with pytest.raises(ValueError, match="separation"):
            meeglet_specparam_reconstruct(
                signal, sfreq, component="aperiodic", fit_stride=50,
                separation="invalid",
            )

    def test_periodic_component_ignores_separation(self, sfreq):
        n_samples = int(4 * sfreq)
        signal = make_pink_noise(n_samples, sfreq, exponent_half=0.75)
        result = meeglet_specparam_reconstruct(
            signal, sfreq, component="periodic", fit_stride=50,
            separation="wiener",
        )
        assert result.method == "weight"
        assert result.bias_estimate is None

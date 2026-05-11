"""Tests for state_space module."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tests.conftest import make_pink_noise
from scipy.linalg import solve_discrete_lyapunov

from meeglet_specparam_weights.state_space import (
    StateSpaceModel,
    StateSpaceDecomposition,
    ar_coefficients_from_exponent,
    select_ar_order,
    build_state_space_model,
    build_matrices,
    kalman_filter,
    rts_smoother,
    state_space_decompose,
    state_space_separate,
)
from meeglet_specparam_weights.wavelet_analysis import wavelet_decompose
from meeglet_specparam_weights.time_resolved_fit import time_resolved_fit
from meeglet_specparam_weights.pipeline import meeglet_specparam_reconstruct, ReconstructionResult


@pytest.fixture
def sfreq():
    return 256.0


@pytest.fixture
def simple_fit(sfreq):
    """Fit from a pink+alpha signal for model construction."""
    n_samples = int(4 * sfreq)
    pink = make_pink_noise(n_samples, sfreq, exponent_half=0.75)
    t = np.arange(n_samples) / sfreq
    alpha = 3.0 * np.sin(2 * np.pi * 10 * t)
    signal = pink + alpha
    decomposition = wavelet_decompose(signal, sfreq)
    fit = time_resolved_fit(decomposition, fit_stride=50)
    return signal, decomposition, fit


class TestARCoefficients:
    def test_output_shapes(self, sfreq):
        coeffs, sigma2 = ar_coefficients_from_exponent(1.5, sfreq, 10)
        assert coeffs.shape == (10,)
        assert sigma2 > 0

    def test_stability(self, sfreq):
        """All AR roots must be inside the unit circle."""
        coeffs, _ = ar_coefficients_from_exponent(1.5, sfreq, 15)
        companion = np.zeros((15, 15))
        companion[0, :] = coeffs
        companion[1:, :-1] = np.eye(14)
        eigenvalues = np.linalg.eigvals(companion)
        assert np.all(np.abs(eigenvalues) < 1.0 + 1e-10), (
            f"AR model unstable: max |eigenvalue| = {np.max(np.abs(eigenvalues)):.4f}"
        )

    def test_psd_matches_target(self, sfreq):
        """AR model PSD should approximate 1/f^exponent within 3 dB."""
        exponent = 1.5
        coeffs, sigma2 = ar_coefficients_from_exponent(exponent, sfreq, 20)

        n_fft = 4096
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)
        freqs[0] = freqs[1]
        target_psd = 1.0 / freqs ** exponent

        z = np.exp(-1j * 2 * np.pi * freqs / sfreq)
        denom = np.ones(len(freqs), dtype=complex)
        for k, a in enumerate(coeffs):
            denom -= a * z ** (k + 1)
        ar_psd = sigma2 / np.abs(denom) ** 2

        ratio = ar_psd / target_psd
        mask = (freqs > 2) & (freqs < sfreq / 4)
        log_ratio = np.abs(10 * np.log10(ratio[mask]))
        assert np.median(log_ratio) < 3.0, (
            f"AR PSD deviates from target by {np.median(log_ratio):.1f} dB (median)"
        )

    def test_order_selection(self, sfreq):
        order = select_ar_order(1.5, sfreq, max_order=30)
        assert 2 <= order <= 30


class TestModelConstruction:
    def test_shapes(self, simple_fit, sfreq):
        _, _, fit = simple_fit
        model = build_state_space_model(fit, sfreq, ar_order=10)
        assert model.n_states == 2 * model.n_oscillators + model.ar_order

    def test_block_diagonal(self, simple_fit, sfreq):
        _, _, fit = simple_fit
        model = build_state_space_model(fit, sfreq, ar_order=5)
        F, H, Q, R = build_matrices(model)
        assert F.shape == (model.n_states, model.n_states)
        assert Q.shape == (model.n_states, model.n_states)
        assert H.shape == (model.n_states,)
        assert R > 0

    def test_observation_vector(self, simple_fit, sfreq):
        _, _, fit = simple_fit
        model = build_state_space_model(fit, sfreq, ar_order=5)
        _, H, _, _ = build_matrices(model)
        for k in range(model.n_oscillators):
            assert H[2 * k] == 1.0
            assert H[2 * k + 1] == 0.0
        ar_start = 2 * model.n_oscillators
        assert H[ar_start] == 1.0

    def test_oscillator_frequency(self, simple_fit, sfreq):
        _, _, fit = simple_fit
        model = build_state_space_model(fit, sfreq, ar_order=5)
        if model.n_oscillators > 0:
            F, _, _, _ = build_matrices(model)
            R_block = F[:2, :2]
            eigenvalues = np.linalg.eigvals(R_block)
            freq = np.abs(np.angle(eigenvalues[0])) * sfreq / (2 * np.pi)
            assert abs(freq - model.center_freqs[0]) < 1.0

    def test_initial_covariance(self, simple_fit, sfreq):
        _, _, fit = simple_fit
        model = build_state_space_model(fit, sfreq, ar_order=5)
        F, _, Q, _ = build_matrices(model)
        P0 = solve_discrete_lyapunov(F, Q)
        residual = F @ P0 @ F.T + Q - P0
        assert np.max(np.abs(residual)) < 1e-8


class TestKalmanFilter:
    def test_output_shapes(self, sfreq):
        n = 4
        T = 100
        y = np.random.randn(T)
        F = np.eye(n) * 0.9
        H = np.array([1.0, 0, 0, 0])
        Q = np.eye(n) * 0.1
        R = 1.0
        x0 = np.zeros(n)
        P0 = np.eye(n)
        x_filt, P_filt, x_pred, P_pred, ll = kalman_filter(y, F, H, Q, R, x0, P0)
        assert x_filt.shape == (n, T)
        assert P_filt.shape == (n, n, T)
        assert x_pred.shape == (n, T)
        assert P_pred.shape == (n, n, T)
        assert np.isfinite(ll)

    def test_known_signal(self, sfreq):
        """Filter should track a constant signal in noise."""
        n = 1
        T = 200
        true_val = 5.0
        y = true_val + np.random.randn(T) * 0.1
        F = np.array([[1.0]])
        H = np.array([1.0])
        Q = np.array([[0.0001]])
        R = 0.01
        x0 = np.array([0.0])
        P0 = np.array([[10.0]])
        x_filt, _, _, _, _ = kalman_filter(y, F, H, Q, R, x0, P0)
        assert abs(x_filt[0, -1] - true_val) < 0.5

    def test_variance_reduction(self, sfreq):
        n = 2
        T = 50
        y = np.random.randn(T)
        F = np.eye(n) * 0.95
        H = np.array([1.0, 0.0])
        Q = np.eye(n) * 0.1
        R = 1.0
        x0 = np.zeros(n)
        P0 = np.eye(n) * 10
        _, P_filt, _, P_pred, _ = kalman_filter(y, F, H, Q, R, x0, P0)
        assert np.trace(P_filt[:, :, -1]) < np.trace(P_pred[:, :, -1])

    def test_log_likelihood_finite(self, sfreq):
        n = 2
        T = 50
        y = np.random.randn(T)
        F = np.eye(n) * 0.95
        H = np.array([1.0, 0.0])
        Q = np.eye(n) * 0.1
        R = 1.0
        x0 = np.zeros(n)
        P0 = np.eye(n)
        _, _, _, _, ll = kalman_filter(y, F, H, Q, R, x0, P0)
        assert np.isfinite(ll)


class TestRTSSmoother:
    def _run_filter_and_smooth(self, T=100):
        n = 2
        rng = np.random.default_rng(42)
        y = rng.standard_normal(T)
        F = np.eye(n) * 0.95
        H = np.array([1.0, 0.0])
        Q = np.eye(n) * 0.1
        R = 1.0
        x0 = np.zeros(n)
        P0 = np.eye(n)
        x_filt, P_filt, x_pred, P_pred, _ = kalman_filter(y, F, H, Q, R, x0, P0)
        x_smooth, P_smooth = rts_smoother(x_filt, P_filt, x_pred, P_pred, F)
        return x_filt, P_filt, x_smooth, P_smooth

    def test_output_shapes(self):
        x_filt, _, x_smooth, P_smooth = self._run_filter_and_smooth()
        assert x_smooth.shape == x_filt.shape
        assert P_smooth.shape[0] == x_filt.shape[0]

    def test_variance_reduction(self):
        _, P_filt, _, P_smooth = self._run_filter_and_smooth()
        mid = P_filt.shape[2] // 2
        assert np.trace(P_smooth[:, :, mid]) <= np.trace(P_filt[:, :, mid]) + 1e-10

    def test_matches_filter_at_end(self):
        x_filt, _, x_smooth, _ = self._run_filter_and_smooth()
        np.testing.assert_allclose(x_smooth[:, -1], x_filt[:, -1], atol=1e-10)


class TestStateSpaceDecompose:
    def test_output_structure(self, simple_fit, sfreq):
        signal, _, fit = simple_fit
        result = state_space_decompose(signal, sfreq, fit, ar_order=10)
        assert isinstance(result, StateSpaceDecomposition)
        assert result.aperiodic.shape == signal.shape
        assert result.measurement_noise.shape == signal.shape
        assert np.isfinite(result.log_likelihood)

    def test_components_sum_to_signal(self, simple_fit, sfreq):
        signal, _, fit = simple_fit
        result = state_space_decompose(signal, sfreq, fit, ar_order=10)
        reconstructed = (
            np.sum(result.oscillators, axis=0)
            + result.aperiodic
            + result.measurement_noise
        )
        np.testing.assert_allclose(reconstructed, signal, atol=1e-8)

    def test_oscillator_at_correct_frequency(self, sfreq):
        """The oscillator should capture the alpha peak."""
        n_samples = int(10 * sfreq)
        pink = make_pink_noise(n_samples, sfreq, exponent_half=0.75)
        t = np.arange(n_samples) / sfreq
        alpha = 5.0 * np.sin(2 * np.pi * 10 * t)
        signal = pink + alpha

        decomposition = wavelet_decompose(signal, sfreq)
        fit = time_resolved_fit(decomposition, fit_stride=50)
        result = state_space_decompose(signal, sfreq, fit, ar_order=10)

        if result.model.n_oscillators > 0:
            osc_sum = np.sum(result.oscillators, axis=0)
            edge = int(1.0 * sfreq)
            osc_trimmed = osc_sum[edge:-edge]
            freqs = np.fft.rfftfreq(len(osc_trimmed), d=1.0 / sfreq)
            psd = np.abs(np.fft.rfft(osc_trimmed)) ** 2
            peak_freq = freqs[np.argmax(psd[1:]) + 1]
            assert abs(peak_freq - 10.0) < 3.0, (
                f"Oscillator should peak near 10 Hz, got {peak_freq:.1f} Hz"
            )

    def test_aperiodic_spectrum_broadband(self, sfreq):
        """The aperiodic component should be broadband (not oscillatory)."""
        n_samples = int(10 * sfreq)
        pink = make_pink_noise(n_samples, sfreq, exponent_half=0.75)
        t = np.arange(n_samples) / sfreq
        alpha = 5.0 * np.sin(2 * np.pi * 10 * t)
        signal = pink + alpha

        decomposition = wavelet_decompose(signal, sfreq)
        fit = time_resolved_fit(decomposition, fit_stride=50)
        result = state_space_decompose(signal, sfreq, fit, ar_order=10)

        edge = int(1.0 * sfreq)
        ap_trimmed = result.aperiodic[edge:-edge]
        freqs = np.fft.rfftfreq(len(ap_trimmed), d=1.0 / sfreq)
        psd = np.abs(np.fft.rfft(ap_trimmed)) ** 2

        idx_10 = np.argmin(np.abs(freqs - 10))
        idx_5 = np.argmin(np.abs(freqs - 5))
        idx_20 = np.argmin(np.abs(freqs - 20))
        window = 5
        psd_10 = np.mean(psd[max(0, idx_10 - window):idx_10 + window])
        psd_5 = np.mean(psd[max(0, idx_5 - window):idx_5 + window])
        psd_20 = np.mean(psd[max(0, idx_20 - window):idx_20 + window])

        ratio_5_to_10 = psd_5 / max(psd_10, 1e-30)
        ratio_20_to_10 = psd_20 / max(psd_10, 1e-30)
        assert ratio_5_to_10 < 100, "Aperiodic should not have a huge dip at 10 Hz"


class TestEndToEnd:
    def test_returns_reconstruction_result(self, simple_fit, sfreq):
        signal, _, _ = simple_fit
        result = meeglet_specparam_reconstruct(
            signal, sfreq, component="aperiodic", fit_stride=50,
            separation="state_space",
        )
        assert isinstance(result, ReconstructionResult)
        assert result.method == "state_space"

    def test_component_aperiodic(self, simple_fit, sfreq):
        signal, _, _ = simple_fit
        result = meeglet_specparam_reconstruct(
            signal, sfreq, component="aperiodic", fit_stride=50,
            separation="state_space",
        )
        assert result.reconstruction.shape == signal.shape
        assert result.bias_estimate is not None
        np.testing.assert_allclose(result.bias_estimate, 1.0)

    def test_method_field(self, simple_fit, sfreq):
        signal, _, _ = simple_fit
        result = meeglet_specparam_reconstruct(
            signal, sfreq, component="aperiodic", fit_stride=50,
            separation="state_space",
        )
        assert result.method == "state_space"

"""Tests for synthesis module."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tests.conftest import make_pink_noise
from meeglet_specparam_weights.wavelet_analysis import wavelet_decompose
from meeglet_specparam_weights.weight_surface import WeightSurface
from meeglet_specparam_weights.synthesis import synthesize


@pytest.fixture
def sine_decomposition(sfreq):
    """Decomposition of a 10 Hz sine wave."""
    n_samples = int(4 * sfreq)
    t = np.arange(n_samples) / sfreq
    signal = np.sin(2 * np.pi * 10 * t)
    return wavelet_decompose(signal, sfreq, foi_start=2, foi_end=32), signal


@pytest.fixture
def pink_decomposition(sfreq):
    """Decomposition of pink noise."""
    signal = make_pink_noise(int(4 * sfreq), sfreq, exponent_half=0.75)
    return wavelet_decompose(signal, sfreq, foi_start=2, foi_end=32), signal


def _unity_weights(decomp):
    """Create a WeightSurface of all ones (identity transform)."""
    return WeightSurface(
        weights=np.ones_like(decomp.coefficients, dtype=np.float64),
        component="full",
        eps=1e-20,
        max_weight=100.0,
    )


class TestUnweightedReconstruction:
    """With weights=1, synthesis should approximate the original signal."""

    def test_reconstruction_shape(self, sine_decomposition, sfreq):
        decomp, signal = sine_decomposition
        ws = _unity_weights(decomp)
        recon, energy_ratio, _ = synthesize(decomp, ws)
        assert recon.shape == signal.shape

    def test_reconstruction_real(self, sine_decomposition, sfreq):
        decomp, signal = sine_decomposition
        ws = _unity_weights(decomp)
        recon, _, _ = synthesize(decomp, ws)
        assert not np.iscomplexobj(recon)

    def test_energy_ratio_reported(self, sine_decomposition, sfreq):
        decomp, signal = sine_decomposition
        ws = _unity_weights(decomp)
        _, energy_ratio, _ = synthesize(decomp, ws)
        assert isinstance(energy_ratio, float)
        assert energy_ratio > 0

    def test_sine_reconstruction_correlation(self, sine_decomposition, sfreq):
        """Unweighted synthesis of sine should correlate well with original."""
        decomp, signal = sine_decomposition
        ws = _unity_weights(decomp)
        recon, _, _ = synthesize(decomp, ws, edge_taper=True)

        # Use interior samples (avoid edge taper region)
        edge = int(0.5 * sfreq)
        corr = np.corrcoef(signal[edge:-edge], recon[edge:-edge])[0, 1]
        assert corr > 0.5, f"Correlation {corr:.3f} too low for sine reconstruction"

    def test_pink_reconstruction_correlation(self, pink_decomposition, sfreq):
        """Unweighted synthesis of broadband signal should correlate with original."""
        decomp, signal = pink_decomposition
        ws = _unity_weights(decomp)
        recon, _, _ = synthesize(decomp, ws, edge_taper=True)

        edge = int(0.5 * sfreq)
        corr = np.corrcoef(signal[edge:-edge], recon[edge:-edge])[0, 1]
        assert corr > 0.3, f"Correlation {corr:.3f} too low for pink reconstruction"


class TestEnergyRatio:
    """Verify energy ratio is reasonable."""

    def test_energy_ratio_sane(self, sine_decomposition, sfreq):
        decomp, signal = sine_decomposition
        ws = _unity_weights(decomp)
        _, energy_ratio, _ = synthesize(decomp, ws)
        assert energy_ratio > 0, "Energy ratio should be positive"

    def test_frame_condition_positive(self, sine_decomposition, sfreq):
        decomp, signal = sine_decomposition
        ws = _unity_weights(decomp)
        _, _, frame_condition = synthesize(decomp, ws)
        assert frame_condition >= 1.0, "Frame condition B/A must be >= 1"

    def test_zero_weights_zero_energy(self, sine_decomposition, sfreq):
        decomp, signal = sine_decomposition
        ws = WeightSurface(
            weights=np.zeros_like(decomp.coefficients, dtype=np.float64),
            component="full",
            eps=1e-20,
            max_weight=100.0,
        )
        recon, energy_ratio, _ = synthesize(decomp, ws)
        assert np.allclose(recon, 0.0)


class TestPhasePreservation:
    """Verify phase is preserved through synthesis."""

    def test_phase_lag_near_zero(self, sfreq):
        """Cross-correlation peak at lag 0 for sine through analysis+synthesis."""
        n_samples = int(4 * sfreq)
        t = np.arange(n_samples) / sfreq
        signal = np.sin(2 * np.pi * 10 * t)

        decomp = wavelet_decompose(signal, sfreq, foi_start=2, foi_end=32)
        ws = _unity_weights(decomp)
        recon, _, _ = synthesize(decomp, ws, edge_taper=True)

        edge = int(0.5 * sfreq)
        sig_interior = signal[edge:-edge]
        rec_interior = recon[edge:-edge]

        # Normalize
        sig_interior = sig_interior / np.std(sig_interior)
        rec_interior = rec_interior / np.std(rec_interior)

        xcorr = np.correlate(sig_interior, rec_interior, mode="full")
        lags = np.arange(-len(sig_interior) + 1, len(sig_interior))
        peak_lag = lags[np.argmax(xcorr)]

        max_lag_samples = int(0.02 * sfreq)  # 20ms tolerance
        assert abs(peak_lag) <= max_lag_samples, (
            f"Phase lag {peak_lag} samples, expected within ±{max_lag_samples}"
        )


class TestMultiChannel:
    """Verify multi-channel synthesis."""

    def test_multichannel_reconstruction_shape(self, sfreq):
        n_samples = int(4 * sfreq)
        t = np.arange(n_samples) / sfreq
        signal_2d = np.stack([
            np.sin(2 * np.pi * 10 * t),
            np.sin(2 * np.pi * 20 * t),
        ])
        decomp = wavelet_decompose(signal_2d, sfreq, foi_start=2, foi_end=32)
        ws = WeightSurface(
            weights=np.ones_like(decomp.coefficients, dtype=np.float64),
            component="full", eps=1e-20, max_weight=100.0,
        )
        recon, energy_ratio, _ = synthesize(decomp, ws)
        assert recon.shape == (2, n_samples)
        assert energy_ratio > 0

    def test_multichannel_per_channel_correctness(self, sfreq):
        n_samples = int(4 * sfreq)
        t = np.arange(n_samples) / sfreq
        ch0 = np.sin(2 * np.pi * 10 * t)
        ch1 = np.sin(2 * np.pi * 20 * t)
        signal_2d = np.stack([ch0, ch1])

        decomp_multi = wavelet_decompose(signal_2d, sfreq, foi_start=2, foi_end=32)
        decomp_ch0 = wavelet_decompose(ch0, sfreq, foi_start=2, foi_end=32)

        ws_multi = WeightSurface(
            weights=np.ones_like(decomp_multi.coefficients, dtype=np.float64),
            component="full", eps=1e-20, max_weight=100.0,
        )
        ws_ch0 = WeightSurface(
            weights=np.ones_like(decomp_ch0.coefficients, dtype=np.float64),
            component="full", eps=1e-20, max_weight=100.0,
        )

        recon_multi, _, _ = synthesize(decomp_multi, ws_multi)
        recon_single, _, _ = synthesize(decomp_ch0, ws_ch0)

        np.testing.assert_array_almost_equal(recon_multi[0], recon_single)

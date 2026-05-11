"""Tests for pipeline module."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tests.conftest import make_pink_noise
from meeglet_specparam_weights.pipeline import (
    ReconstructionResult,
    meeglet_specparam_reconstruct,
)


@pytest.fixture
def pink_plus_alpha_signal(sfreq):
    """Pink noise + 10 Hz sine, with references."""
    n_samples = int(4 * sfreq)
    pink = make_pink_noise(n_samples, sfreq, exponent_half=0.75)
    t = np.arange(n_samples) / sfreq
    alpha = 3.0 * np.sin(2 * np.pi * 10 * t)
    return pink + alpha, pink, alpha


class TestPipelineOutput:
    """Verify pipeline output structure."""

    def test_output_type(self, pink_plus_alpha_signal, sfreq):
        signal, _, _ = pink_plus_alpha_signal
        result = meeglet_specparam_reconstruct(
            signal, sfreq, component="aperiodic", fit_stride=50
        )
        assert isinstance(result, ReconstructionResult)

    def test_reconstruction_shape(self, pink_plus_alpha_signal, sfreq):
        signal, _, _ = pink_plus_alpha_signal
        result = meeglet_specparam_reconstruct(
            signal, sfreq, component="aperiodic", fit_stride=50
        )
        assert result.reconstruction.shape == signal.shape

    def test_residual_shape(self, pink_plus_alpha_signal, sfreq):
        signal, _, _ = pink_plus_alpha_signal
        result = meeglet_specparam_reconstruct(
            signal, sfreq, component="aperiodic", fit_stride=50
        )
        assert result.residual.shape == signal.shape

    def test_residual_equals_original_minus_recon(self, pink_plus_alpha_signal, sfreq):
        signal, _, _ = pink_plus_alpha_signal
        result = meeglet_specparam_reconstruct(
            signal, sfreq, component="aperiodic", fit_stride=50
        )
        np.testing.assert_allclose(
            result.residual, signal - result.reconstruction, atol=1e-10
        )

    def test_energy_ratio_positive(self, pink_plus_alpha_signal, sfreq):
        signal, _, _ = pink_plus_alpha_signal
        result = meeglet_specparam_reconstruct(
            signal, sfreq, component="aperiodic", fit_stride=50
        )
        assert result.energy_ratio > 0

    def test_subcomponents_present(self, pink_plus_alpha_signal, sfreq):
        signal, _, _ = pink_plus_alpha_signal
        result = meeglet_specparam_reconstruct(
            signal, sfreq, component="aperiodic", fit_stride=50
        )
        assert result.fit is not None
        assert result.weights is not None
        assert result.decomposition is not None


class TestAperiodicReconstruction:
    """Verify aperiodic reconstruction captures the 1/f component."""

    def test_aperiodic_rms_similar_to_pink(self, pink_plus_alpha_signal, sfreq):
        """Aperiodic reconstruction RMS should be in the ballpark of pink-only RMS."""
        signal, pink, alpha = pink_plus_alpha_signal
        result = meeglet_specparam_reconstruct(
            signal, sfreq, component="aperiodic", fit_stride=50
        )

        # Interior samples
        edge = int(0.5 * sfreq)
        rms_pink = np.sqrt(np.mean(pink[edge:-edge] ** 2))
        rms_recon = np.sqrt(np.mean(result.reconstruction[edge:-edge] ** 2))

        ratio = rms_recon / rms_pink
        assert 0.05 < ratio < 20.0, (
            f"RMS ratio {ratio:.2f} (recon={rms_recon:.4f}, pink={rms_pink:.4f})"
        )


class TestResidualContent:
    """Verify residual contains oscillatory content."""

    def test_residual_has_alpha_power(self, pink_plus_alpha_signal, sfreq):
        """After removing aperiodic, residual should contain 10 Hz."""
        signal, _, _ = pink_plus_alpha_signal
        result = meeglet_specparam_reconstruct(
            signal, sfreq, component="aperiodic", fit_stride=50
        )

        edge = int(0.5 * sfreq)
        residual = result.residual[edge:-edge]

        # Check spectrum of residual
        n = len(residual)
        freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
        spectrum = np.abs(np.fft.rfft(residual)) ** 2

        idx_10 = np.argmin(np.abs(freqs - 10))
        idx_25 = np.argmin(np.abs(freqs - 25))

        power_band_10 = np.mean(spectrum[max(0, idx_10 - 2):idx_10 + 3])
        power_band_25 = np.mean(spectrum[max(0, idx_25 - 2):idx_25 + 3])

        assert power_band_10 > power_band_25, (
            f"Residual should have more 10 Hz than 25 Hz power: "
            f"10Hz={power_band_10:.2f}, 25Hz={power_band_25:.2f}"
        )


class TestPeriodicReconstruction:
    """Verify periodic reconstruction captures oscillations."""

    def test_periodic_has_alpha(self, pink_plus_alpha_signal, sfreq):
        signal, _, _ = pink_plus_alpha_signal
        result = meeglet_specparam_reconstruct(
            signal, sfreq, component="periodic", fit_stride=50
        )

        edge = int(0.5 * sfreq)
        recon = result.reconstruction[edge:-edge]

        n = len(recon)
        freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
        spectrum = np.abs(np.fft.rfft(recon)) ** 2

        idx_10 = np.argmin(np.abs(freqs - 10))
        idx_25 = np.argmin(np.abs(freqs - 25))

        power_band_10 = np.mean(spectrum[max(0, idx_10 - 2):idx_10 + 3])
        power_band_25 = np.mean(spectrum[max(0, idx_25 - 2):idx_25 + 3])

        assert power_band_10 > power_band_25, (
            f"Periodic recon should have 10 Hz peak: "
            f"10Hz={power_band_10:.2f}, 25Hz={power_band_25:.2f}"
        )


class TestSubtractionMethod:
    """Verify subtraction-based aperiodic decomposition (Phase 16)."""

    def test_method_field_subtraction(self, pink_plus_alpha_signal, sfreq):
        signal, _, _ = pink_plus_alpha_signal
        result = meeglet_specparam_reconstruct(
            signal, sfreq, component="aperiodic", fit_stride=50
        )
        assert result.method == "subtraction"

    def test_method_field_wiener(self, pink_plus_alpha_signal, sfreq):
        signal, _, _ = pink_plus_alpha_signal
        result = meeglet_specparam_reconstruct(
            signal, sfreq, component="aperiodic", fit_stride=50,
            aperiodic_method="wiener",
        )
        assert result.method == "wiener"

    def test_method_field_periodic(self, pink_plus_alpha_signal, sfreq):
        signal, _, _ = pink_plus_alpha_signal
        result = meeglet_specparam_reconstruct(
            signal, sfreq, component="periodic", fit_stride=50
        )
        assert result.method == "weight"

    def test_decomposition_sums_to_original(self, pink_plus_alpha_signal, sfreq):
        """periodic_recon + aperiodic_recon = original signal."""
        signal, _, _ = pink_plus_alpha_signal
        result = meeglet_specparam_reconstruct(
            signal, sfreq, component="aperiodic", fit_stride=50
        )
        np.testing.assert_allclose(
            result.reconstruction + result.residual, signal, atol=1e-10
        )

    def test_periodic_contains_only_excess(self, pink_plus_alpha_signal, sfreq):
        """The periodic residual (from aperiodic subtraction) should peak at alpha."""
        signal, _, _ = pink_plus_alpha_signal
        result = meeglet_specparam_reconstruct(
            signal, sfreq, component="aperiodic", fit_stride=50
        )
        edge = int(0.5 * sfreq)
        periodic = result.residual[edge:-edge]

        n = len(periodic)
        freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
        spectrum = np.abs(np.fft.rfft(periodic)) ** 2

        idx_10 = np.argmin(np.abs(freqs - 10))
        idx_5 = np.argmin(np.abs(freqs - 5))
        assert spectrum[idx_10] > spectrum[idx_5] * 10, (
            "Periodic residual should have strong 10 Hz peak"
        )

    def test_excess_weights_bounded(self, pink_plus_alpha_signal, sfreq):
        """Excess weights used for subtraction must be in [0, 1]."""
        signal, _, _ = pink_plus_alpha_signal
        result = meeglet_specparam_reconstruct(
            signal, sfreq, component="aperiodic", fit_stride=50
        )
        assert result.weights.weights.min() >= 0.0
        assert result.weights.weights.max() <= 1.0

    def test_invalid_aperiodic_method_raises(self, pink_plus_alpha_signal, sfreq):
        signal, _, _ = pink_plus_alpha_signal
        with pytest.raises(ValueError, match="aperiodic_method"):
            meeglet_specparam_reconstruct(
                signal, sfreq, component="aperiodic", fit_stride=50,
                aperiodic_method="invalid",
            )


class TestMultiChannelPipeline:
    """Verify multi-channel end-to-end pipeline."""

    def test_multichannel_output_shapes(self, sfreq):
        rng = np.random.default_rng(42)
        n_samples = int(4 * sfreq)
        signal_2d = rng.standard_normal((2, n_samples))

        result = meeglet_specparam_reconstruct(
            signal_2d, sfreq, component="aperiodic", fit_stride=50
        )

        assert result.reconstruction.shape == (2, n_samples)
        assert result.residual.shape == (2, n_samples)
        assert result.energy_ratio > 0

    def test_multichannel_residual_equals_diff(self, sfreq):
        rng = np.random.default_rng(42)
        n_samples = int(4 * sfreq)
        signal_2d = rng.standard_normal((2, n_samples))

        result = meeglet_specparam_reconstruct(
            signal_2d, sfreq, component="aperiodic", fit_stride=50
        )

        np.testing.assert_allclose(
            result.residual, signal_2d - result.reconstruction, atol=1e-10
        )

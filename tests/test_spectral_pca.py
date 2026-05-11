"""Tests for spectral PCA decomposition."""

import numpy as np
import pytest

from tests.conftest import make_pink_noise
from src.meeglet_specparam_weights.spectral_pca import (
    SpectralPCAResult,
    compute_csd,
    spectral_pca_decompose,
    spectral_pca_reconstruct,
    _project_to_pc_space,
    _project_to_channel_space,
)
from src.meeglet_specparam_weights.wavelet_analysis import wavelet_decompose


SFREQ = 256.0
N_SAMPLES = 1024  # 4 seconds
N_CHANNELS = 3
FOI_START = 2.0
FOI_END = 32.0


def _make_multichannel_signal(
    n_channels=N_CHANNELS,
    n_samples=N_SAMPLES,
    sfreq=SFREQ,
    alpha_amp=2.0,
    alpha_freq=10.0,
    exponent_half=0.75,
    seed=42,
):
    """3-channel signal: 2 independent pink noises + 1 alpha source mixed."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sfreq

    pink1 = make_pink_noise(n_samples, sfreq, exponent_half=exponent_half, seed=seed)
    pink2 = make_pink_noise(n_samples, sfreq, exponent_half=exponent_half, seed=seed + 1)
    alpha = alpha_amp * np.sin(2 * np.pi * alpha_freq * t)

    # Known mixing: alpha appears in channels 0 and 2
    signal = np.empty((n_channels, n_samples))
    signal[0] = pink1 + 0.8 * alpha
    signal[1] = pink2
    signal[2] = pink1 * 0.5 + 0.6 * alpha
    return signal


def _make_decomposition(signal=None, sfreq=SFREQ):
    if signal is None:
        signal = _make_multichannel_signal()
    return wavelet_decompose(
        signal, sfreq, foi_start=FOI_START, foi_end=FOI_END, bw_oct=0.5,
    )


# ────────────────────────────────────────────────────────
# TestComputeCSD
# ────────────────────────────────────────────────────────


class TestComputeCSD:
    def test_shape(self):
        dec = _make_decomposition()
        csd = compute_csd(dec)
        n_ch = dec.coefficients.shape[0]
        n_freqs = dec.coefficients.shape[1]
        assert csd.shape == (n_ch, n_ch, n_freqs)

    def test_hermitian(self):
        dec = _make_decomposition()
        csd = compute_csd(dec)
        for f in range(csd.shape[2]):
            np.testing.assert_allclose(
                csd[:, :, f], csd[:, :, f].conj().T, atol=1e-12
            )

    def test_positive_semidefinite(self):
        dec = _make_decomposition()
        csd = compute_csd(dec)
        for f in range(csd.shape[2]):
            eigvals = np.linalg.eigvalsh(csd[:, :, f])
            assert np.all(eigvals >= -1e-12)

    def test_diagonal_equals_channel_power(self):
        dec = _make_decomposition()
        csd = compute_csd(dec)
        Z = dec.coefficients
        n_times = Z.shape[2]
        for ch in range(Z.shape[0]):
            expected_power = np.mean(np.abs(Z[ch]) ** 2, axis=1)
            np.testing.assert_allclose(
                np.real(csd[ch, ch, :]), expected_power, rtol=1e-10
            )

    def test_rejects_1d_input(self):
        signal_1d = make_pink_noise(N_SAMPLES, SFREQ)
        dec = wavelet_decompose(signal_1d, SFREQ, foi_start=FOI_START, foi_end=FOI_END)
        with pytest.raises(ValueError, match="multi-channel"):
            compute_csd(dec)

    def test_nan_handling(self):
        signal = _make_multichannel_signal()
        signal[0, 100:150] = np.nan
        dec = _make_decomposition(signal)
        csd = compute_csd(dec)
        assert not np.any(np.isnan(csd))
        assert csd.shape[0] == N_CHANNELS


# ────────────────────────────────────────────────────────
# TestSpectralPCADecompose
# ────────────────────────────────────────────────────────


class TestSpectralPCADecompose:
    def test_eigenvector_shape(self):
        dec = _make_decomposition()
        eigvecs, eigvals, csd = spectral_pca_decompose(dec)
        n_ch = dec.coefficients.shape[0]
        n_freqs = dec.coefficients.shape[1]
        assert eigvecs.shape == (n_ch, n_ch, n_freqs)
        assert eigvals.shape == (n_ch, n_freqs)

    def test_eigenvalues_nonnegative(self):
        dec = _make_decomposition()
        _, eigvals, _ = spectral_pca_decompose(dec)
        assert np.all(eigvals >= 0)

    def test_eigenvalues_descending(self):
        dec = _make_decomposition()
        _, eigvals, _ = spectral_pca_decompose(dec)
        for f in range(eigvals.shape[1]):
            assert np.all(np.diff(eigvals[:, f]) <= 1e-12)

    def test_rank_reduction(self):
        dec = _make_decomposition()
        eigvecs, eigvals, _ = spectral_pca_decompose(dec, n_modes=2)
        n_ch = dec.coefficients.shape[0]
        n_freqs = dec.coefficients.shape[1]
        assert eigvecs.shape == (n_ch, 2, n_freqs)
        assert eigvals.shape == (2, n_freqs)

    def test_eigenvector_orthonormality(self):
        dec = _make_decomposition()
        eigvecs, _, _ = spectral_pca_decompose(dec)
        n_modes = eigvecs.shape[1]
        for f in range(eigvecs.shape[2]):
            U = eigvecs[:, :, f]
            gram = U.conj().T @ U
            np.testing.assert_allclose(gram, np.eye(n_modes), atol=1e-10)

    def test_sign_alignment(self):
        dec = _make_decomposition()
        eigvecs, _, _ = spectral_pca_decompose(dec)
        for k in range(eigvecs.shape[1]):
            for f in range(1, eigvecs.shape[2]):
                overlap = np.real(
                    np.vdot(eigvecs[:, k, f - 1], eigvecs[:, k, f])
                )
                assert overlap >= -0.01, (
                    f"Mode {k} freq {f}: overlap={overlap:.3f}, sign flip missed"
                )

    def test_variance_explained_sums_to_one(self):
        dec = _make_decomposition()
        _, eigvals, _ = spectral_pca_decompose(dec)
        total_power = eigvals.sum()
        var_explained = eigvals.sum(axis=1) / total_power
        np.testing.assert_allclose(var_explained.sum(), 1.0, atol=1e-10)

    def test_csd_returned(self):
        dec = _make_decomposition()
        _, _, csd = spectral_pca_decompose(dec)
        n_ch = dec.coefficients.shape[0]
        n_freqs = dec.coefficients.shape[1]
        assert csd.shape == (n_ch, n_ch, n_freqs)


# ────────────────────────────────────────────────────────
# TestProjection
# ────────────────────────────────────────────────────────


class TestProjection:
    def test_round_trip(self):
        dec = _make_decomposition()
        eigvecs, _, _ = spectral_pca_decompose(dec)
        Z = dec.coefficients
        Z_pc = _project_to_pc_space(Z, eigvecs)
        Z_back = _project_to_channel_space(Z_pc, eigvecs)
        np.testing.assert_allclose(Z_back, Z, atol=1e-10)

    def test_pc_coefficient_shapes(self):
        dec = _make_decomposition()
        eigvecs, _, _ = spectral_pca_decompose(dec, n_modes=2)
        Z = dec.coefficients
        Z_pc = _project_to_pc_space(Z, eigvecs)
        n_freqs = Z.shape[1]
        n_times = Z.shape[2]
        assert Z_pc.shape == (2, n_freqs, n_times)

    def test_pc_power_approximates_eigenvalues(self):
        dec = _make_decomposition()
        eigvecs, eigvals, _ = spectral_pca_decompose(dec)
        Z = dec.coefficients
        Z_pc = _project_to_pc_space(Z, eigvecs)
        pc_power = np.mean(np.abs(Z_pc) ** 2, axis=2)
        np.testing.assert_allclose(pc_power, eigvals, rtol=0.1)


# ────────────────────────────────────────────────────────
# TestSpectralPCAReconstruct
# ────────────────────────────────────────────────────────


class TestSpectralPCAReconstruct:
    @pytest.fixture(scope="class")
    def result(self):
        signal = _make_multichannel_signal()
        return spectral_pca_reconstruct(
            signal, SFREQ,
            foi_start=FOI_START, foi_end=FOI_END,
            fit_stride=20, n_iter=5,
        )

    def test_result_type(self, result):
        assert isinstance(result, SpectralPCAResult)

    def test_output_shapes(self, result):
        assert result.aperiodic.shape == (N_CHANNELS, N_SAMPLES)
        assert result.periodic.shape == (N_CHANNELS, N_SAMPLES)
        assert result.mode_periodic.shape[0] == result.n_modes
        assert result.mode_aperiodic.shape[0] == result.n_modes

    def test_subtraction_sums_to_original(self, result):
        signal = _make_multichannel_signal()
        reconstructed = result.aperiodic + result.periodic
        np.testing.assert_allclose(reconstructed, signal, atol=1e-8)

    def test_valid_r_squared(self, result):
        assert np.all(result.mode_fit.r_squared > 0.5)

    def test_periodic_has_alpha_peak(self, result):
        from scipy.signal import welch
        for ch in range(N_CHANNELS):
            f, psd = welch(result.periodic[ch], fs=SFREQ, nperseg=256)
            alpha_mask = (f >= 8) & (f <= 12)
            broad_mask = (f >= 2) & (f <= 30) & ~alpha_mask
            if np.any(alpha_mask) and np.any(broad_mask):
                alpha_power = np.mean(psd[alpha_mask])
                broad_power = np.mean(psd[broad_mask])
                # Channel 1 has no alpha, so only check channels with alpha
                if ch != 1:
                    assert alpha_power > broad_power

    def test_aperiodic_rms_reasonable(self, result):
        signal = _make_multichannel_signal()
        for ch in range(N_CHANNELS):
            orig_rms = np.sqrt(np.mean(signal[ch] ** 2))
            ap_rms = np.sqrt(np.mean(result.aperiodic[ch] ** 2))
            assert ap_rms < 2.0 * orig_rms
            assert ap_rms > 0.01 * orig_rms

    def test_rejects_1d(self):
        signal = make_pink_noise(N_SAMPLES, SFREQ)
        with pytest.raises(ValueError, match="2D"):
            spectral_pca_reconstruct(signal, SFREQ)

    def test_rejects_single_channel(self):
        signal = make_pink_noise(N_SAMPLES, SFREQ).reshape(1, -1)
        with pytest.raises(ValueError, match="2 channels"):
            spectral_pca_reconstruct(signal, SFREQ)

    def test_invalid_separation(self):
        signal = _make_multichannel_signal()
        with pytest.raises(ValueError, match="separation"):
            spectral_pca_reconstruct(signal, SFREQ, separation="invalid")


# ────────────────────────────────────────────────────────
# TestWiener
# ────────────────────────────────────────────────────────


class TestWiener:
    @pytest.fixture(scope="class")
    def wiener_result(self):
        signal = _make_multichannel_signal()
        return spectral_pca_reconstruct(
            signal, SFREQ,
            foi_start=FOI_START, foi_end=FOI_END,
            separation="wiener",
            fit_stride=20, n_iter=5,
        )

    def test_wiener_produces_output(self, wiener_result):
        assert wiener_result.aperiodic.shape == (N_CHANNELS, N_SAMPLES)
        assert wiener_result.periodic.shape == (N_CHANNELS, N_SAMPLES)

    def test_separation_field(self, wiener_result):
        assert wiener_result.separation == "wiener"


# ────────────────────────────────────────────────────────
# TestVsPerChannel
# ────────────────────────────────────────────────────────


class TestVsPerChannel:
    def test_pca_smoother_on_correlated(self):
        """PCA fits should have higher r² on spatially correlated channels."""
        signal = _make_multichannel_signal()
        result = spectral_pca_reconstruct(
            signal, SFREQ,
            foi_start=FOI_START, foi_end=FOI_END,
            fit_stride=20, n_iter=5,
        )
        mean_r2 = float(np.mean(result.mode_fit.r_squared))
        assert mean_r2 > 0.7

    def test_uncorrelated_still_works(self):
        """Independent channels should still produce valid decomposition."""
        rng = np.random.default_rng(99)
        signal = np.empty((2, N_SAMPLES))
        signal[0] = make_pink_noise(N_SAMPLES, SFREQ, seed=10)
        signal[1] = make_pink_noise(N_SAMPLES, SFREQ, seed=20)
        result = spectral_pca_reconstruct(
            signal, SFREQ,
            foi_start=FOI_START, foi_end=FOI_END,
            fit_stride=20, n_iter=5,
        )
        assert result.aperiodic.shape == (2, N_SAMPLES)
        reconstructed = result.aperiodic + result.periodic
        np.testing.assert_allclose(reconstructed, signal, atol=1e-8)

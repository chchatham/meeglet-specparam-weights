"""Tests for multi-epoch ensemble estimation."""

import numpy as np
import pytest

from tests.conftest import make_pink_noise
from src.meeglet_specparam_weights.epochs import (
    EpochDecompositionResult,
    ensemble_decompose,
    meeglet_specparam_reconstruct_epochs,
)


SFREQ = 256.0
N_SAMPLES = 512  # 2 seconds
N_EPOCHS = 8


def _make_epochs(
    n_epochs=N_EPOCHS,
    n_samples=N_SAMPLES,
    sfreq=SFREQ,
    exponent_half=0.75,
    alpha_amp=2.0,
    alpha_freq=10.0,
    seed=42,
):
    """Create epochs of pink noise + alpha sine with random phases."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sfreq
    epochs = np.empty((n_epochs, n_samples))
    for k in range(n_epochs):
        pink = make_pink_noise(n_samples, sfreq, exponent_half=exponent_half, seed=seed + k)
        phase = rng.uniform(0, 2 * np.pi)
        alpha = alpha_amp * np.sin(2 * np.pi * alpha_freq * t + phase)
        epochs[k] = pink + alpha
    return epochs


def _make_evoked_epochs(
    n_epochs=N_EPOCHS,
    n_samples=N_SAMPLES,
    sfreq=SFREQ,
    seed=42,
):
    """Create epochs with a phase-locked evoked response + random noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sfreq
    t_center = t[n_samples // 4]
    evoked = 3.0 * np.exp(-((t - t_center) ** 2) / (2 * 0.02 ** 2))

    epochs = np.empty((n_epochs, n_samples))
    for k in range(n_epochs):
        pink = make_pink_noise(n_samples, sfreq, exponent_half=0.75, seed=seed + k)
        phase = rng.uniform(0, 2 * np.pi)
        alpha = 1.0 * np.sin(2 * np.pi * 10 * t + phase)
        epochs[k] = evoked + pink + alpha
    return epochs, evoked


# --- ensemble_decompose ---


class TestEnsembleDecompose:
    def test_returns_correct_types(self):
        epochs = _make_epochs(n_epochs=4)
        decompositions, ensemble_power = ensemble_decompose(epochs, SFREQ)
        assert isinstance(decompositions, list)
        assert len(decompositions) == 4
        assert isinstance(ensemble_power, np.ndarray)

    def test_shapes(self):
        epochs = _make_epochs(n_epochs=4)
        decompositions, ensemble_power = ensemble_decompose(epochs, SFREQ)
        n_freqs = len(decompositions[0].foi)
        n_times = decompositions[0].coefficients.shape[1]
        assert ensemble_power.shape == (n_freqs, n_times)
        for dec in decompositions:
            assert dec.coefficients.shape == (n_freqs, n_times)

    def test_ensemble_power_is_average(self):
        epochs = _make_epochs(n_epochs=4)
        decompositions, ensemble_power = ensemble_decompose(epochs, SFREQ)
        manual_avg = np.mean(
            [np.abs(d.coefficients) ** 2 for d in decompositions], axis=0
        )
        np.testing.assert_allclose(ensemble_power, manual_avg, rtol=1e-10)

    def test_ensemble_power_smoother_than_single_trial(self):
        epochs = _make_epochs(n_epochs=16)
        decompositions, ensemble_power = ensemble_decompose(epochs, SFREQ)
        single_power = np.abs(decompositions[0].coefficients) ** 2
        ensemble_cv = np.std(ensemble_power, axis=1) / np.mean(ensemble_power, axis=1)
        single_cv = np.std(single_power, axis=1) / np.mean(single_power, axis=1)
        assert np.mean(ensemble_cv) < np.mean(single_cv)

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2D"):
            ensemble_decompose(np.zeros(100), SFREQ)

    def test_rejects_3d_input(self):
        with pytest.raises(ValueError, match="2D"):
            ensemble_decompose(np.zeros((3, 2, 100)), SFREQ)


# --- meeglet_specparam_reconstruct_epochs ---


class TestEpochReconstruction:
    @pytest.fixture
    def result(self):
        epochs = _make_epochs(n_epochs=6)
        return meeglet_specparam_reconstruct_epochs(
            epochs, SFREQ, separation="subtraction", n_iter=5,
        ), epochs

    def test_result_type(self, result):
        res, _ = result
        assert isinstance(res, EpochDecompositionResult)

    def test_output_shapes(self, result):
        res, epochs = result
        n_epochs, n_samples = epochs.shape
        assert res.aperiodic.shape == (n_epochs, n_samples)
        assert res.periodic.shape == (n_epochs, n_samples)
        assert res.energy_ratios.shape == (n_epochs,)
        assert res.evoked is None

    def test_subtraction_sums_to_epoch(self, result):
        res, epochs = result
        for k in range(epochs.shape[0]):
            reconstructed = res.aperiodic[k] + res.periodic[k]
            np.testing.assert_allclose(
                reconstructed, epochs[k], atol=1e-10,
                err_msg=f"Epoch {k}: aperiodic + periodic != original",
            )

    def test_ensemble_fit_has_valid_r_squared(self, result):
        res, _ = result
        valid_r2 = res.ensemble_fit.r_squared[~np.isnan(res.ensemble_fit.r_squared)]
        assert len(valid_r2) > 0
        assert np.mean(valid_r2) > 0.8

    def test_ensemble_power_shape(self, result):
        res, _ = result
        n_freqs = len(res.ensemble_fit.foi)
        assert res.ensemble_power.shape[0] == n_freqs

    def test_separation_field(self, result):
        res, _ = result
        assert res.separation == "subtraction"


class TestEpochExponentRecovery:
    def test_recovers_known_exponent(self):
        epochs = _make_epochs(exponent_half=0.75, alpha_amp=0.0)
        res = meeglet_specparam_reconstruct_epochs(
            epochs, SFREQ, separation="subtraction",
        )
        ap = res.ensemble_fit.aperiodic_params
        valid = ~np.isnan(ap[:, 1])
        median_exp = np.median(ap[valid, 1])
        assert 1.0 < median_exp < 2.2, f"Expected ~1.5, got {median_exp}"


class TestWienerEpochs:
    def test_wiener_produces_output(self):
        epochs = _make_epochs(n_epochs=4)
        res = meeglet_specparam_reconstruct_epochs(
            epochs, SFREQ, separation="wiener",
        )
        assert res.aperiodic.shape == epochs.shape
        assert res.separation == "wiener"

    def test_wiener_energy_ratios_sane(self):
        epochs = _make_epochs(n_epochs=4)
        res = meeglet_specparam_reconstruct_epochs(
            epochs, SFREQ, separation="wiener",
        )
        assert np.all(res.energy_ratios > 0)
        assert np.all(res.energy_ratios < 10)


class TestEvokedSeparation:
    @pytest.fixture
    def evoked_result(self):
        epochs, true_evoked = _make_evoked_epochs(n_epochs=10)
        res = meeglet_specparam_reconstruct_epochs(
            epochs, SFREQ, separation="subtraction",
            separate_evoked=True, n_iter=5,
        )
        return res, epochs, true_evoked

    def test_evoked_returned(self, evoked_result):
        res, _, _ = evoked_result
        assert res.evoked is not None
        assert res.evoked.ndim == 1

    def test_evoked_shape(self, evoked_result):
        res, epochs, _ = evoked_result
        assert res.evoked.shape == (epochs.shape[1],)

    def test_evoked_correlates_with_true(self, evoked_result):
        res, _, true_evoked = evoked_result
        interior = slice(N_SAMPLES // 4, 3 * N_SAMPLES // 4)
        corr = np.corrcoef(res.evoked[interior], true_evoked[interior])[0, 1]
        assert corr > 0.5, f"Evoked correlation with true: {corr:.3f}"

    def test_evoked_correlates_with_time_average(self, evoked_result):
        res, epochs, _ = evoked_result
        time_avg = np.mean(epochs, axis=0)
        interior = slice(N_SAMPLES // 4, 3 * N_SAMPLES // 4)
        corr = np.corrcoef(res.evoked[interior], time_avg[interior])[0, 1]
        assert corr > 0.7, f"Evoked vs time-avg correlation: {corr:.3f}"

    def test_induced_decomposition_shapes(self, evoked_result):
        res, epochs, _ = evoked_result
        n_epochs, n_samples = epochs.shape
        assert res.aperiodic.shape == (n_epochs, n_samples)
        assert res.periodic.shape == (n_epochs, n_samples)

    def test_induced_sums_to_induced_signal(self, evoked_result):
        res, epochs, _ = evoked_result
        for k in range(epochs.shape[0]):
            induced_k = epochs[k] - res.evoked
            reconstructed = res.aperiodic[k] + res.periodic[k]
            np.testing.assert_allclose(
                reconstructed, induced_k, atol=1e-10,
                err_msg=f"Epoch {k}: induced aperiodic + periodic != induced signal",
            )


class TestEpochValidation:
    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2D"):
            meeglet_specparam_reconstruct_epochs(np.zeros(100), SFREQ)

    def test_rejects_3d_input(self):
        with pytest.raises(ValueError, match="2D"):
            meeglet_specparam_reconstruct_epochs(np.zeros((2, 3, 100)), SFREQ)

    def test_invalid_separation(self):
        epochs = _make_epochs(n_epochs=2)
        with pytest.raises(ValueError, match="separation"):
            meeglet_specparam_reconstruct_epochs(
                epochs, SFREQ, separation="invalid"
            )


class TestPeriodicDetection:
    def test_periodic_has_alpha_power(self):
        epochs = _make_epochs(n_epochs=6, alpha_amp=3.0)
        res = meeglet_specparam_reconstruct_epochs(
            epochs, SFREQ, separation="subtraction", n_iter=5,
        )
        alpha_idx = np.argmin(np.abs(res.ensemble_fit.foi - 10.0))
        periodic_alpha_power = np.mean(
            [np.abs(np.fft.rfft(res.periodic[k])) ** 2 for k in range(6)],
            axis=0,
        )
        freqs = np.fft.rfftfreq(N_SAMPLES, d=1.0 / SFREQ)
        alpha_band = (freqs >= 8) & (freqs <= 12)
        non_alpha = (freqs >= 2) & (freqs < 8) | (freqs > 12) & (freqs <= 30)
        alpha_power = np.mean(periodic_alpha_power[alpha_band])
        baseline_power = np.mean(periodic_alpha_power[non_alpha])
        assert alpha_power > 2 * baseline_power

"""Multi-epoch ensemble estimation: stable specparam fits from trial-averaged power.

For epoched M/EEG data, single-trial wavelet power is noisy (chi-squared(2)
per time-frequency bin). Averaging power across trials before fitting specparam
yields much more stable aperiodic/periodic parameter estimates. The ensemble
fit is then applied to each individual epoch for per-trial reconstruction.

Optional evoked separation subtracts trial-averaged wavelet coefficients
(phase-locked activity) before fitting, so the decomposition targets only
the induced (non-phase-locked) part of the signal.
"""

from dataclasses import dataclass

import numpy as np

from .wavelet_analysis import WaveletDecomposition, wavelet_decompose
from .time_resolved_fit import TimeResolvedFit, time_resolved_fit
from .separation import subtraction_separate, wiener_separate
from .state_space import state_space_separate
from .weight_surface import WeightSurface
from .synthesis import synthesize


@dataclass
class EpochDecompositionResult:
    """Result of multi-epoch ensemble decomposition."""

    aperiodic: np.ndarray  # (n_epochs, n_samples)
    periodic: np.ndarray  # (n_epochs, n_samples)
    evoked: np.ndarray | None  # (n_samples,) if separate_evoked=True
    ensemble_fit: TimeResolvedFit
    ensemble_power: np.ndarray  # (n_freqs, n_times) trial-averaged power
    separation: str
    energy_ratios: np.ndarray  # (n_epochs,)


def ensemble_decompose(
    epochs: np.ndarray,
    sfreq: float,
    foi_start: float = 2.0,
    foi_end: float = 32.0,
    bw_oct: float = 0.5,
    delta_oct: float | None = None,
) -> tuple[list[WaveletDecomposition], np.ndarray]:
    """Wavelet-decompose each epoch and compute trial-averaged power.

    Parameters
    ----------
    epochs : np.ndarray
        (n_epochs, n_samples) input array.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    decompositions : list[WaveletDecomposition]
        One per epoch.
    ensemble_power : np.ndarray
        (n_freqs, n_times) trial-averaged |Z|².
    """
    epochs = np.asarray(epochs, dtype=np.float64)
    if epochs.ndim != 2:
        raise ValueError(
            f"epochs must be 2D (n_epochs, n_samples), got {epochs.ndim}D"
        )

    n_epochs = epochs.shape[0]
    decompositions = []
    ensemble_power = None

    for k in range(n_epochs):
        dec = wavelet_decompose(
            epochs[k],
            sfreq,
            foi_start=foi_start,
            foi_end=foi_end,
            bw_oct=bw_oct,
            delta_oct=delta_oct,
        )
        decompositions.append(dec)
        power_k = np.abs(dec.coefficients) ** 2
        if ensemble_power is None:
            ensemble_power = power_k.copy()
        else:
            ensemble_power += power_k

    ensemble_power /= n_epochs
    return decompositions, ensemble_power


def meeglet_specparam_reconstruct_epochs(
    epochs: np.ndarray,
    sfreq: float,
    separation: str = "subtraction",
    separate_evoked: bool = False,
    foi_start: float = 2.0,
    foi_end: float = 32.0,
    bw_oct: float = 0.5,
    delta_oct: float | None = None,
    fit_stride: int = 10,
    power_window: int | None = None,
    smooth_sigma: float | None = None,
    eps: float = 1e-20,
    max_weight: float = 100.0,
    freq_range: list[float] | None = None,
    peak_width_limits: tuple[float, float] = (0.5, 12.0),
    max_n_peaks: int = 8,
    min_peak_height: float = 0.0,
    aperiodic_mode: str = "fixed",
    edge_taper: bool = True,
    n_iter: int = 1,
) -> EpochDecompositionResult:
    """Ensemble specparam fit + per-epoch aperiodic/periodic reconstruction.

    Fits specparam to trial-averaged wavelet power for stable parameter
    estimation, then applies the ensemble fit to each individual epoch.

    Parameters
    ----------
    epochs : np.ndarray
        (n_epochs, n_samples) input. Each row is one epoch/trial.
    sfreq : float
        Sampling frequency in Hz.
    separation : str
        Separation strategy: 'subtraction', 'wiener', or 'state_space'.
    separate_evoked : bool
        If True, subtract trial-averaged wavelet coefficients (evoked/ERP)
        before fitting and reconstruction. The evoked signal is synthesized
        and returned separately. Only the induced (non-phase-locked) part
        is decomposed into aperiodic/periodic.
    """
    epochs = np.asarray(epochs, dtype=np.float64)
    if epochs.ndim != 2:
        raise ValueError(
            f"epochs must be 2D (n_epochs, n_samples), got {epochs.ndim}D"
        )

    valid_sep = ("subtraction", "wiener", "state_space")
    if separation not in valid_sep:
        raise ValueError(
            f"separation must be one of {valid_sep}, got '{separation}'"
        )

    n_epochs, n_samples = epochs.shape

    wavelet_kwargs = dict(
        foi_start=foi_start,
        foi_end=foi_end,
        bw_oct=bw_oct,
        delta_oct=delta_oct,
    )

    decompositions, ensemble_power = ensemble_decompose(
        epochs, sfreq, **wavelet_kwargs
    )

    evoked = None
    if separate_evoked:
        evoked_coefficients = np.mean(
            [d.coefficients for d in decompositions], axis=0
        )

        ref = decompositions[0]
        evoked_dec = WaveletDecomposition(
            coefficients=evoked_coefficients,
            foi=ref.foi,
            sigma_time=ref.sigma_time,
            sigma_freq=ref.sigma_freq,
            times=ref.times,
            sfreq=sfreq,
            bw_oct=ref.bw_oct,
            delta_oct=ref.delta_oct,
            kernel_width=ref.kernel_width,
            density=ref.density,
            n_channels=1,
        )
        unit_weights = WeightSurface(
            weights=np.ones_like(np.abs(evoked_coefficients)),
            component="full",
            eps=eps,
            max_weight=1.0,
        )
        evoked, _, _ = synthesize(
            evoked_dec, unit_weights, edge_taper=edge_taper
        )

        induced_decompositions = []
        for k in range(n_epochs):
            induced_coeff = decompositions[k].coefficients - evoked_coefficients
            induced_dec = WaveletDecomposition(
                coefficients=induced_coeff,
                foi=ref.foi,
                sigma_time=ref.sigma_time,
                sigma_freq=ref.sigma_freq,
                times=ref.times,
                sfreq=sfreq,
                bw_oct=ref.bw_oct,
                delta_oct=ref.delta_oct,
                kernel_width=ref.kernel_width,
                density=ref.density,
                n_channels=1,
            )
            induced_decompositions.append(induced_dec)

        ensemble_power = np.mean(
            [np.abs(d.coefficients) ** 2 for d in induced_decompositions],
            axis=0,
        )
        decompositions = induced_decompositions
        work_signals = epochs - evoked[np.newaxis, :]
    else:
        work_signals = epochs

    ref = decompositions[0]
    ensemble_coefficients = np.sqrt(
        np.maximum(ensemble_power, 0.0)
    ).astype(np.complex128)
    ensemble_dec = WaveletDecomposition(
        coefficients=ensemble_coefficients,
        foi=ref.foi,
        sigma_time=ref.sigma_time,
        sigma_freq=ref.sigma_freq,
        times=ref.times,
        sfreq=sfreq,
        bw_oct=ref.bw_oct,
        delta_oct=ref.delta_oct,
        kernel_width=ref.kernel_width,
        density=ref.density,
        n_channels=1,
    )

    ens_fit = time_resolved_fit(
        ensemble_dec,
        fit_stride=fit_stride,
        power_window=power_window,
        smooth_sigma=smooth_sigma,
        freq_range=freq_range,
        peak_width_limits=peak_width_limits,
        max_n_peaks=max_n_peaks,
        min_peak_height=min_peak_height,
        aperiodic_mode=aperiodic_mode,
    )

    aperiodic_all = np.empty((n_epochs, n_samples))
    periodic_all = np.empty((n_epochs, n_samples))
    energy_ratios = np.empty(n_epochs)

    for k in range(n_epochs):
        ap, per, er = _reconstruct_epoch(
            work_signals[k],
            decompositions[k],
            ens_fit,
            separation=separation,
            eps=eps,
            max_weight=max_weight,
            n_iter=n_iter,
            edge_taper=edge_taper,
            sfreq=sfreq,
        )
        aperiodic_all[k] = ap
        periodic_all[k] = per
        energy_ratios[k] = er

    return EpochDecompositionResult(
        aperiodic=aperiodic_all,
        periodic=periodic_all,
        evoked=evoked,
        ensemble_fit=ens_fit,
        ensemble_power=ensemble_power,
        separation=separation,
        energy_ratios=energy_ratios,
    )


def _reconstruct_epoch(
    signal: np.ndarray,
    decomposition: WaveletDecomposition,
    fit: TimeResolvedFit,
    separation: str,
    eps: float,
    max_weight: float,
    n_iter: int,
    edge_taper: bool,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply ensemble fit to a single epoch."""
    if separation == "subtraction":
        sep = subtraction_separate(
            signal, decomposition, fit,
            eps=eps, max_weight=max_weight,
            n_iter=n_iter, edge_taper=edge_taper,
        )
    elif separation == "wiener":
        sep = wiener_separate(
            signal, decomposition, fit,
            eps=eps, max_weight=max_weight,
            n_iter=n_iter, edge_taper=edge_taper,
        )
    else:
        sep = state_space_separate(
            signal, decomposition, fit, sfreq,
            n_iter=n_iter,
        )
    return sep.aperiodic, sep.periodic, sep.energy_ratio

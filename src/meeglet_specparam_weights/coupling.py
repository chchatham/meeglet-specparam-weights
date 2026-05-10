"""Aperiodic-oscillatory coupling via virtual-channel CSD and amplitude correlation.

WARNING: Aperiodic parameters are derived FROM wavelet power via specparam fitting.
Correlating them back to the same wavelet features creates inherent circularity.
Surrogate testing (block-permuted aperiodic trajectory) is required for any inference.
"""

from dataclasses import dataclass, field

import numpy as np

from .wavelet_analysis import WaveletDecomposition, wavelet_decompose
from .time_resolved_fit import TimeResolvedFit


@dataclass
class AperiodicCouplingResult:
    csd: np.ndarray  # complex, (n_aug, n_aug, n_freqs) where n_aug = n_channels + 2
    amplitude_correlation: np.ndarray  # (n_channels, n_freqs)
    virtual_coefficients: np.ndarray  # complex, (2, n_freqs, n_times) z-scored
    effective_nyquist: float  # Hz
    foi: np.ndarray
    channel_labels: list[str] = field(default_factory=list)


def aperiodic_virtual_channels(
    fit: TimeResolvedFit,
    decomposition: WaveletDecomposition,
) -> tuple[np.ndarray, float]:
    """Wavelet-decompose exponent/offset trajectories into virtual channels.

    Returns z-scored, band-limited virtual channel coefficients and the
    effective Nyquist frequency. For multi-channel fits, aperiodic params
    are averaged across channels.

    Returns
    -------
    virtual_coefficients : np.ndarray
        Complex, shape (2, n_freqs, n_times). Row 0 = exponent, row 1 = offset.
    effective_nyquist : float
        Nyquist frequency in Hz for the aperiodic trajectories.
    """
    sfreq = decomposition.sfreq
    effective_nyq = sfreq / (2 * fit.fit_stride)

    multichannel = fit.aperiodic_params.ndim == 3
    if multichannel:
        exponent = np.nanmean(fit.aperiodic_params[:, :, 1], axis=0)
        offset = np.nanmean(fit.aperiodic_params[:, :, 0], axis=0)
    else:
        exponent = fit.aperiodic_params[:, 1].copy()
        offset = fit.aperiodic_params[:, 0].copy()

    exponent = _interpolate_nans(exponent)
    offset = _interpolate_nans(offset)

    foi = decomposition.foi
    stacked = np.stack([exponent, offset])
    both_decomp = wavelet_decompose(
        stacked, sfreq,
        foi_start=foi[0], foi_end=foi[-1],
        bw_oct=decomposition.bw_oct, delta_oct=decomposition.delta_oct,
    )
    virtual = both_decomp.coefficients

    for f_idx in range(len(foi)):
        if foi[f_idx] > effective_nyq:
            virtual[:, f_idx, :] = 0.0
            continue
        for v in range(2):
            coeffs = virtual[v, f_idx, :].copy()
            coeffs -= np.mean(coeffs)
            rms = np.sqrt(np.mean(np.abs(coeffs) ** 2))
            if rms > 1e-30:
                virtual[v, f_idx, :] = coeffs / rms

    return virtual, effective_nyq


def compute_aperiodic_csd(
    decomposition: WaveletDecomposition,
    fit: TimeResolvedFit,
) -> AperiodicCouplingResult:
    """Compute augmented CSD with aperiodic virtual channels.

    Augments the real wavelet coefficients with 2 virtual channels (exponent
    and offset trajectories, wavelet-decomposed) and computes the
    time-averaged cross-spectral density at each frequency.

    Returns
    -------
    AperiodicCouplingResult
        Contains the CSD matrix, amplitude correlations, virtual coefficients,
        and metadata.
    """
    virtual, effective_nyq = aperiodic_virtual_channels(fit, decomposition)

    Z = decomposition.coefficients
    foi = decomposition.foi
    multichannel = Z.ndim == 3

    if multichannel:
        n_ch = Z.shape[0]
        n_freqs = Z.shape[1]
        n_times = Z.shape[2]
    else:
        n_ch = 1
        n_freqs = Z.shape[0]
        n_times = Z.shape[1]
        Z = Z[np.newaxis, :]

    n_aug = n_ch + 2
    augmented = np.concatenate([Z, virtual], axis=0)

    has_nans = np.any(np.isnan(augmented))
    if not has_nans:
        csd = np.einsum("ift,jft->ijf", augmented, augmented.conj()) / n_times
    else:
        csd = np.zeros((n_aug, n_aug, n_freqs), dtype=np.complex128)
        for f_idx in range(n_freqs):
            data_f = augmented[:, f_idx, :]
            valid = ~np.any(np.isnan(data_f), axis=0)
            n_valid = int(np.sum(valid))
            if n_valid > 0:
                data_valid = data_f[:, valid]
                csd[:, :, f_idx] = data_valid @ data_valid.conj().T / n_valid

    amp_corr = aperiodic_amplitude_correlation(decomposition, fit)

    ch_labels = [f"ch{i}" for i in range(n_ch)] + ["exponent", "offset"]

    return AperiodicCouplingResult(
        csd=csd,
        amplitude_correlation=amp_corr,
        virtual_coefficients=virtual,
        effective_nyquist=effective_nyq,
        foi=foi,
        channel_labels=ch_labels,
    )


def aperiodic_amplitude_correlation(
    decomposition: WaveletDecomposition,
    fit: TimeResolvedFit,
) -> np.ndarray:
    """Compute correlation between exponent trajectory and wavelet amplitude.

    Returns Pearson correlation corr(exponent(t), |Z(ch, f, t)|) for each
    channel and frequency.

    Returns
    -------
    np.ndarray
        Shape (n_channels, n_freqs).
    """
    Z = decomposition.coefficients
    multichannel = Z.ndim == 3

    if multichannel:
        n_ch = Z.shape[0]
        exponent = np.nanmean(fit.aperiodic_params[:, :, 1], axis=0)
    else:
        n_ch = 1
        Z = Z[np.newaxis, :]
        exponent = fit.aperiodic_params[:, 1].copy()

    exponent = _interpolate_nans(exponent)
    n_freqs = Z.shape[1]

    amp_corr = np.zeros((n_ch, n_freqs))
    exp_valid = ~np.isnan(exponent)

    for ch in range(n_ch):
        amplitudes = np.abs(Z[ch])
        has_amp_nans = np.any(np.isnan(amplitudes))

        if not has_amp_nans and np.all(exp_valid):
            exp_centered = exponent - np.mean(exponent)
            amp_centered = amplitudes - amplitudes.mean(axis=1, keepdims=True)
            numerator = (exp_centered[np.newaxis, :] * amp_centered).sum(axis=1)
            denom = np.sqrt(np.sum(exp_centered ** 2) * (amp_centered ** 2).sum(axis=1))
            valid_denom = denom > 0
            amp_corr[ch, valid_denom] = numerator[valid_denom] / denom[valid_denom]
        else:
            for f_idx in range(n_freqs):
                valid = exp_valid & ~np.isnan(amplitudes[f_idx])
                if np.sum(valid) > 10:
                    e = exponent[valid]
                    a = amplitudes[f_idx, valid]
                    e_c = e - e.mean()
                    a_c = a - a.mean()
                    d = np.sqrt(np.sum(e_c ** 2) * np.sum(a_c ** 2))
                    if d > 0:
                        amp_corr[ch, f_idx] = np.sum(e_c * a_c) / d

    return amp_corr


def effective_dof(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate effective degrees of freedom using Bartlett's formula.

    Accounts for temporal autocorrelation in both series to give a
    corrected sample size for significance testing.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if len(y) != n:
        raise ValueError(f"x and y must have same length, got {n} and {len(y)}")

    x = x - np.mean(x)
    y = y - np.mean(y)

    var_x = np.mean(x ** 2)
    var_y = np.mean(y ** 2)

    if var_x < 1e-30 or var_y < 1e-30:
        return 2.0

    max_lag = n // 3

    acf_x_full = np.correlate(x, x, mode="full") / (n * var_x)
    acf_y_full = np.correlate(y, y, mode="full") / (n * var_y)
    rho_x = acf_x_full[n:n + max_lag]
    rho_y = acf_y_full[n:n + max_lag]

    lags = np.arange(1, max_lag + 1)
    weights = 1 - lags / n

    correction = np.sum(weights * rho_x * rho_y)
    n_eff = n / max(1 + 2 * correction, 1.0)
    return max(n_eff, 2.0)


def wavelet_effective_dof(
    sigma_time: np.ndarray,
    sfreq: float,
    n_samples: int,
) -> np.ndarray:
    """Frequency-dependent effective DOF from wavelet temporal resolution.

    Each wavelet at frequency f_i has temporal std sigma_time[i]. Consecutive
    samples separated by less than ~2*sigma_time are not independent. The
    effective number of independent observations is T / (2 * sigma_time[i]).

    Returns
    -------
    np.ndarray
        Shape (n_freqs,). Effective DOF per frequency, clamped to [2, n_samples].
    """
    T = n_samples / sfreq
    n_eff = T / (2 * sigma_time)
    return np.clip(n_eff, 2.0, float(n_samples))


def _interpolate_nans(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values in a 1D array."""
    x = x.copy()
    nans = np.isnan(x)
    if not np.any(nans):
        return x
    if np.all(nans):
        return np.zeros_like(x)
    indices = np.arange(len(x))
    x[nans] = np.interp(indices[nans], indices[~nans], x[~nans])
    return x

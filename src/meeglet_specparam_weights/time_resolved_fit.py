"""Time-resolved parametric fitting: fit specparam to wavelet power at each time step."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from specparam import SpectralModel

from .wavelet_analysis import WaveletDecomposition


@dataclass
class TimeResolvedFit:
    aperiodic_params: np.ndarray  # (n_times, 2) or (n_channels, n_times, 2)
    peak_params: list  # list[ndarray] or list[list[ndarray]] for multi-channel
    model_power: np.ndarray  # (n_freqs, n_times) or (n_channels, n_freqs, n_times)
    r_squared: np.ndarray  # (n_times,) or (n_channels, n_times)
    foi: np.ndarray  # center frequencies used for fitting
    times: np.ndarray  # time points (at fitted positions)
    fit_stride: int  # stride in samples between fits
    n_channels: int = 1


def time_resolved_fit(
    decomposition: WaveletDecomposition,
    fit_stride: int = 10,
    power_window: int | None = None,
    smooth_sigma: float | None = None,
    freq_range: list[float] | None = None,
    n_freqs_linear: int = 100,
    peak_width_limits: tuple[float, float] = (0.5, 12.0),
    max_n_peaks: int = 8,
    min_peak_height: float = 0.0,
    aperiodic_mode: str = "fixed",
) -> TimeResolvedFit:
    """Fit specparam to time-averaged wavelet power at strided positions.

    Accepts single-channel (2D coefficients) or multi-channel (3D coefficients).
    For multi-channel, each channel is fitted independently.

    Parameters
    ----------
    decomposition : WaveletDecomposition
        Output from wavelet_decompose().
    fit_stride : int
        Fit every `fit_stride` samples. Intermediate points are interpolated.
    power_window : int or None
        Number of samples to average power over (centered at each fit point).
        Defaults to fit_stride * 2 (ensuring overlap between windows).
    smooth_sigma : float or None
        If set, apply Gaussian smoothing (in fit-stride units) to aperiodic
        parameter trajectories after fitting.
    freq_range : list of float or None
        Frequency range for fitting [lo, hi]. Defaults to full foi range.
    n_freqs_linear : int
        Number of linearly-spaced frequency points for specparam fitting.
    """
    Z = decomposition.coefficients
    foi = decomposition.foi
    multichannel = Z.ndim == 3

    if multichannel:
        n_channels = Z.shape[0]
        n_times = Z.shape[2]
    else:
        n_channels = 1
        n_times = Z.shape[1]

    if freq_range is None:
        freq_range = [foi[0], foi[-1]]

    if power_window is None:
        power_window = max(fit_stride * 2, 10)

    fit_kwargs = dict(
        foi=foi, n_times=n_times, fit_stride=fit_stride,
        power_window=power_window, smooth_sigma=smooth_sigma,
        freq_range=freq_range, n_freqs_linear=n_freqs_linear,
        peak_width_limits=peak_width_limits, max_n_peaks=max_n_peaks,
        min_peak_height=min_peak_height, aperiodic_mode=aperiodic_mode,
    )

    if not multichannel:
        ap, pk, mp, r2 = _fit_single_channel(Z, **fit_kwargs)
        return TimeResolvedFit(
            aperiodic_params=ap, peak_params=pk, model_power=mp,
            r_squared=r2, foi=foi, times=decomposition.times,
            fit_stride=fit_stride, n_channels=1,
        )

    all_ap = np.empty((n_channels, n_times, 2))
    all_mp = np.empty((n_channels, len(foi), n_times))
    all_r2 = np.empty((n_channels, n_times))
    all_pk: list[list[np.ndarray]] = []

    for ch in range(n_channels):
        ap, pk, mp, r2 = _fit_single_channel(Z[ch], **fit_kwargs)
        all_ap[ch] = ap
        all_mp[ch] = mp
        all_r2[ch] = r2
        all_pk.append(pk)

    return TimeResolvedFit(
        aperiodic_params=all_ap, peak_params=all_pk, model_power=all_mp,
        r_squared=all_r2, foi=foi, times=decomposition.times,
        fit_stride=fit_stride, n_channels=n_channels,
    )


def _fit_single_channel(
    Z_ch: np.ndarray,
    foi: np.ndarray,
    n_times: int,
    fit_stride: int,
    power_window: int,
    smooth_sigma: float | None,
    freq_range: list[float],
    n_freqs_linear: int,
    peak_width_limits: tuple[float, float],
    max_n_peaks: int,
    min_peak_height: float,
    aperiodic_mode: str,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """Fit a single channel and return (aperiodic_all, peak_params_all, model_power, r_squared)."""
    linear_freqs = np.linspace(freq_range[0], freq_range[1], n_freqs_linear)

    fit_indices = np.arange(0, n_times, fit_stride)
    n_fits = len(fit_indices)

    aperiodic_fitted = np.full((n_fits, 2), np.nan)
    peak_params_fitted: list[np.ndarray] = []
    r_squared_fitted = np.full(n_fits, np.nan)

    fm = SpectralModel(
        peak_width_limits=peak_width_limits,
        max_n_peaks=max_n_peaks,
        min_peak_height=min_peak_height,
        aperiodic_mode=aperiodic_mode,
        verbose=False,
    )

    instantaneous_power = np.abs(Z_ch) ** 2

    for i, t_idx in enumerate(fit_indices):
        half_win = power_window // 2
        t_lo = max(0, t_idx - half_win)
        t_hi = min(n_times, t_idx + half_win + 1)

        window_power = instantaneous_power[:, t_lo:t_hi]

        if np.any(np.isnan(window_power)):
            nan_frac = np.mean(np.isnan(window_power))
            if nan_frac > 0.5:
                peak_params_fitted.append(np.empty((0, 3)))
                continue
            avg_power = np.nanmean(window_power, axis=1)
        else:
            avg_power = np.mean(window_power, axis=1)

        if np.all(avg_power < 1e-30):
            peak_params_fitted.append(np.empty((0, 3)))
            continue

        avg_power = np.maximum(avg_power, 1e-30)

        avg_power_hz = avg_power / (foi * np.log(2))

        power_on_linear = np.maximum(
            np.interp(linear_freqs, foi, avg_power_hz), 1e-30
        )

        try:
            fm.fit(linear_freqs, power_on_linear, freq_range)
        except Exception:
            peak_params_fitted.append(np.empty((0, 3)))
            continue

        if not fm.results.has_model:
            peak_params_fitted.append(np.empty((0, 3)))
            continue

        r2 = fm.results.metrics.results["gof_rsquared"]
        ap = fm.results.params.aperiodic.params
        if r2 < 0.5 or ap[1] < -0.5 or ap[1] > 10.0:
            peak_params_fitted.append(np.empty((0, 3)))
            continue

        aperiodic_fitted[i] = ap
        pk = fm.results.params.periodic.params
        if pk.ndim == 1:
            pk = pk.reshape(-1, 3)
        peak_params_fitted.append(pk)
        r_squared_fitted[i] = r2

    if smooth_sigma is not None and smooth_sigma > 0:
        valid = ~np.isnan(aperiodic_fitted[:, 0])
        if np.sum(valid) > 3:
            for col in range(2):
                smoothed = gaussian_filter1d(
                    aperiodic_fitted[valid, col], sigma=smooth_sigma
                )
                aperiodic_fitted[valid, col] = smoothed

    aperiodic_all, r_squared_all, peak_params_all = _interpolate_to_all_times(
        aperiodic_fitted, r_squared_fitted, peak_params_fitted,
        fit_indices, n_times,
    )

    model_power = _reconstruct_model_power(
        aperiodic_all, peak_params_all, foi, n_times,
    )

    return aperiodic_all, peak_params_all, model_power, r_squared_all


def _reconstruct_model_power(
    aperiodic_all: np.ndarray,
    peak_params_all: list[np.ndarray],
    foi: np.ndarray,
    n_times: int,
) -> np.ndarray:
    """Reconstruct model power (linear, oct units) on the log-freq grid.

    specparam's model is in µV²/Hz. We convert back to µV²/oct by multiplying
    by foi * ln(2) to match the wavelet coefficient power units.
    """
    n_freqs = len(foi)
    model_power = np.zeros((n_freqs, n_times))
    log_foi = np.log10(foi)
    hz_to_oct = foi * np.log(2)

    valid = ~np.isnan(aperiodic_all[:, 0])
    if np.any(valid):
        offsets = aperiodic_all[valid, 0]
        exponents = aperiodic_all[valid, 1]
        ap_hz = 10.0 ** (offsets[np.newaxis, :] - exponents[np.newaxis, :] * log_foi[:, np.newaxis])

        pk_log = np.zeros((n_freqs, int(np.sum(valid))))
        valid_indices = np.where(valid)[0]
        for i, t in enumerate(valid_indices):
            pk = peak_params_all[t]
            if len(pk) > 0:
                fit_pk = _converted_to_fit_bw(pk)
                for cf, pw, bw in fit_pk:
                    pk_log[:, i] += pw * np.exp(-(foi - cf) ** 2 / (2 * bw ** 2))

        model_power[:, valid] = ap_hz * (10.0 ** pk_log) * hz_to_oct[:, np.newaxis]

    return model_power


def aperiodic_power_hz(offset: float, exponent: float, log_foi: np.ndarray) -> np.ndarray:
    """Compute aperiodic power in Hz units: 10^(offset - exponent * log10(foi))."""
    return 10.0 ** (offset - exponent * log_foi)


def _converted_to_fit_bw(peak_params: np.ndarray) -> np.ndarray:
    """Convert specparam converted peak BW (FWHM) back to fit BW (Gaussian std)."""
    result = peak_params.copy()
    result[:, 2] = result[:, 2] / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return result


def _interpolate_to_all_times(
    aperiodic_fitted, r_squared_fitted, peak_params_fitted,
    fit_indices, n_times,
):
    """Interpolate fitted parameters to all time points."""
    valid = ~np.isnan(aperiodic_fitted[:, 0])
    valid_indices = fit_indices[valid]

    aperiodic_all = np.full((n_times, 2), np.nan)
    r_squared_all = np.full(n_times, np.nan)

    if len(valid_indices) == 0:
        return aperiodic_all, r_squared_all, \
            [np.empty((0, 3)) for _ in range(n_times)]

    all_t = np.arange(n_times)

    for col in range(2):
        aperiodic_all[:, col] = np.interp(
            all_t, valid_indices, aperiodic_fitted[valid, col]
        )
    r_squared_all[:] = np.interp(
        all_t, valid_indices, r_squared_fitted[valid]
    )

    valid_fit_indices = np.where(valid)[0]
    peak_params_all: list[np.ndarray] = []
    for t in range(n_times):
        left_idx = np.searchsorted(valid_indices, t, side="right") - 1
        right_idx = left_idx + 1
        left_idx = np.clip(left_idx, 0, len(valid_indices) - 1)
        right_idx = np.clip(right_idx, 0, len(valid_indices) - 1)

        fit_left = valid_fit_indices[left_idx]
        fit_right = valid_fit_indices[right_idx]

        if fit_left == fit_right:
            peak_params_all.append(peak_params_fitted[fit_left])
        else:
            t_left = valid_indices[left_idx]
            t_right = valid_indices[right_idx]
            alpha = (t - t_left) / max(t_right - t_left, 1)
            pk_left = peak_params_fitted[fit_left]
            pk_right = peak_params_fitted[fit_right]
            pk = _interpolate_peaks(pk_left, pk_right, alpha)
            peak_params_all.append(pk)

    return aperiodic_all, r_squared_all, peak_params_all


def _interpolate_peaks(
    pk_left: np.ndarray, pk_right: np.ndarray, alpha: float
) -> np.ndarray:
    """Interpolate between two sets of peak parameters."""
    if len(pk_left) == 0 and len(pk_right) == 0:
        return np.empty((0, 3))
    if len(pk_left) == 0:
        result = pk_right.copy()
        result[:, 1] *= alpha  # fade in amplitude
        return result
    if len(pk_right) == 0:
        result = pk_left.copy()
        result[:, 1] *= (1 - alpha)
        return result
    if len(pk_left) == len(pk_right):
        return pk_left * (1 - alpha) + pk_right * alpha
    return pk_left if alpha < 0.5 else pk_right

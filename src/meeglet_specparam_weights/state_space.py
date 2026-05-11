"""State-space decomposition via Kalman smoother.

Models the signal as a superposition of K damped oscillators (periodic)
and an AR(p) process (aperiodic), then uses the Rauch-Tung-Striebel
smoother to optimally estimate each component. This is the only
single-trial method that can properly separate induced oscillations
from the 1/f background, because it uses the temporal structure of
each component (narrowband vs. broadband) as a separating constraint.

Initialization comes from specparam: peak frequencies and bandwidths
define the oscillators, and the aperiodic exponent defines the AR model.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_toeplitz, solve_discrete_lyapunov, cho_factor, cho_solve

from .wavelet_analysis import WaveletDecomposition
from .time_resolved_fit import TimeResolvedFit
from .weight_surface import WeightSurface, compute_weight_surface
from .separation import SeparationResult, decomposition_bias_estimate


@dataclass
class StateSpaceModel:
    n_oscillators: int
    ar_order: int
    n_states: int
    center_freqs: np.ndarray
    damping: np.ndarray
    osc_noise_var: np.ndarray
    ar_coeffs: np.ndarray
    ar_noise_var: float
    measurement_noise_var: float
    sfreq: float


@dataclass
class StateSpaceDecomposition:
    oscillators: np.ndarray
    aperiodic: np.ndarray
    measurement_noise: np.ndarray
    smoothed_states: np.ndarray
    smoothed_covs: np.ndarray
    log_likelihood: float
    model: StateSpaceModel


# ---------------------------------------------------------------------------
# AR coefficient computation
# ---------------------------------------------------------------------------

def ar_coefficients_from_exponent(
    exponent: float,
    sfreq: float,
    ar_order: int,
    freq_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, float]:
    """Compute AR coefficients producing a 1/f^exponent power spectrum.

    Uses Yule-Walker equations on the autocorrelation of the target PSD.
    """
    n_fft = max(4096, 2 * ar_order + 2)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)
    freqs[0] = freqs[1]

    psd = 1.0 / freqs ** exponent

    if freq_range is not None:
        lo, hi = freq_range
        if hi is None:
            hi = sfreq / 2
        mask = (freqs < lo) | (freqs > hi)
        psd[mask] = psd[~mask].min()

    psd_full = np.concatenate([psd, psd[-2:0:-1]])
    acf = np.fft.ifft(psd_full).real
    acf = acf[: ar_order + 1]

    r_col = acf[: ar_order]
    r_rhs = acf[1: ar_order + 1]
    ar_coeffs = solve_toeplitz(r_col, r_rhs)

    innovation_var = acf[0] - np.dot(ar_coeffs, r_rhs)
    innovation_var = max(innovation_var, 1e-30)

    return ar_coeffs, innovation_var


def select_ar_order(
    exponent: float,
    sfreq: float,
    max_order: int = 30,
    criterion: str = "aic",
) -> int:
    """Select AR order for 1/f approximation via information criterion."""
    n_ref = int(10 * sfreq)
    best_order = 1
    best_score = np.inf

    for p in range(1, max_order + 1):
        _, sigma2 = ar_coefficients_from_exponent(exponent, sfreq, p)
        if criterion == "aic":
            score = n_ref * np.log(sigma2) + 2 * p
        else:
            score = n_ref * np.log(sigma2) + p * np.log(n_ref)
        if score < best_score:
            best_score = score
            best_order = p

    return best_order


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_state_space_model(
    fit: TimeResolvedFit,
    sfreq: float,
    ar_order: int | str = "auto",
    max_ar_order: int = 30,
    measurement_noise_fraction: float = 0.05,
    time_index: int | None = None,
) -> StateSpaceModel:
    """Build a state-space model from specparam parameters."""
    ap = fit.aperiodic_params
    if ap.ndim == 3:
        ap = np.mean(ap, axis=0)
    if time_index is not None:
        offset = ap[time_index, 0]
        exponent = ap[time_index, 1]
    else:
        offset = float(np.nanmedian(ap[:, 0]))
        exponent = float(np.nanmedian(ap[:, 1]))

    pk_list = fit.peak_params
    if pk_list and isinstance(pk_list[0], list):
        flat_pk = []
        for ch_pks in pk_list:
            for t_pk in ch_pks:
                if t_pk.shape[0] > 0:
                    flat_pk.append(t_pk)
    else:
        flat_pk = [p for p in pk_list if p.shape[0] > 0]

    if flat_pk:
        all_peaks = np.vstack(flat_pk)
        center_freqs = _cluster_peak_frequencies(all_peaks[:, 0])
    else:
        center_freqs = np.array([], dtype=np.float64)

    n_osc = len(center_freqs)

    damping = np.empty(n_osc)
    osc_noise_var = np.empty(n_osc)

    for k, cf in enumerate(center_freqs):
        bw_list = []
        amp_list = []
        for t_pk in (flat_pk if flat_pk else []):
            if t_pk.shape[0] == 0:
                continue
            dists = np.abs(t_pk[:, 0] - cf)
            closest = np.argmin(dists)
            if dists[closest] < 2.0:
                bw_list.append(t_pk[closest, 2])
                amp_list.append(t_pk[closest, 1])
        median_bw = float(np.median(bw_list)) if bw_list else 2.0
        median_amp = float(np.median(amp_list)) if amp_list else 0.1

        r_k = np.exp(-np.pi * median_bw / sfreq)
        damping[k] = min(r_k, 0.999)

        peak_power = 10.0 ** median_amp * median_bw * np.sqrt(2 * np.pi)
        osc_noise_var[k] = max(peak_power * (1.0 - damping[k] ** 2), 1e-20)

    if isinstance(ar_order, str) and ar_order == "auto":
        ar_order_val = select_ar_order(exponent, sfreq, max_ar_order)
    else:
        ar_order_val = int(ar_order)

    ar_coeffs, ar_noise_var = ar_coefficients_from_exponent(
        exponent, sfreq, ar_order_val,
    )

    total_power = 10.0 ** offset
    ar_noise_var *= total_power / max(ar_noise_var / (1.0 - np.sum(ar_coeffs ** 2) + 1e-30), 1e-30)
    ar_noise_var = max(ar_noise_var, 1e-30)

    signal_var = total_power + sum(osc_noise_var / (1.0 - damping ** 2 + 1e-30))
    measurement_noise_var = max(measurement_noise_fraction * signal_var, 1e-30)

    n_states = 2 * n_osc + ar_order_val

    return StateSpaceModel(
        n_oscillators=n_osc,
        ar_order=ar_order_val,
        n_states=n_states,
        center_freqs=center_freqs,
        damping=damping,
        osc_noise_var=osc_noise_var,
        ar_coeffs=ar_coeffs,
        ar_noise_var=ar_noise_var,
        measurement_noise_var=measurement_noise_var,
        sfreq=sfreq,
    )


def _cluster_peak_frequencies(
    freqs: np.ndarray,
    tolerance: float = 2.0,
) -> np.ndarray:
    """Cluster peak frequencies into distinct oscillator slots."""
    if len(freqs) == 0:
        return np.array([], dtype=np.float64)

    sorted_f = np.sort(freqs)
    clusters = [[sorted_f[0]]]
    for f in sorted_f[1:]:
        if f - np.mean(clusters[-1]) < tolerance:
            clusters[-1].append(f)
        else:
            clusters.append([f])

    return np.array([np.median(c) for c in clusters])


def build_matrices(model: StateSpaceModel) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build static F, H, Q, R matrices from a StateSpaceModel."""
    n = model.n_states
    F = np.zeros((n, n))
    Q = np.zeros((n, n))
    H = np.zeros(n)

    for k in range(model.n_oscillators):
        i = 2 * k
        w = 2 * np.pi * model.center_freqs[k] / model.sfreq
        r = model.damping[k]
        F[i, i] = r * np.cos(w)
        F[i, i + 1] = -r * np.sin(w)
        F[i + 1, i] = r * np.sin(w)
        F[i + 1, i + 1] = r * np.cos(w)
        Q[i, i] = model.osc_noise_var[k]
        Q[i + 1, i + 1] = model.osc_noise_var[k]
        H[i] = 1.0

    ar_start = 2 * model.n_oscillators
    p = model.ar_order
    if p > 0:
        F[ar_start, ar_start: ar_start + p] = model.ar_coeffs
        for j in range(1, p):
            F[ar_start + j, ar_start + j - 1] = 1.0
        Q[ar_start, ar_start] = model.ar_noise_var
        H[ar_start] = 1.0

    R = model.measurement_noise_var
    return F, H, Q, R


# ---------------------------------------------------------------------------
# Kalman filter and RTS smoother
# ---------------------------------------------------------------------------

def kalman_filter(
    y: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: float,
    x0: np.ndarray,
    P0: np.ndarray,
    F_t: np.ndarray | None = None,
    Q_t: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Forward Kalman filter."""
    T = len(y)
    n = len(x0)

    x_filt = np.zeros((n, T))
    P_filt = np.zeros((n, n, T))
    x_pred = np.zeros((n, T))
    P_pred = np.zeros((n, n, T))

    x_curr = x0.copy()
    P_curr = P0.copy()
    log_lik = 0.0

    for t in range(T):
        F_use = F_t[t] if F_t is not None else F
        Q_use = Q_t[t] if Q_t is not None else Q

        xp = F_use @ x_curr
        Pp = F_use @ P_curr @ F_use.T + Q_use

        x_pred[:, t] = xp
        P_pred[:, :, t] = Pp

        S = H @ Pp @ H + R
        v = y[t] - H @ xp
        K = (Pp @ H) / S

        x_curr = xp + K * v
        P_curr = Pp - np.outer(K, H @ Pp)

        x_filt[:, t] = x_curr
        P_filt[:, :, t] = P_curr

        log_lik += -0.5 * (np.log(2 * np.pi) + np.log(S) + v ** 2 / S)

    return x_filt, P_filt, x_pred, P_pred, float(log_lik)


def rts_smoother(
    x_filt: np.ndarray,
    P_filt: np.ndarray,
    x_pred: np.ndarray,
    P_pred: np.ndarray,
    F: np.ndarray,
    F_t: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rauch-Tung-Striebel backward smoother."""
    n, T = x_filt.shape
    x_smooth = np.zeros((n, T))
    P_smooth = np.zeros((n, n, T))

    x_smooth[:, T - 1] = x_filt[:, T - 1]
    P_smooth[:, :, T - 1] = P_filt[:, :, T - 1]

    for t in range(T - 2, -1, -1):
        F_use = F_t[t + 1] if F_t is not None else F

        Pp_next = P_pred[:, :, t + 1]
        try:
            cf = cho_factor(Pp_next)
            G = cho_solve(cf, (F_use @ P_filt[:, :, t]).T).T
        except np.linalg.LinAlgError:
            G = P_filt[:, :, t] @ F_use.T @ np.linalg.pinv(Pp_next)

        x_smooth[:, t] = x_filt[:, t] + G @ (x_smooth[:, t + 1] - x_pred[:, t + 1])
        P_smooth[:, :, t] = P_filt[:, :, t] + G @ (P_smooth[:, :, t + 1] - Pp_next) @ G.T

    return x_smooth, P_smooth


# ---------------------------------------------------------------------------
# Component extraction and public API
# ---------------------------------------------------------------------------

def state_space_decompose(
    signal: np.ndarray,
    sfreq: float,
    fit: TimeResolvedFit,
    ar_order: int | str = "auto",
    max_ar_order: int = 30,
    measurement_noise_fraction: float = 0.05,
) -> StateSpaceDecomposition:
    """Decompose a single-channel signal using the Kalman smoother."""
    model = build_state_space_model(
        fit, sfreq,
        ar_order=ar_order,
        max_ar_order=max_ar_order,
        measurement_noise_fraction=measurement_noise_fraction,
    )
    F, H, Q, R = build_matrices(model)

    try:
        P0 = solve_discrete_lyapunov(F, Q)
    except np.linalg.LinAlgError:
        P0 = np.eye(model.n_states) * np.trace(Q)

    x0 = np.zeros(model.n_states)

    x_filt, P_filt, x_pred, P_pred, log_lik = kalman_filter(
        signal, F, H, Q, R, x0, P0,
    )
    x_smooth, P_smooth = rts_smoother(
        x_filt, P_filt, x_pred, P_pred, F,
    )

    n_osc = model.n_oscillators
    oscillators = np.zeros((max(n_osc, 1), len(signal)))
    for k in range(n_osc):
        oscillators[k] = x_smooth[2 * k]

    ar_start = 2 * n_osc
    aperiodic = x_smooth[ar_start] if model.ar_order > 0 else np.zeros(len(signal))

    predicted = H @ x_smooth
    measurement_noise = signal - predicted

    return StateSpaceDecomposition(
        oscillators=oscillators,
        aperiodic=aperiodic,
        measurement_noise=measurement_noise,
        smoothed_states=x_smooth,
        smoothed_covs=P_smooth,
        log_likelihood=log_lik,
        model=model,
    )


def state_space_separate(
    signal: np.ndarray,
    decomposition: WaveletDecomposition,
    fit: TimeResolvedFit,
    sfreq: float,
    ar_order: int | str = "auto",
    max_ar_order: int = 30,
    measurement_noise_fraction: float = 0.05,
    n_iter: int = 1,
) -> SeparationResult:
    """Separation strategy using the state-space Kalman smoother.

    Called by the pipeline when separation='state_space'.
    """
    sig = signal
    multichannel = signal.ndim == 2

    if multichannel:
        n_channels = signal.shape[0]
        aperiodic_all = np.zeros_like(signal)
        periodic_all = np.zeros_like(signal)

        for ch in range(n_channels):
            ch_fit = _extract_channel_fit(fit, ch)
            ss_result = state_space_decompose(
                signal[ch], sfreq, ch_fit,
                ar_order=ar_order,
                max_ar_order=max_ar_order,
                measurement_noise_fraction=measurement_noise_fraction,
            )
            aperiodic_all[ch] = ss_result.aperiodic + ss_result.measurement_noise
            periodic_all[ch] = np.sum(ss_result.oscillators, axis=0)

        aperiodic_recon = aperiodic_all
        periodic_recon = periodic_all
    else:
        ss_result = state_space_decompose(
            signal, sfreq, fit,
            ar_order=ar_order,
            max_ar_order=max_ar_order,
            measurement_noise_fraction=measurement_noise_fraction,
        )
        aperiodic_recon = ss_result.aperiodic + ss_result.measurement_noise
        periodic_recon = np.sum(ss_result.oscillators, axis=0)

    recon_energy = float(np.sum(aperiodic_recon ** 2))
    empirical_energy = float(np.sum(signal ** 2))
    energy_ratio = recon_energy / max(empirical_energy, 1e-30)

    bias = decomposition_bias_estimate(decomposition, fit, method="state_space")

    return SeparationResult(
        aperiodic=aperiodic_recon,
        periodic=periodic_recon,
        method="state_space",
        bias_estimate=bias,
        weights=None,
        energy_ratio=energy_ratio,
        frame_condition=1.0,
    )


def _extract_channel_fit(fit: TimeResolvedFit, ch: int) -> TimeResolvedFit:
    """Extract single-channel fit from a multi-channel TimeResolvedFit."""
    if fit.aperiodic_params.ndim == 2:
        return fit

    return TimeResolvedFit(
        aperiodic_params=fit.aperiodic_params[ch],
        peak_params=fit.peak_params[ch] if isinstance(fit.peak_params[0], list) else fit.peak_params,
        model_power=fit.model_power[ch],
        r_squared=fit.r_squared[ch],
        foi=fit.foi,
        times=fit.times,
        fit_stride=fit.fit_stride,
        n_channels=1,
    )

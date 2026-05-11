"""Separation strategies for decomposing signals into aperiodic and periodic components.

Each strategy takes a signal, its wavelet decomposition, and a specparam fit,
and returns a SeparationResult with the aperiodic and periodic time-domain signals.

Available strategies:
  - subtraction: aperiodic = original - synthesize(excess_weights * Z).
    Exact decomposition (original = ap + per) but suppresses aperiodic power
    at peak frequencies by ~(1-sqrt(r))^2/(1-r) where r = P_per/P_total.
  - wiener: aperiodic = synthesize(sqrt(P_ap/|Z|^2) * Z).
    Preserves correct aperiodic POWER at all frequencies but contaminates
    the waveform with periodic phase structure.
  - state_space: Kalman smoother with damped oscillators + AR(p) aperiodic.
    Uses temporal structure to properly separate induced oscillations from
    broadband noise. (Implemented in state_space.py.)
"""

from dataclasses import dataclass

import numpy as np

from .wavelet_analysis import WaveletDecomposition
from .time_resolved_fit import TimeResolvedFit
from .weight_surface import WeightSurface, compute_weight_surface
from .synthesis import synthesize


@dataclass
class SeparationResult:
    """Result of aperiodic/periodic separation."""

    aperiodic: np.ndarray  # (n_samples,) or (n_channels, n_samples)
    periodic: np.ndarray  # (n_samples,) or (n_channels, n_samples)
    method: str  # "subtraction", "wiener", "state_space"
    bias_estimate: np.ndarray  # (n_freqs,) expected power bias at each frequency
    weights: WeightSurface | None  # weight surface used (None for state_space)
    energy_ratio: float
    frame_condition: float


def subtraction_separate(
    signal: np.ndarray,
    decomposition: WaveletDecomposition,
    fit: TimeResolvedFit,
    eps: float = 1e-20,
    max_weight: float = 100.0,
    n_iter: int = 1,
    edge_taper: bool = True,
) -> SeparationResult:
    """Separate via subtraction: aperiodic = original - periodic_excess.

    Computes excess weights w_excess = sqrt(max(0, 1 - P_ap/|Z|^2)), synthesizes
    the periodic excess, and subtracts from the original signal.

    Guarantees original = aperiodic + periodic exactly. However, the aperiodic
    power at peak frequencies is suppressed because the excess weight scales the
    full coefficient (both aperiodic and periodic parts) proportionally.
    """
    aperiodic_weights = compute_weight_surface(
        decomposition, fit,
        component="aperiodic",
        eps=eps,
        max_weight=max_weight,
    )
    w_ap = aperiodic_weights.weights
    excess_w = np.sqrt(np.maximum(0.0, 1.0 - w_ap ** 2))
    excess_weights = WeightSurface(
        weights=excess_w,
        component="periodic",
        eps=eps,
        max_weight=1.0,
    )
    periodic_recon, _, frame_condition = synthesize(
        decomposition, excess_weights,
        edge_taper=edge_taper,
        n_iter=max(n_iter, 5),
    )
    aperiodic_recon = signal - periodic_recon
    recon_energy = float(np.sum(aperiodic_recon ** 2))
    empirical_energy = float(np.sum(signal ** 2))
    energy_ratio = recon_energy / max(empirical_energy, 1e-30)

    bias = decomposition_bias_estimate(decomposition, fit, method="subtraction")

    return SeparationResult(
        aperiodic=aperiodic_recon,
        periodic=periodic_recon,
        method="subtraction",
        bias_estimate=bias,
        weights=excess_weights,
        energy_ratio=energy_ratio,
        frame_condition=frame_condition,
    )


def wiener_separate(
    signal: np.ndarray,
    decomposition: WaveletDecomposition,
    fit: TimeResolvedFit,
    eps: float = 1e-20,
    max_weight: float = 100.0,
    n_iter: int = 1,
    edge_taper: bool = True,
) -> SeparationResult:
    """Separate via Wiener filter: aperiodic = synthesize(sqrt(P_ap/|Z|^2) * Z).

    The Wiener filter preserves correct aperiodic POWER at all frequencies
    (|Z * w_ap|^2 = P_ap by construction). However, the waveform is a scaled
    version of the full signal at peak frequencies, retaining periodic phase
    structure in the aperiodic reconstruction.
    """
    aperiodic_weights = compute_weight_surface(
        decomposition, fit,
        component="aperiodic",
        eps=eps,
        max_weight=max_weight,
    )
    aperiodic_recon, energy_ratio, frame_condition = synthesize(
        decomposition, aperiodic_weights,
        edge_taper=edge_taper,
        n_iter=n_iter,
    )
    periodic_recon = signal - aperiodic_recon

    bias = decomposition_bias_estimate(decomposition, fit, method="wiener")

    return SeparationResult(
        aperiodic=aperiodic_recon,
        periodic=periodic_recon,
        method="wiener",
        bias_estimate=bias,
        weights=aperiodic_weights,
        energy_ratio=energy_ratio,
        frame_condition=frame_condition,
    )


def decomposition_bias_estimate(
    decomposition: WaveletDecomposition,
    fit: TimeResolvedFit,
    method: str = "subtraction",
) -> np.ndarray:
    """Estimate the expected power bias factor at each frequency.

    Returns an array of shape (n_freqs,) where each value is the expected
    ratio of reconstructed aperiodic power to true aperiodic power at that
    frequency.  Values near 1.0 mean unbiased; values << 1.0 mean the
    aperiodic is suppressed at that frequency.

    For subtraction: bias(f) = (1 - sqrt(r))^2 / (1 - r) where r = P_per/P_total.
    For wiener: bias(f) = 1.0 (unbiased power by construction).
    For state_space: bias(f) = 1.0 (Kalman smoother provides optimal estimate).
    """
    if method in ("wiener", "state_space"):
        return np.ones(len(decomposition.foi))

    empirical_power = np.abs(decomposition.coefficients) ** 2
    if empirical_power.ndim == 3:
        empirical_power = np.mean(empirical_power, axis=0)

    model_power = fit.model_power
    if model_power.ndim == 3:
        model_power = np.mean(model_power, axis=0)

    log_foi = np.log10(decomposition.foi)
    hz_to_oct = decomposition.foi * np.log(2)

    ap_params = fit.aperiodic_params
    if ap_params.ndim == 3:
        ap_params = np.mean(ap_params, axis=0)
    offsets = np.nanmean(ap_params[:, 0])
    exponents = np.nanmean(ap_params[:, 1])

    aperiodic_power = 10.0 ** (offsets - exponents * log_foi) * hz_to_oct
    total_power = np.mean(model_power, axis=1)

    r = np.clip(1.0 - aperiodic_power / np.maximum(total_power, 1e-30), 0.0, 1.0)

    bias = np.where(
        r < 1e-10,
        1.0,
        (1.0 - np.sqrt(r)) ** 2 / np.maximum(1.0 - r, 1e-30),
    )
    return bias

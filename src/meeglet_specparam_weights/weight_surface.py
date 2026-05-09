"""Weight surface computation: w(f,t) = sqrt(P_model(f,t) / |Z(f,t)|²)."""

from dataclasses import dataclass

import numpy as np

from .wavelet_analysis import WaveletDecomposition
from .time_resolved_fit import TimeResolvedFit, aperiodic_power_hz


@dataclass
class WeightSurface:
    weights: np.ndarray  # real, non-negative, shape (n_freqs, n_times)
    component: str  # 'full', 'aperiodic', 'periodic'
    eps: float  # floor used for denominator
    max_weight: float  # clamp used


def compute_weight_surface(
    decomposition: WaveletDecomposition,
    fit: TimeResolvedFit,
    component: str = "aperiodic",
    eps: float = 1e-20,
    max_weight: float = 100.0,
) -> WeightSurface:
    """Compute spectral weights from model power and empirical power.

    Parameters
    ----------
    decomposition : WaveletDecomposition
        Wavelet coefficients Z(f,t).
    fit : TimeResolvedFit
        Parametric fit with model_power on the same frequency grid.
    component : str
        Which model component to use: 'full', 'aperiodic', or 'periodic'.
    eps : float
        Floor for denominator to prevent division by zero.
    max_weight : float
        Maximum allowed weight value.
    """
    if component not in ("full", "aperiodic", "periodic"):
        raise ValueError(f"component must be 'full', 'aperiodic', or 'periodic', got '{component}'")

    empirical_power = np.abs(decomposition.coefficients) ** 2

    model_power = _extract_component_power(fit, component)

    denominator = np.maximum(empirical_power, eps)
    numerator = np.maximum(model_power, 0.0)

    weights = np.sqrt(numerator / denominator)

    weights = np.minimum(weights, max_weight)

    nan_mask = np.isnan(decomposition.coefficients)
    weights[nan_mask] = 0.0

    weights = np.where(np.isnan(weights) | np.isinf(weights), 0.0, weights)

    return WeightSurface(
        weights=weights,
        component=component,
        eps=eps,
        max_weight=max_weight,
    )


def _extract_component_power(fit: TimeResolvedFit, component: str) -> np.ndarray:
    """Extract the appropriate model power component."""
    if component == "full":
        return fit.model_power

    foi = fit.foi
    log_foi = np.log10(foi)
    n_freqs = len(foi)
    n_times = len(fit.times)

    aperiodic_power = np.zeros((n_freqs, n_times))
    for t in range(n_times):
        if np.isnan(fit.aperiodic_params[t, 0]):
            continue
        offset, exponent = fit.aperiodic_params[t]
        aperiodic_power[:, t] = aperiodic_power_hz(offset, exponent, log_foi)

    if component == "aperiodic":
        return aperiodic_power

    # periodic = full - aperiodic
    periodic_power = np.maximum(fit.model_power - aperiodic_power, 0.0)
    return periodic_power

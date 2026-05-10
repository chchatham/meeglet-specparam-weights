"""Weight surface: w(f,t) = sqrt(P_model / |Z|²), an amplitude-domain Wiener filter.

The weight at each time-frequency point is the ratio of model amplitude to
empirical amplitude. Multiplying Z(f,t) by w(f,t) is equivalent to applying
a non-negative, phase-preserving Wiener filter in the wavelet domain.
"""

from dataclasses import dataclass

import numpy as np

from .wavelet_analysis import WaveletDecomposition
from .time_resolved_fit import TimeResolvedFit, aperiodic_power_hz


@dataclass
class WeightSurface:
    weights: np.ndarray  # (n_freqs, n_times) or (n_channels, n_freqs, n_times)
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

    multichannel = decomposition.coefficients.ndim == 3

    empirical_power = np.abs(decomposition.coefficients) ** 2

    if multichannel:
        n_channels = decomposition.coefficients.shape[0]
        model_power = np.stack([
            _extract_component_power_single(
                fit.model_power[ch], fit.aperiodic_params[ch], fit.foi, fit.times, component
            )
            for ch in range(n_channels)
        ])
    else:
        model_power = _extract_component_power_single(
            fit.model_power, fit.aperiodic_params, fit.foi, fit.times, component
        )

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


def _extract_component_power_single(
    model_power: np.ndarray,
    aperiodic_params: np.ndarray,
    foi: np.ndarray,
    times: np.ndarray,
    component: str,
) -> np.ndarray:
    """Extract model power component for a single channel.

    All returned power is in oct units (µV²/oct) to match model_power and
    the empirical wavelet power |Z|².
    """
    if component == "full":
        return model_power

    log_foi = np.log10(foi)
    hz_to_oct = foi * np.log(2)
    n_times = len(times)

    valid = ~np.isnan(aperiodic_params[:, 0])
    offsets = aperiodic_params[:, 0]
    exponents = aperiodic_params[:, 1]

    aperiodic_power = np.zeros((len(foi), n_times))
    if np.any(valid):
        ap_hz = 10.0 ** (offsets[valid][np.newaxis, :] - exponents[valid][np.newaxis, :] * log_foi[:, np.newaxis])
        aperiodic_power[:, valid] = ap_hz * hz_to_oct[:, np.newaxis]

    if component == "aperiodic":
        return aperiodic_power

    periodic_power = np.maximum(model_power - aperiodic_power, 0.0)
    return periodic_power

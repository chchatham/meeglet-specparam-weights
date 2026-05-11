"""End-to-end pipeline: signal in → ReconstructionResult out.

Supports multiple separation strategies for the aperiodic component:
  - 'subtraction' (default): aperiodic = original - periodic_excess.
    Exact decomposition but suppresses aperiodic power at peak frequencies.
  - 'wiener': aperiodic via Wiener filter weights. Preserves correct aperiodic
    power at all frequencies but contaminates the waveform with periodic phase.
  - 'state_space': Kalman smoother with damped oscillators + AR(p) aperiodic.
    Uses temporal structure for proper separation of induced oscillations.

For periodic and full components, wavelet-domain weighting is used directly.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np

from .wavelet_analysis import WaveletDecomposition, wavelet_decompose
from .time_resolved_fit import TimeResolvedFit, time_resolved_fit
from .weight_surface import WeightSurface, compute_weight_surface
from .synthesis import synthesize
from .separation import (
    SeparationResult,
    subtraction_separate,
    wiener_separate,
    decomposition_bias_estimate,
)
from .state_space import state_space_separate


@dataclass
class ReconstructionResult:
    reconstruction: np.ndarray  # (n_samples,) or (n_channels, n_samples)
    residual: np.ndarray  # (n_samples,) or (n_channels, n_samples)
    fit: TimeResolvedFit
    weights: WeightSurface
    energy_ratio: float
    decomposition: WaveletDecomposition
    frame_condition: float = 1.0
    method: str = "weight"
    bias_estimate: np.ndarray | None = field(default=None, repr=False)


def meeglet_specparam_reconstruct(
    signal: np.ndarray,
    sfreq: float,
    component: str = "aperiodic",
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
    separation: str | None = None,
    aperiodic_method: str | None = None,
) -> ReconstructionResult:
    """End-to-end spectral decomposition: signal → aperiodic/periodic time-domain signal.

    Parameters
    ----------
    signal : np.ndarray
        1D (n_samples,) or 2D (n_channels, n_samples) input signal.
    sfreq : float
        Sampling frequency in Hz.
    component : str
        Which component to reconstruct: 'aperiodic', 'periodic', or 'full'.
    foi_start, foi_end : float
        Frequency range for wavelet analysis.
    bw_oct : float
        Wavelet bandwidth in octaves.
    delta_oct : float or None
        Frequency spacing in octaves. Defaults to bw_oct / 4.
    fit_stride : int
        Fit specparam every N samples.
    power_window : int or None
        Samples to average power over for fitting.
    smooth_sigma : float or None
        Gaussian smoothing of aperiodic parameter trajectories.
    eps : float
        Floor for weight denominator.
    max_weight : float
        Maximum allowed weight.
    freq_range : list or None
        Frequency range for specparam fitting.
    peak_width_limits, max_n_peaks, min_peak_height : specparam settings.
    aperiodic_mode : str
        'fixed' or 'knee'.
    edge_taper : bool
        Taper reconstruction at signal edges.
    n_iter : int
        Iterative refinement steps for OLA synthesis.
    separation : str or None
        Separation strategy for component='aperiodic':
        'subtraction' (default): aperiodic = original - periodic_reconstruction.
        'wiener': aperiodic via Wiener filter weights directly.
        'state_space': Kalman oscillator + AR(p) decomposition.
        Ignored for component='periodic' or 'full'.
    aperiodic_method : str or None
        Deprecated. Use ``separation`` instead.
    """
    resolved_sep = _resolve_separation(separation, aperiodic_method)

    signal = np.asarray(signal, dtype=np.float64)

    decomposition = wavelet_decompose(
        signal, sfreq,
        foi_start=foi_start, foi_end=foi_end,
        bw_oct=bw_oct, delta_oct=delta_oct,
    )

    fit = time_resolved_fit(
        decomposition,
        fit_stride=fit_stride,
        power_window=power_window,
        smooth_sigma=smooth_sigma,
        freq_range=freq_range,
        peak_width_limits=peak_width_limits,
        max_n_peaks=max_n_peaks,
        min_peak_height=min_peak_height,
        aperiodic_mode=aperiodic_mode,
    )

    use_separation = component == "aperiodic" and resolved_sep in ("subtraction", "wiener", "state_space")

    if use_separation:
        if resolved_sep == "state_space":
            sep_result = state_space_separate(
                signal, decomposition, fit, sfreq,
                n_iter=n_iter,
            )
        elif resolved_sep == "subtraction":
            sep_result = subtraction_separate(
                signal, decomposition, fit,
                eps=eps, max_weight=max_weight,
                n_iter=n_iter, edge_taper=edge_taper,
            )
        else:
            sep_result = wiener_separate(
                signal, decomposition, fit,
                eps=eps, max_weight=max_weight,
                n_iter=n_iter, edge_taper=edge_taper,
            )

        weights = sep_result.weights
        if weights is None:
            weights = compute_weight_surface(
                decomposition, fit,
                component="aperiodic",
                eps=eps, max_weight=max_weight,
            )

        return ReconstructionResult(
            reconstruction=sep_result.aperiodic,
            residual=sep_result.periodic,
            fit=fit,
            weights=weights,
            energy_ratio=sep_result.energy_ratio,
            decomposition=decomposition,
            frame_condition=sep_result.frame_condition,
            method=sep_result.method,
            bias_estimate=sep_result.bias_estimate,
        )

    weights = compute_weight_surface(
        decomposition, fit,
        component=component,
        eps=eps,
        max_weight=max_weight,
    )

    reconstruction, energy_ratio, frame_condition = synthesize(
        decomposition, weights,
        edge_taper=edge_taper,
        n_iter=n_iter,
    )

    residual = signal - reconstruction

    return ReconstructionResult(
        reconstruction=reconstruction,
        residual=residual,
        fit=fit,
        weights=weights,
        energy_ratio=energy_ratio,
        decomposition=decomposition,
        frame_condition=frame_condition,
        method="weight",
        bias_estimate=None,
    )


def _resolve_separation(
    separation: str | None,
    aperiodic_method: str | None,
) -> str:
    """Resolve the separation parameter, handling the deprecated aperiodic_method."""
    if separation is not None and aperiodic_method is not None:
        raise ValueError(
            "Cannot specify both 'separation' and 'aperiodic_method'. "
            "Use 'separation' (aperiodic_method is deprecated)."
        )

    if aperiodic_method is not None:
        warnings.warn(
            "aperiodic_method is deprecated; use separation instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        result = aperiodic_method
    elif separation is not None:
        result = separation
    else:
        result = "subtraction"

    valid = ("subtraction", "wiener", "state_space")
    if result not in valid:
        param_name = "aperiodic_method" if aperiodic_method is not None else "separation"
        raise ValueError(
            f"{param_name} must be one of {valid}, got '{result}'"
        )
    return result

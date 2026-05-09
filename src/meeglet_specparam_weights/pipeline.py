"""End-to-end pipeline: signal in → ReconstructionResult out."""

from dataclasses import dataclass

import numpy as np

from .wavelet_analysis import WaveletDecomposition, wavelet_decompose
from .time_resolved_fit import TimeResolvedFit, time_resolved_fit
from .weight_surface import WeightSurface, compute_weight_surface
from .synthesis import synthesize


@dataclass
class ReconstructionResult:
    reconstruction: np.ndarray  # time-domain signal, shape (n_samples,)
    residual: np.ndarray  # original - reconstruction, shape (n_samples,)
    fit: TimeResolvedFit
    weights: WeightSurface
    energy_ratio: float
    decomposition: WaveletDecomposition


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
) -> ReconstructionResult:
    """End-to-end spectral decomposition: signal → aperiodic/periodic time-domain signal.

    Parameters
    ----------
    signal : np.ndarray
        1D input signal.
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
    """
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

    weights = compute_weight_surface(
        decomposition, fit,
        component=component,
        eps=eps,
        max_weight=max_weight,
    )

    reconstruction, energy_ratio = synthesize(
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
    )

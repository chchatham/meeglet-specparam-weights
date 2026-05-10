"""End-to-end pipeline: signal in → ReconstructionResult out.

For the aperiodic component (default), uses a subtraction approach:
  1. Compute excess weights: w_excess = sqrt(max(0, 1 - P_aperiodic/|Z|²))
  2. Synthesize the periodic excess: periodic = OLA(Z * w_excess)
  3. Define aperiodic by subtraction: aperiodic = original - periodic

This guarantees original = aperiodic + periodic exactly and preserves the
full 1/f power at peak frequencies in the aperiodic reconstruction. The
legacy Wiener filter approach (aperiodic_method='wiener') is also available.
"""

from dataclasses import dataclass

import numpy as np

from .wavelet_analysis import WaveletDecomposition, wavelet_decompose
from .time_resolved_fit import TimeResolvedFit, time_resolved_fit
from .weight_surface import WeightSurface, compute_weight_surface
from .synthesis import synthesize


@dataclass
class ReconstructionResult:
    reconstruction: np.ndarray  # (n_samples,) or (n_channels, n_samples)
    residual: np.ndarray  # (n_samples,) or (n_channels, n_samples)
    fit: TimeResolvedFit
    weights: WeightSurface
    energy_ratio: float
    decomposition: WaveletDecomposition
    frame_condition: float = 1.0  # B/A of the frame operator; 1.0 = tight frame
    method: str = "weight"  # "weight" or "subtraction"


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
    aperiodic_method: str = "subtraction",
) -> ReconstructionResult:
    """End-to-end spectral decomposition: signal → aperiodic/periodic time-domain signal.

    For component='aperiodic', the default method ('subtraction') extracts the
    periodic excess first via periodic weights, then subtracts it from the original
    signal. This preserves the full 1/f power at peak frequencies. The legacy
    'wiener' method scales coefficients by sqrt(P_aperiodic / |Z|²), which
    attenuates oscillations in the aperiodic reconstruction.

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
    aperiodic_method : str
        'subtraction' (default): aperiodic = original - periodic_reconstruction.
        'wiener' (legacy): aperiodic via Wiener filter weights directly.
        Only affects component='aperiodic'; ignored for other components.
    """
    if aperiodic_method not in ("subtraction", "wiener"):
        raise ValueError(
            f"aperiodic_method must be 'subtraction' or 'wiener', got '{aperiodic_method}'"
        )

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

    use_subtraction = (component == "aperiodic" and aperiodic_method == "subtraction")

    if use_subtraction:
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
        reconstruction = signal - periodic_recon
        residual = periodic_recon
        recon_energy = float(np.sum(reconstruction ** 2))
        empirical_energy = float(np.sum(signal ** 2))
        energy_ratio = recon_energy / max(empirical_energy, 1e-30)
        return ReconstructionResult(
            reconstruction=reconstruction,
            residual=residual,
            fit=fit,
            weights=excess_weights,
            energy_ratio=energy_ratio,
            decomposition=decomposition,
            frame_condition=frame_condition,
            method="subtraction",
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
    )

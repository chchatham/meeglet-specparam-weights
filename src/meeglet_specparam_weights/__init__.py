from .wavelet_analysis import WaveletDecomposition, wavelet_decompose
from .time_resolved_fit import TimeResolvedFit, time_resolved_fit
from .weight_surface import WeightSurface, compute_weight_surface
from .synthesis import synthesize
from .pipeline import ReconstructionResult, meeglet_specparam_reconstruct
from .coupling import (
    AperiodicCouplingResult,
    aperiodic_virtual_channels,
    compute_aperiodic_csd,
    aperiodic_amplitude_correlation,
    effective_dof,
    wavelet_effective_dof,
)
from .diagnostics import (
    plot_fit_quality,
    plot_weight_surface,
    plot_decomposition,
    plot_parameter_trajectories,
    plot_aperiodic_coupling,
)

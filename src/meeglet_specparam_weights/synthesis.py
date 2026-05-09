"""Synthesis: weighted wavelet coefficients → time-domain signal via overlap-add."""

import numpy as np
from scipy.signal import fftconvolve

import meeglet

from .wavelet_analysis import WaveletDecomposition
from .weight_surface import WeightSurface


def _get_wavelets(decomposition: WaveletDecomposition):
    return meeglet.define_wavelets(
        foi=decomposition.foi,
        sigma_time=decomposition.sigma_time,
        sfreq=decomposition.sfreq,
        kernel_width=decomposition.kernel_width,
        window_shift=1.0,
        density=decomposition.density,
    )


def _ola_synthesis(
    coefficients: np.ndarray,
    wavelets,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-pass OLA synthesis with normalization envelope."""
    reconstruction = np.zeros(n_samples)
    norm_envelope = np.zeros(n_samples)

    for i_freq, (kernel, scaling, _n_samp_eff, _n_shift) in enumerate(wavelets):
        k = kernel[:, 0]
        k_conj = np.conj(k)

        recon_freq = fftconvolve(
            np.real(coefficients[i_freq] * scaling),
            np.real(k_conj),
            mode="same",
        ) + fftconvolve(
            np.imag(coefficients[i_freq] * scaling),
            np.imag(k_conj),
            mode="same",
        )
        reconstruction += recon_freq

        norm_freq = fftconvolve(
            np.ones(n_samples) * scaling ** 2,
            np.abs(k) ** 2,
            mode="same",
        )
        norm_envelope += norm_freq

    return reconstruction, norm_envelope


def _ola_synthesis_signal_only(
    coefficients: np.ndarray,
    wavelets,
    n_samples: int,
) -> np.ndarray:
    """OLA synthesis without recomputing the normalization envelope."""
    reconstruction = np.zeros(n_samples)
    for i_freq, (kernel, scaling, _n_samp_eff, _n_shift) in enumerate(wavelets):
        k = kernel[:, 0]
        k_conj = np.conj(k)
        recon_freq = fftconvolve(
            np.real(coefficients[i_freq] * scaling),
            np.real(k_conj),
            mode="same",
        ) + fftconvolve(
            np.imag(coefficients[i_freq] * scaling),
            np.imag(k_conj),
            mode="same",
        )
        reconstruction += recon_freq
    return reconstruction


def _analyze(signal: np.ndarray, wavelets) -> np.ndarray:
    """Forward wavelet analysis (matching wavelet_decompose but without dataclass)."""
    n_freqs = len(wavelets)
    n_samples = len(signal)
    coefficients = np.empty((n_freqs, n_samples), dtype=np.complex128)
    for i_freq, (kernel, scaling, _n_samp_eff, _n_shift) in enumerate(wavelets):
        k = kernel[:, 0]
        coefficients[i_freq] = fftconvolve(signal, k, mode="same") * scaling
    return coefficients


def synthesize(
    decomposition: WaveletDecomposition,
    weights: WeightSurface,
    edge_taper: bool = True,
    n_iter: int = 1,
) -> tuple[np.ndarray, float]:
    """Synthesize a time-domain signal from weighted wavelet coefficients.

    Uses OLA synthesis with optional iterative refinement (Landweber iteration)
    to compensate for the non-tight frame. Each iteration re-analyzes the
    current reconstruction, computes residual coefficients, and adds a
    damped correction.

    Parameters
    ----------
    decomposition : WaveletDecomposition
        Original wavelet decomposition.
    weights : WeightSurface
        Weight surface to apply.
    edge_taper : bool
        If True, taper edge samples where normalization is weak.
    n_iter : int
        Number of iterative refinement steps. 1 = single-pass OLA (default).

    Returns
    -------
    reconstruction : np.ndarray
        Reconstructed time-domain signal, shape (n_samples,).
    energy_ratio : float
        ||reconstruction||² / ||original_energy|| where original_energy
        is estimated from the wavelet coefficients.
    """
    Z = decomposition.coefficients
    w = weights.weights
    n_freqs, n_samples = Z.shape

    target_Z = Z * w
    wavelets = _get_wavelets(decomposition)

    raw_recon, norm_envelope = _ola_synthesis(target_Z, wavelets, n_samples)
    safe_norm = np.maximum(norm_envelope, 1e-30)
    reconstruction = raw_recon / safe_norm

    if n_iter > 1:
        reanalyzed = _analyze(reconstruction, wavelets)
        frame_norm = np.sqrt(
            np.sum(np.abs(reanalyzed) ** 2)
            / max(np.sum(np.abs(target_Z) ** 2), 1e-30)
        )
        mu = 1.0 / max(frame_norm, 1.0)

        for _ in range(n_iter - 1):
            reanalyzed = _analyze(reconstruction, wavelets)
            residual_Z = target_Z - reanalyzed
            correction_raw = _ola_synthesis_signal_only(residual_Z, wavelets, n_samples)
            reconstruction = reconstruction + mu * correction_raw / safe_norm

    if edge_taper:
        max_kernel_half = max(len(wk[0]) for wk, *_ in wavelets) // 2
        taper = np.ones(n_samples)
        if max_kernel_half > 0 and n_samples > 2 * max_kernel_half:
            ramp = np.linspace(0, 1, max_kernel_half)
            taper[:max_kernel_half] = ramp
            taper[-max_kernel_half:] = ramp[::-1]
            reconstruction *= taper

    empirical_energy = np.sum(np.abs(Z) ** 2)
    recon_energy = np.sum(reconstruction ** 2)
    energy_ratio = recon_energy / max(empirical_energy, 1e-30)

    return reconstruction, energy_ratio

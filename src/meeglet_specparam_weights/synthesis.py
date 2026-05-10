"""Synthesis via overlap-add, implementing a frame multiplier (Balazs 2007).

The pipeline Z → w·Z → synthesis is formally a frame multiplier: a diagonal
operator in the wavelet coefficient domain followed by frame reconstruction.
The normalization envelope is the diagonal of the frame operator; its
min/max give the frame bounds A, B. The condition number B/A measures how
far the wavelet family deviates from a tight frame (B/A = 1).
"""

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
    compute_norm: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """OLA synthesis with optional normalization envelope computation."""
    reconstruction = np.zeros(n_samples)
    norm_envelope = np.zeros(n_samples) if compute_norm else None

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

        if compute_norm:
            norm_freq = fftconvolve(
                np.ones(n_samples) * scaling ** 2,
                np.abs(k) ** 2,
                mode="same",
            )
            norm_envelope += norm_freq

    return reconstruction, norm_envelope


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
) -> tuple[np.ndarray, float, float]:
    """Synthesize a time-domain signal from weighted wavelet coefficients.

    Accepts single-channel (2D coefficients) or multi-channel (3D coefficients).
    For multi-channel, each channel is synthesized independently.

    Returns
    -------
    reconstruction : np.ndarray
        (n_samples,) for single-channel or (n_channels, n_samples) for multi-channel.
    energy_ratio : float
        Aggregate ||reconstruction||² / ||original_energy||.
    frame_condition : float
        B/A ratio of the frame operator (normalization envelope). Values near 1.0
        indicate a tight frame; larger values indicate poorer reconstruction fidelity.
    """
    Z = decomposition.coefficients
    w = weights.weights
    multichannel = Z.ndim == 3

    if not multichannel:
        return _synthesize_single(Z, w, decomposition, edge_taper, n_iter)

    n_channels = Z.shape[0]
    n_samples = Z.shape[2]
    all_recon = np.empty((n_channels, n_samples))
    total_recon_energy = 0.0
    total_empirical_energy = 0.0
    max_frame_condition = 1.0

    for ch in range(n_channels):
        recon_ch, _, fc = _synthesize_single(Z[ch], w[ch], decomposition, edge_taper, n_iter)
        all_recon[ch] = recon_ch
        total_recon_energy += np.sum(recon_ch ** 2)
        total_empirical_energy += np.sum(np.abs(Z[ch]) ** 2)
        max_frame_condition = max(max_frame_condition, fc)

    energy_ratio = total_recon_energy / max(total_empirical_energy, 1e-30)
    return all_recon, energy_ratio, max_frame_condition


def _synthesize_single(
    Z_ch: np.ndarray,
    w_ch: np.ndarray,
    decomposition: WaveletDecomposition,
    edge_taper: bool,
    n_iter: int,
) -> tuple[np.ndarray, float, float]:
    """Synthesize a single channel."""
    n_freqs, n_samples = Z_ch.shape

    target_Z = Z_ch * w_ch
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
            correction_raw = _ola_synthesis(residual_Z, wavelets, n_samples, compute_norm=False)[0]
            reconstruction = reconstruction + mu * correction_raw / safe_norm

    if edge_taper:
        max_kernel_half = max(len(wk[0]) for wk, *_ in wavelets) // 2
        taper = np.ones(n_samples)
        if max_kernel_half > 0 and n_samples > 2 * max_kernel_half:
            ramp = np.linspace(0, 1, max_kernel_half)
            taper[:max_kernel_half] = ramp
            taper[-max_kernel_half:] = ramp[::-1]
            reconstruction *= taper

    # Frame bounds from normalization envelope: A = min, B = max, condition = B/A
    interior = slice(n_samples // 4, 3 * n_samples // 4)
    A = float(np.min(norm_envelope[interior]))
    B = float(np.max(norm_envelope[interior]))
    frame_condition = B / max(A, 1e-30)

    empirical_energy = np.sum(np.abs(Z_ch) ** 2)
    recon_energy = np.sum(reconstruction ** 2)
    energy_ratio = recon_energy / max(empirical_energy, 1e-30)

    return reconstruction, energy_ratio, frame_condition

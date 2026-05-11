"""Spectral PCA: multivariate aperiodic/periodic decomposition via CSD eigenmodes.

Eigendecomposes the cross-spectral density matrix at each frequency to obtain
orthogonal spatial modes whose eigenvalue spectra are real, non-negative power
spectra — ideal input for specparam fitting. Separation in this decorrelated
basis is optimal (MMSE) and produces smoother, more stable parameter estimates
than independent per-channel fitting.

Algorithm:
  1. CSD: S(f) = (1/T) Σ_t Z(:,f,t) · Z(:,f,t)^H
  2. Eigendecompose: S(f) = U(f) Λ(f) U(f)^H
  3. Project to PC space: Z_pc = U^H · Z
  4. Fit specparam to |Z_pc|² per mode
  5. Weight in PC space, project back, synthesize, subtract
"""

from dataclasses import dataclass

import numpy as np

from .wavelet_analysis import WaveletDecomposition, wavelet_decompose
from .time_resolved_fit import TimeResolvedFit, time_resolved_fit
from .weight_surface import WeightSurface, compute_weight_surface
from .synthesis import synthesize


@dataclass
class SpectralPCAResult:
    """Result of spectral PCA decomposition."""

    aperiodic: np.ndarray  # (n_channels, n_samples)
    periodic: np.ndarray  # (n_channels, n_samples)
    eigenvectors: np.ndarray  # (n_channels, n_modes, n_freqs) — spatial modes
    eigenvalues: np.ndarray  # (n_modes, n_freqs) — eigenvalue spectra
    mode_fit: TimeResolvedFit  # specparam fit in PC space
    mode_periodic: np.ndarray  # (n_modes, n_samples)
    mode_aperiodic: np.ndarray  # (n_modes, n_samples)
    n_modes: int
    variance_explained: np.ndarray  # (n_modes,)
    csd: np.ndarray  # (n_ch, n_ch, n_freqs)
    decomposition: WaveletDecomposition
    energy_ratio: float
    frame_condition: float
    separation: str


def compute_csd(
    decomposition: WaveletDecomposition,
) -> np.ndarray:
    """Compute the cross-spectral density matrix at each frequency.

    Parameters
    ----------
    decomposition : WaveletDecomposition
        Must be multi-channel (coefficients shape: n_channels, n_freqs, n_times).

    Returns
    -------
    csd : np.ndarray
        Complex, shape (n_channels, n_channels, n_freqs). Hermitian PSD.
    """
    Z = decomposition.coefficients
    if Z.ndim != 3:
        raise ValueError(
            f"compute_csd requires multi-channel input (3D coefficients), "
            f"got {Z.ndim}D"
        )

    n_ch, n_freqs, n_times = Z.shape

    has_nans = np.any(np.isnan(Z))
    if not has_nans:
        csd = np.einsum("ift,jft->ijf", Z, Z.conj()) / n_times
    else:
        csd = np.zeros((n_ch, n_ch, n_freqs), dtype=np.complex128)
        for f_idx in range(n_freqs):
            data_f = Z[:, f_idx, :]
            valid = ~np.any(np.isnan(data_f), axis=0)
            n_valid = int(np.sum(valid))
            if n_valid > 0:
                data_valid = data_f[:, valid]
                csd[:, :, f_idx] = data_valid @ data_valid.conj().T / n_valid

    return csd


def spectral_pca_decompose(
    decomposition: WaveletDecomposition,
    n_modes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eigendecompose the CSD to obtain spatial modes and eigenvalue spectra.

    Parameters
    ----------
    decomposition : WaveletDecomposition
        Multi-channel wavelet decomposition.
    n_modes : int or None
        Number of modes to retain. None retains all (n_channels).

    Returns
    -------
    eigenvectors : np.ndarray
        Complex, (n_channels, n_modes, n_freqs). Spatial modes.
    eigenvalues : np.ndarray
        Real, (n_modes, n_freqs). Power spectrum of each mode.
    csd : np.ndarray
        Complex, (n_channels, n_channels, n_freqs).
    """
    csd = compute_csd(decomposition)
    n_ch, _, n_freqs = csd.shape

    if n_modes is None:
        n_modes = n_ch
    n_modes = min(n_modes, n_ch)

    eigenvalues = np.empty((n_modes, n_freqs))
    eigenvectors = np.empty((n_ch, n_modes, n_freqs), dtype=np.complex128)

    for f_idx in range(n_freqs):
        vals, vecs = np.linalg.eigh(csd[:, :, f_idx])
        # eigh returns ascending order; reverse to descending
        vals = vals[::-1]
        vecs = vecs[:, ::-1]
        eigenvalues[:, f_idx] = vals[:n_modes]
        eigenvectors[:, :, f_idx] = vecs[:, :n_modes]

    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Sign-align eigenvectors across frequencies
    for k in range(n_modes):
        for f_idx in range(1, n_freqs):
            overlap = np.real(
                np.vdot(eigenvectors[:, k, f_idx - 1], eigenvectors[:, k, f_idx])
            )
            if overlap < 0:
                eigenvectors[:, k, f_idx] *= -1

    return eigenvectors, eigenvalues, csd


def _project_to_pc_space(
    Z: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Project channel-space coefficients to PC space.

    Z: (n_ch, n_freqs, n_times)
    eigenvectors: (n_ch, n_modes, n_freqs)
    Returns: (n_modes, n_freqs, n_times)
    """
    return np.einsum("cmf,cft->mft", eigenvectors.conj(), Z)


def _project_to_channel_space(
    Z_pc: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Project PC-space coefficients back to channel space.

    Z_pc: (n_modes, n_freqs, n_times)
    eigenvectors: (n_ch, n_modes, n_freqs)
    Returns: (n_ch, n_freqs, n_times)
    """
    return np.einsum("cmf,mft->cft", eigenvectors, Z_pc)


def spectral_pca_reconstruct(
    signal: np.ndarray,
    sfreq: float,
    n_modes: int | None = None,
    separation: str = "subtraction",
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
) -> SpectralPCAResult:
    """Spectral PCA decomposition: multi-channel signal in, aperiodic + periodic out.

    Eigendecomposes the cross-spectral density, fits specparam in the
    decorrelated PC space, and reconstructs per-channel time-domain signals.

    Parameters
    ----------
    signal : np.ndarray
        2D input (n_channels, n_samples). Must have >= 2 channels.
    sfreq : float
        Sampling frequency in Hz.
    n_modes : int or None
        Number of PCA modes. None = all channels (no rank reduction).
    separation : str
        'subtraction' (default) or 'wiener'.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2:
        raise ValueError(
            f"spectral_pca_reconstruct requires 2D (n_channels, n_samples) "
            f"input, got {signal.ndim}D"
        )
    if signal.shape[0] < 2:
        raise ValueError(
            f"spectral_pca_reconstruct requires >= 2 channels, "
            f"got {signal.shape[0]}"
        )

    valid_sep = ("subtraction", "wiener")
    if separation not in valid_sep:
        raise ValueError(
            f"separation must be one of {valid_sep}, got '{separation}'"
        )

    n_channels, n_samples = signal.shape

    # Step 1: Wavelet decomposition
    decomposition = wavelet_decompose(
        signal, sfreq,
        foi_start=foi_start, foi_end=foi_end,
        bw_oct=bw_oct, delta_oct=delta_oct,
    )

    # Step 2: CSD eigendecomposition
    eigenvectors, eigenvalues, csd = spectral_pca_decompose(
        decomposition, n_modes=n_modes,
    )
    n_modes_actual = eigenvectors.shape[1]

    # Step 3: Project to PC space
    Z = decomposition.coefficients
    Z_pc = _project_to_pc_space(Z, eigenvectors)

    # Step 4: Create PC-space WaveletDecomposition (modes as "channels")
    ref = decomposition
    pc_decomposition = WaveletDecomposition(
        coefficients=Z_pc,
        foi=ref.foi,
        sigma_time=ref.sigma_time,
        sigma_freq=ref.sigma_freq,
        times=ref.times,
        sfreq=sfreq,
        bw_oct=ref.bw_oct,
        delta_oct=ref.delta_oct,
        kernel_width=ref.kernel_width,
        density=ref.density,
        n_channels=n_modes_actual,
    )

    # Step 5: Fit specparam in PC space
    mode_fit = time_resolved_fit(
        pc_decomposition,
        fit_stride=fit_stride,
        power_window=power_window,
        smooth_sigma=smooth_sigma,
        freq_range=freq_range,
        peak_width_limits=peak_width_limits,
        max_n_peaks=max_n_peaks,
        min_peak_height=min_peak_height,
        aperiodic_mode=aperiodic_mode,
    )

    # Step 6: Compute weights in PC space
    aperiodic_weights = compute_weight_surface(
        pc_decomposition, mode_fit,
        component="aperiodic",
        eps=eps, max_weight=max_weight,
    )

    # Step 7: Weight coefficients in PC space and project back
    synthesis_n_iter = max(n_iter, 5) if separation == "subtraction" else n_iter

    if separation == "subtraction":
        w_ap = aperiodic_weights.weights
        excess_w = np.sqrt(np.maximum(0.0, 1.0 - w_ap ** 2))
        Z_pc_periodic = Z_pc * excess_w
        Z_ch_periodic = _project_to_channel_space(Z_pc_periodic, eigenvectors)

        per_decomp = WaveletDecomposition(
            coefficients=Z_ch_periodic,
            foi=ref.foi, sigma_time=ref.sigma_time, sigma_freq=ref.sigma_freq,
            times=ref.times, sfreq=sfreq, bw_oct=ref.bw_oct,
            delta_oct=ref.delta_oct, kernel_width=ref.kernel_width,
            density=ref.density, n_channels=n_channels,
        )
        unit_weights = WeightSurface(
            weights=np.ones(Z_ch_periodic.shape, dtype=np.float64),
            component="periodic", eps=eps, max_weight=1.0,
        )
        periodic_recon, _, frame_condition = synthesize(
            per_decomp, unit_weights,
            edge_taper=edge_taper, n_iter=synthesis_n_iter,
        )
        aperiodic_recon = signal - periodic_recon

    else:  # wiener
        Z_pc_aperiodic = Z_pc * aperiodic_weights.weights
        Z_ch_aperiodic = _project_to_channel_space(Z_pc_aperiodic, eigenvectors)

        ap_decomp = WaveletDecomposition(
            coefficients=Z_ch_aperiodic,
            foi=ref.foi, sigma_time=ref.sigma_time, sigma_freq=ref.sigma_freq,
            times=ref.times, sfreq=sfreq, bw_oct=ref.bw_oct,
            delta_oct=ref.delta_oct, kernel_width=ref.kernel_width,
            density=ref.density, n_channels=n_channels,
        )
        unit_weights = WeightSurface(
            weights=np.ones(Z_ch_aperiodic.shape, dtype=np.float64),
            component="aperiodic", eps=eps, max_weight=1.0,
        )
        aperiodic_recon, _, frame_condition = synthesize(
            ap_decomp, unit_weights,
            edge_taper=edge_taper, n_iter=synthesis_n_iter,
        )
        periodic_recon = signal - aperiodic_recon

    # Step 8: Per-mode time-domain signals (approximate — no "original" in mode space)
    mode_periodic = np.empty((n_modes_actual, n_samples))
    mode_aperiodic = np.empty((n_modes_actual, n_samples))

    if separation == "subtraction":
        Z_pc_ap = Z_pc * aperiodic_weights.weights
    else:
        Z_pc_periodic = Z_pc * np.sqrt(
            np.maximum(0.0, 1.0 - aperiodic_weights.weights ** 2)
        )
        Z_pc_ap = Z_pc * aperiodic_weights.weights

    for k in range(n_modes_actual):
        # Periodic mode signal
        mode_per_dec = WaveletDecomposition(
            coefficients=Z_pc_periodic[k],
            foi=ref.foi, sigma_time=ref.sigma_time, sigma_freq=ref.sigma_freq,
            times=ref.times, sfreq=sfreq, bw_oct=ref.bw_oct,
            delta_oct=ref.delta_oct, kernel_width=ref.kernel_width,
            density=ref.density, n_channels=1,
        )
        mode_per_ws = WeightSurface(
            weights=np.ones_like(np.abs(Z_pc_periodic[k]), dtype=np.float64),
            component="periodic", eps=eps, max_weight=1.0,
        )
        mode_periodic[k], _, _ = synthesize(
            mode_per_dec, mode_per_ws,
            edge_taper=edge_taper, n_iter=synthesis_n_iter,
        )

        # Aperiodic mode signal
        mode_ap_dec = WaveletDecomposition(
            coefficients=Z_pc_ap[k],
            foi=ref.foi, sigma_time=ref.sigma_time, sigma_freq=ref.sigma_freq,
            times=ref.times, sfreq=sfreq, bw_oct=ref.bw_oct,
            delta_oct=ref.delta_oct, kernel_width=ref.kernel_width,
            density=ref.density, n_channels=1,
        )
        mode_ap_ws = WeightSurface(
            weights=np.ones_like(np.abs(Z_pc_ap[k]), dtype=np.float64),
            component="aperiodic", eps=eps, max_weight=1.0,
        )
        mode_aperiodic[k], _, _ = synthesize(
            mode_ap_dec, mode_ap_ws,
            edge_taper=edge_taper, n_iter=synthesis_n_iter,
        )

    # Variance explained
    total_power = eigenvalues.sum()
    variance_explained = eigenvalues.sum(axis=1) / max(total_power, 1e-30)

    # Energy ratio
    recon_energy = float(np.sum(aperiodic_recon ** 2))
    signal_energy = float(np.sum(signal ** 2))
    energy_ratio = recon_energy / max(signal_energy, 1e-30)

    return SpectralPCAResult(
        aperiodic=aperiodic_recon,
        periodic=periodic_recon,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        mode_fit=mode_fit,
        mode_periodic=mode_periodic,
        mode_aperiodic=mode_aperiodic,
        n_modes=n_modes_actual,
        variance_explained=variance_explained,
        csd=csd,
        decomposition=decomposition,
        energy_ratio=energy_ratio,
        frame_condition=frame_condition,
        separation=separation,
    )

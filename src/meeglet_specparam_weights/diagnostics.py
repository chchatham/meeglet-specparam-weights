"""Diagnostic visualizations for meeglet-specparam-weights results."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .pipeline import ReconstructionResult
from .coupling import AperiodicCouplingResult
from .separation import decomposition_bias_estimate


def plot_fit_quality(result: ReconstructionResult, ax: plt.Axes | None = None) -> Figure:
    """Plot r² over time as a line plot with quality threshold."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.get_figure()

    times = result.fit.times
    r2 = result.fit.r_squared

    ax.plot(times, r2, color="steelblue", linewidth=1)
    ax.axhline(0.85, color="orange", linestyle="--", linewidth=0.8, label="r²=0.85")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("r²")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Fit quality over time")
    fig.tight_layout()
    return fig


def plot_weight_surface(result: ReconstructionResult, ax: plt.Axes | None = None) -> Figure:
    """Plot the 2D weight surface as a heatmap (frequency × time)."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    W = result.weights.weights
    times = result.decomposition.times
    foi = result.decomposition.foi

    im = ax.pcolormesh(
        times, foi, W,
        shading="auto", cmap="viridis",
    )
    ax.set_yscale("log")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Weight surface ({result.weights.component})")
    plt.colorbar(im, ax=ax, label="weight")
    fig.tight_layout()
    return fig


def plot_decomposition(
    result: ReconstructionResult,
    time_range: tuple[float, float] | None = None,
    sfreq: float | None = None,
) -> Figure:
    """Plot original, reconstruction, and residual signals."""
    if sfreq is None:
        sfreq = result.decomposition.sfreq

    original = result.reconstruction + result.residual
    recon = result.reconstruction
    residual = result.residual

    n_samples = len(original)
    t = np.arange(n_samples) / sfreq

    if time_range is not None:
        mask = (t >= time_range[0]) & (t <= time_range[1])
    else:
        mask = np.ones(n_samples, dtype=bool)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t[mask], original[mask], color="black", linewidth=0.5)
    axes[0].set_ylabel("Original")
    axes[0].set_title("Signal decomposition")

    axes[1].plot(t[mask], recon[mask], color="steelblue", linewidth=0.5)
    axes[1].set_ylabel(f"{result.weights.component.capitalize()}")

    axes[2].plot(t[mask], residual[mask], color="coral", linewidth=0.5)
    axes[2].set_ylabel("Residual")
    axes[2].set_xlabel("Time (s)")

    fig.tight_layout()
    return fig


def plot_parameter_trajectories(result: ReconstructionResult) -> Figure:
    """Plot aperiodic parameters (offset, exponent) and peak params over time."""
    times = result.fit.times
    ap = result.fit.aperiodic_params

    n_plots = 2
    has_peaks = any(p.shape[0] > 0 for p in result.fit.peak_params)
    if has_peaks:
        n_plots = 3

    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=True)

    axes[0].plot(times, ap[:, 0], color="steelblue", linewidth=1)
    axes[0].set_ylabel("Offset")
    axes[0].set_title("Parameter trajectories")

    axes[1].plot(times, ap[:, 1], color="darkorange", linewidth=1)
    axes[1].set_ylabel("Exponent")

    if has_peaks:
        xs, ys, ss = [], [], []
        for t_idx, peaks in enumerate(result.fit.peak_params):
            if peaks.shape[0] > 0:
                xs.extend([times[t_idx]] * peaks.shape[0])
                ys.extend(peaks[:, 0])
                ss.extend(peaks[:, 1] * 50 + 1)
        if xs:
            axes[2].scatter(xs, ys, s=ss, c="green", alpha=0.3, edgecolors="none")
        axes[2].set_ylabel("Peak CF (Hz)")

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def plot_decomposition_bias(
    result: ReconstructionResult,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot expected power bias factor across frequencies.

    Shows the theoretical ratio of reconstructed to true aperiodic power
    at each frequency. Values near 1.0 are unbiased; values << 1.0 indicate
    the aperiodic is suppressed at that frequency (typically at peak locations).
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.get_figure()

    foi = result.decomposition.foi
    if result.bias_estimate is not None:
        bias = result.bias_estimate
    else:
        bias = decomposition_bias_estimate(
            result.decomposition, result.fit, method=result.method,
        )

    ax.semilogy(foi, bias, color="steelblue", linewidth=1.5)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0.5, color="orange", linestyle=":", linewidth=0.8, label="50% power retained")
    suppressed = bias < 0.5
    if np.any(suppressed):
        ax.fill_between(
            foi, bias, 0.5,
            where=suppressed, alpha=0.2, color="red", label="suppressed region",
        )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Bias factor (P_recon / P_true)")
    ax.set_xscale("log")
    ax.set_title(f"Decomposition bias ({result.method})")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_ylim(bottom=1e-3)
    fig.tight_layout()
    return fig


def plot_aperiodic_coupling(
    coupling_result: AperiodicCouplingResult,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot amplitude correlation heatmap with Nyquist cutoff line.

    Shows corr(exponent(t), |Z(ch, f, t)|) for each channel and frequency,
    with a vertical line at the effective Nyquist frequency above which
    coupling is not meaningful.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    corr = coupling_result.amplitude_correlation
    foi = coupling_result.foi
    nyquist = coupling_result.effective_nyquist

    if corr.ndim == 1:
        corr = corr[np.newaxis, :]

    n_ch = corr.shape[0]
    ch_labels = coupling_result.channel_labels[:n_ch]

    im = ax.imshow(
        corr, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
        extent=[np.log10(foi[0]), np.log10(foi[-1]), n_ch - 0.5, -0.5],
        interpolation="nearest",
    )
    ax.axvline(np.log10(nyquist), color="yellow", linestyle="--", linewidth=1.5,
               label=f"Nyquist = {nyquist:.1f} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Channel")
    ax.set_yticks(np.arange(n_ch))
    ax.set_yticklabels(ch_labels)
    tick_freqs = [f for f in [1, 2, 5, 10, 20, 50] if foi[0] <= f <= foi[-1]]
    ax.set_xticks([np.log10(f) for f in tick_freqs])
    ax.set_xticklabels([str(f) for f in tick_freqs])
    ax.set_title("Aperiodic-oscillatory amplitude correlation")
    ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(im, ax=ax, label="corr(exponent, amplitude)")
    fig.tight_layout()
    return fig

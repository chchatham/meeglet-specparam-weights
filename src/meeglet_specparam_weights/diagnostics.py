"""Diagnostic visualizations for meeglet-specparam-weights results."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .pipeline import ReconstructionResult


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

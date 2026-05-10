"""Generate all figures for the GitHub Pages site."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import welch, hilbert, butter, sosfilt
from scipy.ndimage import median_filter

from meeglet_specparam_weights import meeglet_specparam_reconstruct
from validation.metrics import generate_pink_noise, correlation

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")

DARK_BG = "#0d1117"
DARK_SURFACE = "#161b22"
DARK_TEXT = "#c9d1d9"
DARK_MUTED = "#8b949e"
DARK_BORDER = "#30363d"
BLUE = "#58a6ff"
GREEN = "#7ee787"
PURPLE = "#d2a8ff"
ORANGE = "#ffa657"
RED = "#ff7b72"
CYAN = "#79c0ff"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": DARK_SURFACE,
    "axes.edgecolor": DARK_BORDER,
    "axes.labelcolor": DARK_TEXT,
    "text.color": DARK_TEXT,
    "xtick.color": DARK_MUTED,
    "ytick.color": DARK_MUTED,
    "grid.color": DARK_BORDER,
    "grid.alpha": 0.5,
    "legend.facecolor": DARK_SURFACE,
    "legend.edgecolor": DARK_BORDER,
    "legend.labelcolor": DARK_TEXT,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "savefig.facecolor": DARK_BG,
})


def fig_decomposition():
    """Generate the hero decomposition figure: original → aperiodic + periodic."""
    print("  [1/7] Decomposition demo...")
    sfreq = 256.0
    n_samples = int(10 * sfreq)
    t = np.arange(n_samples) / sfreq
    rng = np.random.default_rng(42)
    pink = generate_pink_noise(sfreq, n_samples, 1.5, rng)
    alpha = 2.0 * np.sin(2 * np.pi * 10 * t)
    signal = pink + alpha

    result_ap = meeglet_specparam_reconstruct(
        signal, sfreq, component="aperiodic",
        foi_start=2.0, foi_end=50.0, bw_oct=0.5,
        fit_stride=50, power_window=400,
        freq_range=[1, 50], n_iter=5, edge_taper=True,
    )
    result_per = meeglet_specparam_reconstruct(
        signal, sfreq, component="periodic",
        foi_start=2.0, foi_end=50.0, bw_oct=0.5,
        fit_stride=50, power_window=400,
        freq_range=[1, 50], n_iter=5, edge_taper=True,
    )

    t0, t1 = 2.0, 5.0
    mask = (t >= t0) & (t <= t1)
    ts = t[mask]

    fig, axes = plt.subplots(4, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(ts, signal[mask], color=DARK_TEXT, linewidth=0.6, alpha=0.9)
    axes[0].set_ylabel("Original")
    axes[0].set_title("Signal Decomposition: Pink Noise (1/f^1.5) + 10 Hz Alpha")

    axes[1].plot(ts, result_ap.reconstruction[mask], color=BLUE, linewidth=0.6)
    axes[1].set_ylabel("Aperiodic")

    axes[2].plot(ts, result_per.reconstruction[mask], color=GREEN, linewidth=0.6)
    axes[2].set_ylabel("Periodic")

    axes[3].plot(ts, result_ap.residual[mask], color=ORANGE, linewidth=0.6, alpha=0.7)
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(True, alpha=0.2)

    fig.tight_layout(h_pad=0.4)
    fig.savefig(os.path.join(FIGDIR, "decomposition.png"))
    plt.close(fig)


def fig_weight_surface():
    """Generate the weight surface heatmap."""
    print("  [2/7] Weight surface...")
    sfreq = 256.0
    n_samples = int(10 * sfreq)
    t = np.arange(n_samples) / sfreq
    rng = np.random.default_rng(42)
    pink = generate_pink_noise(sfreq, n_samples, 1.5, rng)
    alpha = 2.0 * np.sin(2 * np.pi * 10 * t)
    signal = pink + alpha

    result = meeglet_specparam_reconstruct(
        signal, sfreq, component="aperiodic",
        foi_start=2.0, foi_end=50.0, bw_oct=0.5,
        fit_stride=50, power_window=400,
        freq_range=[1, 50], n_iter=5, edge_taper=True,
    )

    W = result.weights.weights
    times = result.decomposition.times
    foi = result.decomposition.foi

    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), gridspec_kw={"height_ratios": [3, 1]})

    im = axes[0].pcolormesh(times, foi, W, shading="auto", cmap="viridis", vmin=0, vmax=2.5)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_title("Aperiodic Weight Surface w(f, t)")
    axes[0].set_yticks([2, 5, 10, 20, 50])
    axes[0].set_yticklabels(["2", "5", "10", "20", "50"])
    cb = plt.colorbar(im, ax=axes[0], label="Weight", pad=0.01)
    cb.ax.yaxis.label.set_color(DARK_TEXT)
    cb.ax.tick_params(colors=DARK_MUTED)

    axes[1].plot(times, result.fit.r_squared, color=BLUE, linewidth=0.8)
    axes[1].axhline(0.85, color=ORANGE, linestyle="--", linewidth=0.8, alpha=0.7, label="r²=0.85")
    axes[1].set_ylabel("r²")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylim(0.5, 1.05)
    axes[1].legend(loc="lower left", fontsize=9)
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout(h_pad=0.6)
    fig.savefig(os.path.join(FIGDIR, "weight_surface.png"))
    plt.close(fig)


def fig_exponent_tracking():
    """Generate the non-stationary exponent tracking figure."""
    print("  [3/7] Exponent tracking...")
    sfreq = 256.0
    duration = 10.0
    n_samples = int(sfreq * duration)
    rng = np.random.default_rng(7)
    exp_start, exp_end = 1.5, 2.5
    true_exponent = np.linspace(exp_start, exp_end, n_samples)

    seg_len = int(sfreq * 2.0)
    hop = int(sfreq * 1.0)
    signal = np.zeros(n_samples)
    norm = np.zeros(n_samples)
    for start in range(0, n_samples - seg_len + 1, hop):
        end = start + seg_len
        mid = (start + end) // 2
        exp_local = true_exponent[mid]
        seg_white = rng.standard_normal(seg_len)
        seg_fft = np.fft.rfft(seg_white)
        seg_freqs = np.fft.rfftfreq(seg_len, d=1.0 / sfreq)
        seg_freqs[0] = 1.0
        seg_fft *= 1.0 / np.power(seg_freqs, exp_local / 2.0)
        seg = np.fft.irfft(seg_fft, n=seg_len)
        seg = seg / np.std(seg)
        window = np.hanning(seg_len)
        signal[start:end] += seg * window
        norm[start:end] += window
    norm = np.maximum(norm, 1e-30)
    signal = signal / norm

    result = meeglet_specparam_reconstruct(
        signal, sfreq, component="aperiodic",
        foi_start=2.0, foi_end=50.0, bw_oct=0.5,
        fit_stride=50, power_window=400, smooth_sigma=5.0,
        freq_range=[1, 50], edge_taper=True, n_iter=5,
    )

    t_sig = np.arange(n_samples) / sfreq
    t_fit = result.fit.times
    rec_exp = result.fit.aperiodic_params[:, 1]

    edge = int(2.0 * sfreq)
    edge_t = edge / sfreq

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, gridspec_kw={"height_ratios": [1, 1.5]})

    axes[0].plot(t_sig, signal, color=DARK_TEXT, linewidth=0.4, alpha=0.7)
    axes[0].set_ylabel("Signal")
    axes[0].set_title("Non-Stationary Tracking: Aperiodic Exponent Ramp (1.5 → 2.5)")
    axes[0].grid(True, alpha=0.2)

    true_interp = np.interp(t_fit, t_sig, true_exponent)
    axes[1].plot(t_fit, true_interp, color=GREEN, linewidth=2.0, label="True exponent", alpha=0.9)
    axes[1].plot(t_fit, rec_exp, color=BLUE, linewidth=1.5, label="Recovered exponent", alpha=0.9)
    axes[1].axvspan(0, edge_t, color=RED, alpha=0.08)
    axes[1].axvspan(duration - edge_t, duration, color=RED, alpha=0.08)
    axes[1].set_ylabel("Exponent")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper left", fontsize=10)
    axes[1].grid(True, alpha=0.2)

    n = min(len(true_interp), len(rec_exp))
    e = int(2.0 * sfreq / max(result.fit.fit_stride, 1))
    corr = correlation(true_interp[e:n - e], rec_exp[e:n - e])
    axes[1].text(0.98, 0.08, f"r = {corr:.2f}", transform=axes[1].transAxes,
                 ha="right", va="bottom", fontsize=12, fontweight="bold",
                 color=GREEN, bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_SURFACE, edgecolor=DARK_BORDER))

    fig.tight_layout(h_pad=0.4)
    fig.savefig(os.path.join(FIGDIR, "exponent_tracking.png"))
    plt.close(fig)


def fig_alpha_onoff():
    """Generate the alpha on/off tracking figure."""
    print("  [4/7] Alpha on/off tracking...")
    sfreq = 256.0
    duration = 10.0
    n_samples = int(sfreq * duration)
    t = np.arange(n_samples) / sfreq
    rng = np.random.default_rng(42)
    pink = generate_pink_noise(sfreq, n_samples, 1.5, rng)

    block_dur = 2.0
    block_samples = int(block_dur * sfreq)
    alpha_envelope = np.zeros(n_samples)
    on = True
    for start in range(0, n_samples, block_samples):
        end = min(start + block_samples, n_samples)
        if on:
            alpha_envelope[start:end] = 1.0
        on = not on

    alpha = 2.0 * alpha_envelope * np.sin(2 * np.pi * 10 * t)
    signal = pink + alpha

    result = meeglet_specparam_reconstruct(
        signal, sfreq, component="periodic",
        foi_start=2.0, foi_end=50.0, bw_oct=0.5,
        fit_stride=50, power_window=400, smooth_sigma=5.0,
        freq_range=[1, 50], edge_taper=True, n_iter=5,
    )

    recon = result.reconstruction

    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(t, signal, color=DARK_TEXT, linewidth=0.4, alpha=0.7)
    for start in range(0, n_samples, block_samples * 2):
        end = min(start + block_samples, n_samples)
        axes[0].axvspan(start / sfreq, end / sfreq, color=GREEN, alpha=0.07)
    axes[0].set_ylabel("Original")
    axes[0].set_title("Alpha On/Off Tracking: 2s Alternating Blocks")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(t, recon, color=GREEN, linewidth=0.5)
    for start in range(0, n_samples, block_samples * 2):
        end = min(start + block_samples, n_samples)
        axes[1].axvspan(start / sfreq, end / sfreq, color=GREEN, alpha=0.07)
    axes[1].set_ylabel("Periodic\nreconstruction")
    axes[1].grid(True, alpha=0.2)

    window_s = int(0.1 * sfreq)
    power_env = np.sqrt(np.convolve(recon ** 2, np.ones(window_s) / window_s, mode="same"))
    axes[2].plot(t, power_env, color=PURPLE, linewidth=1.0)
    for start in range(0, n_samples, block_samples * 2):
        end = min(start + block_samples, n_samples)
        axes[2].axvspan(start / sfreq, end / sfreq, color=GREEN, alpha=0.07)
    axes[2].set_ylabel("Power envelope")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.2)

    edge = int(1.0 * sfreq)
    recon_t = recon[edge:-edge]
    env_t = alpha_envelope[edge:-edge]
    on_power = np.mean(recon_t[env_t > 0.5] ** 2)
    off_power = np.mean(recon_t[env_t < 0.5] ** 2)
    contrast = on_power / max(off_power, 1e-30)
    axes[2].text(0.98, 0.85, f"Contrast: {contrast:.1f}x", transform=axes[2].transAxes,
                 ha="right", va="top", fontsize=12, fontweight="bold",
                 color=GREEN, bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_SURFACE, edgecolor=DARK_BORDER))

    fig.tight_layout(h_pad=0.4)
    fig.savefig(os.path.join(FIGDIR, "alpha_onoff.png"))
    plt.close(fig)


def fig_beta_bursts():
    """Generate the transient beta burst detection figure."""
    print("  [5/7] Beta burst detection...")
    sfreq = 256.0
    duration = 10.0
    n_samples = int(sfreq * duration)
    t = np.arange(n_samples) / sfreq
    rng = np.random.default_rng(42)
    pink = generate_pink_noise(sfreq, n_samples, 1.5, rng)

    burst_freq = 20.0
    burst_dur = 0.2
    burst_amp = 3.0
    n_bursts = 10
    burst_samples = int(burst_dur * sfreq)
    margin = int(1.0 * sfreq)
    available = n_samples - 2 * margin
    spacing = available // (n_bursts + 1)

    burst_centers = []
    signal = pink.copy()
    for i in range(n_bursts):
        center = margin + (i + 1) * spacing
        start = center - burst_samples // 2
        end = start + burst_samples
        window = np.hanning(burst_samples)
        burst = burst_amp * window * np.sin(2 * np.pi * burst_freq * t[start:end])
        signal[start:end] += burst
        burst_centers.append(center / sfreq)

    result = meeglet_specparam_reconstruct(
        signal, sfreq, component="periodic",
        foi_start=2.0, foi_end=50.0, bw_oct=0.5,
        fit_stride=50, power_window=400, smooth_sigma=5.0,
        freq_range=[1, 50], edge_taper=True, n_iter=5,
    )

    recon = result.reconstruction

    lo, hi = 15, 25
    sos = butter(4, [lo / (sfreq / 2), hi / (sfreq / 2)], btype="band", output="sos")
    filtered = sosfilt(sos, recon)
    envelope = np.abs(hilbert(filtered))
    baseline = median_filter(envelope, size=int(sfreq * 1))
    deviation = envelope - baseline
    mad = np.median(np.abs(deviation))
    threshold = 3.0 * mad
    detected = deviation > threshold

    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(t, signal, color=DARK_TEXT, linewidth=0.4, alpha=0.7)
    for bc in burst_centers:
        axes[0].axvspan(bc - 0.1, bc + 0.1, color=PURPLE, alpha=0.15)
    axes[0].set_ylabel("Original")
    axes[0].set_title("Transient Beta Burst Detection (200ms, 20 Hz)")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(t, recon, color=GREEN, linewidth=0.5)
    for bc in burst_centers:
        axes[1].axvspan(bc - 0.1, bc + 0.1, color=PURPLE, alpha=0.15)
    axes[1].set_ylabel("Periodic\nreconstruction")
    axes[1].grid(True, alpha=0.2)

    axes[2].plot(t, envelope, color=CYAN, linewidth=0.7, label="Beta envelope")
    axes[2].plot(t, baseline + threshold, color=RED, linewidth=0.8, linestyle="--", alpha=0.7, label="Threshold")
    axes[2].fill_between(t, 0, envelope.max() * 1.2, where=detected, color=GREEN, alpha=0.15)
    for bc in burst_centers:
        axes[2].axvline(bc, color=PURPLE, linewidth=0.5, alpha=0.4)
    axes[2].set_ylabel("Beta envelope")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(True, alpha=0.2)

    burst_half_width = int(0.15 * sfreq)
    hits = 0
    for bc in burst_centers:
        csamp = int(bc * sfreq)
        lo_s = max(0, csamp - burst_half_width)
        hi_s = min(len(detected), csamp + burst_half_width)
        if np.any(detected[lo_s:hi_s]):
            hits += 1
    rate = hits / len(burst_centers) * 100
    axes[2].text(0.98, 0.85, f"Detection: {rate:.0f}%", transform=axes[2].transAxes,
                 ha="right", va="top", fontsize=12, fontweight="bold",
                 color=GREEN, bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_SURFACE, edgecolor=DARK_BORDER))

    fig.tight_layout(h_pad=0.4)
    fig.savefig(os.path.join(FIGDIR, "beta_bursts.png"))
    plt.close(fig)


def fig_snr_robustness():
    """Generate the SNR robustness sweep figure."""
    print("  [6/7] SNR robustness sweep...")
    sfreq = 256.0
    duration = 10.0
    n_samples = int(sfreq * duration)
    snr_range = np.arange(-10, 25, 5)
    n_seeds = 3

    mean_r2s = []
    min_r2s = []
    alpha_sups = []

    for snr in snr_range:
        r2s_seed = []
        sups_seed = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            pink = generate_pink_noise(sfreq, n_samples, 1.5, rng)
            t_arr = np.arange(n_samples) / sfreq
            alpha_amp = np.sqrt(10.0 ** (snr / 10.0))
            alpha = alpha_amp * np.sin(2 * np.pi * 10 * t_arr)
            sig = pink + alpha

            result = meeglet_specparam_reconstruct(
                sig, sfreq, component="aperiodic",
                foi_start=2.0, foi_end=50.0, bw_oct=0.5,
                fit_stride=50, power_window=400, smooth_sigma=5.0,
                freq_range=[1, 50], edge_taper=True, n_iter=5,
            )
            r2s_seed.append(np.mean(result.fit.r_squared))

            f_orig, psd_orig = welch(sig, fs=sfreq, nperseg=512)
            f_rec, psd_rec = welch(result.reconstruction, fs=sfreq, nperseg=512)
            i10 = np.argmin(np.abs(f_orig - 10))
            if psd_orig[i10] > 1e-30:
                sups_seed.append((1 - psd_rec[i10] / psd_orig[i10]) * 100)
            else:
                sups_seed.append(100.0)

        mean_r2s.append(np.mean(r2s_seed))
        min_r2s.append(np.min(r2s_seed))
        alpha_sups.append(np.mean(sups_seed))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(snr_range, mean_r2s, "o-", color=BLUE, linewidth=2, markersize=7, label="Mean r²")
    axes[0].plot(snr_range, min_r2s, "s--", color=CYAN, linewidth=1.2, markersize=5, alpha=0.7, label="Min r²")
    axes[0].axhline(0.85, color=ORANGE, linestyle=":", linewidth=1.0, alpha=0.6, label="r²=0.85 threshold")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("r²")
    axes[0].set_ylim(0.8, 1.01)
    axes[0].set_title("Fit Quality vs SNR")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(True, alpha=0.2)

    bars = axes[1].bar(snr_range, alpha_sups, width=3.5, color=GREEN, alpha=0.8, edgecolor=DARK_BORDER)
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Alpha Suppression (%)")
    axes[1].set_ylim(90, 101)
    axes[1].set_title("Alpha (10 Hz) Suppression vs SNR")
    axes[1].grid(True, alpha=0.2, axis="y")

    for bar, val in zip(bars, alpha_sups):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f"{val:.0f}%", ha="center", va="bottom", fontsize=8, color=DARK_MUTED)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "snr_robustness.png"))
    plt.close(fig)


def fig_spectral_comparison():
    """Generate PSD comparison: original vs aperiodic vs periodic."""
    print("  [7/7] Spectral comparison...")
    sfreq = 256.0
    n_samples = int(10 * sfreq)
    t = np.arange(n_samples) / sfreq
    rng = np.random.default_rng(42)
    pink = generate_pink_noise(sfreq, n_samples, 1.5, rng)
    alpha = 2.0 * np.sin(2 * np.pi * 10 * t)
    signal = pink + alpha

    result_ap = meeglet_specparam_reconstruct(
        signal, sfreq, component="aperiodic",
        foi_start=2.0, foi_end=50.0, bw_oct=0.5,
        fit_stride=50, power_window=400,
        freq_range=[1, 50], n_iter=5, edge_taper=True,
    )
    result_per = meeglet_specparam_reconstruct(
        signal, sfreq, component="periodic",
        foi_start=2.0, foi_end=50.0, bw_oct=0.5,
        fit_stride=50, power_window=400,
        freq_range=[1, 50], n_iter=5, edge_taper=True,
    )

    f_orig, psd_orig = welch(signal, fs=sfreq, nperseg=512)
    f_ap, psd_ap = welch(result_ap.reconstruction, fs=sfreq, nperseg=512)
    f_per, psd_per = welch(result_per.reconstruction, fs=sfreq, nperseg=512)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.semilogy(f_orig, psd_orig, color=DARK_TEXT, linewidth=1.5, alpha=0.8, label="Original")
    ax.semilogy(f_ap, psd_ap, color=BLUE, linewidth=1.5, label="Aperiodic reconstruction")
    ax.semilogy(f_per, psd_per, color=GREEN, linewidth=1.5, label="Periodic reconstruction")
    ax.axvline(10, color=ORANGE, linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(11, psd_orig.max() * 0.5, "10 Hz", color=ORANGE, fontsize=9, alpha=0.7)
    ax.set_xlim(1, 60)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title("Spectral Comparison: Original vs Decomposed Components")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "spectral_comparison.png"))
    plt.close(fig)


if __name__ == "__main__":
    print("Generating figures for GitHub Pages...")
    fig_decomposition()
    fig_weight_surface()
    fig_exponent_tracking()
    fig_alpha_onoff()
    fig_beta_bursts()
    fig_snr_robustness()
    fig_spectral_comparison()
    print("Done! Figures saved to docs/figures/")

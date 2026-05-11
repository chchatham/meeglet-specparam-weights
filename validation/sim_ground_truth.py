"""Validation: Ground Truth Comparison of Separation Methods.

Head-to-head comparison of subtraction, Wiener, and state-space separation
on signals with independently known aperiodic and periodic components.

The key insight: we generate aperiodic (pink noise) and periodic (sine with
random phase) SEPARATELY, then sum them. This gives us ground truth for both
components, allowing direct evaluation of each separation method.

Validation matrix: sweep P_per/P_total across {0.1, 0.3, 0.5, 0.7, 0.9}.

Metrics:
  - alpha_power_ratio: PSD_recon(10 Hz) / PSD_true(10 Hz) — 1.0 is perfect
  - waveform_correlation: corr(recon, true) over full time series
  - spectral_shape_error: RMS of log10(PSD_recon) - log10(PSD_true) across freqs
"""

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from meeglet_specparam_weights.wavelet_analysis import wavelet_decompose
from meeglet_specparam_weights.time_resolved_fit import time_resolved_fit
from meeglet_specparam_weights.separation import subtraction_separate, wiener_separate
from meeglet_specparam_weights.state_space import state_space_separate

from validation.metrics import (
    generate_pink_noise,
    correlation,
    alpha_power_ratio,
    spectral_shape_error,
)

WAVELET_KWARGS = dict(foi_start=2.0, foi_end=50.0, bw_oct=0.5)
FIT_KWARGS = dict(
    fit_stride=50, power_window=400, smooth_sigma=5.0,
    freq_range=[1, 50],
)


def generate_ground_truth_signal(
    sfreq: float = 256.0,
    duration: float = 10.0,
    exponent: float = 1.5,
    alpha_freq: float = 10.0,
    periodic_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate signal with independently known aperiodic and periodic components.

    Returns (signal, aperiodic_true, periodic_true).
    """
    rng = np.random.default_rng(seed)
    n_samples = int(sfreq * duration)
    t = np.arange(n_samples) / sfreq

    aperiodic = generate_pink_noise(sfreq, n_samples, exponent, rng)

    phase = rng.uniform(0, 2 * np.pi)
    periodic_raw = np.sin(2 * np.pi * alpha_freq * t + phase)

    ap_power = np.mean(aperiodic ** 2)
    target_per_power = periodic_fraction * ap_power / max(1.0 - periodic_fraction, 1e-10)
    scale = np.sqrt(target_per_power / max(np.mean(periodic_raw ** 2), 1e-30))
    periodic = scale * periodic_raw

    return aperiodic + periodic, aperiodic, periodic


def run_separation(signal, decomposition, fit, sfreq, method):
    """Run a separation method using shared decomposition and fit."""
    if method == "subtraction":
        sep = subtraction_separate(signal, decomposition, fit, n_iter=5, edge_taper=True)
    elif method == "wiener":
        sep = wiener_separate(signal, decomposition, fit, n_iter=5, edge_taper=True)
    else:
        sep = state_space_separate(signal, decomposition, fit, sfreq)
    return sep.aperiodic


def evaluate_method(
    reconstruction: np.ndarray,
    ground_truth: np.ndarray,
    sfreq: float,
    peak_freq: float = 10.0,
) -> dict:
    """Evaluate a reconstruction against ground truth (1s edge trim)."""
    edge = int(1.0 * sfreq)
    recon_t = reconstruction[edge:-edge]
    truth_t = ground_truth[edge:-edge]
    return {
        "alpha_power_ratio": alpha_power_ratio(recon_t, truth_t, sfreq, peak_freq),
        "waveform_correlation": correlation(recon_t, truth_t),
        "spectral_shape_error": spectral_shape_error(recon_t, truth_t, sfreq),
    }


def run_validation():
    """Run ground truth validation across methods and periodic fractions."""
    sfreq = 256.0
    duration = 10.0
    n_seeds = 5
    periodic_fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
    methods = ["subtraction", "wiener", "state_space"]

    print("Ground Truth Validation: Method Comparison")
    print("=" * 70)
    print(f"Signal: pink noise (exp=1.5) + 10 Hz sine, {duration}s @ {sfreq} Hz")
    print(f"Seeds: {n_seeds}, periodic fractions: {periodic_fractions}")
    print(f"Methods: {methods}")
    print()

    all_results = {}

    for pf in periodic_fractions:
        print(f"--- P_per/P_total = {pf:.1f} ---")
        method_metrics = {m: [] for m in methods}
        t0 = time.time()

        for seed in range(n_seeds):
            signal, ap_true, _ = generate_ground_truth_signal(
                sfreq=sfreq, duration=duration,
                periodic_fraction=pf, seed=seed,
            )
            decomposition = wavelet_decompose(signal, sfreq, **WAVELET_KWARGS)
            fit = time_resolved_fit(decomposition, **FIT_KWARGS)

            for method in methods:
                try:
                    ap_recon = run_separation(signal, decomposition, fit, sfreq, method)
                    metrics = evaluate_method(ap_recon, ap_true, sfreq)
                    method_metrics[method].append(metrics)
                except Exception as e:
                    print(f"  {method}: ERROR — {e}")

        elapsed = time.time() - t0
        method_results = {}

        for method in methods:
            ml = method_metrics[method]
            if ml:
                avg = {k: np.mean([m[k] for m in ml]) for k in ml[0]}
                std = {k: np.std([m[k] for m in ml]) for k in ml[0]}
                method_results[method] = {"mean": avg, "std": std}
                print(f"  {method:14s}  "
                      f"alpha_ratio={avg['alpha_power_ratio']:.3f}±{std['alpha_power_ratio']:.3f}  "
                      f"waveform_corr={avg['waveform_correlation']:.3f}±{std['waveform_correlation']:.3f}  "
                      f"shape_err={avg['spectral_shape_error']:.3f}±{std['spectral_shape_error']:.3f}")

        print(f"  ({elapsed:.1f}s total)")
        all_results[pf] = method_results
        print()

    print_summary_table(all_results, periodic_fractions, methods)
    return all_results


def print_summary_table(all_results, periodic_fractions, methods):
    """Print a formatted comparison table."""
    header = f"{'P_per/P_total':>12s}"
    for m in methods:
        header += f"  {m:>14s}"

    for title, metric_key in [
        ("Alpha Power Ratio (1.0 = perfect)", "alpha_power_ratio"),
        ("Waveform Correlation (1.0 = perfect)", "waveform_correlation"),
        ("Spectral Shape Error (0.0 = perfect)", "spectral_shape_error"),
    ]:
        print()
        print("=" * 70)
        print(f"SUMMARY: {title}")
        print("-" * 70)
        print(header)
        for pf in periodic_fractions:
            row = f"{pf:>12.1f}"
            for m in methods:
                if m in all_results.get(pf, {}):
                    val = all_results[pf][m]["mean"][metric_key]
                    row += f"  {val:>14.3f}"
                else:
                    row += f"  {'N/A':>14s}"
            print(row)

    print()


if __name__ == "__main__":
    results = run_validation()

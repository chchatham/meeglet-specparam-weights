"""Validation: Non-Stationary Tracking.

Test 1: Time-varying aperiodic exponent (ramp 1.0 → 2.0 over 10s).
  - Metric: recovered exponent trajectory correlates with ground truth > 0.85

Test 2: Time-varying alpha amplitude (on/off blocks, 2s each).
  - Metric: periodic reconstruction power tracks alpha on/off with contrast ratio > 3:1
"""

import sys
import os

import numpy as np
from scipy.signal import welch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from meeglet_specparam_weights import meeglet_specparam_reconstruct

from validation.metrics import correlation, rms, generate_pink_noise


def generate_ramped_exponent(
    sfreq: float = 256.0,
    duration: float = 10.0,
    exp_start: float = 1.5,
    exp_end: float = 2.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate noise with linearly ramping aperiodic exponent.

    Uses overlapping FFT-shaped segments with Hanning windows to produce a
    signal whose local aperiodic exponent varies smoothly over time.

    Returns (signal, true_exponent_trajectory).
    """
    rng = np.random.default_rng(seed)
    n_samples = int(sfreq * duration)

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

    return signal, true_exponent


def generate_alpha_blocks(
    sfreq: float = 256.0,
    duration: float = 10.0,
    block_dur: float = 2.0,
    alpha_freq: float = 10.0,
    alpha_amp: float = 2.0,
    exponent: float = 1.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate pink noise + alpha that turns on/off in blocks.

    Returns (signal, alpha_envelope) where envelope is 1 during on-blocks, 0 during off.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(sfreq * duration)
    t = np.arange(n_samples) / sfreq
    pink = generate_pink_noise(sfreq, n_samples, exponent, rng)

    block_samples = int(block_dur * sfreq)
    alpha_envelope = np.zeros(n_samples)
    on = True
    for start in range(0, n_samples, block_samples):
        end = min(start + block_samples, n_samples)
        if on:
            alpha_envelope[start:end] = 1.0
        on = not on

    alpha = alpha_amp * alpha_envelope * np.sin(2 * np.pi * alpha_freq * t)

    return pink + alpha, alpha_envelope


def test_exponent_tracking():
    """Test 1: Does the recovered exponent trajectory track the true ramp?"""
    sfreq = 256.0
    duration = 10.0

    print("Test 1: Time-varying exponent tracking")
    print("-" * 50)

    corrs = []
    for seed in range(5):
        signal, true_exp = generate_ramped_exponent(
            sfreq=sfreq, duration=duration, seed=seed
        )

        result = meeglet_specparam_reconstruct(
            signal, sfreq,
            component="aperiodic",
            foi_start=2.0, foi_end=50.0, bw_oct=0.5,
            fit_stride=50, power_window=400, smooth_sigma=5.0,
            freq_range=[1, 50], edge_taper=True, n_iter=5,
        )

        recovered_exp = result.fit.aperiodic_params[:, 1]
        n = min(len(true_exp), len(recovered_exp))

        edge = int(2.0 * sfreq)
        true_t = true_exp[edge:n - edge]
        rec_t = recovered_exp[edge:n - edge]

        corr = correlation(true_t, rec_t)
        corrs.append(corr)

    mean_corr = np.mean(corrs)
    print(f"  Exponent trajectory correlation: {mean_corr:.4f} ± {np.std(corrs):.4f}")
    print(f"  Target: > 0.85")

    passed = mean_corr > 0.85
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    return {"correlation": mean_corr, "passed": passed}


def test_alpha_tracking():
    """Test 2: Does the periodic reconstruction track alpha on/off blocks?"""
    sfreq = 256.0
    duration = 10.0

    print("Test 2: Alpha on/off tracking")
    print("-" * 50)

    contrast_ratios = []
    for seed in range(5):
        signal, alpha_env = generate_alpha_blocks(
            sfreq=sfreq, duration=duration, seed=seed
        )

        result = meeglet_specparam_reconstruct(
            signal, sfreq,
            component="periodic",
            foi_start=2.0, foi_end=50.0, bw_oct=0.5,
            fit_stride=50, power_window=400, smooth_sigma=5.0,
            freq_range=[1, 50], edge_taper=True, n_iter=5,
        )

        recon = result.reconstruction
        edge = int(1.0 * sfreq)
        recon_t = recon[edge:-edge]
        env_t = alpha_env[edge:-edge]

        on_power = np.mean(recon_t[env_t > 0.5] ** 2)
        off_power = np.mean(recon_t[env_t < 0.5] ** 2)

        contrast = on_power / max(off_power, 1e-30)
        contrast_ratios.append(contrast)

    mean_contrast = np.mean(contrast_ratios)
    print(f"  On/off contrast ratio: {mean_contrast:.2f} ± {np.std(contrast_ratios):.2f}")
    print(f"  Target: > 3.0")

    passed = mean_contrast > 3.0
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    return {"contrast_ratio": mean_contrast, "passed": passed}


def run_validation():
    """Run all non-stationary tracking tests."""
    print("Non-Stationary Tracking Validation")
    print("=" * 60)
    print()

    result1 = test_exponent_tracking()
    result2 = test_alpha_tracking()

    all_pass = result1["passed"] and result2["passed"]
    if all_pass:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")

    return {
        "exponent_tracking": result1,
        "alpha_tracking": result2,
        "passed": all_pass,
    }


if __name__ == "__main__":
    results = run_validation()
    sys.exit(0 if results["passed"] else 1)

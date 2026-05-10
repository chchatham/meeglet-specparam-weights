"""Validation: SNR robustness sweep.

Sweep the signal-to-noise ratio from -10dB to +20dB and identify the threshold
where the fit quality (r²) drops below 0.85. Reports periodic excess extraction
quality: the periodic residual (from subtraction) should concentrate power at
the alpha frequency.
"""

import sys
import os

import numpy as np
from scipy.signal import welch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from meeglet_specparam_weights import meeglet_specparam_reconstruct

from validation.metrics import generate_pink_noise


def generate_signal_at_snr(
    sfreq: float = 256.0,
    duration: float = 10.0,
    exponent: float = 1.5,
    alpha_freq: float = 10.0,
    snr_db: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate pink noise + alpha at a specified SNR.

    SNR is defined as the ratio of alpha power to pink noise power.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(sfreq * duration)
    t = np.arange(n_samples) / sfreq
    pink = generate_pink_noise(sfreq, n_samples, exponent, rng)

    alpha_amp = np.sqrt(10.0 ** (snr_db / 10.0))
    alpha = alpha_amp * np.sin(2 * np.pi * alpha_freq * t)

    return pink + alpha


def run_validation():
    """Sweep SNR and report fit quality."""
    sfreq = 256.0
    duration = 10.0
    n_seeds = 3
    snr_range = np.arange(-10, 25, 5)

    print("SNR Robustness Sweep")
    print("=" * 60)
    print(f"Signal: pink noise (exp=1.5) + 10 Hz sine at varying SNR")
    print(f"Duration: {duration}s @ {sfreq} Hz, {n_seeds} seeds per SNR")
    print()

    print(f"{'SNR (dB)':>10s}  {'Mean r²':>8s}  {'Min r²':>8s}  {'Peak ratio':>10s}")
    print("-" * 45)

    results = {}
    threshold_snr = None

    for snr in snr_range:
        r2s = []
        peak_ratios = []

        for seed in range(n_seeds):
            signal = generate_signal_at_snr(
                sfreq=sfreq, duration=duration, snr_db=float(snr), seed=seed
            )

            result = meeglet_specparam_reconstruct(
                signal, sfreq,
                component="aperiodic",
                foi_start=2.0, foi_end=50.0, bw_oct=0.5,
                fit_stride=50, power_window=400, smooth_sigma=5.0,
                freq_range=[1, 50], edge_taper=True, n_iter=5,
            )

            r2s.append(np.mean(result.fit.r_squared))

            # Periodic excess quality: residual should peak at alpha
            residual = result.residual
            f_res, psd_res = welch(residual, fs=sfreq, nperseg=512)
            i10 = np.argmin(np.abs(f_res - 10))
            i5 = np.argmin(np.abs(f_res - 5))
            if psd_res[i5] > 1e-30:
                peak_ratios.append(psd_res[i10] / psd_res[i5])
            else:
                peak_ratios.append(float("nan"))

        mean_r2 = np.mean(r2s)
        min_r2 = np.min(r2s)
        mean_peak = np.nanmean(peak_ratios)

        print(f"{snr:10d}  {mean_r2:8.3f}  {min_r2:8.3f}  {mean_peak:10.1f}")

        results[int(snr)] = {
            "mean_r2": mean_r2,
            "min_r2": min_r2,
            "peak_ratio": float(mean_peak),
        }

        if threshold_snr is None and mean_r2 >= 0.85:
            threshold_snr = int(snr)

    print()
    if threshold_snr is not None:
        print(f"r² threshold (0.85) first met at SNR = {threshold_snr} dB")
    else:
        print("r² never reached 0.85 across tested range")

    passed = threshold_snr is not None
    print()
    if passed:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")

    return {
        "results": results,
        "threshold_snr": threshold_snr,
        "passed": passed,
    }


if __name__ == "__main__":
    results = run_validation()
    sys.exit(0 if results["passed"] else 1)

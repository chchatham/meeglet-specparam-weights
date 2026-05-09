"""Validation: Transient beta burst detection.

Embed short (200ms) beta bursts (20 Hz) in pink noise and check whether the
periodic reconstruction detects them.

Metric: >80% of bursts detected (SNR permitting).
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from meeglet_specparam_weights import meeglet_specparam_reconstruct

from validation.metrics import rms, generate_pink_noise


def generate_beta_bursts(
    sfreq: float = 256.0,
    duration: float = 10.0,
    exponent: float = 1.5,
    burst_freq: float = 20.0,
    burst_dur: float = 0.2,
    burst_amp: float = 3.0,
    n_bursts: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate pink noise with embedded beta bursts.

    Returns (signal, burst_centers_sec, burst_envelope).
    """
    rng = np.random.default_rng(seed)
    n_samples = int(sfreq * duration)
    t = np.arange(n_samples) / sfreq
    pink = generate_pink_noise(sfreq, n_samples, exponent, rng)

    burst_samples = int(burst_dur * sfreq)
    margin = int(1.0 * sfreq)
    available = n_samples - 2 * margin
    spacing = available // (n_bursts + 1)

    burst_centers = []
    burst_envelope = np.zeros(n_samples)

    for i in range(n_bursts):
        center = margin + (i + 1) * spacing
        start = center - burst_samples // 2
        end = start + burst_samples
        if start < 0 or end > n_samples:
            continue

        window = np.hanning(burst_samples)
        burst = burst_amp * window * np.sin(2 * np.pi * burst_freq * t[start:end])
        pink[start:end] += burst
        burst_envelope[start:end] = 1.0
        burst_centers.append(center / sfreq)

    return pink, np.array(burst_centers), burst_envelope


def detect_bursts(
    periodic_recon: np.ndarray,
    sfreq: float,
    burst_freq: float = 20.0,
    threshold_factor: float = 3.0,
) -> np.ndarray:
    """Detect bursts in the periodic reconstruction using envelope thresholding.

    Uses a local median baseline with MAD-based threshold for robustness
    against non-stationary envelope amplitude.
    """
    from scipy.signal import hilbert, butter, sosfilt
    from scipy.ndimage import median_filter

    lo = max(burst_freq - 5, 1)
    hi = min(burst_freq + 5, sfreq / 2 - 1)
    sos = butter(4, [lo / (sfreq / 2), hi / (sfreq / 2)], btype="band", output="sos")
    filtered = sosfilt(sos, periodic_recon)

    envelope = np.abs(hilbert(filtered))

    baseline = median_filter(envelope, size=int(sfreq * 1))
    deviation = envelope - baseline
    mad = np.median(np.abs(deviation))
    threshold = threshold_factor * mad
    detected = deviation > threshold
    return detected


def run_validation():
    """Run transient burst detection validation."""
    sfreq = 256.0
    duration = 10.0
    n_seeds = 5

    print("Transient Beta Burst Detection Validation")
    print("=" * 60)
    print(f"Signal: pink noise (exp=1.5) + {10} x 200ms beta bursts (20 Hz, amp=3)")
    print(f"Duration: {duration}s @ {sfreq} Hz, {n_seeds} seeds")
    print()

    detection_rates = []

    for seed in range(n_seeds):
        signal, burst_centers, burst_env = generate_beta_bursts(
            sfreq=sfreq, duration=duration, seed=seed
        )

        result = meeglet_specparam_reconstruct(
            signal, sfreq,
            component="periodic",
            foi_start=2.0, foi_end=50.0, bw_oct=0.5,
            fit_stride=50, power_window=400, smooth_sigma=5.0,
            freq_range=[1, 50], edge_taper=True, n_iter=5,
        )

        detected = detect_bursts(result.reconstruction, sfreq)

        hits = 0
        burst_half_width = int(0.15 * sfreq)
        for center_sec in burst_centers:
            center_samp = int(center_sec * sfreq)
            lo = max(0, center_samp - burst_half_width)
            hi = min(len(detected), center_samp + burst_half_width)
            if np.any(detected[lo:hi]):
                hits += 1

        rate = hits / len(burst_centers) if len(burst_centers) > 0 else 0
        detection_rates.append(rate)

    mean_rate = np.mean(detection_rates)
    print(f"Results:")
    print(f"  Detection rate: {mean_rate * 100:.1f}% ± {np.std(detection_rates) * 100:.1f}%")
    print(f"  Target: > 80%")
    print()

    passed = mean_rate > 0.80
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()

    if passed:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")

    return {
        "detection_rate": mean_rate,
        "passed": passed,
    }


if __name__ == "__main__":
    results = run_validation()
    sys.exit(0 if results["passed"] else 1)

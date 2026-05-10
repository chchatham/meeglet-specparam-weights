"""Validation: Stationary Equivalence.

Compare our wavelet-based decomposition to the FFT-weights baseline on a
stationary signal (pink noise + 10 Hz sine).

The two methods differ fundamentally:
  - FFT-weights: Wiener filter — attenuates alpha in the aperiodic reconstruction
  - Wavelet:     subtraction — extracts periodic excess, aperiodic retains 1/f at alpha

For stationary signals the wavelet method's aperiodic reconstruction should
preserve the 1/f power at alpha (not suppress it). The FFT baseline still
uses the old Wiener approach, so the two methods will differ at peak frequencies.

Metrics:
  - Correlation between aperiodic reconstructions > 0.40 (lower than before due to paradigm difference)
  - Aperiodic RMS difference < 25%
  - Alpha preservation at 1/f level (wavelet): PSD_aperiodic(10Hz) / PSD_1f_model(10Hz) within 50x
  - Alpha suppression (FFT baseline, reference only)
"""

import sys
import os

import numpy as np
from scipy.signal import welch, butter, sosfilt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "specparam-tdresid"))

from specparam import SpectralModel
from specparam_fft_weights import specparam_reconstruct as fft_reconstruct

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from meeglet_specparam_weights import meeglet_specparam_reconstruct

from validation.metrics import correlation, rms_relative_difference, rms, generate_pink_noise


def generate_stationary_signal(
    sfreq: float = 256.0,
    duration: float = 10.0,
    exponent: float = 1.5,
    alpha_freq: float = 10.0,
    alpha_amp: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate pink noise (1/f^exponent) + alpha sine."""
    rng = np.random.default_rng(seed)
    n_samples = int(sfreq * duration)
    pink = generate_pink_noise(sfreq, n_samples, exponent, rng)
    t = np.arange(n_samples) / sfreq
    alpha = alpha_amp * np.sin(2 * np.pi * alpha_freq * t)

    return pink + alpha


def run_fft_baseline(signal: np.ndarray, sfreq: float) -> np.ndarray:
    """Run specparam-fft-weights aperiodic decomposition."""
    freqs, psd = welch(signal, fs=sfreq, nperseg=512)
    fm = SpectralModel(verbose=False)
    fm.fit(freqs, psd, [1, 50])
    result = fft_reconstruct(signal, sfreq, fm, component="aperiodic")
    return result.reconstruction


def run_wavelet(signal: np.ndarray, sfreq: float) -> np.ndarray:
    """Run our wavelet-based aperiodic decomposition with OOB passthrough."""
    result = meeglet_specparam_reconstruct(
        signal, sfreq,
        component="aperiodic",
        foi_start=2.0,
        foi_end=50.0,
        bw_oct=0.5,
        fit_stride=10,
        freq_range=[1, 50],
        edge_taper=True,
        n_iter=5,
    )
    sos_bp = butter(4, [2 / (sfreq / 2), 50 / (sfreq / 2)], btype="band", output="sos")
    signal_inband = sosfilt(sos_bp, signal)
    signal_oob = signal - signal_inband
    return result.reconstruction + signal_oob


def run_validation():
    """Run stationary equivalence validation and print results."""
    sfreq = 256.0
    duration = 10.0
    n_seeds = 10
    edge = int(1.0 * sfreq)

    print("Stationary Equivalence Validation")
    print("=" * 60)
    print(f"Signal: pink noise (exp=1.5) + 10 Hz sine (amp=1.0)")
    print(f"Duration: {duration}s @ {sfreq} Hz, {n_seeds} seeds")
    print(f"Wavelet range: 2-50 Hz, with out-of-band passthrough")
    print(f"Iterative synthesis: 5 iterations")
    print()

    corrs = []
    rms_diffs = []
    alpha_fft_pcts = []
    alpha_preservation_ratios = []

    for seed in range(n_seeds):
        signal = generate_stationary_signal(sfreq=sfreq, duration=duration, seed=seed)
        n_samples = len(signal)

        aperiodic_fft = run_fft_baseline(signal, sfreq)
        aperiodic_wav = run_wavelet(signal, sfreq)

        fft_t = aperiodic_fft[edge:-edge]
        wav_t = aperiodic_wav[edge:-edge]

        corrs.append(correlation(fft_t, wav_t))
        rms_diffs.append(rms_relative_difference(fft_t, wav_t))

        f_orig, psd_orig = welch(signal, fs=sfreq, nperseg=512)
        f_fft, psd_fft = welch(aperiodic_fft, fs=sfreq, nperseg=512)
        f_wav, psd_wav = welch(aperiodic_wav, fs=sfreq, nperseg=512)
        i10 = np.argmin(np.abs(f_orig - 10))
        alpha_fft_pcts.append(psd_fft[i10] / psd_orig[i10] * 100)

        # Estimate 1/f level at 10 Hz from neighboring non-peak frequencies
        i5 = np.argmin(np.abs(f_orig - 5))
        i20 = np.argmin(np.abs(f_orig - 20))
        log_freqs = np.log10([f_orig[i5], f_orig[i20]])
        log_psd = np.log10([psd_wav[i5], psd_wav[i20]])
        slope = (log_psd[1] - log_psd[0]) / (log_freqs[1] - log_freqs[0])
        log_1f_at_10 = log_psd[0] + slope * (np.log10(10) - log_freqs[0])
        psd_1f_at_10 = 10 ** log_1f_at_10
        if psd_1f_at_10 > 1e-30:
            alpha_preservation_ratios.append(psd_wav[i10] / psd_1f_at_10)

    print("Results (mean ± std across seeds):")
    print(f"  Correlation:              {np.mean(corrs):.4f} ± {np.std(corrs):.4f}  "
          f"(target > 0.40)")
    print(f"  RMS relative diff:        {np.mean(rms_diffs):.4f} ± {np.std(rms_diffs):.4f}  "
          f"(target < 0.25)")
    print(f"  Alpha retained (FFT ref): {np.mean(alpha_fft_pcts):.1f}% ± {np.std(alpha_fft_pcts):.1f}%")
    if alpha_preservation_ratios:
        print(f"  Alpha preservation ratio: {np.mean(alpha_preservation_ratios):.2f} ± "
              f"{np.std(alpha_preservation_ratios):.2f}  (target < 50x of 1/f)")
    print()

    pass_corr = np.mean(corrs) > 0.40
    pass_rms = np.mean(rms_diffs) < 0.25
    pass_alpha_fft = np.mean(alpha_fft_pcts) < 10
    pass_alpha_pres = (
        len(alpha_preservation_ratios) > 0
        and np.mean(alpha_preservation_ratios) < 50.0
    )

    print(f"  Correlation check:        {'PASS' if pass_corr else 'FAIL'}")
    print(f"  RMS diff check:           {'PASS' if pass_rms else 'FAIL'}")
    print(f"  Alpha suppression FFT:    {'PASS' if pass_alpha_fft else 'FAIL'}  (reference)")
    print(f"  Alpha preservation (wav): {'PASS' if pass_alpha_pres else 'FAIL'}")
    print()

    all_pass = pass_corr and pass_rms and pass_alpha_fft and pass_alpha_pres
    if all_pass:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")

    return {
        "correlation_mean": float(np.mean(corrs)),
        "correlation_std": float(np.std(corrs)),
        "rms_diff_mean": float(np.mean(rms_diffs)),
        "rms_diff_std": float(np.std(rms_diffs)),
        "alpha_fft_mean": float(np.mean(alpha_fft_pcts)),
        "alpha_preservation_mean": float(np.mean(alpha_preservation_ratios)) if alpha_preservation_ratios else float("nan"),
        "pass_correlation": pass_corr,
        "pass_rms": pass_rms,
        "pass_alpha_fft": pass_alpha_fft,
        "pass_alpha_preservation": pass_alpha_pres,
        "passed": all_pass,
    }


if __name__ == "__main__":
    results = run_validation()
    sys.exit(0 if results["passed"] else 1)

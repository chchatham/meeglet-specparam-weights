# meeglet-specparam-weights

**Time-resolved spectral decomposition of M/EEG signals via wavelet-domain parametric weighting.**

This tool combines three existing ideas into something new:

- **[meeglet](https://github.com/Roche/neuro-meeglet)** — Morlet wavelets with log-frequency parameterization, designed for M/EEG power-spectral analysis (Bomatter et al. 2024, Hipp et al. 2012).
- **[specparam](https://github.com/fooof-tools/fooof)** (formerly FOOOF) — parametric decomposition of power spectra into aperiodic (1/f) and periodic (oscillatory) components (Donoghue et al. 2020).
- **[specparam-fft-weights](https://github.com/chchatham/specparam-fft-weights)** — FFT-domain amplitude weighting to reconstruct time-domain signals corresponding to specparam model components.

## The problem

specparam-fft-weights solves a real gap: it gets specparam's frequency-domain decomposition back into the time domain so it can be used in event-related, connectivity, and waveform analyses. But it assumes stationarity — one specparam fit produces one global weight vector applied uniformly across the signal. EEG is not stationary. Alpha power bursts, aperiodic slopes vary over time, transient beta events come and go. A static decomposition cannot track any of this.

## The approach

Replace the global FFT with meeglet's Morlet wavelet decomposition. This gives you:

1. **Time-resolved complex coefficients** Z(f, t) at each frequency and time point — no need to retrofit time resolution onto a global transform.
2. **A log-frequency grid** that matches the natural 1/f structure of neural signals and gives uniform leverage on specparam's aperiodic fit.
3. **NaN-aware convolution** that handles artifact-marked segments gracefully.

Then fit specparam to the instantaneous power profile |Z(f, t)|² at each time step (or at a configurable stride with interpolation), producing time-varying parametric models. Compute a 2D weight surface w(f, t) = √(P_model(f, t) / |Z(f, t)|²), apply it to the complex wavelet coefficients (preserving phase exactly), and synthesize back to the time domain via overlap-add.

The result: aperiodic and periodic time-domain signals that track the dynamics of the original signal.

## Installation

```bash
pip install numpy scipy meeglet specparam matplotlib
git clone https://github.com/[you]/meeglet-specparam-weights.git
cd meeglet-specparam-weights
pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np
from meeglet_specparam_weights import meeglet_specparam_reconstruct

# Your signal: 1D (n_samples,) or 2D (n_channels, n_samples) numpy array
sfreq = 256.0
t = np.arange(int(10 * sfreq)) / sfreq
signal = make_your_signal(t)  # pink noise + oscillations

# Decompose into aperiodic and periodic time-domain components
result = meeglet_specparam_reconstruct(
    signal, sfreq,
    component='aperiodic',
    foi_start=2, foi_end=50,
    bw_oct=0.5,
    fit_stride=50,
    power_window=400,
    freq_range=[1, 50],
    n_iter=5,
)

aperiodic_signal = result.reconstruction
oscillatory_residual = result.residual

# Check fit quality over time
print(f"Mean r²: {result.fit.r_squared.mean():.3f}")
print(f"Energy ratio: {result.energy_ratio:.3f}")
print(f"Frame condition: {result.frame_condition:.3f}")  # B/A ≈ 1.0 = tight frame
```

## Diagnostics

```python
from meeglet_specparam_weights import (
    plot_fit_quality,
    plot_weight_surface,
    plot_decomposition,
    plot_parameter_trajectories,
    plot_aperiodic_coupling,
)

plot_fit_quality(result)            # r² over time
plot_weight_surface(result)         # 2D weight heatmap (freq × time)
plot_decomposition(result)          # original / reconstruction / residual
plot_parameter_trajectories(result) # exponent, offset, peak CFs over time
# plot_aperiodic_coupling(coupling)  # coupling heatmap (see Coupling section)
```

## Advanced workflows

These workflows illustrate research scenarios where time-resolved decomposition provides capabilities beyond what static FFT-based methods can offer. Each draws inspiration from common [specparam-fft-weights](https://github.com/chchatham/specparam-fft-weights) use cases but exploits the wavelet method's ability to track spectral dynamics continuously.

### Workflow 1: Time-varying oscillatory waveform morphology

Extract the periodic component to study waveform shape (e.g., alpha peak–trough asymmetry) without 1/f contamination. Unlike a static correction, the aperiodic model adapts at each time step, so the periodic extraction stays clean even as the background slope drifts over the course of a recording.

```python
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from meeglet_specparam_weights import meeglet_specparam_reconstruct

# Decompose with time-adaptive 1/f correction
result = meeglet_specparam_reconstruct(
    signal, sfreq,
    component="periodic",
    foi_start=2.0, foi_end=50.0,
    bw_oct=0.5,
    fit_stride=50,
    power_window=400,
    freq_range=[1, 50],
    n_iter=5,
)

# Bandpass for alpha waveform analysis
sos = butter(4, [7, 14], btype="band", fs=sfreq, output="sos")
alpha_waveform = sosfiltfilt(sos, result.reconstruction)

# Measure peak-trough asymmetry
analytic = hilbert(alpha_waveform)
phase = np.angle(analytic)

peak_idx = np.where(np.diff(np.sign(phase)) < 0)[0]
trough_idx = np.where(np.diff(np.sign(phase - np.pi)) < 0)[0]

asymmetry = np.mean(alpha_waveform[peak_idx]) / np.mean(np.abs(alpha_waveform[trough_idx]))
print(f"Peak-trough asymmetry: {asymmetry:.2f}")
```

### Workflow 2: Within-trial event-related decomposition

Traditional 1/f-corrected ERP analysis fits one specparam model per epoch and subtracts a static aperiodic signal. The wavelet method resolves spectral changes *within* each trial — alpha desynchronization, aperiodic slope changes — giving you a time-varying periodic power envelope rather than a single per-epoch number.

```python
import numpy as np
from meeglet_specparam_weights import meeglet_specparam_reconstruct
from scipy.signal import hilbert, butter, sosfiltfilt

# epochs: (n_epochs, n_samples) — e.g., -0.5 to 1.5s around stimulus
sfreq = 256.0
t_epoch = np.arange(epochs.shape[1]) / sfreq - 0.5

alpha_envelopes = []
exponent_trajectories = []

for epoch in epochs:
    result = meeglet_specparam_reconstruct(
        epoch, sfreq,
        component="periodic",
        foi_start=2.0, foi_end=40.0,
        bw_oct=0.5,
        fit_stride=25,
        power_window=200,
        freq_range=[1, 40],
        n_iter=5,
    )

    # Alpha envelope within this trial
    sos = butter(4, [8, 13], btype="band", fs=sfreq, output="sos")
    alpha_bp = sosfiltfilt(sos, result.reconstruction)
    alpha_envelopes.append(np.abs(hilbert(alpha_bp)))

    # Track aperiodic exponent within the trial
    exponent_trajectories.append(result.fit.aperiodic_params[:, 1])

# Event-related alpha envelope
mean_alpha_env = np.mean(alpha_envelopes, axis=0)
baseline = mean_alpha_env[t_epoch < 0].mean()
post_stim = mean_alpha_env[(t_epoch > 0.2) & (t_epoch < 0.8)].mean()
erd_pct = 100 * (baseline - post_stim) / baseline
print(f"Alpha ERD: {erd_pct:.1f}%")
```

### Workflow 3: Continuous aperiodic state tracking

The aperiodic exponent varies across time and experimental conditions. With FFT-based methods, you compare discrete epochs (rest vs. task). The wavelet method gives you a continuous exponent trajectory, enabling detection of transitions and gradual drifts within a single recording.

```python
import numpy as np
from meeglet_specparam_weights import meeglet_specparam_reconstruct
from scipy.ndimage import uniform_filter1d

# Long recording (e.g., 5-minute resting state)
result = meeglet_specparam_reconstruct(
    signal, sfreq,
    component="aperiodic",
    foi_start=2.0, foi_end=50.0,
    bw_oct=0.5,
    fit_stride=100,        # fit every ~0.4s for long recordings
    power_window=800,       # ~3s averaging for stable fits
    smooth_sigma=5.0,       # smooth parameter trajectory
    freq_range=[1, 50],
    n_iter=5,
)

# Continuous exponent trajectory
times = result.fit.times
exponent = result.fit.aperiodic_params[:, 1]

# Smooth for state-level analysis (10s moving average)
exponent_smooth = uniform_filter1d(exponent, size=int(10 * sfreq / 100))

# Detect state transitions
d_exp = np.gradient(exponent_smooth, times)
transition_mask = np.abs(d_exp) > np.std(d_exp) * 2
print(f"State transitions detected at: {times[transition_mask]} s")
```

### Workflow 4: Transient oscillatory burst detection

Transient oscillatory bursts (beta events, sleep spindles, gamma bursts) are invisible to methods that assume stationarity — a 200 ms burst is smeared into the global spectrum. The wavelet method's periodic reconstruction preserves burst timing and shape, enabling envelope-based detection with precise onset, duration, and amplitude estimates.

```python
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from meeglet_specparam_weights import meeglet_specparam_reconstruct

# Fine-grained periodic extraction for burst detection
result = meeglet_specparam_reconstruct(
    signal, sfreq,
    component="periodic",
    foi_start=2.0, foi_end=50.0,
    bw_oct=0.5,
    fit_stride=25,        # fine stride for transient detection
    power_window=100,     # short window to preserve burst timing
    freq_range=[1, 50],
    n_iter=5,
)

# Beta envelope from the periodic reconstruction
sos = butter(4, [13, 30], btype="band", fs=sfreq, output="sos")
beta_periodic = sosfiltfilt(sos, result.reconstruction)
beta_envelope = np.abs(hilbert(beta_periodic))

# Detect bursts: median + 2×MAD threshold
median_env = np.median(beta_envelope)
mad = np.median(np.abs(beta_envelope - median_env)) * 1.4826
threshold = median_env + 2 * mad
burst_mask = beta_envelope > threshold

# Extract burst events
diff_mask = np.diff(burst_mask.astype(int))
onsets = np.where(diff_mask == 1)[0]
offsets = np.where(diff_mask == -1)[0]

if len(offsets) > 0 and (len(onsets) == 0 or offsets[0] < onsets[0]):
    offsets = offsets[1:]
if len(onsets) > len(offsets):
    onsets = onsets[:len(offsets)]

for i, (on, off) in enumerate(zip(onsets, offsets)):
    duration_ms = (off - on) / sfreq * 1000
    peak_amp = beta_envelope[on:off].max()
    print(f"Burst {i}: onset={on / sfreq:.3f}s, "
          f"duration={duration_ms:.0f}ms, peak={peak_amp:.3f}")
```

## Multi-channel support

All pipeline functions accept 2D input `(n_channels, n_samples)` in addition to 1D. Each channel is processed independently.

```python
import numpy as np
from meeglet_specparam_weights import meeglet_specparam_reconstruct

# Multi-channel signal: (n_channels, n_samples)
signal = np.stack([channel_1, channel_2, channel_3])  # (3, n_samples)

result = meeglet_specparam_reconstruct(
    signal, sfreq,
    component="aperiodic",
    foi_start=2, foi_end=50,
    bw_oct=0.5,
    fit_stride=50,
    power_window=400,
    freq_range=[1, 50],
    n_iter=5,
)

# Multi-channel output shapes:
# result.reconstruction:                (n_channels, n_samples)
# result.residual:                      (n_channels, n_samples)
# result.fit.aperiodic_params:          (n_channels, n_times, 2)
# result.fit.r_squared:                 (n_channels, n_times)
# result.fit.model_power:               (n_channels, n_freqs, n_times)
# result.weights.weights:               (n_channels, n_freqs, n_times)
# result.decomposition.coefficients:    (n_channels, n_freqs, n_times)
```

Single-channel input continues to produce the same shapes as before (`(n_samples,)`, `(n_times, 2)`, etc.).

## Aperiodic-oscillatory coupling

The coupling module tests whether slow fluctuations in aperiodic parameters (exponent, offset) are correlated with oscillatory amplitude at specific frequencies. This is a common question in systems neuroscience — do changes in the "background" 1/f slope predict changes in alpha or beta power?

```python
from meeglet_specparam_weights import (
    wavelet_decompose, time_resolved_fit,
    aperiodic_virtual_channels,
    compute_aperiodic_csd, aperiodic_amplitude_correlation,
    effective_dof, wavelet_effective_dof,
    plot_aperiodic_coupling,
)

# Multi-channel signal: (n_channels, n_samples)
decomp = wavelet_decompose(signal, sfreq, foi_start=2, foi_end=50)
fit = time_resolved_fit(decomp, fit_stride=25, power_window=200)

# Simple amplitude correlation: corr(exponent(t), |Z(ch, f, t)|)
amp_corr = aperiodic_amplitude_correlation(decomp, fit)
# amp_corr shape: (n_channels, n_freqs)

# Full augmented CSD with virtual channels
coupling = compute_aperiodic_csd(decomp, fit)
print(f"CSD shape: {coupling.csd.shape}")           # (n_ch+2, n_ch+2, n_freqs)
print(f"Effective Nyquist: {coupling.effective_nyquist:.1f} Hz")
print(f"Labels: {coupling.channel_labels}")          # ['ch0', ..., 'exponent', 'offset']

# Correct DOF for autocorrelated signals (Bartlett)
exponent = fit.aperiodic_params[:, 1]
amplitude_10hz = np.abs(decomp.coefficients[idx_10, :])
n_eff = effective_dof(exponent, amplitude_10hz)

# Frequency-dependent DOF from wavelet temporal resolution
n_eff_wavelet = wavelet_effective_dof(decomp.sigma_time, decomp.sfreq, signal.shape[-1])
# n_eff_wavelet shape: (n_freqs,) — lower freqs have fewer independent samples

# Virtual channels: wavelet-decomposed exponent/offset trajectories
virtual_coeffs, eff_nyq = aperiodic_virtual_channels(fit, decomp)
# virtual_coeffs shape: (2, n_freqs, n_times) — z-scored, band-limited

# Visualize
plot_aperiodic_coupling(coupling)
```

**Important caveats:**
- Aperiodic parameters are derived from wavelet power via specparam. Correlating them back to the same wavelet features creates inherent circularity. Surrogate testing (block-permuted aperiodic trajectory) is required for inference.
- Coupling is only meaningful below the effective Nyquist frequency: `sfreq / (2 * fit_stride)`. With default parameters (stride=50, sfreq=256), this is ~2.5 Hz — infraslow modulations only.

## What this is not

- **Not an exact reconstruction.** Wavelet synthesis via overlap-add is approximate. Energy ratios are reported; sample-exact recovery is not guaranteed. For stationary signals where exact reconstruction matters, use [specparam-fft-weights](https://github.com/chchatham/specparam-fft-weights) directly.
- **Not a replacement for specparam.** We call specparam's fitting algorithm. If the fit is poor (low r²), our decomposition is poor. Always check diagnostics.
- **Not validated on real EEG data** (in this repo). Validation uses synthetic signals with known ground truth. Real-data validation belongs in downstream studies.

## Design rationale

See `CLAUDE.md` for the full architecture and design principles. The key decisions:

1. Wavelet coefficients (not FFT bins) are the canonical representation.
2. Phase is preserved exactly — weights are real and non-negative. Multiplying complex wavelet coefficients by a real positive scalar preserves their phase by construction. The weight surface controls amplitude at each (frequency, time) point without rotating phase; the original signal's phase structure is carried through the complex coefficients and recovered in synthesis.
3. specparam does the fitting; we do the bridging.
4. Log-frequency is the native grid (with interpolation to linear grids for specparam fitting).
5. Energy accounting is transparent, not hidden.

## Validation results

All validation uses synthetic signals with known ground truth. See `validation/RESULTS.md` for details.

| Test | Metric | Result | Target |
|------|--------|--------|--------|
| Stationary equivalence | Correlation with FFT-weights | 0.85 ± 0.03 | > 0.65 |
| Stationary equivalence | Alpha suppression | 99.3% | > 90% |
| Non-stationary tracking | Exponent trajectory correlation | 0.92 ± 0.09 | > 0.85 |
| Non-stationary tracking | Alpha on/off contrast ratio | 7.2 ± 2.4 | > 3.0 |
| Transient detection | Beta burst detection rate | 96% ± 5% | > 80% |
| SNR robustness | r² at -10 dB SNR | 0.969 | > 0.85 |

Run validation:
```bash
python -m validation.sim_stationary
python -m validation.sim_nonstationary
python -m validation.sim_transient
python -m validation.sim_noise_sweep
```

## Testing

```bash
pytest tests/ -v  # 105 tests
```

## Citations

If you use this tool, please cite:

- Donoghue T, et al. (2020). Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience*, 23, 1655-1665.
- Bomatter P, et al. (2024). Machine learning of brain-specific biomarkers from EEG. *eBioMedicine*, 106, 105259.
- Hipp JF, et al. (2012). Large-scale cortical correlation structure of spontaneous oscillatory activity. *Nature Neuroscience*, 15(6), 884-890.

## License

MIT

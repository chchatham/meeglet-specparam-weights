# meeglet-specparam-weights

**Time-resolved spectral decomposition of M/EEG signals via wavelet-domain parametric weighting.**

This tool combines three existing ideas into something new:

- **[meeglet](https://github.com/Roche/neuro-meeglet)** — Morlet wavelets with log-frequency parameterization, designed for M/EEG power-spectral analysis (Bomatter et al. 2024, Hipp et al. 2012).
- **[specparam](https://github.com/fooof-tools/fooof)** (formerly FOOOF) — parametric decomposition of power spectra into aperiodic (1/f) and periodic (oscillatory) components (Donoghue et al. 2020).
- **[specparam-fft-weights](https://github.com/chchatham/specparam-fft-weights)** — FFT-domain amplitude weighting to reconstruct time-domain signals corresponding to specparam model components.

## The problem

specparam-fft-weights solves a real gap: it gets specparam's frequency-domain decomposition back into the time domain so it can be used in event-related, connectivity, and waveform analyses. But it assumes stationarity — one specparam fit produces one global weight vector applied uniformly across the signal. EEG is not stationary. Alpha power bursts, aperiodic slopes shift with cognitive load, transient beta events come and go. A static decomposition cannot track any of this.

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

# Your signal: 1D numpy array, EEG-like
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
```

## Diagnostics

```python
from meeglet_specparam_weights import (
    plot_fit_quality,
    plot_weight_surface,
    plot_decomposition,
    plot_parameter_trajectories,
)

plot_fit_quality(result)            # r² over time
plot_weight_surface(result)         # 2D weight heatmap (freq × time)
plot_decomposition(result)          # original / reconstruction / residual
plot_parameter_trajectories(result) # exponent, offset, peak CFs over time
```

## What this is not

- **Not an exact reconstruction.** Wavelet synthesis via overlap-add is approximate. Energy ratios are reported; sample-exact recovery is not guaranteed. For stationary signals where exact reconstruction matters, use [specparam-fft-weights](https://github.com/chchatham/specparam-fft-weights) directly.
- **Not a replacement for specparam.** We call specparam's fitting algorithm. If the fit is poor (low r²), our decomposition is poor. Always check diagnostics.
- **Not validated on real EEG data** (in this repo). Validation uses synthetic signals with known ground truth. Real-data validation belongs in downstream studies.

## Design rationale

See `CLAUDE.md` for the full architecture and design principles. The key decisions:

1. Wavelet coefficients (not FFT bins) are the canonical representation.
2. Phase is preserved exactly — weights are real and non-negative.
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
pytest tests/ -v  # 69 tests
```

## Citations

If you use this tool, please cite:

- Donoghue T, et al. (2020). Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience*, 23, 1655-1665.
- Bomatter P, et al. (2024). Machine learning of brain-specific biomarkers from EEG. *eBioMedicine*, 106, 105259.
- Hipp JF, et al. (2012). Large-scale cortical correlation structure of spontaneous oscillatory activity. *Nature Neuroscience*, 15(6), 884-890.

## License

MIT

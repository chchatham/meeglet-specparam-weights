# meeglet-specparam-weights

## Session Recovery (READ THIS FIRST)
If you're starting a new session, recovering from compaction, or running in a Ralph loop:
1. Read `.ralph/ralph_task.md` — the anchor. Has all checkboxes. Defines "done."
2. Read `.ralph/progress.md` — what exists, what's broken, what to do next.
3. Read `.ralph/guardrails.md` — learned constraints. Follow every sign.
4. Do NOT re-read the full codebase unless progress.md says something is broken. Trust the files.
5. Pick up the "Current Focus" from progress.md and work on it.
6. Before exiting or if context feels heavy: update progress.md with what you did and what's next.

## Compaction Instructions
When compacting this conversation, preserve:
- Current task and its completion state
- Any new guardrails discovered this session
- Any new known issues
- The exact next step to take
Do NOT preserve: file contents already read, full API/command outputs, failed approaches
(log failures to .ralph/errors.log instead).

## Project Purpose

Time-resolved spectral decomposition of M/EEG signals into aperiodic and periodic
time-domain components, combining meeglet's log-frequency Morlet wavelet representation
with specparam's parametric spectral modeling and the FFT-weighting reconstruction
approach from specparam-fft-weights.

The tool decomposes signals into aperiodic and periodic components using a
subtraction approach: it first extracts the periodic excess above the 1/f floor
via bounded wavelet-domain weights, synthesizes the periodic signal, then defines
the aperiodic as original minus periodic. This guarantees `original = aperiodic +
periodic` exactly and preserves the full 1/f power at peak frequencies in the
aperiodic component.

## Architecture

```
src/meeglet_specparam_weights/
├── __init__.py                  # Public API re-exports
├── wavelet_analysis.py          # meeglet wrapper: signal → complex coefficients Z(f,t)
├── time_resolved_fit.py         # Fit specparam to instantaneous power at each time step
├── weight_surface.py            # Compute w(f,t); for aperiodic: excess weights via subtraction
├── synthesis.py                 # Weighted coefficients → time-domain signal via OLA
├── pipeline.py                  # End-to-end: signal in → ReconstructionResult out
├── coupling.py                  # Aperiodic-oscillatory coupling: virtual-channel CSD
└── diagnostics.py               # Fit quality, energy accounting, visualization helpers

tests/
├── test_wavelet_analysis.py     # Meeglet integration, coefficient shapes, NaN handling
├── test_time_resolved_fit.py    # Parameter recovery on synthetic signals
├── test_weight_surface.py       # Numerical properties: clamping, eps, edge bins
├── test_synthesis.py            # OLA reconstruction accuracy, energy preservation
├── test_pipeline.py             # End-to-end on synthetic signals with known ground truth
├── test_diagnostics.py          # Plotting smoke tests, metric calculations
├── test_coupling.py             # CSD shape, Hermitian, Nyquist, coupling recovery
└── conftest.py                  # Shared fixtures: synthetic signals, standard wavelet configs

validation/
├── sim_stationary.py            # Baseline: stationary signal, compare to specparam-fft-weights
├── sim_nonstationary.py         # Core validation: time-varying exponent and/or peak amplitude
├── sim_transient.py             # Transient oscillation bursts (e.g., beta events)
├── sim_noise_sweep.py           # SNR sweep: where does the method break down?
├── metrics.py                   # R², correlation, energy ratios, phase error metrics
└── figures.py                   # Publication-quality figure generation
```

## Key Schemas / Interfaces

### WaveletDecomposition (wavelet_analysis → weight_surface, synthesis, coupling)
```python
@dataclass
class WaveletDecomposition:
    coefficients: np.ndarray    # complex, (n_freqs, n_times) or (n_channels, n_freqs, n_times)
    foi: np.ndarray             # center frequencies, shape (n_freqs,)
    sigma_time: np.ndarray      # temporal std per freq, shape (n_freqs,)
    sigma_freq: np.ndarray      # spectral std per freq, shape (n_freqs,)
    times: np.ndarray           # time points in seconds, shape (n_times,)
    sfreq: float                # sampling frequency
    bw_oct: float               # bandwidth in octaves used
    delta_oct: float            # frequency spacing in octaves used
    kernel_width: int = 5       # wavelet kernel width in sigma_time units
    density: str = "oct"        # power density units ("oct" or "Hz")
    n_channels: int = 1         # 1 for single-channel, >1 for multi-channel
```

### TimeResolvedFit (time_resolved_fit → weight_surface, coupling)
```python
@dataclass
class TimeResolvedFit:
    aperiodic_params: np.ndarray   # (n_times, 2) or (n_channels, n_times, 2)
    peak_params: list              # list[ndarray] or list[list[ndarray]]
    model_power: np.ndarray        # (n_freqs, n_times) or (n_channels, n_freqs, n_times)
    r_squared: np.ndarray          # (n_times,) or (n_channels, n_times)
    foi: np.ndarray                # center frequencies used for fitting
    times: np.ndarray              # time points
    fit_stride: int                # stride in samples between fits
    n_channels: int = 1
```

### WeightSurface (weight_surface → synthesis)
```python
@dataclass
class WeightSurface:
    weights: np.ndarray         # (n_freqs, n_times) or (n_channels, n_freqs, n_times)
    component: str              # 'full', 'aperiodic', 'periodic'
    eps: float                  # floor used
    max_weight: float           # clamp used
```

### ReconstructionResult (pipeline output)
```python
@dataclass
class ReconstructionResult:
    reconstruction: np.ndarray  # (n_samples,) or (n_channels, n_samples)
    residual: np.ndarray        # (n_samples,) or (n_channels, n_samples)
    fit: TimeResolvedFit        # the underlying parametric fit
    weights: WeightSurface      # the weight surface applied (excess weights for subtraction)
    energy_ratio: float         # ||reconstruction||² / ||original||² (sanity check)
    decomposition: WaveletDecomposition  # the wavelet decomposition used
    frame_condition: float      # B/A of the frame operator; 1.0 = tight frame
    method: str                 # "subtraction" or "weight"
```

### AperiodicCouplingResult (coupling output)
```python
@dataclass
class AperiodicCouplingResult:
    csd: np.ndarray              # complex, (n_ch+2, n_ch+2, n_freqs) — augmented CSD
    amplitude_correlation: np.ndarray  # (n_channels, n_freqs) — corr(exponent, |Z|)
    virtual_coefficients: np.ndarray   # complex, (2, n_freqs, n_times) — z-scored
    effective_nyquist: float     # Hz — coupling only meaningful below this
    foi: np.ndarray              # center frequencies
    channel_labels: list[str]    # ['ch0', ..., 'exponent', 'offset']
```

## Environment

- Python 3.10+
- Key dependencies:
  - `numpy`, `scipy`
  - `meeglet` (pip install meeglet)
  - `specparam>=2.0` (pip install specparam)
  - `matplotlib` (visualization / diagnostics)
  - `pytest`, `pytest-cov` (testing)
- No required env vars
- No required data files (all validation uses synthetic signals)
- Test command: `pytest tests/ -v --tb=short`
- Validation command: `python -m validation.sim_nonstationary`

## Design Principles

1. **Wavelet coefficients are the canonical representation.**
   All weighting happens in the wavelet domain (f, t), never via global FFT.
   The signal is decomposed once via meeglet; synthesis inverts that decomposition.

2. **Phase is sacred; aperiodic uses subtraction.**
   Periodic extraction uses real, non-negative weights (preserving phase).
   The aperiodic is defined as `original - periodic`, which preserves the full
   1/f power at peak frequencies. The legacy Wiener filter approach
   (`aperiodic_method="wiener"`) is still available but attenuates the aperiodic
   at peak frequencies.

3. **Parametric model comes from specparam, not from us.**
   We do not reimplement specparam's fitting. We call SpectralModel.fit() on
   wavelet-derived power profiles. If specparam's fit is bad, our decomposition
   is bad — and we surface that clearly via r² diagnostics.

4. **Log-frequency is the native grid.**
   meeglet's octave-spaced frequencies are used throughout. No interpolation to
   linear grids. specparam fits on this log grid directly.

5. **Synthesis is a frame multiplier (Balazs 2007) with OLA reconstruction.**
   Reconstruction quality is bounded by the frame condition number B/A, computed
   from the normalization envelope and reported in `ReconstructionResult.frame_condition`.
   Values near 1.0 indicate a tight frame; we also report energy ratios.

6. **NaN-awareness from meeglet propagates everywhere.**
   Bad segments marked NaN in the input are handled by meeglet's convolution.
   Weights at NaN time points are set to 0. Reconstruction at those points is 0.
   The residual preserves the original NaN-marked samples.

7. **Validation is against synthetic ground truth with known parameters.**
   We never validate by "it looks right." Every validation script compares
   recovered parameters or waveforms against analytically known targets.
   The four validation tiers are: stationary equivalence, non-stationary tracking,
   transient detection, and noise robustness.

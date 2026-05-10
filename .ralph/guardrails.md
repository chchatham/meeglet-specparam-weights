# Guardrails

Append-only. If you hit a wall, leave a sign so the next iteration doesn't hit it too.

## Domain Constraints

### 🚧 specparam requires linear-power input, not log10
specparam's SpectralModel.fit(freqs, powers, freq_range) expects `powers` in
linear scale (µV²/Hz or similar), NOT log10-transformed. meeglet's wavelet power
|Z(f,t)|² is already linear. Do not log-transform before passing to specparam.

### 🚧 specparam v2 REQUIRES linearly-spaced frequencies
~~specparam does not require linearly-spaced frequency arrays.~~ WRONG.
specparam v2 (2.0.0rc6) raises DataError if frequencies are not evenly spaced.
meeglet's octave-spaced `foi` array CANNOT be passed directly to specparam.fit().
MUST interpolate wavelet power to a linearly-spaced grid before fitting.
After fitting, reconstruct model power on the original log grid from parameters.

### 🚧 specparam v2 uses SpectralModel, not FOOOF — new API structure
The class is `from specparam import SpectralModel`. v2 restructured all attributes:
- `fm.results.params.aperiodic.params` → [offset, exponent]
- `fm.results.params.periodic.params` → converted params (FWHM bandwidth)
- `fm.results.params.periodic.get_params('fit')` → raw Gaussian std params
- `fm.results.model.modeled_spectrum` → log10 power
- `fm.results.model._ap_fit` / `_peak_fit` → component spectra in log10
- `fm.results.metrics.results['gof_rsquared']` → r²
- Model: log10(P) = (offset - exp*log10(f)) + sum(Gaussians on LINEAR freq)
- Peaks use LINEAR freq for center, but BW is Gaussian std (fit) or FWHM (converted).
  FWHM = 2*sqrt(2*ln2)*std ≈ 2.355*std. Use get_params('fit') for reconstruction.

### 🚧 Wavelet coefficients have frequency-dependent time support
A wavelet at 2 Hz spans ~2.5 seconds; at 32 Hz it spans ~0.15 seconds.
Edge effects are frequency-dependent. The synthesis normalization envelope
must account for this: edge samples at low frequencies have fewer contributing
wavelets and lower normalization denominators.

### 🚧 meeglet's power is in µV²/oct — MUST convert before specparam fitting
SOLVED: time_resolved_fit.py now divides wavelet power by (foi × ln2) before
interpolating to the linear grid and fitting with specparam. _reconstruct_model_power
multiplies the model power back by (foi × ln2) to return oct-unit power for weight
computation. Without this correction, the recovered exponent is ~1 unit too low.

### 🚧 Weights can explode where empirical power is near zero
Always floor the denominator with eps (default 1e-20) and clamp weights with
max_weight (default 100.0). This is inherited from specparam-fft-weights and
equally critical here.

### 🚧 NaN handling must be consistent across all modules
If input signal has NaN at sample k, then:
- wavelet coefficients at (f, t) are NaN for all t within kernel_width of k
- weights at those (f, t) points are 0
- reconstruction at those time points is 0
- residual at those time points preserves the original value (NaN)
Test this explicitly. A single inconsistency will produce silent corruption.

## Code Constraints

### 🚧 No global FFT anywhere in the pipeline
The whole point is wavelet-domain weighting. If you find yourself writing
np.fft.rfft on the full signal, you've left the architecture. The only FFTs
should be inside meeglet's own convolution (if it uses frequency-domain
convolution internally — that's meeglet's business, not ours).

### 🚧 Dataclass interfaces are the contract
If you need to change a field in WaveletDecomposition, TimeResolvedFit,
WeightSurface, or ReconstructionResult — update CLAUDE.md schemas AND
every module that produces or consumes that dataclass. Grep first.

### 🚧 Tests before implementation changes
If a test fails, do not delete it. Fix the implementation or update the test
with a comment explaining why the expectation changed. Log the reasoning in
errors.log.

## Performance Constraints

### 🚧 Single-sample wavelet power is too noisy for specparam
|Z(f,t)|² at a single time point is a chi-squared(2) estimate — extremely noisy.
specparam needs smooth spectra. MUST average power over a time window (default:
2 × fit_stride samples) before fitting. This is the `power_window` parameter
in time_resolved_fit().

### 🚧 meeglet.define_frequencies returns 5 values, not 4
`define_frequencies()` returns `(foi, sigma_time, sigma_freq, bw_oct, qt)`.
The type hint says 4, but the code returns 5. Always unpack all 5.

### 🚧 Synthesis is approximate — use n_iter≥5 for quality
OLA with normalization envelope alone has ~0.76 in-band correlation. Iterative
refinement (n_iter=5) significantly improves spectral fidelity — especially for
periodic suppression (37% → 5% alpha retention). The step size is computed
adaptively as 1/frame_norm. Do NOT use n_iter=1 (default) without mu; the
undamped iteration diverges. For validation, always use n_iter=5.

### 🚧 Wavelet reconstruction only covers foi_start–foi_end
The wavelet method only reconstructs content in the wavelet frequency range.
For comparison with FFT-weights (which passes through all frequencies), add
out-of-band signal content back via bandpass filter subtraction. The correlation
ceiling between methods is ~0.76 on 10s signals due to this fundamental difference.

### 🚧 Python version mismatch on this machine
`python3` resolves to Python 3.14 but dependencies are installed under 3.12.
Always use `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`
for running tests and scripts.

### 🚧 specparam fitting is slow — stride matters
SpectralModel.fit() takes ~5-50ms per call depending on frequency resolution
and number of peaks. A 10-second signal at 256 Hz has 2560 time points.
Fitting every point = 12-128 seconds. Default stride should be >= 10.
Interpolation between fitted points is mandatory for usable performance.

### 🚧 Figure generation uses matplotlib Agg backend
docs/generate_figures.py must use `matplotlib.use("Agg")` before importing
pyplot because the script runs headless. Takes ~2 minutes for all 7 figures
due to multiple specparam fits. Regenerate with:
`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 docs/generate_figures.py`

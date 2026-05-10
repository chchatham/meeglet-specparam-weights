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

### 🚧 .gitignore blocks *.png — docs/figures/ needs an exception
The `.gitignore` has a blanket `*.png` rule. To track figures for the GitHub Pages
site, `!docs/figures/*.png` was added as an exception. If new image directories are
added (e.g., `docs/screenshots/`), they will also need explicit exceptions.

## Multi-Channel Constraints (Phases 10–14)

### 🚧 Backward compatibility is non-negotiable
All 69 existing tests must pass at every step. Single-channel input must produce
identical output shapes and values. Detect dimensionality at entry points and
branch accordingly. Do NOT change existing dataclass field shapes for single-channel.

### 🚧 Multi-channel coefficients shape: (n_channels, n_freqs, n_times)
NOT (n_freqs, n_channels, n_times) or (n_freqs, n_times, n_channels).
Channel-first matches meeglet's (n_channels, n_samples) input convention
and numpy broadcasting norms.

### 🚧 Aperiodic trajectory effective bandwidth is limited by fit_stride
The interpolated exponent(t) and offset(t) have effective Nyquist at
sfreq / (2 * fit_stride). Content above this frequency is interpolation
artifact. Virtual-channel wavelet coefficients MUST be zeroed above this
cutoff. With default params (stride=50, sfreq=256), coupling is only
meaningful below ~2.5 Hz — infraslow modulations.

### 🚧 Circularity in aperiodic-oscillatory coupling
Aperiodic params are derived FROM wavelet power via specparam fitting.
Correlating them back to the same wavelet features creates spurious coupling.
This is NOT a bug — it is an inherent property. Document prominently.
Surrogate testing (block-permuted aperiodic trajectory) is required for
any inference. Consider computing exponent from a freq subset and testing
coupling against a disjoint set.

### 🚧 meeglet CSD uses time-averaged outer products
meeglet's CSD = data_conv @ data_conv.conj().T / n_valid, producing one
matrix per frequency (not time-resolved). Our augmented CSD must match
this convention exactly. The formula, normalization, and output namespace
format must be compatible with meeglet's SimpleNamespace output.

### 🚧 Virtual channel unit normalization
Exponent is unitless, offset is log10(power), wavelet coefficients are in
V²/oct. Z-score the virtual channel coefficients before computing CSD so
the off-diagonal coupling values are interpretable as correlation-like
quantities. Without normalization, the CSD matrix has wildly different
scales in different blocks.

### 🚧 test_rejects_2d_input must be updated
The existing test `test_rejects_2d_input` explicitly checks that 2D input
raises ValueError. This test must be updated (not deleted) to reflect the
new multi-channel behavior. Keep a test that rejects 3D+ input instead.
RESOLVED: renamed to `test_accepts_2d_input` and added `test_rejects_3d_input`.

## Simplification Constraints (Phase 15)

### 🚧 synthesize() returns a 3-tuple, not 2-tuple
`synthesize()` now returns `(reconstruction, energy_ratio, frame_condition)`.
All callers must unpack three values. If you add a new caller or test that
calls `synthesize()`, remember the third return value.

### 🚧 Unit consistency: aperiodic power must be in oct units
`_extract_component_power_single` in weight_surface.py must multiply Hz-unit
aperiodic power by `hz_to_oct = foi * np.log(2)` to match model_power and
empirical |Z|² which are in µV²/oct. This was a pre-existing bug fixed in
Phase 15. If you ever reconstruct aperiodic power from specparam parameters,
always apply this conversion.

### 🚧 conftest.py cannot be imported directly in test files
pytest auto-loads conftest.py but it is not importable as `from conftest import`.
Use `from tests.conftest import make_pink_noise` (the tests/ directory has
`__init__.py`). Alternatively, use it only as a pytest fixture.

### 🚧 Merged OLA functions — use compute_norm flag
`_ola_synthesis_signal_only` no longer exists. Use
`_ola_synthesis(coefficients, wavelets, n_samples, compute_norm=False)[0]`
for signal-only synthesis (e.g., in iterative refinement).

## Decomposition Paradigm (Phase 16)

### 🚧 Aperiodic reconstruction uses SUBTRACTION, not Wiener weighting
The Wiener filter w=sqrt(P_ap/|Z|²) attenuates alpha in the aperiodic reconstruction.
This is conceptually wrong. The correct approach:
1. Compute aperiodic weight: w_ap = sqrt(P_ap / |Z|²)
2. Derive bounded excess weight: w_excess = sqrt(max(0, 1 - w_ap²))  — always in [0, 1]
3. Synthesize periodic excess: periodic = synthesize(Z * w_excess)
4. Subtract: aperiodic = original - periodic
This guarantees original = periodic + aperiodic exactly and preserves the full 1/f
power at peak frequencies in the aperiodic component. The Wiener weight for
component="aperiodic" remains in weight_surface.py for diagnostic visualization
but is NOT used for synthesis.

### 🚧 Do NOT use model-based periodic weights for subtraction
The naive periodic weight sqrt(P_periodic / |Z|²) can exceed 1.0 (observed max: 56.7)
because the model can overestimate power at noisy time points (|Z|² is chi²(2), very
noisy). Weights > 1 amplify coefficients, causing OLA synthesis to explode (periodic
RMS 22 when signal RMS is 2.1). Always use the bounded excess weight formulation
w_excess = sqrt(max(0, 1 - w_ap²)) which is guaranteed ∈ [0, 1].

### 🚧 Periodic synthesis quality is critical for subtraction
Since aperiodic = original - periodic, any artifacts in the periodic synthesis
appear INVERTED in the aperiodic reconstruction. The pipeline forces
n_iter = max(n_iter, 5) in the subtraction path. Do not lower this.

### 🚧 "Alpha suppression" is the WRONG metric after Phase 16
The old validation metric "alpha suppression > 90%" measured how much alpha power
was removed from the aperiodic reconstruction. This was the old (incorrect) goal.
The new metric is "alpha preservation at 1/f level": the aperiodic reconstruction
should have alpha-band power matching the 1/f model prediction, not zero.

# Progress

## Current Focus
Phase 16 complete. 112 tests passing. No open work items.

## What Exists
- `src/meeglet_specparam_weights/__init__.py` — public API re-exports (includes `wavelet_effective_dof`)
- `src/meeglet_specparam_weights/wavelet_analysis.py` — Phase 1 ✓ (vectorized NaN propagation via binary_dilation)
- `src/meeglet_specparam_weights/time_resolved_fit.py` — Phase 2 ✓ (vectorized `_reconstruct_model_power`)
- `src/meeglet_specparam_weights/weight_surface.py` — Phase 3 ✓ (aperiodic weights deprecated for synthesis, kept for diagnostics)
- `src/meeglet_specparam_weights/synthesis.py` — Phase 4 ✓ (frame multiplier docstring, frame_condition diagnostic)
- `src/meeglet_specparam_weights/pipeline.py` — Phase 5+16 ✓ (subtraction aperiodic, `method` field, `aperiodic_method` param)
- `src/meeglet_specparam_weights/coupling.py` — Phase 13 ✓ (einsum CSD, np.correlate DOF, `wavelet_effective_dof`)
- `src/meeglet_specparam_weights/diagnostics.py` — Phase 9 ✓
- `tests/conftest.py` — shared fixtures + `make_pink_noise()` helper
- `tests/test_wavelet_analysis.py` — 20 tests ✓
- `tests/test_time_resolved_fit.py` — 16 tests ✓
- `tests/test_weight_surface.py` — 13 tests ✓
- `tests/test_synthesis.py` — 11 tests ✓ (includes frame_condition test)
- `tests/test_pipeline.py` — 18 tests ✓ (includes 7 Phase 16 subtraction tests)
- `tests/test_coupling.py` — 17 tests ✓ (includes 3 wavelet_effective_dof tests)
- `tests/test_diagnostics.py` — 8 tests ✓
- `validation/` — 4 simulation scripts + metrics.py + RESULTS.md ✓
- `docs/` — GitHub Pages site, figures, generation script ✓
- **Total: 112 tests, all passing**

## Key Implementation Details
- Wavelet convolution uses scipy.signal.fftconvolve in 'same' mode at every sample
- specparam v2 requires linearly-spaced freqs — wavelet power is interpolated to linear grid
- Single-sample power is too noisy — averaged over power_window before fitting
- Model power is reconstructed on the original log-freq grid from fitted parameters
- Synthesis uses OLA with normalization envelope (frame multiplier, Balazs 2007)
- Aperiodic uses subtraction: excess_w = sqrt(1 - P_ap/|Z|²), aperiodic = orig - synth(Z*excess_w)
- Frame condition B/A computed from normalization envelope interior, reported in ReconstructionResult
- `synthesize()` returns 3-tuple: `(reconstruction, energy_ratio, frame_condition)`
- Peak params: specparam returns FWHM bandwidth, must convert to std for reconstruction
- Python 3.12 at /Library/Frameworks/Python.framework/Versions/3.12/bin/python3

## What's Done This Session (simplify + Phase 15)
- **Bug fix**: `_ola_synthesis_signal_only` was deleted but still called in `_synthesize_single` — replaced with `_ola_synthesis(..., compute_norm=False)[0]`
- **Bug fix**: Unit mismatch in `_extract_component_power_single` — aperiodic/periodic power was in Hz units but model_power is in oct units. Added `hz_to_oct = foi * np.log(2)` multiplication.
- **Vectorizations applied**:
  - `_propagate_nans`: nested loop → `scipy.ndimage.binary_dilation`
  - `_reconstruct_model_power`: per-time loop → batch aperiodic across all valid times
  - `effective_dof` autocorrelation: per-lag list comprehension → `np.correlate(mode="full")`
  - `compute_aperiodic_csd`: per-freq loop → `np.einsum` fast path (fallback for NaN case)
  - `aperiodic_amplitude_correlation`: hoisted `np.abs(Z[ch])` out of inner loop
- **Phase 15a**: `wavelet_effective_dof(sigma_time, sfreq, n_samples)` — frequency-dependent DOF = T / (2 * sigma_time(f))
- **Phase 15b**: Frame bounds A, B from normalization envelope → `frame_condition` (B/A) in `ReconstructionResult`
- **Phase 15c**: `make_pink_noise(n_samples, sfreq, exponent_half, seed)` extracted to conftest.py, replaced 6+ inline copies across 6 test files
- **Phase 15d**: Module docstrings updated — weight_surface.py (Wiener filter), synthesis.py (frame multiplier, Balazs 2007)
- **CLAUDE.md**: Updated ReconstructionResult schema (added `frame_condition`), updated design principle #5
- **105 tests passing** (101 original + 4 new: 3 wavelet_effective_dof + 1 frame_condition)

## What's Done This Session (Phase 16: Subtraction Paradigm)
- **Core change**: `pipeline.py` — when `component="aperiodic"`, computes excess weights
  `w_excess = sqrt(max(0, 1 - w_ap²))`, synthesizes periodic excess, subtracts from original
- **Key insight**: Original periodic weights (`sqrt(P_periodic / |Z|²)`) could exceed 1.0
  (max 56.7!), causing OLA synthesis to explode. Excess weights are bounded [0, 1].
- **Forced `n_iter >= 5`** for the periodic synthesis in subtraction mode
- **New `ReconstructionResult.method` field**: "subtraction" (default for aperiodic) or "weight"
- **Legacy `aperiodic_method="wiener"` parameter** for backward compatibility
- **7 new tests**: method field, decomposition sums to original, periodic excess peak,
  bounded weights, invalid method raises ValueError
- **Validation**: All 4 scripts pass. Replaced alpha suppression metric with alpha
  preservation metric in sim_stationary.py and sim_noise_sweep.py
- **Docs**: Updated CLAUDE.md, README.md, guardrails.md, module docstrings, RESULTS.md
- **112 tests passing** (105 original + 7 new)

## What's Next
All planned phases (1-16) complete. Potential future work:
- pip-installable package (pyproject.toml)
- Real EEG data examples
- Surrogate testing implementation for coupling inference

## Decisions Made (Do Not Revisit)
1. Use meeglet's Morlet wavelets (not STFT) as the time-frequency representation.
2. Use specparam's SpectralModel.fit() directly — do not reimplement the optimizer.
3. Synthesis via overlap-add with normalization envelope, not exact dual frame.
4. Validation on synthetic signals only (no real EEG data in the repo).
5. Interpolate wavelet power to linear-freq grid for specparam fitting.
6. Use scipy.signal.fftconvolve in 'same' mode for sample-level wavelet coefficients.
7. Average power over a window before specparam fitting (single-sample too noisy).
8. Dark-themed figures matching the GitHub Pages site color scheme.
9. Multi-channel support via per-channel processing (no cross-channel fitting).
10. Aperiodic-oscillatory coupling via virtual-channel CSD approach.
11. Virtual channels band-limited to effective Nyquist (sfreq / 2*fit_stride).
12. Z-score virtual channel coefficients before CSD for unit normalization.
13. Simple amplitude correlation as secondary utility (not CSD-based).
14. Skip complex cross-covariance (B) and phase-binning (D) approaches.
15. Frame condition B/A as the synthesis quality diagnostic (not energy ratio alone).
16. Wavelet-aware DOF n_eff(f) = T/(2*sigma_time) alongside generic Bartlett DOF.
17. Aperiodic via subtraction (not Wiener): excess_w = sqrt(1-P_ap/|Z|²), aperiodic = orig - periodic.

## Known Issues
- Synthesis is approximate (OLA with normalization) — frame_condition and energy_ratio track quality
- Aperiodic exponent from specparam depends on wavelet density ('oct' vs 'Hz')
- specparam v2 API uses deeply nested attribute access (results.params.periodic.params)

## Decisions Deferred
- Exact smoothing kernel width for parameter trajectories
- Surrogate testing implementation details (block permutation scheme)
- Whether to support cross-channel aperiodic fitting (fit channel A, couple to channel B)

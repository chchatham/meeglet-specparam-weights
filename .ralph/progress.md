# Progress

## Current Focus
All phases complete (1–9). GitHub Pages site elaborated with validation figures.

## What Exists
- `src/meeglet_specparam_weights/__init__.py` — public API re-exports
- `src/meeglet_specparam_weights/wavelet_analysis.py` — Phase 1 ✓
- `src/meeglet_specparam_weights/time_resolved_fit.py` — Phase 2 ✓
- `src/meeglet_specparam_weights/weight_surface.py` — Phase 3 ✓
- `src/meeglet_specparam_weights/synthesis.py` — Phase 4 ✓
- `src/meeglet_specparam_weights/pipeline.py` — Phase 5 ✓
- `src/meeglet_specparam_weights/diagnostics.py` — Phase 9 ✓
- `tests/conftest.py` — shared fixtures
- `tests/test_wavelet_analysis.py` — 20 tests ✓
- `tests/test_time_resolved_fit.py` — 12 tests ✓
- `tests/test_weight_surface.py` — 13 tests ✓
- `tests/test_synthesis.py` — 8 tests ✓
- `tests/test_pipeline.py` — 9 tests ✓
- `tests/test_diagnostics.py` — 7 tests ✓
- `validation/__init__.py` — package marker
- `validation/metrics.py` — R², correlation, energy ratios, SNR
- `validation/sim_stationary.py` — Phase 6: stationary equivalence ✓
- `validation/sim_nonstationary.py` — Phase 7: exponent + alpha tracking ✓
- `validation/sim_transient.py` — Phase 8: beta burst detection ✓
- `validation/sim_noise_sweep.py` — Phase 8: SNR robustness ✓
- `validation/RESULTS.md` — documented results
- `docs/index.html` — GitHub Pages site with figures ✓
- `docs/generate_figures.py` — reproducible figure generation script ✓
- `docs/figures/` — 9 dark-themed PNG figures ✓
- **Total: 69 tests, all passing**

## Key Implementation Details
- Wavelet convolution uses scipy.signal.fftconvolve in 'same' mode at every sample
- specparam v2 requires linearly-spaced freqs — wavelet power is interpolated to linear grid
- Single-sample power is too noisy — averaged over power_window before fitting
- Model power is reconstructed on the original log-freq grid from fitted parameters
- Synthesis uses overlap-add with normalization envelope
- Peak params: specparam returns FWHM bandwidth, must convert to std for reconstruction
- Python 3.12 at /Library/Frameworks/Python.framework/Versions/3.12/bin/python3

## What's Done This Session
- Added periodic contribution surface figure (`periodic_weight_surface.png`) to devpage
  - Shows effective periodic amplitude w(f,t)×|Z(f,t)| normalized to [0,1]
  - Alpha band (10 Hz) appears as bright horizontal stripe on dark background
  - Power-weighted view avoids noisy raw weights at high frequencies
- Added phase preservation section to devpage "What It Produces"
  - Explains wavelet-domain phase preservation: arg(Z×w)=arg(Z) for real w≥0
  - Clarifies that weight surfaces are amplitude envelopes, not spectrograms
  - Explains time-domain phase fidelity depends on OLA synthesis quality
- Added phase preservation verification figure (`phase_preservation.png`)
  - Waveform overlay: original and periodic alpha align precisely
  - Phase difference: circular mean -1.3°, circular std 3.3°
  - Cross-correlation: peak at 0.0 ms lag (no systematic phase shift)
- Updated `generate_figures.py` to produce 9 figures (was 7)
- Expanded README phase preservation note in Design rationale section
- All 69 tests passing, all 9 figures generated successfully

## What's Next
All phases complete. Potential future work:
- Multi-channel input support
- pip-installable package (setup.py / pyproject.toml)
- Real EEG data examples

## Decisions Made (Do Not Revisit)
1. Use meeglet's Morlet wavelets (not STFT) as the time-frequency representation.
2. Use specparam's SpectralModel.fit() directly — do not reimplement the optimizer.
3. Synthesis via overlap-add with normalization envelope, not exact dual frame.
4. Validation on synthetic signals only (no real EEG data in the repo).
5. Interpolate wavelet power to linear-freq grid for specparam fitting.
6. Use scipy.signal.fftconvolve in 'same' mode for sample-level wavelet coefficients.
7. Average power over a window before specparam fitting (single-sample too noisy).
8. Dark-themed figures matching the GitHub Pages site color scheme.

## Known Issues
- Synthesis is approximate (OLA with normalization) — energy ratio tracks quality
- Aperiodic exponent from specparam depends on wavelet density ('oct' vs 'Hz')
- specparam v2 API uses deeply nested attribute access (results.params.periodic.params)

## Decisions Deferred
- Exact smoothing kernel width for parameter trajectories
- Whether to support multi-channel input

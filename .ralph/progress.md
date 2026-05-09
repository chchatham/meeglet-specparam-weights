# Progress

## Current Focus
All phases complete (1–9). Project is feature-complete.

## What Exists
- `src/meeglet_specparam_weights/__init__.py` — public API re-exports
- `src/meeglet_specparam_weights/wavelet_analysis.py` — Phase 1 ✓
- `src/meeglet_specparam_weights/time_resolved_fit.py` — Phase 2 ✓
- `src/meeglet_specparam_weights/weight_surface.py` — Phase 3 ✓
- `src/meeglet_specparam_weights/synthesis.py` — Phase 4 ✓
- `src/meeglet_specparam_weights/pipeline.py` — Phase 5 ✓
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
- `validation/RESULTS.md` — documented results
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
- Added iterative refinement to synthesis.py (n_iter parameter, adaptive step size)
- Added n_iter parameter to pipeline.py
- Applied Jacobian correction in time_resolved_fit.py: wavelet power (µV²/oct) is converted
  to µV²/Hz before specparam fitting, and model power is converted back to oct units. This
  fixes exponent recovery (was off by ~1 unit) and improves fit quality (r² 0.64 → 0.97).
- Added fit quality gating: reject fits with r² < 0.5 or exponent outside [-0.5, 10.0]
- Created validation/metrics.py, validation/sim_stationary.py, validation/sim_nonstationary.py
- Stationary validation: corr=0.85, RMS diff=10%, alpha suppression=99.3%
- Non-stationary validation: exponent tracking corr=0.92, alpha contrast ratio=7.2
- specparam-fft-weights is at ../specparam-tdresid/specparam_fft_weights.py (not pip-installed)
- Phase 8 complete: transient burst detection 96% ± 5%, SNR sweep r² never below 0.95
- Burst detection uses local median baseline (1s window) + MAD threshold (factor=3.0)
- Phase 9 complete: diagnostics.py (4 plot functions), test_diagnostics.py (7 tests), README updated

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

## Known Issues
- Synthesis is approximate (OLA with normalization) — energy ratio tracks quality
- Aperiodic exponent from specparam depends on wavelet density ('oct' vs 'Hz')
- specparam v2 API uses deeply nested attribute access (results.params.periodic.params)

## Decisions Deferred
- Exact smoothing kernel width for parameter trajectories
- Whether to support multi-channel input

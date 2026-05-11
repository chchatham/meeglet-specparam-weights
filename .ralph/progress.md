# Progress

## Current Focus
Phase 21 (Spectral PCA) complete. 211 tests passing.

## What Exists
- `src/meeglet_specparam_weights/__init__.py` — public API re-exports (includes state-space, separation, epochs)
- `src/meeglet_specparam_weights/separation.py` — Phase 17 ✓ (SeparationResult, subtraction/wiener/bias diagnostics)
- `src/meeglet_specparam_weights/state_space.py` — Phase 18 ✓ (Kalman oscillator + AR(p) decomposition)
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
- `validation/` — 5 simulation scripts + metrics.py + RESULTS.md ✓ (includes Phase 19 ground truth comparison)
- `docs/` — GitHub Pages site, figures, generation script ✓
- `tests/test_separation.py` — 21 tests ✓ (Phase 17 separation framework)
- `tests/test_state_space.py` — 23 tests ✓ (Phase 18 Kalman + AR decomposition)
- `src/meeglet_specparam_weights/epochs.py` — Phase 20 ✓ (ensemble decompose, epoch reconstruction, evoked separation)
- `tests/test_epochs.py` — 25 tests ✓ (Phase 20 multi-epoch ensemble estimation)
- `src/meeglet_specparam_weights/spectral_pca.py` — Phase 21 ✓ (CSD eigendecomposition, PC-space fitting, channel-space reconstruction)
- `tests/test_spectral_pca.py` — 30 tests ✓ (CSD, eigendecompose, projection, reconstruct, wiener, vs-per-channel)
- **Total: 211 tests, all passing**

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

## What's Done This Session (Phase 20)
- **Phase 20**: Multi-epoch ensemble estimation
  - `epochs.py`: `ensemble_decompose()` and `meeglet_specparam_reconstruct_epochs()`
  - Ensemble power averaging: trial-averaged |Z|² for stable specparam fits
  - Evoked separation: trial-averaged wavelet coefficients subtracted → induced-only decomposition
  - Per-epoch reconstruction: ensemble fit applied to each individual epoch via chosen separation strategy
  - `EpochDecompositionResult` dataclass: aperiodic, periodic, evoked, ensemble_fit, ensemble_power
  - 25 tests covering shapes, subtraction sum-to-original, exponent recovery, evoked correlation, validation
  - All 3 separation methods (subtraction, wiener, state_space) supported

## What's Done This Session (Phase 21)
- **Phase 21**: Spectral PCA Decomposition
  - `spectral_pca.py`: CSD eigendecomposition of multi-channel wavelet coefficients
  - Eigenvalue spectra as real, non-negative power spectra — ideal specparam input
  - PC-space fitting and weighting, back-projection to channel space
  - Subtraction and Wiener separation modes
  - `SpectralPCAResult` dataclass with eigenvectors, eigenvalues, mode fits, variance explained
  - 30 tests: CSD (6), eigendecompose (8), projection (3), reconstruct (9), wiener (2), vs-per-channel (2)
  - `__init__.py` updated with 4 new exports
  - `CLAUDE.md` updated with SpectralPCAResult schema and architecture entries

## What's Next
All 21 phases complete. 211 tests passing.
- Potential future work: pip-installable package, real EEG data examples, surrogate testing

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
18. State-space oscillator absorbs ALL narrowband power near peaks (including aperiodic) — alpha_power_ratio ≈ 0. Fundamental, not a bug.
19. Ground truth validation: subtraction best for waveform fidelity, Wiener best for spectral shape, state-space intermediate. No single method dominates.

## Known Issues
- Synthesis is approximate (OLA with normalization) — frame_condition and energy_ratio track quality
- Aperiodic exponent from specparam depends on wavelet density ('oct' vs 'Hz')
- specparam v2 API uses deeply nested attribute access (results.params.periodic.params)

## Decisions Deferred
- Exact smoothing kernel width for parameter trajectories
- Surrogate testing implementation details (block permutation scheme)
- Whether to support cross-channel aperiodic fitting (fit channel A, couple to channel B)

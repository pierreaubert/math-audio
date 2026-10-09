# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- IEC 60268-16 direct-method STI test signals (`signals::sti`):
  `gen_stipa_signal` / `gen_stipa_signal_seeded` (dual-modulated
  composite) and `gen_full_sti_signal` /
  `gen_full_sti_signal_seeded` (98 concatenated single-modulation
  segments, modulation-major, optional silence gaps). Seeded pink
  carriers filtered into the seven STI octaves with the same
  zero-phase Butterworth recipe as `math_rir::bands`, standard
  speech-spectrum weighting, active RMS normalized to 0.07.
- Drift-correction + array primitives (`req-math-audio-capture.md`):
  `capture_tdoa` (C1 band-limited GCC chirp TDOA with matched/PHAT
  weighting and parabolic interpolation; C2 two-point clock skew with
  validity flags and multi-chirp drift-series check; C4 CRB-form
  post-correction uncertainty in microseconds, infinite when
  uncomputable), `capture_resample` (C3 256-phase polyphase common-clock
  resampler applying offset + skew with exact IR time origin),
  `capture_array` (C5 LS TDOA-multilateration DOA with delay-and-sum
  scoring; C6 geometry container, loud TDOA-vs-tape check, scale
  self-calibration).
- Capture multi-rate coverage: C1/C4, C3 and C5 unit tests across 6, 12,
  44.1, 48, 88.2 and 96 kHz with rate-scaled chirp/analysis bands
  (identical relative band, so 48 kHz defaults are unchanged).
- RoomEQ report primitives (`req-math-audio-report.md`): `rir_early_late`
  (envelope-peak detector, 120 Hz lowpass subwoofer reference, 20 ms
  early/late split, fixed-centre 1/3-octave SPL), `rir_waterfall` (STFT
  decay grid over −5…500 ms with 60 ms resonance picking and per-resonance
  decay times), and `rir_wavelet` (3-cycle Morlet CWT heatmap, −30…0 dB).
- Canonical ESS measurement pipeline (`analysis/measurement.rs`): lag
  alignment with confidence, tail-aware deconvolution, Farina harmonic
  separation, synchronous averaging with outlier rejection, and combined
  measurement quality reports.
- MLS deconvolution, clock-drift estimation/correction, seeded noise
  generation, and log-frequency microphone compensation application to
  measured responses.
- `WavAnalysisConfig::room_slope_db` / `subwoofer`: optional end-to-end
  log-frequency tilt (0 dB at `min_freq`, slope value at `max_freq`),
  skipped for subwoofer measurements; exposed as `wav2csv`
  `--room-slope-db` / `--subwoofer`.

### Fixed
- `true_peak`: `peak` is now a running maximum over all processed samples
  instead of the last window; documented the Catmull-Rom path as approximate
  and the `ebur128` FIR table as canonical.
- `fdn`: saturation reporting for the ±4.0 safety clamp (plus a
  unitary-matrix validation helper) instead of silently masking instability.
- `rir_wavelet`: magnitudes are now referenced to each heatmap's own peak
  cell (0 dB = strongest cell), so scaling the input no longer shifts the
  display; the unit-sine calibration is documented as not surviving this
  display normalization. `rir_early_late` boundary handling fixed.
- `analysis`: lag estimation now reports confidence and rejects
  noise-only/silent recordings instead of returning an arbitrary lag.
- `analysis`: THD now uses the un-padded sweep duration (padded playback
  buffers previously shifted Farina harmonic offsets, silently corrupting
  THD); IR construction uses −60 dB-relative regularization like the FR
  path instead of an absolute 1e-20 floor.
- `analysis`: `deconvolve_sweep` accepts recordings longer than the
  reference (pre-roll + reverb tail) without circular tail wrap.
- `analysis`: replaced NaN-panicking `partial_cmp().unwrap()` sites with
  `total_cmp`; guarded `compute_thd_from_ir` against
  `start_freq >= end_freq`.
- `analysis::load`: tolerates truncated WAV data chunks (reads all complete
  samples past EOF instead of failing), making `wav2csv` robust to
  malformed files.
- `wav2csv`: pink-compensation help corrected to +3 dB/octave for log
  sweeps; pre-allocated buffers to minimise memory copies in analysis.
- `simd`: fixed `compute_covariance_simd` sign bug on the AVX2 path; CI now
  runs tests with `+avx2`.
- `rtpghi`: fixed swapped/mis-scaled phase gradients, wrong gamma
  (0.17 → 0.25645·M²), added relative log-magnitude threshold and phase
  wrapping; cross-validated against phaseret (coherence 0.04 → 1.00).
- `ebur128`: relative gate is now exactly −10 LU (was −9.309); RLB filter
  uses consistent 48 kHz coefficients; channel weights cover 1–8 channels.
- `analysis`: fixed inverted broadband C80 (tests rewritten); `rt60_ms` is
  now actually milliseconds; 16/24-bit WAV normalization fixed (−96 dB bug).
- `audio_features::chroma`: fixed transposed template masks (golden triad
  tests added).
- `esprit`: fixed MDL/AIC sign; auto mode now recovers 2 close tones.
- `adaa`: ADAA2 dilog now uses the Landen identity (fixes quiet-signal
  corruption, removes ~400 iterations/sample).
- `replaygain`: no longer returns +∞ on silence; album gating pools then
  gates once.
- Closed ~15 panic/NaN paths (short-window FR, short chroma input,
  `num_points == 1`, interleave mismatch, empty covariance range, SIMD
  length guards, NaN envelope poisoning, lookahead latency report,
  `DualWindowStft` latency, FDN per-line T60 gains, DC-bin consistency,
  malformed CSV errors, f32 tone phase growth, LCG high bits, pink-noise
  RMS).

### Performance
- Faster FDW Morlet kernels (~1.7–2.5×), ReplayGain (~3.2×), peak-only
  EBU R128 (~2.5×), ESPRIT SVD (−25%), `single_bin_dft`/`extract_tone_phase`
  (~3.2×), IF subbands (~2.8×), psychoacoustics `from_response` (2×) and
  `bark_spectrum` (2.6×), FFT convolution (2.4×), binaural-matrix planner
  caching (~1.6–2.1×); spectrogram/measurement allocation storms removed.
  All verified bit-identical or within pinned tolerance.
- Welch spectrum, chroma STFT, and audio-feature extraction now use
  real-to-complex FFTs with reused scratch buffers (aggregate benchmark
  4.64 ms → 1.72 ms, results unchanged).

## [0.5.26] - 2026-07-10

### Fixed
- `clock drift`: wrong division

## [0.5.22] - 2026-07-10

### Fixed
- `analysis`: Welch, single-FFT, and spectrogram magnitudes are now calibrated
  as peak-amplitude dBFS independently of FFT/window size; Welch also includes
  coherently normalized trailing partial frames.
- `analysis`: lag estimation now correlates the complete unwindowed signals,
  and cross-correlation envelope searches are restricted to valid positive
  lags instead of selecting circular aliases.
- `analysis`: harmonic transfer functions no longer retain an unmatched FFT
  `1/N` factor, restoring absolute THD levels, and silent sweep references no
  longer underflow regularization into NaNs.
- `analysis`: inverse frequency-response transforms now derive their FFT length
  from input-grid density, and spectrograms include an exact single frame.
- `stft`: dual analysis/synthesis windows now satisfy the COLA constraint at
  every hop phase instead of normalizing only the average overlap sum.
- `signals`: direct and windowed tone extraction now return true peak amplitude,
  while non-integral tone periods are accumulated before sample-length rounding
  to minimize leakage.

### Changed
- `simd::fast_inv_sqrt` now uses exact IEEE-754 square-root/reciprocal semantics
  instead of the Quake approximation.
- Clarified dB-domain smoothing, denormal flushing, and Welch amplitude-spectrum
  contracts.

## [0.5.20] - 2026-05-30

### Fixed
- `adaa`: ADAA2 near-coincident fallback now evaluates the published
  three-sample centroid instead of a middle-sample-biased weighted average.
- `adaa`: `dilog_neg` now uses the exact `Li_2(-1)` value near `z = 1`,
  avoiding slow alternating-series convergence at the worst-conditioned point.
- `dynamics_core`: expand-mode gate hold samples are cached when hold time or
  sample rate changes, removing the per-sample hold-time multiply from the hot
  path.
- `ebur128`: true-peak mode now documents the BS.1770-4 48 kHz FIR-table
  assumption and logs a warning for non-48 kHz meters while preserving
  native-rate analysis.

### Changed
- `stft`: removed unused `DualWindowStft` COLA state and added explicit
  coverage for the current analysis-window fill latency contract.
- `rtpghi` and `simd`: clarified zero-allocation RTPGHI scratch ordering,
  compile-time SIMD feature selection, and scoped FTZ/DAZ usage.

## [0.5.19] - 2025-05-13

### Added
- Added `binaural_loudness` module: streaming binaural-loudness meter
  (`BinauralLoudness`) applying ITU-R BS.1770-4 K-weighting and gated
  integration to a 2-channel ear-signal pair. Provides momentary,
  short-term, and integrated LUFS; cumulative sample peak and true peak
  per ear; interleaved or separate L/R input; reset; and a
  `BinauralLoudnessResult` snapshot. One-shot helper `measure_binaural`
  for offline analysis.
- Added surround → binaural downmix path: `BinauralDownmix` carries a
  per-channel `[L_ear, R_ear]` linear gain matrix; preset constructors
  `BinauralDownmix::bs775(SurroundLayout::{FiveZero, FiveOne, SevenOne})`
  implement ITU-R BS.775 stereo-downmix coefficients (centre / surrounds
  at −3 dB, LFE excluded per BS.1770-4). `BinauralLoudness::add_surround_f32`
  and `measure_binaural_from_surround` feed multichannel programmes
  through the matrix into the binaural meter.

## [0.5.18] - 2025-05-13

### Fixed
- `analysis::compute_rt60_broadband` now uses Schroeder backward integration
  with least-squares T30/T20 extrapolation and fit-quality rejection instead
  of first-crossing timing. This makes octave-band RT60 estimates less prone
  to inflated values from noisy or flattened decay tails.
- `analysis::compute_rt60_spectrum` now trims late steady-state noise on each
  band-filtered impulse before fitting RT60 and logs the selected fit method,
  `r²`, and fit window for easier diagnosis.

## [0.5.17] - 2025-05-13

### Fixed
- `ebur128`: `gating_blocks` changed from `Vec` to `VecDeque` to eliminate
  O(n) `remove(0)` shifts on the audio hot path once the 1-hour cap is
  reached (#1).
- `instantaneous_frequency`: phase unwrapping now uses `rem_euclid` instead
  of `%` for robust wrap-to-π behavior with negative differences (#2).
- `audio_features::utils::geometric_mean` now asserts that the input length
  is a multiple of 8. Previously `chunks_exact(8)` silently dropped the
  remainder, producing wrong results for non-multiple-of-8 slices (#3).
- `audio_features::spectral`: spectral flatness no longer hardcodes 256
  bins; it uses the largest multiple of 8 `<= norms.len()` (#4).
- `audio_features::chroma`: `pip_track` now returns empty pitch/mag vectors
  instead of erroring when the frequency mask is empty (#5).
- `analysis::compute_thd_from_ir`: harmonic extraction window minimum is now
  frequency-dependent instead of a fixed 256 samples (#6).
- `analysis::compute_coherence_from_realizations`: now returns `Err` for
  `N < 4` instead of silently returning γ² = 1 (#7).
- `fast_exp2`: documented the silent `[-126, 126]` clamp (#9).
- `fdn`: documented the rationale for the `±4` safety clamp (#10).
- Synchronized version strings in `README.md` and `CLAUDE.md` with
  `Cargo.toml` (#8).

## [0.5.16] - 2025-05-13

### Added
- Added reusable binaural transfer-matrix DSP primitives for RoomEQ CTC:
  regularized and weighted matrix inverse solves, approximate minimax
  worst-position reweighting, per-position reconstruction errors, FIR synthesis
  from half-spectra, sweep deconvolution, loopback/direct-peak alignment,
  harmonic residue suppression, direct-peak windowing, and complex
  frequency-dependent windowing.
- Added reusable psychoacoustic DSP primitives for expensive perceptual losses:
  Bark-scale conversion and aggregation, Zwicker-style specific/total loudness,
  sharpness, listening-level calibration, pairwise sensory roughness, cached
  feature extraction, and stereo HRTF/CTC convolution helpers.
- Added reusable frequency-response helpers for linear DSP modeling: complex
  biquad response, FIR response, and LR4 low/high crossover response.

### Performance
- Added parallelisation in FDW computation.

## [0.5.15] - 2025-05-13

### Added
- Added Frequency-Dependent Windowing (FDW) analysis for impulse responses,
  including Morlet-style frequency-dependent gates, FDW-gated magnitude, and
  direct/total time-frequency energy ratios for correction-depth consumers.

## [0.5.14] - 2025-05-13

### Added
- Added new signals: Dirac and MLS.

## [0.5.13] - 2025-05-13

### Changed
- Switched to `oxiblas-ndarray` for BLAS operations. Replaced ndarray's
  built-in dot product and matrix multiplication with oxiblas-ndarray's
  pure-Rust BLAS implementation for better performance on all platforms
  without requiring external BLAS libraries (OpenBLAS, Accelerate, MKL).

## [0.5.12] - 2025-05-13

### Added
- Added multi-sweep coherence and noise-floor primitives:
  - `compute_coherence_from_realizations` — per-bin γ² across N complex spectra.
  - `deconvolve_sweep` — inverse-filter deconvolution of one recorded log sweep.
  - `estimate_noise_floor_db_from_silence` — per-bin dB over a Hann-windowed FFT.

## [0.5.11] - 2025-05-13

### Added
- Added FCMLA instruction in SIMD (ARM8.3+ works on Apple ARM).

## [0.5.10] - 2025-05-13

### Added
- Added proper signal recording specialized on delays detection with narrowband probe.

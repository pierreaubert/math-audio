# Changelog

Notable changes across the math-audio workspace, newest first.
Per-crate details live in `crates/<name>/CHANGELOG.md`.

## [Unreleased]

Work present in the tree but not yet committed or released:

- `math-rir`: early-reflection picker threshold now lives in the envelope
  domain (relative to the direct burst's envelope peak instead of its
  waveform peak). The old comparison rejected every reflection in sparse
  deconvolved IRs, whose direct envelope sits far below its peak; gains
  remain reported relative to the direct peak.
- `math-rir`: the STIPA analytical-MTF test no longer effectively hangs
  in debug builds: its fixture convolution is FFT-based (was a naive
  ~5e9-operation double loop per channel) and the 64 independent channel
  realizations run in parallel via rayon. Assertions, realization count,
  and tolerances are unchanged; the test now completes in ~1 minute in
  debug instead of running for tens of minutes.

- `math-optimisation`: SACOBRA-style `cobra` surrogate constrained
  optimiser, single-trust-region constrained BO arm, shared
  `RbfSurrogate` models, CMA-ES native inequality constraints with
  adaptive-penalty merit + IPOP restarts + diagonal mode, and the
  `benchmark-constrained` suite (COBRA solves 5/6 vs 2/6 CMA-ES
  baseline).
- `math-test-functions`: CEC constrained fixtures `g04`, `g06`, `g08`,
  `g09`, `g24` (objectives + inequalities, registry metadata, known
  optima).
- `math-analog`: Shockley diode-clipper, Koren-12AX7A triode-stage, and
  Yeh FMV tone-stack component models (IDs 6–8) with the
  `component_report` characterization example.

## [0.5.4] - 2026-10-18

- Implemented the code audit's DSP correctness fixes: sound FIR SIMD type/feature
  boundaries, exact custom taps, whole-program and near-threshold loudness,
  decay-fit headroom, FIR true peaks, clarity invariance, and complete STFT streams.
- Driven small-buffer validation also exposed incorrect held triode plate values
  after grid jumps. The triode solver now uses an analytic derivative and
  safeguarded monotone root bracket, checked against independent f64 bisection.
- Added prepared/reusable FIR, Welch, common-clock resampling, filtfilt, and decay
  APIs; reduced median-selection work and bounded multiband workspace concurrency.
- Made required QA references fail visibly and verified executed comparison
  records against the complete manifest. Isolated hull exports and strengthened
  numerical, allocation, callback-size, and maintained benchmark coverage.
- Versions: math-dsp 0.5.35, math-iir-fir 0.5.27, math-rir 0.5.13,
  math-qa 0.5.4, math-analog 0.5.5, math-convex-hull 0.3.4.
- Broadband clarity now infers arrival by default; use the new explicit-start API
  for an existing array-origin convention. True-peak reconstruction has FIR delay
  and finite-tail completion. Unsupported decay fits now return no measurement.

## math-rir 0.5.12

- `math-rir`: full indirect IR-only Speech Transmission Index using the
  IEC 60268-16:2020 model, with 14 × 7 modulation transfer values and octave MTI.

## 2026-09

- 2026-09-27 — `math-autodiff`: full `f32`/`f64` generics, contiguous
  fast paths (17–156x), FNV-1a fingerprints, fused MSE,
  `forward_into`/`backward_into` allocation-free passes, `evo_bench`
  timing harness.
- 2026-09-23 — `math-dsp`: C1–C6 drift-correction + array primitives
  (chirp TDOA, clock-skew, polyphase resampler, DOA multilateration)
  with Wolfram QA cross-validation; `rir_wavelet` magnitudes go
  peak-relative.
- 2026-09-23 — `math-iir-fir`: faithful Kautz room correction
  (dry-plus-bank model, guarded Gauss-Newton fitting, versioned
  `kautz-correction-v1` bank spec).
- 2026-09-23 — `math-qa`: Kautz correction QA case; capture oracles go
  multi-rate (schema 2, 6–96 kHz); crate re-versioned 0.1.3 → 0.5.3.
- 2026-09-22 — `math-qa`: new crate — Wolfram Engine cross-validation
  harness (oracle scripts + comparison tests + checked-in goldens),
  capture TDOA/array cases, hermetic unit tests.
- 2026-09-22 — `math-dsp`: RoomEQ report primitives (`rir_early_late`,
  `rir_waterfall`, `rir_wavelet`).
- 2026-09-22 — `math-rir`: `report` module (early-reflection table +
  batched octave-band T60).
- 2026-09-09 — `math-iir-fir`: Kirkeby/minimum-phase fixes (original
  frequency coordinates, even-periodic mirroring, SPL-calibration
  invariance). `math-optimisation`: DE uses seeded RNG for
  full-archive replacement (`ExternalArchive::add_with_rng`).
- 2026-09-08 — `math-iir-fir`: unified `Crossover` API (Butterworth,
  Linkwitz–Riley, cascaded Bessel, Neville–Thiele). `math-optimisation`
  0.5.13: CMA-ES lazy eigendecomposition, Levenberg-Marquardt analytic
  Jacobian.
- 2026-09-04 — Workspace: CI coverage gates + per-crate `qa-*` recipes.
  `math-autodiff`: Bark/ERB, spectral-convergence, log-magnitude, and
  multi-scale losses (`Array5` API deprecated for VJP).
  `math-convex-hull`: wrong-hull-on-duplicates fix, scale-aware
  epsilon, degenerate-face rejection. `math-delaunay` 0.5.4: Voronoi
  bounds sanitization, bounds-checked queries. `math-rir`: Lundeby
  noise cutoff, per-band ISO 3382 metrics. `math-dsp`: `true_peak`
  running maximum, FDN saturation reporting. `math-test-functions`
  0.5.3: dimension arity asserts, `happy_cat` alias fix.

## 2026-08

- 2026-08-25 — `math-optimisation` 0.5.12: pre-allocated generation
  buffers, flaky caller-pool test fixed. `math-iir-fir`: pre-ringing
  suppression switches to a smooth cosine time-envelope cap.
- 2026-08-22 — `math-dsp`: in-room slope tilt + subwoofer mode in
  analysis and `wav2csv`; truncated-WAV robustness.
- 2026-08-18 — Major correctness batch. `math-dsp`: RTPGHI
  cross-validated against phaseret (coherence 0.04 → 1.00), AVX2
  covariance sign fix (+avx2 CI job), EBU R128 exact −10 LU gate,
  inverted-C80 fix, ESPRIT MDL/AIC fix, ADAA2 Landen dilog, ~15
  panic/NaN paths closed, broad speedups. `math-autodiff` 0.5.2:
  FFT/shape/saturation/noise fixes, shared plans, cached responses.
  `math-analog` 0.5.2: FFT-backed analysis, ADAA1/ADAA2, feature-gated
  DE→LM fitting with provenance, console preamp, tape/transformer
  defects. Workspace: `qa` aggregator + per-crate recipes, workspace
  clippy, 90% coverage gates.
- 2026-08-17 — `math-dsp`: canonical ESS measurement pipeline (lag
  confidence, tail-aware deconvolution, averaging, quality reports),
  MLS deconvolution, clock-drift estimation. `math-rir`:
  `Iso3382QualityVerdict`.

## 2026-07

- 2026-07-15 — `math-optimisation` 0.5.11: ~7.3x DE/CMA-ES hot-path
  speedups, criterion benches, `rand::RngExt` migration.
- 2026-07-13 — `math-autodiff` 0.5.1: numerical-correctness pass and
  hot-path optimisation (FFT/biquad/recursion), evo throughput merge.
  `math-dsp`: Welch/chroma/audio-features move to real-to-complex FFT
  (4.64 → 1.72 ms).
- 2026-07-10 — `math-dsp` 0.5.26/0.5.22: analysis invariant fixes
  (dBFS calibration, lag-alias rejection, THD levels, STFT COLA).
  `math-iir-fir` 0.5.16: biquad sanitisation, windowed-sinc
  normalization, Kirkeby transition tracking.

## 2026-06

- 2026-06-21 — Memory-copy removal + criterion benches across
  `math-optimisation`, `math-convex-hull`, `math-delaunay`,
  `math-test-functions` (incl. allocation-free `levy`);
  `math-iir-fir` AVX2/FMA + NEON fast paths (~2x filter throughput).
- 2026-06-14 — Proptest property suites + eval benches across
  optimisation, hull, delaunay, and test-functions.
- 2026-06-12 — SotF merge: `math-optimisation` gains the Bayesian
  backend, `continuous_area` loss integration, L-SHADE fixes, and
  ISRES/native COBYLA (nlopt dropped); `math-test-functions`
  literature-formula corrections; `math-delaunay` 0.5.3.

## Earlier

- 2026-05-30 — `math-rir` 0.5.6: `direct_sound_doa()`, public
  `schroeder_curve()`, SSIR 11 dB direct-sound rule, LER multi-maxima.
- 2026-05-13 — `math-rir` 0.5.5: ISO 3382 metrics module
  (EDT/T20/T30/C50/C80/D50/Ts with r²) and octave/third-octave band
  analysis; B-format DOA conventions.
- 2026-01 — Code moved into `crates/`, 0.3 re-versioning,
  `math_audio_*` naming made consistent.
- 2025-12-21 — Imported from AUTOEQ as a standalone workspace.

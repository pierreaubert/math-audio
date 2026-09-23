# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Capture QA oracles go multi-rate (schema 2): `capture_tdoa.wls` and
  `capture_array.wls` emit one C1/C4/C5/C6/C3 block per rate (6, 12,
  44.1, 48, 88.2, 96 kHz) with rate-scaled bands and tones; comparison
  tests loop over every rate entry. C2 stays single-entry: pure
  sample-domain arithmetic with no rate input.

## [0.1.2] - 2026-09-22

### Added
- Capture QA cases (`req-math-audio-capture.md`): `capture_tdoa.wls`
  (direct-summation C1 offsets, C2 skew arithmetic, C4 bound evaluation)
  and `capture_array.wls` (LinearSolve C5 DOA, C6 scale calibration,
  direct textbook C3 interpolation) with checked-in engine goldens and
  `wolfram_capture_tdoa` / `wolfram_capture_array` comparison tests.

## [0.1.1] - 2026-09-22

### Added
- Hermetic harness unit tests: golden load/skip/corrupt branches,
  fixture-engine live-runner tests (last-JSON-line parsing, missing
  binary, failing engine, non-JSON output), assert diagnostics, and
  golden-dir override resolution.

### Changed
- `reference`/`run_wolfram_script` split into env-wired thin wrappers
  over explicit `reference_with`/`run_wolfram_script_with` entry points
  so every branch is unit-testable without the live engine.
- Coverage ratchet raised 70 → 90 (lib at 98% lines).

## [0.1.0] - 2026-09-22

### Added
- Initial `math-qa` crate: Wolfram Engine cross-validation harness
  (`rel_error` metrics, live/golden reference loader, `QA_RESULT:`
  emitter) with 9 manifest cases covering biquad/SVF responses, FFT
  peak, Schroeder/T30, M1 comb dip, M2 third-octave SPL, M4 waterfall
  spots, M5 Morlet spots, and optimisation test functions.

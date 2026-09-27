# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.4] - 2026-09-27

### Added
- `fitting`: documented capture-fit-validate loop with numeric acceptance
  thresholds (`FIT_RMS_ACCEPTANCE`, `FitQualityReport::meets_acceptance_criteria`).
- `stateful`: oversampling guidance in model API docs (when to run 2x/4x,
  latency reporting, prepared-capture time constants).
- Component tier from published classical references (frozen in
  `references/component-references.md`): `DiodeClipperModel` (append-only
  ID 6, Shockley shunt clipper with silicon/germanium flavors),
  `TriodeStageModel` (append-only ID 7, Koren-12AX7A common-cathode stage
  at the documented 350 V / 150 kΩ operating point), and `ToneStackModel`
  (append-only ID 8, Yeh DAFx-06 FMV/TMB network with schematic and
  production value sets). All existing presets keep their meaning.
- `component_report` example with clipper threshold/symmetry, triode
  gain/THD/solver-stat, and tone-stack scoop/sweep evidence rows, plus a
  checked-in `reports/components.md` artifact.
- `solve_stats` accessors on the Newton-solved models reporting total
  solves and loose/hold fallbacks.

### Fixed
- `effects::solve_bounded_nonlinear`: a stalled iterate that already
  satisfies the tolerance now reports convergence (matching the
  exhausted-budget exit) instead of always reporting failure.

### Tests
- Pinned `ControlSmoother` reconfiguration across sample rates.
- Component gates: clipper DC/threshold/symmetry/passivity checks, triode
  Q-point/gain/inversion/THD checks, tone-stack response-shape agreement
  against the Yeh symbolic oracle, and a finite alias characterization
  for IDs 6-8 (the 50% folded-energy guard stays scoped to IDs 0-5).

## [0.5.2] - 2026-08-18

### Added
- FFT-backed offline analysis with window variants, log-chirp generation and
  deconvolution, harmonic/IMD measurements, transient metrics, and calibrated
  level matching.
- ADAA1 and ADAA2 paths across the memoryless, Hammerstein, tape, transformer,
  and console/preamp model families, with derivative regression coverage.
- Feature-gated measurement fitting with independent held-out captures,
  FFT magnitude/phase objectives, differential-evolution plus Levenberg–Marquardt
  fitting, provenance hashes, and fit-quality reports.
- Wiener–Hammerstein console/preamp processing with an optional prepared
  Hammerstein pre-filter and level-matched comparison reporting.
- Tape EQ, head bump, level-dependent high-frequency loss, configurable time
  constants, optional hysteresis, and normalized Jiles–Atherton behavior.
- Optional transformer bounded-flux behavior and individually gated defect
  modules for tape, transformer, and console/preamp models.
- Runtime-dispatched SIMD Chebyshev and static-curve helpers, prepared batch
  processing, denormal stress coverage, and release callback/SIMD performance
  criteria for six- and twelve-channel processing.

### Changed
- Corrected fifth-order Hammerstein ADAA antiderivatives and expanded the
  analysis, spectral, realtime, and model-contract reports.

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

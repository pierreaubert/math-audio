# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Quickhull furthest-point search now uses the deduplicated vertex slice,
  fixing wrong-hull selection on inputs containing duplicates.
- Visibility epsilon is scale-aware throughout (`HullFace` and `Face`),
  fixing misclassification at very large/small coordinate magnitudes.
- Degenerate zero-area faces are rejected instead of falling back to a
  bogus `(0,0,1)` normal.

### Tests
- Added duplicate-points hull regression coverage and
  translated/large-scale property tests.

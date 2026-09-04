# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `loss`: Bark/ERB weighting helpers (`bark_weights`, `erb_weights`,
  `bark_weighted_loss`, `erb_weighted_loss`), spectral-convergence,
  log-magnitude, and multi-scale spectral losses with VJP backward passes.

### Changed
- `Series`/`Parallel` warm the forward cache to avoid first-step
  recompute plus per-module clones.

### Deprecated
- The public `Array5` full-Jacobian response API (`sos_frequency_response_jacobian*`);
  `SosFilter::backward` uses the O(K·M) VJP path.

## [0.5.2] - 2026-08-18

### Fixed
- FFT and IFFT single-channel fast paths now handle non-contiguous tensors
  without panicking.
- Delay and matrix modules now use their current public parameter shapes and
  reject mismatched backward inputs instead of indexing out of bounds.
- Saturated biquad cutoff parameters remain finite, unstable SOS poles are
  rejected, and SOS response helpers return errors for invalid coefficient
  shapes.
- White-noise signals now draw independent values for each channel.

### Performance
- FFT plans and scratch buffers are shared or reused across processing calls.
- Delay responses and composition intermediates are cached across forward and
  backward passes.
- SOS filter coefficient gradients now use the direct VJP path, and
  orthogonal-matrix gradients use an exact block matrix-exponential derivative.

### Changed
- Removed the unused legacy `Gradient`/`Parameters` abstraction.
- Added regression coverage for malformed shapes, non-contiguous FFT inputs,
  saturated filter parameters, unstable poles, and multichannel noise.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Math-Audio is a Rust workspace containing numerical computing libraries for audio processing and acoustic analysis: filters, DSP utilities, optimisation algorithms, test functions, computational geometry, and room-acoustics analysis.

## Build and Test Commands

```bash
# Install dependencies (run once)
just install-macos    # macOS with Homebrew
just install-linux    # Linux (Ubuntu/Debian)

# Build
just build            # Build release (alias for prod)
just dev              # Build debug

# Test
just test             # cargo check + cargo test --lib --release
just ntest            # Run with nextest (parallel, no-fail-fast)

# Run single test
cargo test -p math-dsp test_name
cargo test -p math-iir-fir test_name -- --nocapture

# QA suites
just qa               # Run math QA suite
just qa-math          # math-dsp fuzzer

# Format and lint
just fmt              # Format all code
cargo clippy --workspace  # Lint (strict clippy enabled in workspace)
```

## Architecture

### Workspace Crates

| Crate | Purpose |
|-------|---------|
| `math-convex-hull` | 3D convex hull (Quickhull) |
| `math-delaunay` | 2D Delaunay triangulation and Voronoi diagrams |
| `math-dsp` | DSP utilities: signal generation, FFT analysis, loudness, dynamics, etc. |
| `math-iir-fir` | IIR/FIR filters, biquads, PEQ, crossovers, EqualizerAPO export |
| `math-optimisation` | Non-linear optimisation: DE, Levenberg-Marquardt, COBYLA, ISRES, CMA-ES |
| `math-rir` | Room Impulse Response analysis: SSIR segmentation, ISO 3382 metrics |
| `math-test-functions` | Test functions for optimisation algorithms |

### Key Dependency Flow

```
math-dsp → math-iir-fir
math-rir → math-iir-fir
math-optimisation → math-test-functions
```

### DSP Architecture (math-dsp/src/)

- `analysis/`: Signal analysis (group delay, THD, coherence, etc.)
- `audio_features/`: Chroma, loudness, spectral features, tempo, ZCR
- `binaural_loudness/`: BS.1770 loudness metering
- `dynamics/`: Compressor/limiter/gate core
- `ebur128/`: EBU R128 loudness measurement
- `esprit/`: Frequency estimation
- `fdn/`: Feedback delay networks
- `simd/`: SIMD utilities
- `signals/`: Signal generators
- `stft/`: Short-time Fourier transform

### IIR/FIR Architecture (math-iir-fir/src/)

- `iir/`: Biquad, PEQ, biquad banks, Kautz, warped biquad
- `fir/`: FIR filters and banks
- `filtfilt/`: Zero-phase forward/reverse filtering
- `fir_crossover/`, `lr4_crossover/`, `lr8_crossover/`: Crossovers
- `svf/`: State-variable filter

### Optimisation Architecture (math-optimisation/src/)

- `differential_evolution_mod/`: Differential Evolution variants
- `levenberg_marquardt/`: LM solver
- `cobyla_native/`: COBYLA constraint optimiser
- `isres/`: ISRES evolutionary strategy
- `cmaes/`: CMA-ES
- `nsga/`: NSGA-II multi-objective optimisation
- `bayesian/`: Bayesian optimisation

## Feature Flags

Important features to know when building:

- `parallel`: Explicit rayon parallelization
- `cli`: CLI dependencies (clap, etc.)
- `plotly`: Plotting support for binaries
- `plotly_static`: Static PNG export (requires chromedriver)
- `wasm`: WebAssembly support (requires nightly for parallel)

## Rust Edition and Toolchain

- Edition: 2024
- Toolchain: 1.92.0 (pinned in `rust-toolchain.toml`)
- Strict clippy lints enabled at workspace level

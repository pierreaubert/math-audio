# AGENTS.md

This file provides guidance to coding agents working with code in this repository.

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

- `analysis/`: Signal analysis (frequency response, RT60, C50/C80, THD, group delay, microphone compensation, CSV I/O)
- `audio_features/`: Chroma, loudness, spectral features, tempo, ZCR
- `binaural_loudness/`: BS.1770 binaural loudness metering with surround downmix presets
- `binaural_matrix/`: Transfer-matrix inversion, deconvolution, regularized inverse solutions
- `dynamics_core/`: Compressor/limiter/gate core (ADAA, envelope, detector, lookahead, auto-makeup, channel linking)
- `ebur128/`: EBU R128 / ITU-R BS.1770-4 loudness measurement
- `esprit/`: Sinusoidal frequency estimation
- `fast_math/`: Fast math approximations
- `fdn/`: Feedback delay networks
- `fdw/`: Frequency-dependent windowing for impulse responses
- `instantaneous_frequency/`: Instantaneous frequency extraction
- `psychoacoustics/`: Psychoacoustic utilities
- `replaygain/`: ReplayGain analysis
- `response/`: IIR/FIR complex frequency-response helpers
- `rtpghi/`: Real-Time Phase Gradient Heap Integration
- `signals/`: Signal generators (tones, sweeps, noise)
- `simd/`: SIMD utilities
- `stft/`: Short-time Fourier transform
- `tonal_transient/`: Tonal/transient separation
- `waveform/`: Waveform visualization data

### IIR/FIR Architecture (math-iir-fir/src/)

- `iir/`: Biquad, PEQ, biquad banks, Kautz, warped biquad
- `fir/`: FIR filters and banks
- `fir_design/`: FIR design from frequency response, Kirkeby correction, pre-ringing suppression
- `filtfilt/`: Zero-phase forward/reverse filtering
- `fir_crossover/`, `lr4_crossover/`, `lr8_crossover/`: Crossovers
- `phase_smooth/`: Phase unwrapping and smoothing via group delay
- `svf/`: State-variable filter (Zavalishin TPT zero-delay feedback)

### Optimisation Architecture (math-optimisation/src/)

Solvers:

- `differential_evolution_mod/`: Differential Evolution variants and core DE loop
- `differential_evolution.rs`: Public DE facade
- `levenberg_marquardt/`: LM nonlinear least-squares solver
- `cobyla_native/`: Pure-Rust COBYLA constraint optimiser
- `cobyla.rs`: Public COBYLA facade
- `isres/`: ISRES evolutionary strategy
- `cmaes/`: CMA-ES
- `nsga/`: NSGA-II/III multi-objective optimisation
- `bayesian/`: Gaussian-process Bayesian optimisation

DE machinery:

- `mutation/`, `mutant_*`: Mutation strategies (rand, best, current-to-best, current-to-pbest, etc.)
- `crossover_binomial/`, `crossover_exponential/`: Crossover operators
- `init_latin_hypercube/`, `init_random/`, `init_sobol/`: Initialization strategies
- `lshade/`: L-SHADE adaptive population reduction
- `external_archive/`: L-SHADE archive
- `adaptive_config/`, `adaptive_state/`: SHADE-style adaptive F/CR control
- `parallel_eval/`: Parallel population evaluation
- `linear_constraint_helper/`, `nonlinear_constraint_helper/`: Constraint helpers
- `recorder/`, `run_recorded/`: Optimization recording and replay
- `continuous_area/`: Continuous-prior / area-based loss integration

### RIR Architecture (math-rir/src/)

- `lib/analyze.rs`, `lib/misc.rs`: RIR analysis entry points
- `config.rs`: SSIR configuration
- `detection.rs`: Direct sound and reflection detection
- `segmentation.rs`: SSIR event building
- `mixing_time.rs`: Echo density and mixing time estimation
- `metrics.rs`: ISO 3382 metrics (EDT, T20, T30, C50, C80, D50, Ts)
- `bands.rs`: Octave and third-octave band filtering
- `types.rs`: `RirSegment`, `SsirResult`

### Computational Geometry Architecture

- `math-delaunay/src/`: `delaunay.rs` (triangulation), `voronoi.rs` (Voronoi cells)
- `math-convex-hull/src/`: `quickhull.rs` (3D hull), `geometry.rs`, `types.rs`, `export.rs`, `testdata.rs`

## Binaries and Examples

| Target | Command |
|---|---|
| `plot-functions` | `cargo build --release --bin plot-functions -p math-test-functions --features plotly` |
| `plot-de` | `cargo build --release --bin plot-de -p math-optimisation --features plotly` |
| `run-de` | `cargo build --release --bin run-de -p math-optimisation` |
| `wav2csv` | `cargo build --release --bin wav2csv -p math-dsp` |
| `simd-fuzzer` | `cargo build --release --bin simd-fuzzer -p math-dsp` |

Run example suites with `just examples` (IIR/FIR, optimisation, test functions).

## Feature Flags

Important features to know when building:

- `plotly`: Plotting support for the `plot-functions` and `plot-de` binaries
- `plotly_static`: Static PNG export (requires chromedriver)

Note: `rayon` parallelism and `clap` CLI parsing are used directly where needed;
there are no workspace `parallel`, `cli`, or `wasm` feature flags at this time.

## Rust Edition and Toolchain

- Edition: 2024
- Toolchain: 1.92.0 (pinned in `rust-toolchain.toml`)
- Strict clippy lints enabled at workspace level

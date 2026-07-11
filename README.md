<!-- markdownlint-disable-file MD013 -->

# Math-Audio: a toolkit for audio applications

Math-Audio is a Rust workspace of numerical computing libraries for audio
processing and acoustic analysis: DSP utilities, IIR/FIR filters, optimisation
algorithms, test functions, computational geometry, and room-acoustics analysis.

## Install

Install [rustup](https://rustup.rs/) first, then:

```shell
cargo install just
just
```

## Build and test

```shell
just build   # release build
just dev     # debug build
just test    # cargo check + cargo test --lib --release
just ntest   # nextest, parallel and no-fail-fast
just fmt     # format all code
```

Run the QA suite:

```shell
just qa
```

## Toolkit

| Crate | Purpose | Status |
|---|---|---|
| [`math-convex-hull`](crates/math-convex-hull/README.md) | 3D convex hull (Quickhull) | stable |
| [`math-delaunay`](crates/math-delaunay/README.md) | 2D Delaunay triangulation and Voronoi diagrams | stable |
| [`math-dsp`](crates/math-dsp/README.md) | DSP utilities: signal generation, FFT analysis, loudness (EBU R128, binaural), dynamics, STFT, RTPGHI, ESPRIT | stable |
| [`math-autodiff`](crates/math-autodiff/README.md) | Frequency-domain differentiable audio DSP: FFT/IFFT, delays, gains, matrices, IIR/SOS filters, feedback systems, and analytical gradients | stable |
| [`math-optimisation`](crates/math-optimisation/README.md) | Non-linear optimisation: DE, L-SHADE, CMA-ES, NSGA-II/III, COBYLA, ISRES, Levenberg-Marquardt, Bayesian | stable |
| [`math-iir-fir`](crates/math-iir-fir/README.md) | IIR/FIR/SVF filters, biquads, PEQ, crossovers, filtfilt, EqualizerAPO/RME/etc. export | stable |
| [`math-rir`](crates/math-rir/README.md) | Room Impulse Response analysis: SSIR segmentation, ISO 3382 metrics | stable |
| [`math-test-functions`](crates/math-test-functions/README.md) | 56+ benchmark functions for optimisation algorithms | stable |

### Key dependency flow

```text
math-dsp → math-iir-fir
math-rir → math-iir-fir
math-optimisation → math-test-functions
math-autodiff → math-dsp, math-iir-fir
```

## Repository

<https://github.com/pierreaubert/math-audio>

## License

ISC

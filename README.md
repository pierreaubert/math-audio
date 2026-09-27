<!-- markdownlint-disable-file MD013 -->

# Math-Audio: a toolkit for audio applications

Math-Audio is a Rust workspace of numerical computing libraries for audio
processing and acoustic analysis: DSP utilities, IIR/FIR filters, optimisation
algorithms, analog-style coloration models, test functions, computational
geometry, and room-acoustics analysis.

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

### [`math-convex-hull`](crates/math-convex-hull/README.md) — stable

3D convex hull via the Quickhull algorithm (Barber et al. 1996), ported
from Leo McCormack's `convhull_3d`.

- `ConvexHull3D::build(&[Vertex])` returns triangular faces; `volume()`
  and `surface_area()` properties; O(n log n) expected complexity
- Robustness: duplicate-point dedup, scale-aware visibility epsilon,
  degenerate zero-area face rejection
- Export: OBJ for 3D software plus interactive HTML with synchronized
  point-cloud/mesh views
- Test-data generators: Platonic solids, random/Fibonacci/T-design
  spheres, interior-point and OBJ fixtures

### [`math-delaunay`](crates/math-delaunay/README.md) — stable

2D Delaunay triangulation and Voronoi diagrams; faithful port of
`d3-delaunay` on the `delaunator` backend.

- `Delaunay::from_points` triangulation with point-location queries;
  `Voronoi` derivation with bounds clipping and per-cell polygon
  extraction
- Robustness: coordinate-magnitude-scaled collinearity/circumcenter
  epsilons, bounds-relative edge-code and dedup epsilons, deterministic
  NaN/collinear handling
- Release-mode sanitization of non-finite/unordered Voronoi bounds;
  documented `find` greedy-descent and `cell_polygon` scale-assumption
  contracts

### [`math-dsp`](crates/math-dsp/README.md) — stable

Signal generation, FFT-based analysis, loudness, dynamics, and feature
extraction for audio applications.

- **signals**: tones, log sweeps, white/pink/M-noise, MLS, probes; fades,
  padding, channel utils
- **analysis**: Welch/single-FFT response, recording-vs-reference
  transfer function, RT60, C50/C80, THD, group delay, mic compensation,
  CSV I/O
- **capture_tdoa / capture_resample / capture_array**: chirp TDOA,
  clock-skew estimation, polyphase resampling, array DOA
  multilateration
- **rir_early_late / rir_waterfall / rir_wavelet**: early/late split,
  STFT decay grid, Morlet CWT heatmap
- **ebur128 / binaural_loudness / replaygain / true_peak**: BS.1770-4,
  binaural, ReplayGain, true-peak metering
- **dynamics_core** (+ adaa, detector, envelope, lookahead):
  compressor/limiter/gate building blocks
- **stft / rtpghi / esprit / instantaneous_frequency /
  tonal_transient / fdn / fdw**: time-frequency tools, ESPRIT
  estimation, FDN reverb, frequency-dependent windowing
- **audio_features / psychoacoustics / binaural_matrix / response**:
  chroma, spectral, tempo, loudness models, CTC inverses,
  filter-response helpers
- **Binaries**: `wav2csv` (WAV→CSV analysis), `simd-fuzzer` (SIMD
  validation)

### [`math-analog`](crates/math-analog/README.md) — alpha

Host-independent analog-style coloration models for realtime audio; owns
model math and per-channel state, not plugin schemas or oversampling
wrappers.

- Nine model families: Chebyshev H2/H3 harmonic baseline, memoryless
  static curves (tanh/soft/hard clip), ≤5-branch Hammerstein, stylized
  tape/transformer with memory-flux equations, Wiener–Hammerstein
  console preamp, Shockley diode clipper, Koren-12AX7A triode stage,
  Yeh FMV tone stack
- Realtime contract: checked prepare/reset, allocation-free steady
  state, deterministic reset, NaN/Inf sanitization, DC blocker, zero
  reported latency
- ADAA1/ADAA2 antialiasing on memoryless paths; stateful models rely
  on host-owned 2x/4x oversampling
- Offline analysis: windowed FFT harmonics/IMD with alias marking,
  transient metrics, log-chirp capture/deconvolution, BS.1770
  level-matched comparison
- Feature-gated measurement fitting (DE → LM) with held-out
  validation and immutable coefficient provenance

### [`math-autodiff`](crates/math-autodiff/README.md) — beta

Frequency-domain differentiable audio DSP: LTI modules with analytical
gradients, optimized in the frequency domain with no AD framework.

- Building blocks: real FFT/IFFT wrapper, gain matrices, MIMO /
  per-channel delays, dense and orthogonal learnable matrices,
  `Recursion` closed-loop feedback composition
- Differentiable filters: RBJ biquad, generic SOS cascade, SVF
  (`fc`/`R`/gain), graphic EQ (ISO bands), parametric EQ
  (frequency/Q/gain per section)
- Composition via `Series`, `Parallel`, `Shell`; all modules
  implement `DiffModule` (`forward`/`backward`, parameters,
  gradients, `zero_grad`)
- Losses (MSE, Bark/ERB-weighted, spectral-convergence,
  log-magnitude, multi-scale) with VJP backward passes, signal
  generators, SGD optimizer
- Full `f32`/`f64` generics, allocation-free `_into` passes,
  contiguous fast paths; six magnitude-matching examples (FDN,
  biquad, PEQ, SVF, GEQ, FDN+direct)

### [`math-optimisation`](crates/math-optimisation/README.md) — stable

Pure-Rust non-linear and global optimisation.

- **Differential Evolution**: rand/best/current-to-best/pbest
  mutations, binomial/exponential crossover, SHADE-style adaptive
  F/CR, external archive
- **L-SHADE**: linear population reduction (18×dim → 4) with
  current-to-pbest/1
- **CMA-ES**: native inequality constraints with adaptive-penalty
  merit, IPOP restarts, diagonal-covariance mode
- **Constrained solvers**: SACOBRA-style `cobra` RBF-surrogate
  optimiser, single-trust-region constrained BO, shared
  `RbfSurrogate` models
- **Local / least-squares**: Levenberg-Marquardt (analytic Jacobian
  support), pure-Rust COBYLA, ISRES, Gaussian-process Bayesian
  optimisation (EI, q-EI, Thompson, qEHVI), NSGA-II/III
- **Constraints**: linear/non-linear helpers, penalty stacking,
  mixed-integer rounding; `continuous_area` module for
  continuous-prior loss integration
- **Tooling**: run recording/replay (`recorder`, `run_recorded`),
  parallel evaluation, `benchmark-constrained` suite,
  `plot-de`/`run-de` binaries

### [`math-iir-fir`](crates/math-iir-fir/README.md) — stable

Biquads, PEQ, crossovers, FIR design, and offline filtering for audio,
generic over `f32`/`f64`.

- **Biquads**: Peak, Lowpass, Highpass (+VariableQ), Lowshelf,
  Highshelf, Bandpass, Notch, AllPass; Orfanidis shelves,
  PeakMatched (Vicanek)
- **SVF**: Zavalishin TPT zero-delay-feedback state-variable filter
- **PEQ**: multi-band EQ with SPL response, loudness compensation,
  preamp gain, 9 export formats (APO, RME, CamillaDSP, EasyEffects,
  …)
- **Crossovers**: unified `Crossover` API (Butterworth,
  Linkwitz–Riley, cascaded Bessel, Neville–Thiele) plus LR4/LR8 and
  linear-phase FIR
- **FIR design**: windowed-sinc banks, design from frequency
  response, Kirkeby correction, pre-ringing suppression
- **Kautz**: resonant-basis room correction with dry-plus-bank model
  and Gauss-Newton fitting
- **Offline**: zero-phase `filtfilt`/`sosfilt`, warped LPC, phase
  smoothing via group delay
- **Robustness**: `FirError` + fallible constructors, denormal
  (FTZ/DAZ) policy, x86_64 AVX2/FMA and NEON fast paths

### [`math-rir`](crates/math-rir/README.md) — beta

Segments measured RIRs into directional sound events; ISO 3382
metrics and RoomEQ report primitives.

- **SSIR segmentation**: direct sound + early reflections as
  variable-length events with constant DOA (Pawlak & Lee 2026)
- **Detection**: 11 dB log-magnitude onset, Local Energy Ratio
  reflection picking with DOA/TOA validation, onset refinement
- **Mixing time**: Abel & Huang echo-density estimation
- **ISO 3382 metrics**: EDT, T20, T30 (with r² + quality verdicts),
  C50, C80, D50, Ts; per-band octave/third-octave analysis
- **report**: 1–8 kHz reflection table (< 15 ms post-direct,
  gain/distance/first-dip/comb ripple), batched 63 Hz–16 kHz octave
  T60 with validity flags
- **Bands**: ISO/IEC 61260 octave/third-octave zero-phase
  Butterworth filterbank (rayon-parallel)

### [`math-test-functions`](crates/math-test-functions/README.md) — stable

100+ benchmark functions for optimisation testing.

- **Unimodal**: `sphere`, `rosenbrock`, `elliptic`, `cigar`,
  `zakharov`, `powell`, `dixons_price` — convergence speed and
  precision
- **Multimodal**: `ackley`, `rastrigin`, `griewank`, `schwefel`,
  `eggholder`, `michalewicz`, Hartman 3/4/6-D, `branin`,
  `six_hump_camel` — global search
- **Constrained**: CEC fixtures `g04`, `g06`, `g08`, `g09`, `g24`
  (objectives + inequalities, known optima), Keane's bump,
  Binh-Korn, Rosenbrock-disk
- **Composite/modern**: `happycat`, `katsuura`, `vincent`,
  Xin-She-Yang, expanded Griewank-Rosenbrock, Gramacy-Lee,
  Forrester
- **Catalog hygiene**: registry metadata, bounds helpers,
  documented aliases/near-duplicates (`step`/`de_jong_step2`,
  `tablet`/`discus`)
- **Validation**: fixed-dimension arity asserts, literature-formula
  fixes, proptest property tests, criterion eval benches,
  `plot-functions` binary

### [`math-qa`](crates/math-qa/README.md) — new

Wolfram Engine cross-validation harness for math-audio.

- **Oracle pairs**: each case is a closed-form `wolfram/*.wls`
  script plus a Rust comparison test (`tests/wolfram_*.rs`);
  references resolve live when `WOLFRAMSCRIPT` is set, else from
  checked-in `wolfram/goldens/`
- **Coverage**: biquad/SVF responses, Kautz correction, FFT peak,
  Schroeder/T30, M1 comb first-dip, M2 third-octave SPL, M4
  waterfall spots, M5 Morlet spots, capture TDOA/array (C1–C6),
  optimisation test functions
- **Multi-rate oracles**: capture cases emit per-rate blocks
  (6–96 kHz)
- **Harness**: `rel_error` metrics, `QA_RESULT:` emitter, hermetic
  unit tests, `validation-manifest.toml` case list with tiers and
  tolerances

### Key dependency flow

```text
math-dsp → math-iir-fir
math-rir → math-iir-fir
math-optimisation → math-test-functions
math-analog → math-dsp
math-autodiff → math-dsp, math-iir-fir
math-qa → math-dsp, math-iir-fir, math-rir, math-test-functions
```

## Repository

<https://github.com/pierreaubert/math-audio>

## License

ISC

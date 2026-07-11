# Design: FLAMO-like Differentiable DSP in `math-autodiff`

## 1. Goals and scope

`math-autodiff` already provides a frequency-domain differentiable audio DSP framework in Rust with FFT/iFFT, anti-aliasing, RBJ biquads, gain/magnitude modules, `Series`/`Shell` composition, MSE loss, and SGD. This design extends the crate toward parity with FLAMO (Dal Santo et al., ICASSP 2025) by adding the three missing primitives that unlock feedback systems such as FDNs and reverberators:

1. **Frequency-domain delays** — integer and fractional delays as phase rotation.
2. **Structured matrix gains** — orthogonal/unitary mixing matrices for stable feedback.
3. **Recursion** — closed-loop MIMO composition via direct matrix inverse.

The document also sketches the next two phases (extended filter types and a trainer/dataset layer) so the roadmap is visible, but they are out of scope for the immediate implementation plan.

## 2. Non-goals

- GPU acceleration / CUDA support.
- Time-domain state-space or sample-by-sample recurrence.
- Full neural-hypernetwork conditioning (external parameters passed at forward time).
- Audio file I/O, plotting, or checkpointing inside `math-autodiff` itself.

## 3. Section 1: Delays, Matrix Gains, and Recursion (priority) — Approach A

**Approach A:** implement the new frequency-domain primitives directly inside `crates/math-autodiff`, extending the existing `DiffModule`/`Series`/`Shell` infrastructure.

### 3.1 Requirements

- `Delay` and `ParallelDelay` modules must be differentiable w.r.t. the delay length in samples.
- Integer delays must produce exact phase rotations; fractional delays must be exact in the frequency-domain representation (i.e. `exp(-jωτ)` for arbitrary `τ`).
- `Matrix` must support a learnable orthogonal/unitary parameterization suitable for FDN feedback loops.
- `Recursion` must compose a feedforward path `H_ff` and a feedback path `H_fb` into the closed-loop transfer matrix `(I - H_fb)^-1 @ H_ff` at every frequency bin.
- All new modules must implement `DiffModule<f64>` and plug into `Series` and `Shell` without special casing.
- Gradients must be validated against finite differences.

### 3.2 Architecture

#### 3.2.1 Delay

A delay of `τ` samples has frequency response

```text
H_delay[f] = exp(-j * 2π * f * τ / nfft)
```

where `f = 0 … nfft/2` for the real-FFT positive-frequency bins.

Two module shapes are provided:

- `Delay` — MIMO matrix delay. Parameters have shape `(n_out, n_in)`; each input→output path has its own delay.
- `ParallelDelay` — per-channel diagonal delay. Parameters have shape `(n_channels,)`.

The raw parameter is unbounded. It is mapped to a positive delay in samples through a softplus plus a small minimum:

```text
tau_samples = softplus(raw) + tau_min
```

`tau_min` avoids the singular zero-delay case and can default to `0.0` for full expressivity or a small value (e.g. `1e-3`) for numerical safety.

Forward pass:

```text
output[b, f, o] = sum_i H_delay[f, o, i] * input[b, f, i]
```

Backward pass:

- Parameter gradient: `dL/dτ = Re{ sum_{b,f} (dL/doutput)[b,f] * conj(input[b,f]) * conj(dH/dτ) }`.
- Input gradient: `dL/dinput[b,f] = conj(H[f]) * dL/doutput[b,f]`.

`dH/dτ = -j * (2πf/nfft) * H[f]`.

#### 3.2.2 Matrix

A `Matrix` module multiplies the input channel vector by a complex frequency-independent matrix at each frequency bin:

```text
output[b, f, o] = sum_i M[o, i] * input[b, f, i]
```

Two parameterizations:

- `MatrixType::Dense` — raw `(n_out, n_in)` real parameters, no constraints.
- `MatrixType::Orthogonal` — square `N×N` orthogonal matrix parameterized as the matrix exponential of a skew-symmetric matrix:

```text
M = exp(S - S^T)
```

where `S` is a learnable `(N, N)` real matrix. Gradients flow through the matrix-exponential Jacobian using a first-order approximation or a Padé-based `matrix_exp` derivative. For small `N` (FDN matrices are typically 4–16) a dense finite-difference or Frechet-derivative helper is acceptable.

A `ParallelMatrix` (diagonal scaling per channel) can reuse `ParallelGain`; it is not a new module.

#### 3.2.3 Recursion

`Recursion` implements a single-input, single-output closed loop:

```text
y = H_ff @ x + H_fb @ y
=> y = (I - H_fb)^-1 @ H_ff @ x
```

At each frequency bin `f` we solve an `N×N` linear system:

```text
H_closed[f] = (I - H_fb[f])^-1 @ H_ff[f]
```

`H_ff[f]` has shape `(N_out, N_in)`; `H_fb[f]` has shape `(N_out, N_out)` (or `(N_in, N_out)` depending on convention). We require square feedback paths (`N_out == N_in`) so that `I - H_fb` is invertible.

Forward pass:

```text
output[b, f, o] = sum_i H_closed[f, o, i] * input[b, f, i]
```

Backward pass:

The module itself has no parameters, so it only needs to propagate `dL/dinput`. Because `H_closed` depends on the forward outputs of `H_ff` and `H_fb`, we store the intermediate transfer matrices from the forward pass and compute:

```text
dL/dH_closed[f] = sum_b (dL/doutput)[b,f] * conj(input[b,f])^T
dL/dH_ff[f]    = (I - H_fb[f])^-H @ dL/dH_closed[f]
dL/dH_fb[f]    = (I - H_fb[f])^-H @ dL/dH_closed[f] @ H_ff[f]^H @ (I - H_fb[f])^-H
```

where `^H` denotes the conjugate transpose.

These gradients are then passed as `grad_output` to the `H_ff` and `H_fb` submodules via their own `backward` methods. The submodule structure mirrors `Shell`: `Recursion` owns `Box<dyn DiffModule<f64>>` for feedforward and feedback.

### 3.3 API sketch

```rust
pub mod delay {
    pub struct Delay { /* ... */ }
    pub struct ParallelDelay { /* ... */ }

    impl Delay {
        pub fn new(nfft: usize, n_out: usize, n_in: usize, tau_min: f64) -> Result<Self, AutodiffError>;
    }
    impl ParallelDelay {
        pub fn new(nfft: usize, n_channels: usize, tau_min: f64) -> Result<Self, AutodiffError>;
    }
}

pub mod matrix {
    pub enum MatrixType { Dense, Orthogonal }
    pub struct Matrix { /* ... */ }

    impl Matrix {
        pub fn new(nfft: usize, n_out: usize, n_in: usize, matrix_type: MatrixType) -> Result<Self, AutodiffError>;
    }
}

pub mod recursion {
    pub struct Recursion {
        feedforward: Box<dyn DiffModule<f64>>,
        feedback: Box<dyn DiffModule<f64>>,
    }

    impl Recursion {
        pub fn new(
            feedforward: Box<dyn DiffModule<f64>>,
            feedback: Box<dyn DiffModule<f64>>,
        ) -> Result<Self, AutodiffError>;
    }
}
```

### 3.4 Data flow

A minimal FDN built with the new modules:

```text
input_time  → FFT → input_gain(N×1) → Recursion(
                                          fF = ParallelDelay(N),
                                          fB = Series(Matrix(N×N orthogonal), ParallelGain(N))
                                      ) → output_gain(1×N) → iFFTAntiAlias → output_time
```

All internal signals are `(batch, n_bins, channels)` complex tensors. The anti-aliased iFFT undoes the exponential modulation introduced by modules that use `alias_decay_db`.

### 3.5 Error handling

All new modules return `AutodiffError` on:

- `nfft == 0`
- Incompatible channel dimensions.
- Non-square feedback path for `Recursion`.
- Singular `I - H_fb` (reported as a numerical error with the bin index).
- Orthogonal matrix requested with non-square shape.

### 3.6 Testing strategy

- Unit tests:
  - `Delay` forward matches a manually computed phase rotation.
  - `ParallelDelay` gradient matches finite differences for fractional delays.
  - `Matrix::Dense` forward/backward matches hand-derived MIMO gain.
  - `Matrix::Orthogonal` preserves `M @ M^T ≈ I` after SGD updates.
  - `Recursion` forward matches `(I - H_fb)^-1 @ H_ff` computed independently.
  - `Recursion` gradient matches finite differences on a tiny stable feedback loop.
- Integration tests:
  - A stable one-pole feedback loop recovers the known closed-form response.
  - An FDN-style `Series` built from delays + matrix + gains produces finite output for an impulse input.
- Example:
  - `crates/math-autodiff/examples/fdn_match.rs`: match a target RIR using the FDN topology above.

## 4. Section 2: Extended filter types — Approach B

**Approach B:** reuse the existing filter implementations and frequency-response utilities in `math-iir-fir` and `math-dsp` rather than re-implementing coefficient equations from scratch inside `math-autodiff`.

### 4.1 Goals and scope

Extend the differentiable filter palette to match FLAMO’s broader examples:

1. **`SOSFilter`** — generic, learnable cascade of second-order sections parameterized directly by `b`/`a` coefficients.
2. **`SVF`** — State Variable Filter parameterized by physical controls (`fc`, `R`, optional gain) and mapped to an equivalent SOS representation for frequency-domain evaluation.
3. **`GEQ`** — Graphic EQ: a fixed bank of peaking filters at ISO octave center frequencies with learnable per-band gains.
4. **`PEQ`** — Parametric EQ: a cascade of arbitrary peaking/shelving sections, each with learnable frequency, Q, and gain.

All four modules implement `DiffModule<f64>`, operate on `(batch, n_bins, channels)` complex spectra, and reuse the existing `sos_frequency_response` engine and coefficient-gradient pattern from `Biquad`.

### 4.2 Non-goals

- Sample-level time-domain processing (the modules stay frequency-domain).
- New filter topologies beyond cascaded SOS (e.g., lattice, wave digital).
- Automatic filter-structure optimization (e.g., choosing minimum biquad count).

### 4.3 Architecture

#### 4.3.1 `SOSFilter`

The foundational module. Parameters are the raw numerator and denominator coefficients of `K` cascaded SOS sections:

```text
param shape: (K, 3, N_out, N_in) for b coefficients
param shape: (K, 3, N_out, N_in) for a coefficients
```

Forward:

```text
H[f] = prod_k B_k[f] / A_k[f]
output[b, f, o] = sum_i H[f, o, i] * input[b, f, i]
```

`SOSFilter` calls `sos_frequency_response` directly and stores the analytical Jacobian `dH/db` and `dH/da` already computed by `sos_response`. The backward pass is identical in structure to the existing `Biquad` backward but without the extra physical-parameter chain rule.

Backward:

```text
dL/db_k[t, o, i] = Re{ sum_f (dL/doutput)[b, f, o] * conj(input[b, f, i]) * dH/db_k[t, f, o, i] }
dL/da_k[t, o, i] = Re{ sum_f (dL/doutput)[b, f, o] * conj(input[b, f, i]) * dH/da_k[t, f, o, i] }
dL/dinput[b, f, i] = sum_o conj(H[f, o, i]) * (dL/doutput)[b, f, o]
```

#### 4.3.2 `SVF`

`SVF` is a thin wrapper around `SOSFilter`. It exposes physical parameters (`fc`, `R`, and optionally `gain_db`) and maps them to the equivalent biquad coefficients using the same RBJ-style formulas already present in `math-iir-fir::SvfFilter`. The frequency response of a ZDF SVF is identical to its equivalent direct-form biquad, so reusing `sos_frequency_response` is exact.

Parameters per section:

```text
fc_raw  -> fc  in (0, fs/2) via sigmoid
r_raw   -> R   in (0.01, max_R) via softplus + offset
gain_raw -> gain_db in [-60, 60] (for peak/shelf types)
```

`SVF` builds the SOS coefficients and their parameter derivatives, then delegates to the same frequency-response/Jacobian engine as `SOSFilter`.

#### 4.3.3 `GEQ`

A `GEQ` is a cascade of fixed-frequency peaking filters. Frequencies are taken from the ISO octave center set already available in `math-iir-fir` / `math-dsp`. Only the per-band gain is learnable.

Parameters:

```text
param shape: (N_bands, N_channels)
```

Forward builds each band's peaking biquad at its fixed frequency and Q, scales the numerator by the learned gain, and cascades them via `sos_frequency_response`. Backward uses the stored SOS Jacobians w.r.t. the gain parameter.

#### 4.3.4 `PEQ`

A `PEQ` is a cascade of arbitrary peaking or shelving sections, each with learnable frequency, Q, and gain. It generalizes `GEQ` by making frequency and Q learnable.

Parameters per section:

```text
fc_raw  -> fc  in (0, fs/2)
q_raw   -> Q   in (Q_min, Q_max) via softplus + offset
gain_raw -> gain_db in [-60, 60]
```

Section type (`Peak`, `Lowshelf`, `Highshelf`) is chosen at construction time. `PEQ` maps raw physical parameters to b/a coefficients, computes `db_dparam` and `da_dparam` exactly as the existing `Biquad` module does, and reuses `sos_frequency_response_jacobian`.

### 4.4 API sketch

```rust
pub mod iir {
    pub mod sos {
        pub struct SosFilter { /* ... */ }
        pub struct SvFilter { /* ... */ }
        pub struct GraphicEq { /* ... */ }
        pub struct ParametricEq { /* ... */ }

        pub enum SvfType { Lowpass, Highpass, Bandpass, Notch, Peak, Lowshelf, Highshelf, Allpass }
        pub enum PeqBandType { Peak, Lowshelf, Highshelf }
    }
}
```

### 4.5 Data flow

All four modules follow the same pattern as `Biquad`:

```text
raw param -> physical param -> b/a coefficients -> sos_frequency_response -> H(f) + dH/db + dH/da
```

The difference is only in how the raw parameters map to coefficients:
- `SOSFilter`: raw parameters ARE the coefficients.
- `SVF`: physical `fc`/`R`/`gain` mapped to coefficients.
- `GEQ`: only gain is learned; frequency/Q fixed.
- `PEQ`: full physical parameterization per section.

### 4.6 Error handling

All modules return `AutodiffError` for:

- `nfft == 0`
- Incompatible channel dimensions.
- Invalid parameter counts/shapes.
- Frequency/Q out of the allowed physical range after mapping.

### 4.7 Testing strategy

- Unit tests:
  - `SOSFilter` forward matches a hand-computed single-section response.
  - `SOSFilter` gradient matches finite differences w.r.t. `b` and `a`.
  - `SVF` frequency response matches the equivalent `math-iir-fir::SvfFilter` response.
  - `SVF` gradient matches finite differences.
  - `GEQ` response matches the cascade of fixed biquads.
  - `GEQ` gradient matches finite differences w.r.t. per-band gain.
  - `PEQ` gradient matches finite differences w.r.t. frequency, Q, and gain.
- Integration tests:
  - A `Series(SVF, PEQ, GEQ)` chain has finite output and correct finite-difference gradients.
- Example:
  - `crates/math-autodiff/examples/peq_match.rs`: match a target magnitude response by optimizing a `PEQ`.

### 4.8 Dependencies

No new external crates. Reuse:

- `math-iir-fir::BiquadFilterType`, `math-iir-fir::SvfFilter`, `math-iir-fir` peaking/shelving coefficient helpers.
- `math-autodiff::iir::response::{sos_frequency_response, sos_frequency_response_jacobian}`.
- Existing `Biquad` coefficient-gradient helpers where applicable.

## 5. Section 3: Trainer / dataset layer (roadmap) — Approach A

**Approach A:** build a small, optional trainer/dataset harness directly inside `crates/math-autodiff` on top of the existing `DiffModule` and `Sgd` primitives.

Replace the hand-rolled SGD loop with a small training harness:

- `Dataset` — holds `(input, target)` tensors and supports `expand`/shuffle/split semantics similar to FLAMO.
- `Trainer` — owns the model, an optimizer (start with Adam-like momentum on top of `Sgd`), learning-rate schedule, early stopping, and per-epoch train/validation loops.
- `EagerTrainer` — optimizes a single fixed `(input, target)` pair, which is the common case for LTI system identification.

This layer stays optional: `DiffModule` + `Sgd` remain usable without it.

## 6. Dependencies

- New Rust code stays inside `crates/math-autodiff`.
- Existing workspace dependencies (`ndarray`, `num-complex`, `rustfft`, `realfft`, `math-iir-fir`) are sufficient.
- A dense matrix exponential helper may be added as a private utility; no new external crate is required for `N ≤ 16`.
- `criterion` will be added as a dev-dependency for the benchmark requested in the integration QA.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `Recursion` matrix inverse becomes singular | Validate that feedback path is contractive; check condition number and return a clear error. |
| Orthogonal matrix gradients are expensive | Limit to small `N`; use first-order matrix-exp derivative; document the `N ≤ 32` recommendation. |
| Frequency-domain delay is periodic/circular | Document that `nfft` must exceed the maximum delay; add a validation helper. |
| Numerical drift in anti-aliased loops | Reuse the existing `gamma` envelope convention and test round-trip identity. |

## 8. Success criteria

- `cargo test -p math-autodiff --release` passes with all new tests.
- `cargo clippy -p math-autodiff --tests --examples --benches -- -D warnings` is clean.
- New examples (`fdn_match.rs` for Section 1, `peq_match.rs` for Section 2) run and reduce loss over optimization.
- Design doc and updated README are committed.

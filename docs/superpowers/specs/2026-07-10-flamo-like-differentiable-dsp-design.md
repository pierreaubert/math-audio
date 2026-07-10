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

- Parameter gradient: `dL/dτ = Re{ sum_{b,f} (dL/doutput)[b,f] * conj(input[b,f]) * dH/dτ }`.
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
dL/dH_ff[f]    = (I - H_fb[f])^-T @ dL/dH_closed[f]
dL/dH_fb[f]    = (I - H_fb[f])^-T @ dL/dH_closed[f] @ H_ff[f]^T @ (I - H_fb[f])^-T
```

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

## 4. Section 2: Extended filter types (roadmap) — Approach B

**Approach B:** reuse the existing filter implementations and frequency-response utilities in `math-iir-fir` and `math-dsp` rather than re-implementing coefficient equations from scratch inside `math-autodiff`.

Once Section 1 is merged, extend the filter palette to match FLAMO’s broader examples:

- `SVF` (State Variable Filter) parameterized by `fc` and `R`, mapping to biquad coefficients.
- `GEQ` (Graphic EQ) — cascade of peaking filters at ISO center frequencies.
- `PEQ` (Parametric EQ) — arbitrary peaking/shelving sections.
- `SOSFilter` — generic second-order sections with arbitrary coefficient parameterization.

These reuse the existing `sos_frequency_response` engine and the coefficient-gradient pattern already present in `Biquad`.

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

- `cargo test -p math-autodiff --release` passes with new delay, matrix, and recursion tests.
- `cargo clippy -p math-autodiff --tests --examples --benches -- -D warnings` is clean.
- New `examples/fdn_match.rs` runs and reduces loss over optimization.
- Design doc and updated README are committed.

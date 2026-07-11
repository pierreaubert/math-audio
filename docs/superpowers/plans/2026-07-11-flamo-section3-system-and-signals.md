# FLAMO-like Differentiable DSP — Section 3: System Composition & Signal Generation

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the missing system-composition and signal-generation pieces that FLAMO provides on top of the filter modules from Section 2. This enables richer end-to-end examples (SVF/GEQ matching, direct-path FDN) without yet building a full training framework.

**Architecture:** All new modules live under `crates/math-autodiff/src/`, implement `DiffModule<f64>` where applicable, and are re-exported from `crates/math-autodiff/src/lib.rs`. `Parallel` composes two branches with identical channel shapes. `signals` generates real-valued time-domain signals that can be fed into `Fft`/`Shell`.

**Tech Stack:** Rust 1.92, ndarray, num-complex, rustfft, rand (workspace). Strict clippy enabled at workspace level.

---

## Global Constraints

- Rust edition 2024, toolchain 1.92.0.
- All new public modules live under `crates/math-autodiff/src/` and are re-exported from `crates/math-autodiff/src/lib.rs`.
- `cargo clippy -p math-autodiff --tests --examples --benches -- -D warnings` must pass.
- New parameter-bearing modules must implement `DiffModule<f64>` and plug into `Series`/`Shell`.
- Gradients for parameter-bearing modules must be checked against finite differences where applicable.
- Commit messages follow repo style: `feat(autodiff): ...`, `test(autodiff): ...`, `docs(autodiff): ...`.

---

## Task 1: `Parallel` system combiner

**Files:**
- Modify: `crates/math-autodiff/src/system.rs`
- Test: `crates/math-autodiff/tests/system_tests.rs`

**Interface:**

```rust
pub struct Parallel {
    branch_a: Box<dyn DiffModule<f64>>,
    branch_b: Box<dyn DiffModule<f64>>,
}

impl Parallel {
    pub fn new(
        branch_a: Box<dyn DiffModule<f64>>,
        branch_b: Box<dyn DiffModule<f64>>,
    ) -> Result<Self, AutodiffError>;
}
```

**Behavior:**
- Both branches must have the same `n_bins`, `input_channels`, and `output_channels`.
- `forward(input) = branch_a.forward(input) + branch_b.forward(input)` (element-wise).
- `backward` recomputes forward intermediates and backpropagates the upstream gradient through both branches, summing their input gradients.
- `parameters`, `parameters_mut`, `gradients`, `zero_grad` delegate to both branches.

**Tests:**
- Two `Gain` branches in parallel produce the sum of their individual responses.
- Finite-difference gradient check on a `Parallel(Gain, Gain)` system.

**Commit:** `feat(autodiff): add Parallel system combiner`

---

## Task 2: `signals` module

**Files:**
- Create: `crates/math-autodiff/src/signals.rs`
- Modify: `crates/math-autodiff/src/lib.rs`
- Test: `crates/math-autodiff/tests/signal_tests.rs`
- Modify: `crates/math-autodiff/Cargo.toml` (add `rand = { workspace = true }` to `[dependencies]`)

**Interface:**

```rust
pub enum SignalType {
    Impulse,
    Sine { freq_hz: f64 },
    Sweep { f0_hz: f64, f1_hz: f64 },
    WhiteNoise,
    ExpDecay { rate: f64 },
    VelvetNoise { density: f64 },
}

pub fn signal_gallery(
    signal_type: SignalType,
    n_samples: usize,
    n_channels: usize,
    fs: f64,
) -> DiffTensor<f64>;
```

**Behavior:**
- Returns a real-valued `DiffTensor<f64>` of shape `(1, n_samples, n_channels)`.
- `Impulse`: first sample is 1.0, rest 0.0.
- `Sine`: `sin(2π * freq_hz * n / fs)`.
- `Sweep`: linear chirp from `f0_hz` to `f1_hz` using `scipy.signal.chirp` equivalent (`sin(2π * (f0 + (f1-f0) * t / (2*T)) * t)`).
- `WhiteNoise`: independent uniform `[-1, 1)` samples using `rand`.
- `ExpDecay`: `exp(-rate * n / fs)`.
- `VelvetNoise`: Poisson-spaced impulses with random signs at average spacing `fs/density`, first impulse at sample 0, clamped to bounds.

**Tests:**
- `Impulse` has energy only at sample 0.
- `Sine` peak amplitude is 1.0.
- `ExpDecay` first sample is 1.0 and decays monotonically.
- `VelvetNoise` has expected number of impulses within tolerance.

**Commit:** `feat(autodiff): add signal generation module`

---

## Task 3: `svf_match` example

**Files:**
- Create: `crates/math-autodiff/examples/svf_match.rs`

**Interface:**
- Model: `input_time → Fft → SvFilter(Peak) → Magnitude → output_magnitude`.
- Target: `SvFilter` with known `fc`, `R`, `gain_db`.
- Optimizable: another `SvFilter` initialized to different parameters.
- Run SGD for ~100 epochs and print decreasing loss.

**Commit:** `feat(autodiff): add SVF magnitude-matching example`

---

## Task 4: `geq_match` example

**Files:**
- Create: `crates/math-autodiff/examples/geq_match.rs`

**Interface:**
- Model: `input_time → Fft → GraphicEq(n_bands, n_channels) → Magnitude → output_magnitude`.
- Target: `GraphicEq` with known per-band gains.
- Optimizable: another `GraphicEq` initialized to zero/flat gains.
- Run SGD for ~100 epochs and print decreasing loss.

**Commit:** `feat(autodiff): add GEQ magnitude-matching example`

---

## Task 5: `fdn_direct_match` example

**Files:**
- Create: `crates/math-autodiff/examples/fdn_direct_match.rs`

**Interface:**
- Demonstrates `Parallel` by building an FDN with a direct path:
  - Branch A: `input_gain → Recursion(delays, feedback) → output_gain`.
  - Branch B: `direct_gain`.
  - Combined with `Parallel`.
- Target: known FDN + direct path parameters.
- Optimizable: same structure with different initialization.
- Run SGD and print decreasing loss.

**Commit:** `feat(autodiff): add FDN direct-path matching example`

---

## Task 6: README and Integration QA

**Files:**
- Modify: `crates/math-autodiff/README.md`

**Changes:**
- Add `system::Parallel` and `signals` rows to the modules table.
- Add `svf_match`, `geq_match`, and `fdn_direct_match` run commands.

**QA commands:**
```bash
cargo fmt -p math-autodiff
cargo clippy -p math-autodiff --tests --examples --benches -- -D warnings
cargo test -p math-autodiff --release
cargo run --release --example svf_match -p math-autodiff
cargo run --release --example geq_match -p math-autodiff
cargo run --release --example fdn_direct_match -p math-autodiff
```

**Commit:** `docs(autodiff): document Section 3 system and signal modules`

---

## Spec Coverage

| Spec Requirement | Plan Task |
|---|---|
| `Parallel` combiner with gradient propagation | Task 1 |
| `signals` gallery (impulse, sine, sweep, noise, exp, velvet) | Task 2 |
| SVF magnitude-matching example | Task 3 |
| GEQ magnitude-matching example | Task 4 |
| FDN direct-path example using `Parallel` | Task 5 |
| README and integration QA | Task 6 |

## Placeholder Scan

- No `TBD` or `TODO` remain.
- Every task ends with a concrete commit command.

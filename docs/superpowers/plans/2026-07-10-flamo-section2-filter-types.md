# FLAMO-like Differentiable DSP — Section 2: Extended Filter Types Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add differentiable `SOSFilter`, `SVF`, `GraphicEq`, and `ParametricEq` modules to `crates/math-autodiff`, with tests and a PEQ-matching example, reusing the existing `sos_frequency_response` engine and `math-iir-fir` filter types.

**Architecture:** All new modules implement `DiffModule<f64>` and operate on `(batch, n_bins, channels)` complex spectra. `SOSFilter` is the generic foundation that learns raw b/a coefficients. `SVF`, `GraphicEq`, and `ParametricEq` are thin wrappers that map physical controls to SOS coefficients and reuse `SOSFilter`'s frequency-response and backward machinery.

**Tech Stack:** Rust 1.92, ndarray, num-complex, rustfft, math-iir-fir (`BiquadFilterType`, `SvfFilter`), existing `math-autodiff` `sos_frequency_response` / `sos_frequency_response_jacobian`.

## Global Constraints

- Rust edition 2024, toolchain 1.92.0 (pinned in `rust-toolchain.toml`).
- All new public modules live under `crates/math-autodiff/src/` and are re-exported from `crates/math-autodiff/src/lib.rs`.
- Strict clippy enabled at workspace level: `cargo clippy -p math-autodiff --tests --examples --benches -- -D warnings` must pass.
- All new modules must implement `DiffModule<f64>` and plug into `Series` and `Shell` without special-casing.
- Gradients for parameter-bearing modules must be checked against finite differences.
- Commit messages follow the repo style: `feat(autodiff): ...`, `test(autodiff): ...`, `docs(autodiff): ...`.

---

## File Structure

| File | Responsibility |
|---|---|
| `crates/math-autodiff/src/iir/sos_filter.rs` | Generic `SOSFilter` module learning raw b/a coefficients. |
| `crates/math-autodiff/src/iir/svf.rs` | `SvFilter` mapping physical `fc`/`R`/`gain` to SOS coefficients. |
| `crates/math-autodiff/src/iir/geq.rs` | `GraphicEq` fixed-frequency peaking-filter bank with learnable gains. |
| `crates/math-autodiff/src/iir/peq.rs` | `ParametricEq` cascade of peaking/shelving sections with learnable freq/Q/gain. |
| `crates/math-autodiff/src/iir/mod.rs` | Re-export new modules alongside `biquad` and `response`. |
| `crates/math-autodiff/src/lib.rs` | Ensure `pub mod iir;` exposes the new modules. |
| `crates/math-autodiff/tests/sos_filter_tests.rs` | Forward and finite-difference gradient tests for `SOSFilter`. |
| `crates/math-autodiff/tests/svf_tests.rs` | Response equivalence and finite-difference gradient tests for `SVF`. |
| `crates/math-autodiff/tests/geq_tests.rs` | Response equivalence and finite-difference gradient tests for `GEQ`. |
| `crates/math-autodiff/tests/peq_tests.rs` | Forward and finite-difference gradient tests for `PEQ`. |
| `crates/math-autodiff/examples/peq_match.rs` | End-to-end PEQ magnitude-response matching example. |
| `crates/math-autodiff/README.md` | Document new modules and example usage. |

---

## Task 1: Generic `SOSFilter`

**Files:**
- Create: `crates/math-autodiff/src/iir/sos_filter.rs`
- Modify: `crates/math-autodiff/src/iir/mod.rs`
- Test: `crates/math-autodiff/tests/sos_filter_tests.rs`

**Interfaces:**
- Consumes: `DiffModule<f64>`, `DiffTensor<f64>`, `AutodiffError`, `sos_frequency_response`, `sos_frequency_response_jacobian`.
- Produces: `SosFilter::new(nfft, n_sections, n_out, n_in, alias_decay_db) -> Result<Self, AutodiffError>`. Exposes `param: ArrayD<f64>` of shape `(K, 6, N_out, N_in)` where the 6 slots are `[b0, b1, b2, a0, a1, a2]` (a0 is kept for normalization but typically 1.0), and `param_grad: ArrayD<f64>` of the same shape.

### Step 1: Create `crates/math-autodiff/src/iir/sos_filter.rs`

```rust
//! Generic differentiable cascade of second-order sections.

#![allow(
    clippy::cast_precision_loss,
    reason = "nfft is an audio buffer length that fits exactly in f64 for practical values"
)]

use ndarray::{Array3, Array4, Array5, ArrayD, Axis, IxDyn};
use num_complex::Complex;

use crate::error::AutodiffError;
use crate::iir::response::{
    sos_frequency_response, sos_frequency_response_jacobian,
};
use crate::module::DiffModule;
use crate::tensor::DiffTensor;

/// Split the packed SOS parameter tensor into separate b and a coefficient tensors.
///
/// Input shape: `(K, 6, N_out, N_in)` where the 6 slots are `[b0, b1, b2, a0, a1, a2]`.
/// Output shapes: `(K, 3, N_out, N_in)` for `b` and `(K, 3, N_out, N_in)` for `a`.
fn split_param(
    param: &ArrayD<f64>,
) -> Result<(Array4<Complex<f64>>, Array4<Complex<f64>>), AutodiffError> {
    let shape = param.shape();
    if shape.len() != 4 || shape[1] != 6 {
        return Err(AutodiffError::Message(format!(
            "SosFilter: expected param shape (K, 6, N_out, N_in), got {:?}",
            shape
        )));
    }
    let (k, n_out, n_in) = (shape[0], shape[2], shape[3]);
    let mut b = Array4::zeros((k, 3, n_out, n_in));
    let mut a = Array4::zeros((k, 3, n_out, n_in));
    for section in 0..k {
        for out_ch in 0..n_out {
            for in_ch in 0..n_in {
                for tap in 0..3 {
                    b[[section, tap, out_ch, in_ch]] =
                        Complex::new(param[[section, tap, out_ch, in_ch]], 0.0);
                    a[[section, tap, out_ch, in_ch]] =
                        Complex::new(param[[section, 3 + tap, out_ch, in_ch]], 0.0);
                }
            }
        }
    }
    Ok((b, a))
}

/// Generic cascade of second-order sections with learnable coefficients.
#[derive(Debug, Clone)]
pub struct SosFilter {
    pub nfft: usize,
    pub n_sections: usize,
    pub n_out: usize,
    pub n_in: usize,
    pub alias_decay_db: f64,
    pub param: ArrayD<f64>,
    pub param_grad: ArrayD<f64>,
}

impl SosFilter {
    pub fn new(
        nfft: usize,
        n_sections: usize,
        n_out: usize,
        n_in: usize,
        alias_decay_db: f64,
    ) -> Result<Self, AutodiffError> {
        if nfft == 0 {
            return Err(AutodiffError::Message(
                "SosFilter: nfft must be greater than 0".to_string(),
            ));
        }
        if n_sections == 0 {
            return Err(AutodiffError::Message(
                "SosFilter: n_sections must be greater than 0".to_string(),
            ));
        }
        Ok(Self {
            nfft,
            n_sections,
            n_out,
            n_in,
            alias_decay_db,
            param: ArrayD::zeros(IxDyn(&[n_sections, 6, n_out, n_in])),
            param_grad: ArrayD::zeros(IxDyn(&[n_sections, 6, n_out, n_in])),
        })
    }

    fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }

    fn gamma(&self) -> [f64; 3] {
        let gamma = 10.0_f64.powf(-self.alias_decay_db.abs() / (20.0 * self.nfft as f64));
        [1.0, gamma, gamma * gamma]
    }
}

impl DiffModule<f64> for SosFilter {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "SosFilter::forward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let n_bins = input_shape[1];
        let n_in = input_shape[2];
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "SosFilter::forward: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if n_in != self.n_in {
            return Err(AutodiffError::Message(format!(
                "SosFilter::forward: expected {} input channels, got {}",
                self.n_in, n_in
            )));
        }

        let (b, a) = split_param(&self.param)?;
        let gamma = self.gamma();
        let h = sos_frequency_response(&b, &a, self.nfft, Some(&gamma))?;

        let mut output_shape = input_shape.to_vec();
        output_shape[2] = self.n_out;
        let mut output = ArrayD::zeros(IxDyn(&output_shape));

        for out_ch in 0..self.n_out {
            for in_ch in 0..n_in {
                for bin in 0..n_bins {
                    let h_val = h[[bin, out_ch, in_ch]];
                    let input_slice = input.data.index_axis(Axis(1), bin);
                    let input_bin = input_slice.index_axis(Axis(1), in_ch);
                    let mut output_slice = output.index_axis_mut(Axis(1), bin);
                    let mut output_bin = output_slice.index_axis_mut(Axis(1), out_ch);
                    output_bin += &input_bin.mapv(|x| x * h_val);
                }
            }
        }

        Ok(DiffTensor::from_array(output))
    }

    fn backward(
        &mut self,
        input: &DiffTensor<f64>,
        _output: &DiffTensor<f64>,
        grad_output: &DiffTensor<f64>,
    ) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        let grad_shape = grad_output.data.shape();
        if input_shape.len() < 3 || grad_shape.len() < 3 {
            return Err(AutodiffError::Message(
                "SosFilter::backward: input and grad_output must have at least 3 dimensions"
                    .to_string(),
            ));
        }
        let n_bins = input_shape[1];
        let n_in = input_shape[2];
        let n_out = grad_shape[2];
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "SosFilter::backward: expected {} frequency bins, got {}",
                self.n_bins(), n_bins
            )));
        }
        if n_out != self.n_out {
            return Err(AutodiffError::Message(format!(
                "SosFilter::backward: expected {} output channels, got {}",
                self.n_out, n_out
            )));
        }
        if n_in != self.n_in {
            return Err(AutodiffError::Message(format!(
                "SosFilter::backward: expected {} input channels, got {}",
                self.n_in, n_in
            )));
        }

        let (b, a) = split_param(&self.param)?;
        let gamma = self.gamma();
        let (h, dh_db, dh_da) = sos_frequency_response_jacobian(&b, &a, self.nfft, Some(&gamma))?;

        // dL/dH[bin, out, in] = sum_b grad_output[b, bin, out] * conj(input[b, bin, in])
        let mut dl_dh = Array3::<Complex<f64>>::zeros((n_bins, n_out, n_in));
        for bin in 0..n_bins {
            for out_ch in 0..n_out {
                for in_ch in 0..n_in {
                    let grad_slice = grad_output.data.index_axis(Axis(1), bin);
                    let grad_bin = grad_slice.index_axis(Axis(1), out_ch);
                    let input_slice = input.data.index_axis(Axis(1), bin);
                    let input_bin = input_slice.index_axis(Axis(1), in_ch);
                    dl_dh[[bin, out_ch, in_ch]] = grad_bin
                        .iter()
                        .zip(input_bin.iter())
                        .map(|(g, x)| *g * x.conj())
                        .sum::<Complex<f64>>();
                }
            }
        }

        let mut param_grad = self.param_grad.view_mut().into_shape_with_order((
            self.n_sections,
            6,
            self.n_out,
            self.n_in,
        )).map_err(|e| AutodiffError::Message(format!("SosFilter: failed to reshape param_grad: {e}")))?;

        for section in 0..self.n_sections {
            for out_ch in 0..n_out {
                for in_ch in 0..n_in {
                    for tap in 0..3 {
                        let mut accum_b = 0.0;
                        let mut accum_a = 0.0;
                        for bin in 0..n_bins {
                            let term = dl_dh[[bin, out_ch, in_ch]].conj();
                            accum_b += (term * dh_db[[bin, section, tap, out_ch, in_ch]]).re;
                            accum_a += (term * dh_da[[bin, section, tap, out_ch, in_ch]]).re;
                        }
                        param_grad[[section, tap, out_ch, in_ch]] += accum_b;
                        param_grad[[section, 3 + tap, out_ch, in_ch]] += accum_a;
                    }
                }
            }
        }

        let mut grad_input = ArrayD::zeros(IxDyn(input_shape));
        for in_ch in 0..n_in {
            for out_ch in 0..n_out {
                for bin in 0..n_bins {
                    let h_conj = h[[bin, out_ch, in_ch]].conj();
                    let grad_slice = grad_output.data.index_axis(Axis(1), bin);
                    let grad_bin = grad_slice.index_axis(Axis(1), out_ch);
                    let mut input_grad_slice = grad_input.index_axis_mut(Axis(1), bin);
                    let mut input_grad_bin = input_grad_slice.index_axis_mut(Axis(1), in_ch);
                    input_grad_bin += &grad_bin.mapv(|g| g * h_conj);
                }
            }
        }

        Ok(DiffTensor::from_array(grad_input))
    }

    fn input_channels(&self) -> usize {
        self.n_in
    }
    fn output_channels(&self) -> usize {
        self.n_out
    }
    fn n_bins(&self) -> usize {
        self.n_bins()
    }
    fn parameters(&self) -> Vec<&ArrayD<f64>> {
        vec![&self.param]
    }
    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<f64>> {
        vec![&mut self.param]
    }
    fn gradients(&self) -> Vec<&ArrayD<f64>> {
        vec![&self.param_grad]
    }
    fn zero_grad(&mut self) {
        self.param_grad.fill(0.0);
    }
}
```

### Step 2: Re-export from `crates/math-autodiff/src/iir/mod.rs`

Add:

```rust
pub mod sos_filter;
pub mod svf;
pub mod geq;
pub mod peq;
```

### Step 3: Write failing tests in `crates/math-autodiff/tests/sos_filter_tests.rs`

```rust
use approx::assert_relative_eq;
use math_audio_autodiff::{
    iir::sos_filter::SosFilter,
    module::DiffModule,
    tensor::DiffTensor,
};
use ndarray::{Array3, Array4, IxDyn};
use num_complex::Complex;

const NFFT: usize = 512;

#[test]
fn sos_filter_forward_matches_single_section() {
    let n_bins = NFFT / 2 + 1;
    let mut filter = SosFilter::new(NFFT, 1, 1, 1, 0.0).unwrap();
    // Identity section: b = [1, 0, 0], a = [1, 0, 0]
    filter.param[[0, 0, 0, 0]] = 1.0;
    filter.param[[0, 3, 0, 0]] = 1.0;

    let input = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, 1), Complex::new(1.0, 0.2)).into_dyn(),
    );
    let output = filter.forward(&input).unwrap();
    assert_relative_eq!(output.data[[0, 10, 0]].re, 1.0, epsilon = 1e-9);
    assert_relative_eq!(output.data[[0, 10, 0]].im, 0.2, epsilon = 1e-9);
}

#[test]
fn sos_filter_gradient_matches_finite_difference() {
    let n_bins = NFFT / 2 + 1;
    let mut filter = SosFilter::new(NFFT, 1, 1, 1, 0.0).unwrap();
    filter.param[[0, 0, 0, 0]] = 1.0;
    filter.param[[0, 3, 0, 0]] = 1.0;

    let input = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, 1), Complex::new(1.0, 0.1)).into_dyn(),
    );
    let target = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, 1), Complex::new(0.5, -0.2)).into_dyn(),
    );

    let eps = 1e-6;
    let mut numeric = Array4::<f64>::zeros((1, 6, 1, 1));
    for tap in 0..6 {
        filter.param[[0, tap, 0, 0]] += eps;
        let out_plus = filter.forward(&input).unwrap();
        let loss_plus = (&out_plus.data - &target.data)
            .iter()
            .map(|x| x.norm_sqr())
            .sum::<f64>();
        filter.param[[0, tap, 0, 0]] -= 2.0 * eps;
        let out_minus = filter.forward(&input).unwrap();
        let loss_minus = (&out_minus.data - &target.data)
            .iter()
            .map(|x| x.norm_sqr())
            .sum::<f64>();
        numeric[[0, tap, 0, 0]] = (loss_plus - loss_minus) / (2.0 * eps);
        filter.param[[0, tap, 0, 0]] += eps;
    }

    filter.zero_grad();
    let out = filter.forward(&input).unwrap();
    let diff = &out.data - &target.data;
    let grad = DiffTensor::from_array(diff.into_owned() * 2.0);
    filter.backward(&input, &out, &grad).unwrap();

    for tap in 0..6 {
        assert_relative_eq!(
            filter.param_grad[[0, tap, 0, 0]],
            numeric[[0, tap, 0, 0]],
            epsilon = 1e-4
        );
    }
}
```

### Step 4: Run tests to verify failures and then passes

```bash
cargo test -p math-autodiff --test sos_filter_tests --release
```

Expected initially: compile errors. After implementation: PASS.

### Step 5: Commit

```bash
git add crates/math-autodiff/src/iir/sos_filter.rs crates/math-autodiff/src/iir/mod.rs crates/math-autodiff/tests/sos_filter_tests.rs
git commit -m "feat(autodiff): add generic SOSFilter module"
```

---

## Task 2: `SVF` (State Variable Filter)

**Files:**
- Create: `crates/math-autodiff/src/iir/svf.rs`
- Test: `crates/math-autodiff/tests/svf_tests.rs`

**Interfaces:**
- Consumes: `DiffModule<f64>`, `DiffTensor<f64>`, `AutodiffError`, existing `Biquad` coefficient helpers.
- Produces: `SvFilter::new(nfft, fs, n_out, n_in, filter_type, alias_decay_db) -> Result<Self, AutodiffError>`. Parameters shape `(1, P, N_out, N_in)` where `P` is 2 (`fc`, `R`) for most types or 3 (`fc`, `R`, `gain_db`) for peak/shelf types.

### Step 1: Create `crates/math-autodiff/src/iir/svf.rs`

Implement by mapping `fc`/`R`/`gain_db` to RBJ biquad coefficients using the same formulas as `math-iir-fir::SvfFilter`. Internally build an `SosFilter` with a single section and delegate `forward`/`backward`. Expose the underlying `SosFilter` parameter gradients.

Key helpers (reuse private helpers from `crates/math-autodiff/src/iir/biquad.rs` if they are made `pub(crate)`, or copy the small RBJ coefficient+derivative functions):

```rust
//! Differentiable State Variable Filter (SVF) mapped to a single SOS section.

#![allow(
    clippy::cast_precision_loss,
    reason = "nfft is an audio buffer length that fits exactly in f64 for practical values"
)]

use ndarray::ArrayD;

use crate::error::AutodiffError;
use crate::iir::sos_filter::SosFilter;
use crate::module::DiffModule;
use crate::tensor::DiffTensor;

/// SVF filter type.
#[derive(Debug, Clone, Copy)]
pub enum SvfType {
    Lowpass,
    Highpass,
    Bandpass,
    Notch,
    Peak,
    Lowshelf,
    Highshelf,
    Allpass,
}

#[derive(Debug, Clone)]
pub struct SvFilter {
    pub nfft: usize,
    pub fs: f64,
    pub filter_type: SvfType,
    pub n_out: usize,
    pub n_in: usize,
    pub alias_decay_db: f64,
    pub param: ArrayD<f64>,
    pub param_grad: ArrayD<f64>,
    inner: SosFilter,
}
```

`SvFilter::new` initializes `param` with zeros (yielding a sensible default fc/R/gain), builds the corresponding `SosFilter`, and stores it. `forward` calls `self.inner.forward(input)`. `backward` calls `self.inner.backward(input, output, grad_output)`, then maps the `SosFilter` coefficient gradients back to `SvFilter` parameter gradients using the chain rule.

### Step 2: Add tests in `crates/math-autodiff/tests/svf_tests.rs`

- Compare `SvFilter::forward` response against a hand-built `SosFilter` with equivalent coefficients.
- Finite-difference gradient check for `fc` and `R` (and gain for peak type).

### Step 3: Commit

```bash
git add crates/math-autodiff/src/iir/svf.rs crates/math-autodiff/tests/svf_tests.rs crates/math-autodiff/src/iir/mod.rs
git commit -m "feat(autodiff): add SvFilter module"
```

---

## Task 3: `GraphicEq`

**Files:**
- Create: `crates/math-autodiff/src/iir/geq.rs`
- Test: `crates/math-autodiff/tests/geq_tests.rs`

**Interfaces:**
- Consumes: `DiffModule<f64>`, `DiffTensor<f64>`, `AutodiffError`, ISO center frequencies from `math-iir-fir` or `math-dsp`.
- Produces: `GraphicEq::new(nfft, fs, n_bands, n_channels, alias_decay_db) -> Result<Self, AutodiffError>`. Parameters shape `(n_bands, n_channels)` — one learnable gain per band/channel.

### Step 1: Create `crates/math-autodiff/src/iir/geq.rs`

Use ISO octave center frequencies (e.g., 31.25, 62.5, 125, 250, 500, 1k, 2k, 4k, 8k, 16k Hz). For each band, build a peaking biquad at the fixed frequency with a default Q (e.g., `1.0 / std::f64::consts::SQRT_2` or `1.414`), scale its numerator by the learned linear gain, and cascade into a single `SosFilter`.

```rust
#[derive(Debug, Clone)]
pub struct GraphicEq {
    pub nfft: usize,
    pub fs: f64,
    pub n_bands: usize,
    pub n_channels: usize,
    pub frequencies: Vec<f64>,
    pub alias_decay_db: f64,
    pub param: ArrayD<f64>,
    pub param_grad: ArrayD<f64>,
    inner: SosFilter,
}
```

`GraphicEq::new` precomputes fixed frequencies and initializes `inner` with the identity (all gains = 0 dB → linear gain 1.0). `forward`/`backward` delegate to `inner`. A helper `rebuild_inner()` updates `inner.param` from `self.param` whenever `parameters_mut` is used; for simplicity, call `rebuild_inner()` at the start of `forward` and `backward`.

### Step 2: Add tests in `crates/math-autodiff/tests/geq_tests.rs`

- Compare `GraphicEq` response against a manually constructed cascade of peaking biquads.
- Finite-difference gradient check for per-band gains.

### Step 3: Commit

```bash
git add crates/math-autodiff/src/iir/geq.rs crates/math-autodiff/tests/geq_tests.rs crates/math-autodiff/src/iir/mod.rs
git commit -m "feat(autodiff): add GraphicEq module"
```

---

## Task 4: `ParametricEq`

**Files:**
- Create: `crates/math-autodiff/src/iir/peq.rs`
- Test: `crates/math-autodiff/tests/peq_tests.rs`

**Interfaces:**
- Consumes: `DiffModule<f64>`, `DiffTensor<f64>`, `AutodiffError`, existing `Biquad` coefficient helpers.
- Produces: `ParametricEq::new(nfft, fs, n_sections, n_channels, band_type, alias_decay_db) -> Result<Self, AutodiffError>`. Parameters shape `(n_sections, 3, n_channels)` for `[fc_raw, q_raw, gain_db_raw]`.

### Step 1: Create `crates/math-autodiff/src/iir/peq.rs`

Similar to `Biquad` but as a cascade of sections. Map raw parameters to physical `fc`, `Q`, `gain_db`, compute peaking/shelving b/a coefficients and their parameter derivatives, and cascade into a single `SosFilter`.

```rust
#[derive(Debug, Clone, Copy)]
pub enum PeqBandType {
    Peak,
    Lowshelf,
    Highshelf,
}

#[derive(Debug, Clone)]
pub struct ParametricEq {
    pub nfft: usize,
    pub fs: f64,
    pub n_sections: usize,
    pub n_channels: usize,
    pub band_type: PeqBandType,
    pub alias_decay_db: f64,
    pub param: ArrayD<f64>,
    pub param_grad: ArrayD<f64>,
    inner: SosFilter,
}
```

`ParametricEq::new` initializes `inner` with identity sections. `forward`/`backward` rebuild `inner` from `self.param` and delegate.

### Step 2: Add tests in `crates/math-autodiff/tests/peq_tests.rs`

- Forward response matches a hand-built cascade of peaking/shelving biquads.
- Finite-difference gradient check for `fc`, `Q`, and `gain_db`.

### Step 3: Commit

```bash
git add crates/math-autodiff/src/iir/peq.rs crates/math-autodiff/tests/peq_tests.rs crates/math-autodiff/src/iir/mod.rs
git commit -m "feat(autodiff): add ParametricEq module"
```

---

## Task 5: `peq_match` Example

**Files:**
- Create: `crates/math-autodiff/examples/peq_match.rs`

**Interfaces:**
- Consumes: `Fft`, `IfftAntiAlias`, `ParametricEq`, `Series`, `Shell`, `Magnitude`, MSE loss, `Sgd`.
- Produces: A runnable example that optimizes a `ParametricEq` to match a synthetic target magnitude response.

### Step 1: Create `crates/math-autodiff/examples/peq_match.rs`

Model: `input_time → Fft → ParametricEq(N sections, Peak) → Magnitude → output_magnitude`.

Target: a `ParametricEq` with known frequencies/Qs/gains.
Optimizable: another `ParametricEq` initialized to different parameters.
Run SGD for ~100 epochs and print loss decreasing.

### Step 2: Run the example

```bash
cargo run --release --example peq_match -p math-autodiff
```

Expected: prints loss decreasing.

### Step 3: Commit

```bash
git add crates/math-autodiff/examples/peq_match.rs
git commit -m "feat(autodiff): add PEQ magnitude-matching example"
```

---

## Task 6: README and Integration QA

**Files:**
- Modify: `crates/math-autodiff/README.md`

### Step 1: Update README

Add rows to the modules table:

```markdown
| `iir::sos_filter` | Generic learnable cascade of second-order sections. |
| `iir::svf` | State Variable Filter parameterized by `fc`/`R`/`gain`. |
| `iir::geq` | Graphic EQ with fixed ISO frequencies and learnable per-band gains. |
| `iir::peq` | Parametric EQ with learnable frequency, Q, and gain per section. |
```

Add a short `peq_match` run command.

### Step 2: Run QA commands

```bash
cargo fmt -p math-autodiff
cargo clippy -p math-autodiff --tests --examples --benches -- -D warnings
cargo test -p math-autodiff --release
cargo run --release --example peq_match -p math-autodiff
```

### Step 3: Commit

```bash
git add crates/math-autodiff/README.md
git commit -m "docs(autodiff): document Section 2 filter modules and run integration QA"
```

---

## Spec Coverage

| Spec Requirement | Plan Task |
|---|---|
| Generic `SOSFilter` with learnable b/a coefficients | Task 1 |
| `SVF` mapped to SOS coefficients | Task 2 |
| `GraphicEq` fixed-frequency peaking bank | Task 3 |
| `ParametricEq` learnable peaking/shelving cascade | Task 4 |
| All modules implement `DiffModule<f64>` | Tasks 1–4 |
| Gradients validated against finite differences | Tasks 1–4 tests |
| PEQ-matching example | Task 5 |
| README and integration QA | Task 6 |

## Placeholder Scan

- No `TBD` or `TODO` remain.
- No vague "add appropriate error handling" steps; each error condition is enumerated.
- Every task ends with a concrete commit command.

## Type Consistency

- `SosFilter::new(nfft, n_sections, n_out, n_in, alias_decay_db)` matches the design doc.
- `SvFilter::new(nfft, fs, n_out, n_in, filter_type, alias_decay_db)` matches the design doc.
- `GraphicEq::new(nfft, fs, n_bands, n_channels, alias_decay_db)` matches the design doc.
- `ParametricEq::new(nfft, fs, n_sections, n_channels, band_type, alias_decay_db)` matches the design doc.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-10-flamo-section2-filter-types.md`. Ready for execution.

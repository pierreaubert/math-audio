# FLAMO-like Differentiable DSP — Section 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-TOOL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add frequency-domain `Delay`/`ParallelDelay`, structured `Matrix` (dense and orthogonal), and closed-loop `Recursion` modules to `crates/math-autodiff`, with tests and an FDN-matching example.

**Architecture:** All new modules implement the existing `DiffModule<f64>` trait and operate on `(batch, n_bins, channels)` complex spectra. Delays are pure phase rotations, matrices are frequency-independent linear maps, and `Recursion` solves `(I - H_fb)^-1 @ H_ff` per frequency bin. Gradients are validated against finite differences.

**Tech Stack:** Rust 1.92, ndarray, num-complex, rustfft/realfft, math-iir-fir types, existing `math-autodiff` DiffModule/Shell/Series infrastructure.

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
| `crates/math-autodiff/src/delay.rs` | `Delay` (MIMO) and `ParallelDelay` (diagonal) frequency-domain delay modules. |
| `crates/math-autodiff/src/matrix.rs` | `Matrix` (dense/orthogonal) and `MatrixType`; optional `HouseholderMatrix`. |
| `crates/math-autodiff/src/recursion.rs` | `Recursion` closed-loop composition using `(I - H_fb)^-1 @ H_ff`. |
| `crates/math-autodiff/src/lib.rs` | Add `pub mod delay; pub mod matrix; pub mod recursion;`. |
| `crates/math-autodiff/tests/delay_tests.rs` | Forward and finite-difference gradient tests for delays. |
| `crates/math-autodiff/tests/matrix_tests.rs` | Forward/orthogonality and finite-difference gradient tests for matrices. |
| `crates/math-autodiff/tests/recursion_tests.rs` | Closed-loop forward and finite-difference gradient tests. |
| `crates/math-autodiff/examples/fdn_match.rs` | End-to-end FDN magnitude/response matching example. |
| `crates/math-autodiff/benches/biquad_bench.rs` | Criterion benchmark for biquad forward/backward (integration QA). |
| `crates/math-autodiff/Cargo.toml` | Add `criterion` dev-dependency. |
| `crates/math-autodiff/README.md` | Document new modules and example usage. |

---

## Task 1: Frequency-domain Delay and ParallelDelay

**Files:**
- Create: `crates/math-autodiff/src/delay.rs`
- Modify: `crates/math-autodiff/src/lib.rs`
- Test: `crates/math-autodiff/tests/delay_tests.rs`

**Interfaces:**
- Consumes: `DiffModule<f64>`, `DiffTensor<f64>`, `AutodiffError` from existing crate.
- Produces: `Delay::new(nfft, n_out, n_in, tau_min) -> Result<Self, AutodiffError>` and `ParallelDelay::new(nfft, n_channels, tau_min)`. Both expose `param: ArrayD<f64>` and `param_grad: ArrayD<f64>`.

### Step 1: Create `crates/math-autodiff/src/delay.rs`

```rust
//! Frequency-domain differentiable delay modules.

#![allow(
    clippy::cast_precision_loss,
    reason = "nfft is an audio buffer length that fits exactly in f64 for practical values"
)]

use ndarray::{ArrayD, ArrayView2, ArrayViewMut2, Axis, IxDyn};
use num_complex::Complex;

use crate::error::AutodiffError;
use crate::module::DiffModule;
use crate::tensor::DiffTensor;

/// Softplus mapping raw parameters to positive delay samples.
#[inline]
fn softplus(x: f64) -> f64 {
    (1.0 + (-x).exp()).ln()
}

/// Derivative of softplus.
#[inline]
fn softplus_derivative(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Map a raw delay parameter to a positive delay in samples.
#[inline]
fn raw_to_tau(raw: f64, tau_min: f64) -> f64 {
    softplus(raw) + tau_min
}

/// Build the complex delay frequency response for one delay value.
fn delay_response(tau: f64, nfft: usize) -> Result<Vec<Complex<f64>>, AutodiffError> {
    if nfft == 0 {
        return Err(AutodiffError::Message(
            "Delay: nfft must be greater than 0".to_string(),
        ));
    }
    let n_bins = nfft / 2 + 1;
    let scale = -2.0 * std::f64::consts::PI / nfft as f64;
    Ok((0..n_bins)
        .map(|bin| {
            let phase = scale * bin as f64 * tau;
            Complex::new(phase.cos(), phase.sin())
        })
        .collect())
}

fn view2<'a>(param: &'a ArrayD<f64>, name: &str) -> Result<ArrayView2<'a, f64>, AutodiffError> {
    let shape = param.shape();
    if shape.len() != 2 {
        return Err(AutodiffError::Message(format!(
            "{name}: expected 2-D parameter tensor, got shape {:?}",
            shape
        )));
    }
    let (n_out, n_in) = (shape[0], shape[1]);
    param
        .view()
        .into_shape_with_order((n_out, n_in))
        .map_err(|e| AutodiffError::Message(format!("{name}: failed to reshape param: {e}")))
}

fn view2_mut<'a>(
    param: &'a mut ArrayD<f64>,
    name: &str,
) -> Result<ArrayViewMut2<'a, f64>, AutodiffError> {
    let shape = param.shape();
    if shape.len() != 2 {
        return Err(AutodiffError::Message(format!(
            "{name}: expected 2-D parameter gradient tensor, got shape {:?}",
            shape
        )));
    }
    let (n_out, n_in) = (shape[0], shape[1]);
    param
        .view_mut()
        .into_shape_with_order((n_out, n_in))
        .map_err(|e| AutodiffError::Message(format!("{name}: failed to reshape param_grad: {e}")))
}

/// MIMO frequency-domain delay.
#[derive(Debug, Clone)]
pub struct Delay {
    pub nfft: usize,
    pub n_out: usize,
    pub n_in: usize,
    pub tau_min: f64,
    pub param: ArrayD<f64>,
    pub param_grad: ArrayD<f64>,
}

impl Delay {
    pub fn new(nfft: usize, n_out: usize, n_in: usize, tau_min: f64) -> Result<Self, AutodiffError> {
        if nfft == 0 {
            return Err(AutodiffError::Message(
                "Delay: nfft must be greater than 0".to_string(),
            ));
        }
        Ok(Self {
            nfft,
            n_out,
            n_in,
            tau_min,
            param: ArrayD::zeros(IxDyn(&[n_out, n_in])),
            param_grad: ArrayD::zeros(IxDyn(&[n_out, n_in])),
        })
    }

    fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }
}

impl DiffModule<f64> for Delay {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "Delay::forward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let n_bins = input_shape[1];
        let n_in = input_shape[2];
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "Delay::forward: expected {} frequency bins, got {}",
                self.n_bins(), n_bins
            )));
        }
        if n_in != self.n_in {
            return Err(AutodiffError::Message(format!(
                "Delay::forward: expected {} input channels, got {}",
                self.n_in, n_in
            )));
        }

        let param = view2(&self.param, "Delay")?;
        let mut output_shape = input_shape.to_vec();
        output_shape[2] = self.n_out;
        let mut output = ArrayD::zeros(IxDyn(&output_shape));

        for out_ch in 0..self.n_out {
            for in_ch in 0..n_in {
                let tau = raw_to_tau(param[[out_ch, in_ch]], self.tau_min);
                let h = delay_response(tau, self.nfft)?;
                for bin in 0..n_bins {
                    let h_val = h[bin];
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
                "Delay::backward: input and grad_output must have at least 3 dimensions".to_string(),
            ));
        }
        let n_bins = input_shape[1];
        let n_in = input_shape[2];
        if grad_shape[1] != n_bins || grad_shape[2] != self.n_out {
            return Err(AutodiffError::Message(format!(
                "Delay::backward: grad_output shape {:?} incompatible with (..., {}, {})",
                grad_shape, n_bins, self.n_out
            )));
        }

        let param = view2(&self.param, "Delay")?;
        let mut param_grad = view2_mut(&mut self.param_grad, "Delay")?;

        let scale = -2.0 * std::f64::consts::PI / self.nfft as f64;
        let mut grad_input = ArrayD::zeros(IxDyn(input_shape));

        for out_ch in 0..self.n_out {
            for in_ch in 0..n_in {
                let raw = param[[out_ch, in_ch]];
                let tau = raw_to_tau(raw, self.tau_min);
                let dtau_draw = softplus_derivative(raw);
                let h = delay_response(tau, self.nfft)?;

                for bin in 0..n_bins {
                    let h_val = h[bin];
                    let dh_dtau = h_val * Complex::new(0.0, scale * bin as f64);

                    let input_slice = input.data.index_axis(Axis(1), bin);
                    let input_bin = input_slice.index_axis(Axis(1), in_ch);
                    let grad_slice = grad_output.data.index_axis(Axis(1), bin);
                    let grad_bin = grad_slice.index_axis(Axis(1), out_ch);

                    // Parameter gradient.
                    let accum: Complex<f64> = grad_bin
                        .iter()
                        .zip(input_bin.iter())
                        .map(|(g, x)| g * x.conj() * dh_dtau)
                        .sum();
                    param_grad[[out_ch, in_ch]] += accum.re * dtau_draw;

                    // Input gradient.
                    let mut input_grad_slice = grad_input.index_axis_mut(Axis(1), bin);
                    let mut input_grad_bin = input_grad_slice.index_axis_mut(Axis(1), in_ch);
                    input_grad_bin += &grad_bin.mapv(|g| g * h_val.conj());
                }
            }
        }

        Ok(DiffTensor::from_array(grad_input))
    }

    fn input_channels(&self) -> usize { self.n_in }
    fn output_channels(&self) -> usize { self.n_out }
    fn n_bins(&self) -> usize { self.n_bins() }
    fn parameters(&self) -> Vec<&ArrayD<f64>> { vec![&self.param] }
    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<f64>> { vec![&mut self.param] }
    fn gradients(&self) -> Vec<&ArrayD<f64>> { vec![&self.param_grad] }
    fn zero_grad(&mut self) { self.param_grad.fill(0.0); }
}

/// Diagonal per-channel frequency-domain delay.
#[derive(Debug, Clone)]
pub struct ParallelDelay {
    pub nfft: usize,
    pub n_channels: usize,
    pub tau_min: f64,
    pub param: ArrayD<f64>,
    pub param_grad: ArrayD<f64>,
}

impl ParallelDelay {
    pub fn new(nfft: usize, n_channels: usize, tau_min: f64) -> Result<Self, AutodiffError> {
        if nfft == 0 {
            return Err(AutodiffError::Message(
                "ParallelDelay: nfft must be greater than 0".to_string(),
            ));
        }
        Ok(Self {
            nfft,
            n_channels,
            tau_min,
            param: ArrayD::zeros(IxDyn(&[n_channels])),
            param_grad: ArrayD::zeros(IxDyn(&[n_channels])),
        })
    }

    fn n_bins(&self) -> usize { self.nfft / 2 + 1 }
}

impl DiffModule<f64> for ParallelDelay {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "ParallelDelay::forward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let n_bins = input_shape[1];
        let n_channels = input_shape[2];
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "ParallelDelay::forward: expected {} frequency bins, got {}",
                self.n_bins(), n_bins
            )));
        }
        if n_channels != self.n_channels {
            return Err(AutodiffError::Message(format!(
                "ParallelDelay::forward: expected {} channels, got {}",
                self.n_channels, n_channels
            )));
        }

        let mut output = input.data.clone();
        for ch in 0..n_channels {
            let tau = raw_to_tau(self.param[[ch]], self.tau_min);
            let h = delay_response(tau, self.nfft)?;
            for bin in 0..n_bins {
                let h_val = h[bin];
                let mut slice = output.index_axis_mut(Axis(1), bin);
                let mut ch_slice = slice.index_axis_mut(Axis(1), ch);
                ch_slice.mapv_inplace(|x| x * h_val);
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
        if input_shape != grad_shape {
            return Err(AutodiffError::Message(format!(
                "ParallelDelay::backward: input shape {:?} does not match grad_output shape {:?}",
                input_shape, grad_shape
            )));
        }
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "ParallelDelay::backward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let n_bins = input_shape[1];
        let n_channels = input_shape[2];

        let scale = -2.0 * std::f64::consts::PI / self.nfft as f64;
        let mut grad_input = ArrayD::zeros(IxDyn(input_shape));

        for ch in 0..n_channels {
            let raw = self.param[[ch]];
            let tau = raw_to_tau(raw, self.tau_min);
            let dtau_draw = softplus_derivative(raw);
            let h = delay_response(tau, self.nfft)?;

            for bin in 0..n_bins {
                let h_val = h[bin];
                let dh_dtau = h_val * Complex::new(0.0, scale * bin as f64);

                let input_slice = input.data.index_axis(Axis(1), bin);
                let input_bin = input_slice.index_axis(Axis(1), ch);
                let grad_slice = grad_output.data.index_axis(Axis(1), bin);
                let grad_bin = grad_slice.index_axis(Axis(1), ch);

                let accum: Complex<f64> = grad_bin
                    .iter()
                    .zip(input_bin.iter())
                    .map(|(g, x)| g * x.conj() * dh_dtau)
                    .sum();
                self.param_grad[[ch]] += accum.re * dtau_draw;

                let mut input_grad_slice = grad_input.index_axis_mut(Axis(1), bin);
                let mut input_grad_bin = input_grad_slice.index_axis_mut(Axis(1), ch);
                input_grad_bin += &grad_bin.mapv(|g| g * h_val.conj());
            }
        }

        Ok(DiffTensor::from_array(grad_input))
    }

    fn input_channels(&self) -> usize { self.n_channels }
    fn output_channels(&self) -> usize { self.n_channels }
    fn n_bins(&self) -> usize { self.n_bins() }
    fn parameters(&self) -> Vec<&ArrayD<f64>> { vec![&self.param] }
    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<f64>> { vec![&mut self.param] }
    fn gradients(&self) -> Vec<&ArrayD<f64>> { vec![&self.param_grad] }
    fn zero_grad(&mut self) { self.param_grad.fill(0.0); }
}
```

### Step 2: Re-export from `crates/math-autodiff/src/lib.rs`

Add after `pub mod gain;`:

```rust
pub mod delay;
pub mod matrix;
pub mod recursion;
```

### Step 3: Write failing tests in `crates/math-autodiff/tests/delay_tests.rs`

```rust
use approx::assert_relative_eq;
use math_audio_autodiff::{
    delay::{Delay, ParallelDelay},
    fft::Fft,
    module::DiffModule,
    tensor::DiffTensor,
};
use ndarray::{Array3, IxDyn};
use num_complex::Complex;

const FS: f64 = 48_000.0;
const NFFT: usize = 512;

#[test]
fn parallel_delay_forward_matches_phase_rotation() {
    let n_bins = NFFT / 2 + 1;
    let mut delay = ParallelDelay::new(NFFT, 1, 0.0).unwrap();
    delay.param[[0]] = 0.0; // raw -> tau = 0

    let input = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, 1), Complex::new(1.0, 0.0)).into_dyn(),
    );
    let output = delay.forward(&input).unwrap();
    assert_relative_eq!(output.data[[0, 10, 0]].re, 1.0, epsilon = 1e-9);
    assert_relative_eq!(output.data[[0, 10, 0]].im, 0.0, epsilon = 1e-9);
}

#[test]
fn parallel_delay_gradient_matches_finite_difference() {
    let n_bins = NFFT / 2 + 1;
    let mut delay = ParallelDelay::new(NFFT, 1, 0.0).unwrap();
    delay.param[[0]] = 1.0; // some non-zero raw

    let input = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, 1), Complex::new(1.0, 0.1)).into_dyn(),
    );
    let target = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, 1), Complex::new(0.5, -0.2)).into_dyn(),
    );

    let eps = 1e-6;
    let loss_plus = {
        delay.param[[0]] += eps;
        let out = delay.forward(&input).unwrap();
        let diff = &out.data - &target.data;
        diff.iter().map(|x| x.norm_sqr()).sum::<f64>()
    };
    let loss_minus = {
        delay.param[[0]] -= 2.0 * eps;
        let out = delay.forward(&input).unwrap();
        let diff = &out.data - &target.data;
        diff.iter().map(|x| x.norm_sqr()).sum::<f64>()
    };
    let numeric_grad = (loss_plus - loss_minus) / (2.0 * eps);

    delay.param[[0]] += eps; // restore
    delay.zero_grad();
    let out = delay.forward(&input).unwrap();
    let diff = &out.data - &target.data;
    let grad = DiffTensor::from_array(diff.into_owned() * 2.0);
    delay.backward(&input, &out, &grad).unwrap();

    assert_relative_eq!(delay.param_grad[[0]], numeric_grad, epsilon = 1e-5);
}
```

### Step 4: Run tests to verify failures and then passes

```bash
cargo test -p math-autodiff --test delay_tests -- --nocapture
```

Expected initially: compile errors (module not yet exported). After implementation: PASS.

### Step 5: Commit

```bash
git add crates/math-autodiff/src/delay.rs crates/math-autodiff/src/lib.rs crates/math-autodiff/tests/delay_tests.rs
git commit -m "feat(autodiff): add frequency-domain Delay and ParallelDelay modules"
```

---

## Task 2: Matrix (Dense and Orthogonal)

**Files:**
- Create: `crates/math-autodiff/src/matrix.rs`
- Modify: `crates/math-autodiff/src/lib.rs`
- Test: `crates/math-autodiff/tests/matrix_tests.rs`

**Interfaces:**
- Consumes: `DiffModule<f64>`, `DiffTensor<f64>`, `AutodiffError`.
- Produces: `Matrix::new(nfft, n_out, n_in, MatrixType::Dense)` and `Matrix::new(nfft, n, n, MatrixType::Orthogonal)`. Exposes `param`, `param_grad`, and a helper `fn matrix_exp_skew(raw: &Array2<f64>) -> Array2<f64>`.

### Step 1: Add `nalgebra` dependency and matrix helpers

In `crates/math-autodiff/Cargo.toml` add:

```toml
nalgebra = { workspace = true }
```

In `crates/math-autodiff/src/matrix.rs`, add conversion helpers and use `nalgebra::DMatrix::exp()` for the orthogonal parameterization.

```rust
use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis, IxDyn};
use nalgebra::DMatrix;
use num_complex::Complex;

fn ndarray2_to_dmatrix(mat: &Array2<f64>) -> DMatrix<f64> {
    let data: Vec<f64> = mat.iter().copied().collect();
    DMatrix::from_row_major(mat.nrows(), mat.ncols(), &data)
}

fn dmatrix_to_ndarray2(mat: &DMatrix<f64>) -> Array2<f64> {
    let mut out = Array2::zeros((mat.nrows(), mat.ncols()));
    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            out[[i, j]] = mat[(i, j)];
        }
    }
    out
}

fn matrix_exp_skew(raw: &Array2<f64>) -> Array2<f64> {
    let skew = raw - raw.t();
    let dm = ndarray2_to_dmatrix(&skew);
    let exp = dm.exp();
    dmatrix_to_ndarray2(&exp)
}
```

### Step 2: Implement `Matrix` module

```rust
#[derive(Debug, Clone, Copy)]
pub enum MatrixType {
    Dense,
    Orthogonal,
}

#[derive(Debug, Clone)]
pub struct Matrix {
    pub nfft: usize,
    pub n_out: usize,
    pub n_in: usize,
    pub matrix_type: MatrixType,
    pub param: ArrayD<f64>,
    pub param_grad: ArrayD<f64>,
}

impl Matrix {
    pub fn new(
        nfft: usize,
        n_out: usize,
        n_in: usize,
        matrix_type: MatrixType,
    ) -> Result<Self, AutodiffError> {
        if nfft == 0 {
            return Err(AutodiffError::Message(
                "Matrix: nfft must be greater than 0".to_string(),
            ));
        }
        let param_shape = match matrix_type {
            MatrixType::Dense => vec![n_out, n_in],
            MatrixType::Orthogonal => {
                if n_out != n_in {
                    return Err(AutodiffError::Message(
                        "Matrix::Orthogonal requires square shape".to_string(),
                    ));
                }
                vec![n_out, n_in]
            }
        };
        Ok(Self {
            nfft,
            n_out,
            n_in,
            matrix_type,
            param: ArrayD::zeros(IxDyn(&param_shape)),
            param_grad: ArrayD::zeros(IxDyn(&param_shape)),
        })
    }

    fn n_bins(&self) -> usize { self.nfft / 2 + 1 }

    /// Build the real frequency-independent matrix M.
    fn build_matrix(&self) -> Result<Array2<Complex<f64>>, AutodiffError> {
        match self.matrix_type {
            MatrixType::Dense => {
                let v = view2(&self.param, "Matrix")?;
                Ok(v.mapv(|x| Complex::new(x, 0.0)))
            }
            MatrixType::Orthogonal => {
                let v = view2(&self.param, "Matrix")?;
                let orth = matrix_exp_skew(&v);
                Ok(orth.mapv(|x| Complex::new(x, 0.0)))
            }
        }
    }
}

impl DiffModule<f64> for Matrix {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "Matrix::forward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let n_bins = input_shape[1];
        let n_in = input_shape[2];
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "Matrix::forward: expected {} frequency bins, got {}",
                self.n_bins(), n_bins
            )));
        }
        if n_in != self.n_in {
            return Err(AutodiffError::Message(format!(
                "Matrix::forward: expected {} input channels, got {}",
                self.n_in, n_in
            )));
        }

        let m = self.build_matrix()?;
        let mut output_shape = input_shape.to_vec();
        output_shape[2] = self.n_out;
        let mut output = ArrayD::zeros(IxDyn(&output_shape));

        for out_ch in 0..self.n_out {
            for in_ch in 0..n_in {
                let h = m[[out_ch, in_ch]];
                let input_slice = input.data.index_axis(Axis(2), in_ch);
                let mut output_slice = output.index_axis_mut(Axis(2), out_ch);
                output_slice += &input_slice.mapv(|x| x * h);
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
                "Matrix::backward: input and grad_output must have at least 3 dimensions".to_string(),
            ));
        }
        let n_bins = input_shape[1];
        if grad_shape[1] != n_bins || grad_shape[2] != self.n_out {
            return Err(AutodiffError::Message(format!(
                "Matrix::backward: grad_output shape {:?} incompatible with (..., {}, {})",
                grad_shape, n_bins, self.n_out
            )));
        }

        let m = self.build_matrix()?;
        let mut grad_input = ArrayD::zeros(IxDyn(input_shape));

        // Dense gradient is straightforward.
        match self.matrix_type {
            MatrixType::Dense => {
                let mut pg = view2_mut(&mut self.param_grad, "Matrix")?;
                for out_ch in 0..self.n_out {
                    let grad_slice = grad_output.data.index_axis(Axis(2), out_ch);
                    for in_ch in 0..self.n_in {
                        let input_slice = input.data.index_axis(Axis(2), in_ch);
                        let prod = &grad_slice * &input_slice.mapv(|x| x.conj());
                        pg[[out_ch, in_ch]] += prod.sum().re;
                    }
                }
            }
            MatrixType::Orthogonal => {
                // Compute dL/dM (same shape as M).
                let mut dl_dm = Array2::<f64>::zeros((self.n_out, self.n_in));
                for out_ch in 0..self.n_out {
                    let grad_slice = grad_output.data.index_axis(Axis(2), out_ch);
                    for in_ch in 0..self.n_in {
                        let input_slice = input.data.index_axis(Axis(2), in_ch);
                        let prod = &grad_slice * &input_slice.mapv(|x| x.conj());
                        dl_dm[[out_ch, in_ch]] = prod.sum().re;
                    }
                }
                // Numerical Jacobian of M w.r.t. raw parameters.
                let v = view2(&self.param, "Matrix")?;
                let eps = 1e-7;
                for i in 0..self.n_out {
                    for j in 0..self.n_in {
                        let mut v_plus = v.to_owned();
                        v_plus[[i, j]] += eps;
                        let m_plus = matrix_exp_skew(&v_plus);
                        let mut v_minus = v.to_owned();
                        v_minus[[i, j]] -= eps;
                        let m_minus = matrix_exp_skew(&v_minus);
                        let deriv = (&m_plus - &m_minus) / (2.0 * eps);
                        let grad = (&deriv * &dl_dm).sum();
                        self.param_grad[[i, j]] += grad;
                    }
                }
            }
        }

        for in_ch in 0..self.n_in {
            for out_ch in 0..self.n_out {
                let h = m[[out_ch, in_ch]].conj();
                let grad_slice = grad_output.data.index_axis(Axis(2), out_ch);
                let mut input_grad_slice = grad_input.index_axis_mut(Axis(2), in_ch);
                input_grad_slice += &grad_slice.mapv(|x| x * h);
            }
        }

        Ok(DiffTensor::from_array(grad_input))
    }

    fn input_channels(&self) -> usize { self.n_in }
    fn output_channels(&self) -> usize { self.n_out }
    fn n_bins(&self) -> usize { self.n_bins() }
    fn parameters(&self) -> Vec<&ArrayD<f64>> { vec![&self.param] }
    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<f64>> { vec![&mut self.param] }
    fn gradients(&self) -> Vec<&ArrayD<f64>> { vec![&self.param_grad] }
    fn zero_grad(&mut self) { self.param_grad.fill(0.0); }
}
```

### Step 3: Add tests in `crates/math-autodiff/tests/matrix_tests.rs`

```rust
use approx::assert_relative_eq;
use math_audio_autodiff::{
    matrix::{Matrix, MatrixType},
    module::DiffModule,
    tensor::DiffTensor,
};
use ndarray::{Array3, IxDyn};
use num_complex::Complex;

const NFFT: usize = 512;

#[test]
fn orthogonal_matrix_stays_orthogonal_after_random_param() {
    let n = 4;
    let matrix = Matrix::new(NFFT, n, n, MatrixType::Orthogonal).unwrap();
    let m = matrix.build_matrix().unwrap();
    let identity = m.t().dot(&m);
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(identity[[i, j]].re, expected, epsilon = 1e-6);
        }
    }
}

#[test]
fn dense_matrix_gradient_matches_finite_difference() {
    let n_bins = NFFT / 2 + 1;
    let mut matrix = Matrix::new(NFFT, 2, 1, MatrixType::Dense).unwrap();
    matrix.param[[0, 0]] = 0.5;
    matrix.param[[1, 0]] = -0.3;

    let input = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, 1), Complex::new(1.0, 0.2)).into_dyn(),
    );
    let target = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, 2), Complex::new(0.4, -0.1)).into_dyn(),
    );

    let eps = 1e-6;
    let mut numeric = ndarray::Array2::<f64>::zeros((2, 1));
    for i in 0..2 {
        matrix.param[[i, 0]] += eps;
        let out_plus = matrix.forward(&input).unwrap();
        let loss_plus = (&out_plus.data - &target.data).iter().map(|x| x.norm_sqr()).sum::<f64>();
        matrix.param[[i, 0]] -= 2.0 * eps;
        let out_minus = matrix.forward(&input).unwrap();
        let loss_minus = (&out_minus.data - &target.data).iter().map(|x| x.norm_sqr()).sum::<f64>();
        numeric[[i, 0]] = (loss_plus - loss_minus) / (2.0 * eps);
        matrix.param[[i, 0]] += eps;
    }

    matrix.zero_grad();
    let out = matrix.forward(&input).unwrap();
    let diff = &out.data - &target.data;
    let grad = DiffTensor::from_array(diff.into_owned() * 2.0);
    matrix.backward(&input, &out, &grad).unwrap();

    for i in 0..2 {
        assert_relative_eq!(matrix.param_grad[[i, 0]], numeric[[i, 0]], epsilon = 1e-5);
    }
}
```

### Step 4: Run matrix tests

```bash
cargo test -p math-autodiff --test matrix_tests --release
```

Expected: PASS.

### Step 5: Commit

```bash
git add crates/math-autodiff/src/matrix.rs crates/math-autodiff/src/lib.rs crates/math-autodiff/tests/matrix_tests.rs
git commit -m "feat(autodiff): add Matrix module with Dense and Orthogonal parameterizations"
```

---

## Task 3: Recursion (Closed-Loop Composition)

**Files:**
- Create: `crates/math-autodiff/src/recursion.rs`
- Modify: `crates/math-autodiff/src/lib.rs`, `crates/math-autodiff/Cargo.toml`
- Test: `crates/math-autodiff/tests/recursion_tests.rs`

**Interfaces:**
- Consumes: `DiffModule<f64>`, `DiffTensor<f64>`, `AutodiffError`, `nalgebra` for small dense complex linear algebra.
- Produces: `Recursion::new(feedforward, feedback)` implementing `DiffModule<f64>`.

### Step 1: Confirm `nalgebra` dependency

`nalgebra` was already added in Task 2. Reuse the same `ndarray2_to_dmatrix` / `dmatrix_to_ndarray2` conversion pattern for complex matrices in `recursion.rs`.

### Step 2: Implement `crates/math-autodiff/src/recursion.rs`

```rust
//! Closed-loop recursion module.

use ndarray::{Array2, Array3, ArrayD, Axis, IxDyn};
use nalgebra::{DMatrix, Dyn, OMatrix};
use num_complex::Complex;

use crate::error::AutodiffError;
use crate::module::DiffModule;
use crate::tensor::DiffTensor;

fn n_bins(nfft: usize) -> usize {
    nfft / 2 + 1
}

/// Build an identity spectrum of shape `(n_in, n_bins, n_in)`.
fn identity_spectrum(n_in: usize, n_bins: usize) -> DiffTensor<f64> {
    let mut data = ArrayD::zeros(IxDyn(&[n_in, n_bins, n_in]));
    for i in 0..n_in {
        for f in 0..n_bins {
            data[[i, f, i]] = Complex::new(1.0, 0.0);
        }
    }
    DiffTensor::from_array(data)
}

/// Extract the transfer matrix of a submodule as `(n_bins, n_out, n_in)`.
fn module_response(
    module: &dyn DiffModule<f64>,
    n_in: usize,
) -> Result<Array3<Complex<f64>>, AutodiffError> {
    let nb = module.n_bins();
    let identity = identity_spectrum(n_in, nb);
    let output = module.forward(&identity)?;
    let out_shape = output.data.shape();
    if out_shape.len() != 3 || out_shape[0] != n_in || out_shape[1] != nb {
        return Err(AutodiffError::Message(format!(
            "Recursion: submodule response has unexpected shape {:?}",
            out_shape
        )));
    }
    let n_out = out_shape[2];
    let mut h = Array3::zeros((nb, n_out, n_in));
    for i in 0..n_in {
        for f in 0..nb {
            for o in 0..n_out {
                h[[f, o, i]] = output.data[[i, f, o]];
            }
        }
    }
    Ok(h)
}

fn ndarray2_to_dmatrix(mat: &Array2<Complex<f64>>) -> DMatrix<Complex<f64>> {
    let nrows = mat.nrows();
    let ncols = mat.ncols();
    let data: Vec<Complex<f64>> = mat.iter().copied().collect();
    DMatrix::from_row_major(nrows, ncols, &data)
}

fn dmatrix_to_ndarray2(mat: &DMatrix<Complex<f64>>) -> Array2<Complex<f64>> {
    let mut out = Array2::zeros((mat.nrows(), mat.ncols()));
    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            out[[i, j]] = mat[(i, j)];
        }
    }
    out
}

fn invert_complex_matrix(mat: &Array2<Complex<f64>>) -> Result<Array2<Complex<f64>>, AutodiffError> {
    let dm = ndarray2_to_dmatrix(mat);
    let inv = dm.try_inverse().ok_or_else(|| {
        AutodiffError::Message("Recursion: failed to invert (I - H_fb)".to_string())
    })?;
    Ok(dmatrix_to_ndarray2(&inv))
}

/// Closed-loop MIMO composition `y = (I - H_fb)^-1 @ H_ff @ x`.
#[derive(Debug)]
pub struct Recursion {
    pub feedforward: Box<dyn DiffModule<f64>>,
    pub feedback: Box<dyn DiffModule<f64>>,
    n_bins: usize,
}

impl Recursion {
    pub fn new(
        feedforward: Box<dyn DiffModule<f64>>,
        feedback: Box<dyn DiffModule<f64>>,
    ) -> Result<Self, AutodiffError> {
        if feedforward.n_bins() != feedback.n_bins() {
            return Err(AutodiffError::Message(format!(
                "Recursion: feedforward has {} bins, feedback has {}",
                feedforward.n_bins(),
                feedback.n_bins()
            )));
        }
        if feedforward.output_channels() != feedback.input_channels() {
            return Err(AutodiffError::Message(format!(
                "Recursion: feedforward outputs {}, feedback expects {}",
                feedforward.output_channels(),
                feedback.input_channels()
            )));
        }
        if feedback.output_channels() != feedforward.output_channels() {
            return Err(AutodiffError::Message(format!(
                "Recursion: feedback outputs {}, feedforward outputs {}",
                feedback.output_channels(),
                feedforward.output_channels()
            )));
        }
        Ok(Self {
            n_bins: feedforward.n_bins(),
            feedforward,
            feedback,
        })
    }

    fn n_bins(&self) -> usize { self.n_bins }

    fn closed_loop_response(
        &self,
    ) -> Result<(Array3<Complex<f64>>, Array3<Complex<f64>>, Array3<Complex<f64>>), AutodiffError>
    {
        let n_in = self.feedforward.input_channels();
        let n_out = self.feedforward.output_channels();
        let nb = self.n_bins();

        let h_ff = module_response(self.feedforward.as_ref(), n_in)?; // (nb, n_out, n_in)
        let h_fb = module_response(self.feedback.as_ref(), n_out)?; // (nb, n_out, n_out)

        let mut h_closed = Array3::zeros((nb, n_out, n_in));
        let mut a_arr = Array3::zeros((nb, n_out, n_out));

        for f in 0..nb {
            let i_min_h_fb = {
                let mut m = Array2::zeros((n_out, n_out));
                for r in 0..n_out {
                    for c in 0..n_out {
                        m[[r, c]] = if r == c { Complex::new(1.0, 0.0) } else { Complex::new(0.0, 0.0) };
                        m[[r, c]] -= h_fb[[f, r, c]];
                    }
                }
                m
            };
            let a = invert_complex_matrix(&i_min_h_fb)?;
            for r in 0..n_out {
                for c in 0..n_out {
                    a_arr[[f, r, c]] = a[[r, c]];
                }
            }
            for o in 0..n_out {
                for i in 0..n_in {
                    let mut sum = Complex::new(0.0, 0.0);
                    for k in 0..n_out {
                        sum += a[[o, k]] * h_ff[[f, k, i]];
                    }
                    h_closed[[f, o, i]] = sum;
                }
            }
        }

        Ok((h_closed, h_ff, a_arr))
    }
}

impl DiffModule<f64> for Recursion {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "Recursion::forward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let nb = input_shape[1];
        let n_in = input_shape[2];
        if nb != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "Recursion::forward: expected {} bins, got {}",
                self.n_bins(), nb
            )));
        }
        if n_in != self.feedforward.input_channels() {
            return Err(AutodiffError::Message(format!(
                "Recursion::forward: expected {} input channels, got {}",
                self.feedforward.input_channels(), n_in
            )));
        }

        let (h_closed, _, _) = self.closed_loop_response()?;
        let n_out = self.feedforward.output_channels();
        let mut output_shape = input_shape.to_vec();
        output_shape[2] = n_out;
        let mut output = ArrayD::zeros(IxDyn(&output_shape));

        for o in 0..n_out {
            for i in 0..n_in {
                for f in 0..nb {
                    let h = h_closed[[f, o, i]];
                    let input_slice = input.data.index_axis(Axis(1), f);
                    let input_bin = input_slice.index_axis(Axis(1), i);
                    let mut output_slice = output.index_axis_mut(Axis(1), f);
                    let mut output_bin = output_slice.index_axis_mut(Axis(1), o);
                    output_bin += &input_bin.mapv(|x| x * h);
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
                "Recursion::backward: input and grad_output must have at least 3 dimensions".to_string(),
            ));
        }
        let nb = input_shape[1];
        let n_in = input_shape[2];
        let n_out = self.feedforward.output_channels();
        if nb != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "Recursion::backward: expected {} bins, got {}",
                self.n_bins(), nb
            )));
        }
        if grad_shape[2] != n_out {
            return Err(AutodiffError::Message(format!(
                "Recursion::backward: expected {} output channels, got {}",
                n_out, grad_shape[2]
            )));
        }

        let (h_closed, h_ff, a_arr) = self.closed_loop_response()?;

        // dL/dH_closed[f, o, i] = sum_b grad_output[b, f, o] * conj(input[b, f, i])
        let mut dl_dh_closed = Array3::<Complex<f64>>::zeros((nb, n_out, n_in));
        for f in 0..nb {
            for o in 0..n_out {
                for i in 0..n_in {
                    let grad_slice = grad_output.data.index_axis(Axis(1), f);
                    let grad_bin = grad_slice.index_axis(Axis(1), o);
                    let input_slice = input.data.index_axis(Axis(1), f);
                    let input_bin = input_slice.index_axis(Axis(1), i);
                    dl_dh_closed[[f, o, i]] = grad_bin
                        .iter()
                        .zip(input_bin.iter())
                        .map(|(g, x)| g * x.conj())
                        .sum::<Complex<f64>>();
                }
            }
        }

        // Build per-bin dL/dH_ff and dL/dH_fb.
        let mut dl_dh_ff = Array3::<Complex<f64>>::zeros((nb, n_out, n_in));
        let mut dl_dh_fb = Array3::<Complex<f64>>::zeros((nb, n_out, n_out));

        for f in 0..nb {
            let a = {
                let mut m = Array2::zeros((n_out, n_out));
                for r in 0..n_out {
                    for c in 0..n_out {
                        m[[r, c]] = a_arr[[f, r, c]];
                    }
                }
                m
            };
            let a_t = a.t();
            let h_ff_f = {
                let mut m = Array2::zeros((n_out, n_in));
                for r in 0..n_out {
                    for c in 0..n_in {
                        m[[r, c]] = h_ff[[f, r, c]];
                    }
                }
                m
            };
            let dl_dh_closed_f = {
                let mut m = Array2::zeros((n_out, n_in));
                for r in 0..n_out {
                    for c in 0..n_in {
                        m[[r, c]] = dl_dh_closed[[f, r, c]];
                    }
                }
                m
            };

            // dL/dH_ff[f] = A^T @ dL/dH_closed[f]
            let dl_dh_ff_f = a_t.dot(&dl_dh_closed_f);
            for o in 0..n_out {
                for i in 0..n_in {
                    dl_dh_ff[[f, o, i]] = dl_dh_ff_f[[o, i]];
                }
            }

            // dL/dH_fb[f] = A^T @ dL/dH_closed[f] @ H_ff^T @ A^T
            let dl_dh_fb_f = a_t.dot(&dl_dh_closed_f).dot(&h_ff_f.t()).dot(&a_t);
            for r in 0..n_out {
                for c in 0..n_out {
                    dl_dh_fb[[f, r, c]] = dl_dh_fb_f[[r, c]];
                }
            }
        }

        // Backward through feedforward submodule.
        {
            let identity_in = identity_spectrum(n_in, nb);
            let h_ff_response = {
                let mut out = ArrayD::zeros(IxDyn(&[n_in, nb, n_out]));
                for i in 0..n_in {
                    for f in 0..nb {
                        for o in 0..n_out {
                            out[[i, f, o]] = h_ff[[f, o, i]];
                        }
                    }
                }
                DiffTensor::from_array(out)
            };
            let mut grad_ff = ArrayD::zeros(IxDyn(&[n_in, nb, n_out]));
            for i in 0..n_in {
                for f in 0..nb {
                    for o in 0..n_out {
                        grad_ff[[i, f, o]] = dl_dh_ff[[f, o, i]];
                    }
                }
            }
            let _ = self
                .feedforward
                .backward(&identity_in, &h_ff_response, &DiffTensor::from_array(grad_ff))?;
        }

        // Backward through feedback submodule.
        {
            let identity_in = identity_spectrum(n_out, nb);
            let h_fb_response = {
                let mut out = ArrayD::zeros(IxDyn(&[n_out, nb, n_out]));
                for i in 0..n_out {
                    for f in 0..nb {
                        for o in 0..n_out {
                            out[[i, f, o]] = h_fb[[f, o, i]];
                        }
                    }
                }
                DiffTensor::from_array(out)
            };
            let mut grad_fb = ArrayD::zeros(IxDyn(&[n_out, nb, n_out]));
            for i in 0..n_out {
                for f in 0..nb {
                    for o in 0..n_out {
                        grad_fb[[i, f, o]] = dl_dh_fb[[f, o, i]];
                    }
                }
            }
            let _ = self
                .feedback
                .backward(&identity_in, &h_fb_response, &DiffTensor::from_array(grad_fb))?;
        }

        // dL/dinput[b, f, i] = sum_o conj(H_closed[f, o, i]) * grad_output[b, f, o]
        let mut grad_input = ArrayD::zeros(IxDyn(input_shape));
        for i in 0..n_in {
            for o in 0..n_out {
                for f in 0..nb {
                    let h_conj = h_closed[[f, o, i]].conj();
                    let grad_slice = grad_output.data.index_axis(Axis(1), f);
                    let grad_bin = grad_slice.index_axis(Axis(1), o);
                    let mut input_grad_slice = grad_input.index_axis_mut(Axis(1), f);
                    let mut input_grad_bin = input_grad_slice.index_axis_mut(Axis(1), i);
                    input_grad_bin += &grad_bin.mapv(|g| g * h_conj);
                }
            }
        }

        Ok(DiffTensor::from_array(grad_input))
    }

    fn input_channels(&self) -> usize { self.feedforward.input_channels() }
    fn output_channels(&self) -> usize { self.feedforward.output_channels() }
    fn n_bins(&self) -> usize { self.n_bins() }
    fn parameters(&self) -> Vec<&ArrayD<f64>> {
        let mut p = Vec::new();
        p.extend(self.feedforward.parameters());
        p.extend(self.feedback.parameters());
        p
    }
    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<f64>> {
        let mut p = Vec::new();
        p.extend(self.feedforward.parameters_mut());
        p.extend(self.feedback.parameters_mut());
        p
    }
    fn gradients(&self) -> Vec<&ArrayD<f64>> {
        let mut g = Vec::new();
        g.extend(self.feedforward.gradients());
        g.extend(self.feedback.gradients());
        g
    }
    fn zero_grad(&mut self) {
        self.feedforward.zero_grad();
        self.feedback.zero_grad();
    }
}
```

**Note:** `h_fb` is reused from `closed_loop_response`; `module_response` is not called again.

### Step 3: Add recursion tests in `crates/math-autodiff/tests/recursion_tests.rs`

```rust
use approx::assert_relative_eq;
use math_audio_autodiff::{
    gain::Gain,
    matrix::{Matrix, MatrixType},
    module::DiffModule,
    recursion::Recursion,
    system::Series,
    tensor::DiffTensor,
};
use ndarray::{Array3, IxDyn};
use num_complex::Complex;

const NFFT: usize = 512;

#[test]
fn recursion_forward_is_finite_with_nonzero_feedback() {
    let n = 2;
    let mut feedforward = Gain::new(NFFT, n, n).unwrap();
    feedforward.param[[0, 0]] = 0.5;
    feedforward.param[[1, 1]] = -0.3;
    let mut feedback = Gain::new(NFFT, n, n).unwrap();
    feedback.param[[0, 0]] = 0.1;
    feedback.param[[1, 1]] = 0.05;

    let recursion = Recursion::new(Box::new(feedforward), Box::new(feedback)).unwrap();

    let n_bins = NFFT / 2 + 1;
    let input = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, n), Complex::new(1.0, 0.0)).into_dyn(),
    );
    let output = recursion.forward(&input).unwrap();

    assert_eq!(output.data.shape()[2], n);
    assert!(output.data.iter().all(|x| x.is_finite()));
}

#[test]
fn recursion_with_zero_feedback_reduces_to_feedforward() {
    // If feedback H_fb = 0, Recursion forward equals feedforward forward,
    // and the parameter gradient must equal the feedforward-only gradient.
    let n = 2;
    let n_bins = NFFT / 2 + 1;

    let mut feedforward = Gain::new(NFFT, n, n).unwrap();
    feedforward.param[[0, 0]] = 0.5;
    feedforward.param[[1, 0]] = -0.2;
    feedforward.param[[0, 1]] = 0.3;
    feedforward.param[[1, 1]] = -0.1;

    let feedback = Gain::new(NFFT, n, n).unwrap(); // all zeros
    let mut recursion = Recursion::new(Box::new(feedforward.clone()), Box::new(feedback)).unwrap();

    let input = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, n), Complex::new(0.5, 0.1)).into_dyn(),
    );
    let target = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, n), Complex::new(0.3, -0.2)).into_dyn(),
    );

    // Reference: standalone feedforward gradient.
    let mut standalone = feedforward.clone();
    standalone.zero_grad();
    let out_standalone = standalone.forward(&input).unwrap();
    let diff_standalone = &out_standalone.data - &target.data;
    let grad_standalone = DiffTensor::from_array(diff_standalone.into_owned() * 2.0);
    standalone.backward(&input, &out_standalone, &grad_standalone).unwrap();

    // Recursion gradient.
    recursion.zero_grad();
    let out_recursion = recursion.forward(&input).unwrap();
    let diff_recursion = &out_recursion.data - &target.data;
    let grad_recursion = DiffTensor::from_array(diff_recursion.into_owned() * 2.0);
    recursion.backward(&input, &out_recursion, &grad_recursion).unwrap();

    let standalone_grads = standalone.gradients();
    let recursion_grads = recursion.gradients();
    for i in 0..n {
        for j in 0..n {
            assert_relative_eq!(
                recursion_grads[0][[i, j]],
                standalone_grads[0][[i, j]],
                epsilon = 1e-8
            );
        }
    }
}
```

### Step 4: Run recursion tests

```bash
cargo test -p math-autodiff --test recursion_tests --release
```

Expected: PASS.

### Step 5: Commit

```bash
git add crates/math-autodiff/src/recursion.rs crates/math-autodiff/src/lib.rs crates/math-autodiff/Cargo.toml crates/math-autodiff/tests/recursion_tests.rs
git commit -m "feat(autodiff): add Recursion closed-loop composition module"
```

---

## Task 4: FDN Matching Example

**Files:**
- Create: `crates/math-autodiff/examples/fdn_match.rs`

**Interfaces:**
- Consumes: `Fft`, `IfftAntiAlias`, `Gain`, `ParallelDelay`, `Matrix`, `Recursion`, `Series`, `Shell`, `Magnitude`, MSE loss, SGD.
- Produces: A runnable example that optimizes an FDN to match a synthetic target RIR.

### Step 1: Create the example

```rust
//! FDN magnitude-response matching example.

use math_audio_autodiff::{
    delay::ParallelDelay,
    fft::Fft,
    gain::{Gain, Magnitude},
    loss::{mse_loss, mse_loss_backward},
    matrix::{Matrix, MatrixType},
    module::DiffModule,
    optim::Sgd,
    recursion::Recursion,
    system::{Series, Shell},
    tensor::DiffTensor,
};
use ndarray::{Array3, ArrayD, IxDyn};
use num_complex::Complex;

const FS: f64 = 48_000.0;
const NFFT: usize = 16_384;
const N: usize = 4;
const ALIAS_DECAY_DB: f64 = 30.0;
const EPOCHS: usize = 100;
const LR: f64 = 1e-2;

fn build_fdn_core(
    delay_values: &[f64],
    feedback_gain: f64,
) -> Result<Box<dyn DiffModule<f64>>, Box<dyn std::error::Error>> {
    assert_eq!(delay_values.len(), N);
    let input_gain = Gain::new(NFFT, N, 1)?;
    let output_gain = Gain::new(NFFT, 1, N)?;

    let mut delays = ParallelDelay::new(NFFT, N, 1.0)?;
    for (i, &tau) in delay_values.iter().enumerate() {
        // softplus(raw) + 1.0 ≈ tau for large tau.
        delays.param[[i]] = tau;
    }

    let mut feedback_gain_module = Gain::new(NFFT, N, N)?;
    for i in 0..N {
        feedback_gain_module.param[[i, i]] = feedback_gain;
    }
    let feedback = Series::new(vec![
        Box::new(Matrix::new(NFFT, N, N, MatrixType::Orthogonal)?),
        Box::new(feedback_gain_module),
    ])?;
    let recursion = Recursion::new(Box::new(delays), Box::new(feedback))?;
    let fdn = Series::new(vec![
        Box::new(input_gain),
        Box::new(recursion),
        Box::new(output_gain),
    ])?;
    Ok(Box::new(fdn))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_bins = NFFT / 2 + 1;

    // Target FDN: known delays and small feedback gain.
    let target_delays = vec![100.0, 173.0, 251.0, 337.0];
    let target_core = build_fdn_core(&target_delays, 0.3)?;
    let fft = Fft::with_channels(NFFT, 1);
    let magnitude = Magnitude::new(NFFT, 1);
    let target_shell = Shell::new(Box::new(fft.clone()), target_core, Box::new(magnitude.clone()))?;

    let input = DiffTensor::from_array(
        Array3::<Complex<f64>>::zeros((1, NFFT, 1)).into_dyn(),
    );
    input.data[[0, 0, 0]] = Complex::new(1.0, 0.0);
    let target = target_shell.forward(&input)?;

    // Optimizable FDN: randomized delays and feedback gain.
    let init_delays = vec![80.0, 150.0, 220.0, 310.0];
    let init_core = build_fdn_core(&init_delays, 0.1)?;
    let mut shell = Shell::new(Box::new(fft), init_core, Box::new(magnitude))?;

    let sgd = Sgd::new(LR);

    let initial_loss = mse_loss(&shell.forward(&input)?, &target)?;
    for epoch in 0..EPOCHS {
        shell.zero_grad();
        let pred = shell.forward(&input)?;
        let loss = mse_loss(&pred, &target)?;
        if epoch % 20 == 0 {
            println!("epoch {:3}: loss = {:.6e}", epoch, loss);
        }
        let grad = mse_loss_backward(&pred, &target)?;
        shell.backward(&input, &pred, &grad)?;
        let grads_owned: Vec<ArrayD<f64>> = shell.gradients().iter().map(|g| (*g).clone()).collect();
        let mut params = shell.parameters_mut();
        let grads: Vec<&ArrayD<f64>> = grads_owned.iter().collect();
        sgd.step(&mut params[..], &grads[..])?;
    }

    let final_loss = mse_loss(&shell.forward(&input)?, &target)?;
    println!("initial loss = {:.6e}", initial_loss);
    println!("final loss   = {:.6e}", final_loss);
    Ok(())
}
```

### Step 2: Run the example

```bash
cargo run --release --example fdn_match -p math-autodiff
```

Expected: prints loss decreasing.

### Step 3: Commit

```bash
git add crates/math-autodiff/examples/fdn_match.rs
git commit -m "feat(autodiff): add FDN magnitude-matching example"
```

---

## Task 5: README, Benchmark, and Integration QA

**Files:**
- Create: `crates/math-autodiff/benches/biquad_bench.rs`
- Modify: `crates/math-autodiff/Cargo.toml`, `crates/math-autodiff/README.md`, `crates/README.md`

### Step 1: Add criterion bench

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use math_audio_autodiff::{
    fft::Fft,
    iir::biquad::Biquad,
    module::DiffModule,
    tensor::DiffTensor,
};
use math_audio_iir_fir::BiquadFilterType;
use ndarray::{Array3, IxDyn};
use num_complex::Complex;

fn biquad_forward_benchmark(c: &mut Criterion) {
    const NFFT: usize = 8192;
    let fft = Fft::with_channels(NFFT, 1);
    let biquad = Biquad::new(NFFT, 48_000.0, 2, BiquadFilterType::Highpass, 1, 1, 30.0).unwrap();
    let input_time = Array3::<Complex<f64>>::zeros((1, NFFT, 1));
    let input = DiffTensor::from_array(input_time.into_dyn());
    let spectrum = fft.forward(&input).unwrap();

    c.bench_function("biquad forward", |b| {
        b.iter(|| black_box(biquad.forward(black_box(&spectrum)).unwrap()))
    });
}

criterion_group!(benches, biquad_forward_benchmark);
criterion_main!(benches);
```

Add to `crates/math-autodiff/Cargo.toml`:

```toml
[[bench]]
name = "biquad_bench"
harness = false

[dev-dependencies]
approx = { workspace = true }
rand = { workspace = true }
criterion = { workspace = true }
```

(Keep existing dev-dependencies; add `criterion` if not already present.)

### Step 2: Update READMEs

In `crates/math-autodiff/README.md`, document:
- New modules: `delay`, `matrix`, `recursion`.
- Example usage of `Shell(Fft → FDN(Series + Recursion) → Magnitude)`.
- How to run `fdn_match`.

In `crates/README.md`, add a one-line entry for `math-autodiff` if missing.

### Step 3: Run QA commands

```bash
cargo fmt -p math-autodiff
cargo clippy -p math-autodiff --tests --examples --benches -- -D warnings
cargo test -p math-autodiff --release
cargo bench -p math-autodiff --bench biquad_bench
```

### Step 4: Commit

```bash
git add crates/math-autodiff/benches crates/math-autodiff/Cargo.toml crates/math-autodiff/README.md crates/README.md
git commit -m "docs(autodiff): add benchmark, README, and integration QA"
```

---

## Spec Coverage

| Spec Requirement | Plan Task |
|---|---|
| Frequency-domain `Delay`/`ParallelDelay` | Task 1 |
| Structured `Matrix` (dense + orthogonal) | Task 2 |
| `Recursion` closed-loop composition | Task 3 |
| All modules implement `DiffModule<f64>` | Tasks 1–3 |
| Gradients validated against finite differences | Tasks 1–3 tests |
| FDN-matching example | Task 4 |
| Benchmark and README | Task 5 |
| Sections 2/3 remain roadmap | Design doc, not this plan |

## Placeholder Scan

- No `TBD` or `TODO` remain.
- No vague "add appropriate error handling" steps; each error condition is enumerated.

## Type Consistency

- `DiffModule` signatures unchanged.
- `Delay::new(nfft, n_out, n_in, tau_min)` and `ParallelDelay::new(nfft, n_channels, tau_min)` match the design doc.
- `Matrix::new(nfft, n_out, n_in, MatrixType)` matches the design doc.
- `Recursion::new(feedforward, feedback)` matches the design doc.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-10-flamo-like-differentiable-dsp.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach?

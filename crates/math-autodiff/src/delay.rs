//! Frequency-domain differentiable delay modules.

#![allow(
    clippy::cast_precision_loss,
    reason = "nfft is an audio buffer length that fits exactly in f64 for practical values"
)]
#![allow(
    clippy::uninlined_format_args,
    reason = "format strings are clearer with explicit arguments in error messages"
)]

use ndarray::{ArrayD, ArrayView2, ArrayViewMut2, Axis, IxDyn};
use num_complex::Complex;

use crate::error::AutodiffError;
use crate::module::DiffModule;
use crate::tensor::DiffTensor;

/// Softplus mapping raw parameters to positive delay samples.
#[inline]
fn softplus(x: f64) -> f64 {
    (1.0 + x.exp()).ln()
}

/// Derivative of softplus.
#[inline]
fn softplus_derivative(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Map a raw delay parameter to a positive delay in samples.
///
/// `raw = 0` maps to `tau_min`, so the default initialization yields the
/// minimum (often zero) delay.
#[inline]
fn raw_to_tau(raw: f64, tau_min: f64) -> f64 {
    softplus(raw) - softplus(0.0) + tau_min
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
    /// Create a new MIMO frequency-domain delay module.
    ///
    /// # Errors
    ///
    /// Returns an error if `nfft` is zero.
    pub fn new(
        nfft: usize,
        n_out: usize,
        n_in: usize,
        tau_min: f64,
    ) -> Result<Self, AutodiffError> {
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
                self.n_bins(),
                n_bins
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
                for (bin, &h_val) in h.iter().enumerate() {
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
                "Delay::backward: input and grad_output must have at least 3 dimensions"
                    .to_string(),
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

                for (bin, &h_val) in h.iter().enumerate() {
                    let dh_dtau = h_val * Complex::new(0.0, scale * bin as f64);

                    let input_slice = input.data.index_axis(Axis(1), bin);
                    let input_bin = input_slice.index_axis(Axis(1), in_ch);
                    let grad_slice = grad_output.data.index_axis(Axis(1), bin);
                    let grad_bin = grad_slice.index_axis(Axis(1), out_ch);

                    // Parameter gradient.
                    let accum: Complex<f64> = grad_bin
                        .iter()
                        .zip(input_bin.iter())
                        .map(|(g, x)| g * x.conj() * dh_dtau.conj())
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
    /// Create a new diagonal per-channel frequency-domain delay module.
    ///
    /// # Errors
    ///
    /// Returns an error if `nfft` is zero.
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

    fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }
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
                self.n_bins(),
                n_bins
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
            for (bin, &h_val) in h.iter().enumerate() {
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
        let n_channels = input_shape[2];

        let scale = -2.0 * std::f64::consts::PI / self.nfft as f64;
        let mut grad_input = ArrayD::zeros(IxDyn(input_shape));

        for ch in 0..n_channels {
            let raw = self.param[[ch]];
            let tau = raw_to_tau(raw, self.tau_min);
            let dtau_draw = softplus_derivative(raw);
            let h = delay_response(tau, self.nfft)?;

            for (bin, &h_val) in h.iter().enumerate() {
                let dh_dtau = h_val * Complex::new(0.0, scale * bin as f64);

                let input_slice = input.data.index_axis(Axis(1), bin);
                let input_bin = input_slice.index_axis(Axis(1), ch);
                let grad_slice = grad_output.data.index_axis(Axis(1), bin);
                let grad_bin = grad_slice.index_axis(Axis(1), ch);

                let accum: Complex<f64> = grad_bin
                    .iter()
                    .zip(input_bin.iter())
                    .map(|(g, x)| g * x.conj() * dh_dtau.conj())
                    .sum();
                self.param_grad[[ch]] += accum.re * dtau_draw;

                let mut input_grad_slice = grad_input.index_axis_mut(Axis(1), bin);
                let mut input_grad_bin = input_grad_slice.index_axis_mut(Axis(1), ch);
                input_grad_bin += &grad_bin.mapv(|g| g * h_val.conj());
            }
        }

        Ok(DiffTensor::from_array(grad_input))
    }

    fn input_channels(&self) -> usize {
        self.n_channels
    }
    fn output_channels(&self) -> usize {
        self.n_channels
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

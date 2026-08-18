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
use std::{
    cell::RefCell,
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use crate::error::AutodiffError;
use crate::module::{DiffModule, validate_spectral_gradient_shape};
use crate::tensor::DiffTensor;

/// Softplus mapping raw parameters to positive delay samples.
#[inline]
fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// Derivative of softplus.
#[inline]
fn softplus_derivative(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// Map a raw delay parameter to a positive delay in samples.
///
/// `raw = 0` maps to `tau_min`, so the default initialization yields the
/// minimum (often zero) delay.
#[inline]
fn raw_to_tau(raw: f64, tau_min: f64) -> f64 {
    tau_min + (softplus(raw) - softplus(0.0)).max(0.0)
}

/// Derivative of [`raw_to_tau`]. Negative raw values intentionally remain on
/// the exact `tau_min` plateau; optimizers must cross the initialization point
/// through an explicit parameter update before delay can increase.
#[inline]
fn raw_to_tau_derivative(raw: f64) -> f64 {
    if raw < 0.0 {
        0.0
    } else {
        softplus_derivative(raw)
    }
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

#[derive(Debug, Clone)]
struct DelayResponseCache {
    hash: u64,
    shape: Vec<usize>,
    tau_min_bits: u64,
    responses: Vec<Vec<Complex<f64>>>,
}

fn with_cached_delay_responses<R>(
    cache: &RefCell<Option<DelayResponseCache>>,
    param: &ArrayD<f64>,
    nfft: usize,
    tau_min: f64,
    operation: impl FnOnce(&[Vec<Complex<f64>>]) -> R,
) -> Result<R, AutodiffError> {
    let mut hasher = DefaultHasher::new();
    param.shape().hash(&mut hasher);
    for &value in param {
        value.to_bits().hash(&mut hasher);
    }
    let hash = hasher.finish();
    let shape = param.shape().to_vec();
    let tau_min_bits = tau_min.to_bits();
    let mut cached = cache.borrow_mut();
    let stale = cached.as_ref().is_none_or(|entry| {
        entry.hash != hash
            || entry.shape != shape
            || entry.tau_min_bits != tau_min_bits
            || entry.responses.len() != param.len()
    });
    if stale {
        let responses = param
            .iter()
            .map(|&raw| delay_response(raw_to_tau(raw, tau_min), nfft))
            .collect::<Result<Vec<_>, _>>()?;
        *cached = Some(DelayResponseCache {
            hash,
            shape,
            tau_min_bits,
            responses,
        });
    }
    Ok(operation(
        &cached.as_ref().expect("cache populated").responses,
    ))
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

fn view1<'a>(
    param: &'a ArrayD<f64>,
    name: &str,
) -> Result<ndarray::ArrayView1<'a, f64>, AutodiffError> {
    if param.ndim() != 1 {
        return Err(AutodiffError::Message(format!(
            "{name}: expected 1-D parameter tensor, got shape {:?}",
            param.shape()
        )));
    }
    param
        .view()
        .into_shape_with_order(param.len())
        .map_err(|e| AutodiffError::Message(format!("{name}: failed to reshape param: {e}")))
}

fn view1_mut<'a>(
    param: &'a mut ArrayD<f64>,
    name: &str,
) -> Result<ndarray::ArrayViewMut1<'a, f64>, AutodiffError> {
    if param.ndim() != 1 {
        return Err(AutodiffError::Message(format!(
            "{name}: expected 1-D parameter gradient tensor, got shape {:?}",
            param.shape()
        )));
    }
    let len = param.len();
    param
        .view_mut()
        .into_shape_with_order(len)
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
    response_cache: RefCell<Option<DelayResponseCache>>,
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
        if n_out == 0 || n_in == 0 {
            return Err(AutodiffError::Message(
                "Delay: channel counts must be greater than 0".to_string(),
            ));
        }
        if !tau_min.is_finite() {
            return Err(AutodiffError::Message(
                "Delay: tau_min must be finite".to_string(),
            ));
        }
        Ok(Self {
            nfft,
            n_out,
            n_in,
            tau_min,
            param: ArrayD::zeros(IxDyn(&[n_out, n_in])),
            param_grad: ArrayD::zeros(IxDyn(&[n_out, n_in])),
            response_cache: RefCell::new(None),
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
        let param = view2(&self.param, "Delay")?;
        let (n_out, n_in) = param.dim();
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "Delay::forward: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if input_shape[2] != n_in {
            return Err(AutodiffError::Message(format!(
                "Delay::forward: expected {} input channels, got {}",
                n_in, input_shape[2]
            )));
        }
        let mut output_shape = input_shape.to_vec();
        output_shape[2] = n_out;
        let mut output = ArrayD::zeros(IxDyn(&output_shape));

        with_cached_delay_responses(
            &self.response_cache,
            &self.param,
            self.nfft,
            self.tau_min,
            |responses| {
                for out_ch in 0..n_out {
                    for in_ch in 0..n_in {
                        let h = &responses[out_ch * n_in + in_ch];
                        for (bin, &h_val) in h.iter().enumerate() {
                            let input_slice = input.data.index_axis(Axis(1), bin);
                            let input_bin = input_slice.index_axis(Axis(1), in_ch);
                            let mut output_slice = output.index_axis_mut(Axis(1), bin);
                            let mut output_bin = output_slice.index_axis_mut(Axis(1), out_ch);
                            for (destination, &source) in
                                output_bin.iter_mut().zip(input_bin.iter())
                            {
                                *destination += source * h_val;
                            }
                        }
                    }
                }
            },
        )?;

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
        let param = view2(&self.param, "Delay")?;
        let (n_out, n_in) = param.dim();
        validate_spectral_gradient_shape("Delay::backward", input_shape, grad_shape, n_out)?;
        let n_bins = input_shape[1];
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "Delay::backward: input shape {input_shape:?} is incompatible with the module"
            )));
        }

        let mut param_grad = view2_mut(&mut self.param_grad, "Delay")?;
        if param_grad.dim() != (n_out, n_in) {
            return Err(AutodiffError::Message(format!(
                "Delay::backward: parameter gradient shape {:?} does not match parameter shape {:?}",
                param_grad.dim(),
                (n_out, n_in)
            )));
        }

        let scale = -2.0 * std::f64::consts::PI / self.nfft as f64;
        let mut grad_input = ArrayD::zeros(IxDyn(input_shape));

        with_cached_delay_responses(
            &self.response_cache,
            &self.param,
            self.nfft,
            self.tau_min,
            |responses| {
                for out_ch in 0..n_out {
                    for in_ch in 0..n_in {
                        let raw = param[[out_ch, in_ch]];
                        let dtau_draw = raw_to_tau_derivative(raw);
                        let h = &responses[out_ch * n_in + in_ch];

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
                            let mut input_grad_bin =
                                input_grad_slice.index_axis_mut(Axis(1), in_ch);
                            for (destination, &gradient) in
                                input_grad_bin.iter_mut().zip(grad_bin.iter())
                            {
                                *destination += gradient * h_val.conj();
                            }
                        }
                    }
                }
            },
        )?;

        Ok(DiffTensor::from_array(grad_input))
    }

    fn input_channels(&self) -> usize {
        self.param.shape().get(1).copied().unwrap_or(0)
    }
    fn output_channels(&self) -> usize {
        self.param.shape().first().copied().unwrap_or(0)
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
    response_cache: RefCell<Option<DelayResponseCache>>,
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
        if n_channels == 0 {
            return Err(AutodiffError::Message(
                "ParallelDelay: n_channels must be greater than 0".to_string(),
            ));
        }
        if !tau_min.is_finite() {
            return Err(AutodiffError::Message(
                "ParallelDelay: tau_min must be finite".to_string(),
            ));
        }
        Ok(Self {
            nfft,
            n_channels,
            tau_min,
            param: ArrayD::zeros(IxDyn(&[n_channels])),
            param_grad: ArrayD::zeros(IxDyn(&[n_channels])),
            response_cache: RefCell::new(None),
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
        let param = view1(&self.param, "ParallelDelay")?;
        let n_channels = param.len();
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "ParallelDelay::forward: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if input_shape[2] != n_channels {
            return Err(AutodiffError::Message(format!(
                "ParallelDelay::forward: expected {} channels, got {}",
                n_channels, input_shape[2]
            )));
        }
        let mut output = input.data.clone();
        with_cached_delay_responses(
            &self.response_cache,
            &self.param,
            self.nfft,
            self.tau_min,
            |responses| {
                for (ch, response) in responses.iter().take(n_channels).enumerate() {
                    for (bin, &h_val) in response.iter().enumerate() {
                        let mut slice = output.index_axis_mut(Axis(1), bin);
                        let mut ch_slice = slice.index_axis_mut(Axis(1), ch);
                        ch_slice.mapv_inplace(|x| x * h_val);
                    }
                }
            },
        )?;
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
        let param = view1(&self.param, "ParallelDelay")?;
        let n_channels = param.len();
        if input_shape[1] != self.n_bins() || n_channels != input_shape[2] {
            return Err(AutodiffError::Message(format!(
                "ParallelDelay::backward: input shape {:?} is incompatible with the module",
                input_shape
            )));
        }
        let mut param_grad = view1_mut(&mut self.param_grad, "ParallelDelay")?;
        if param_grad.len() != n_channels {
            return Err(AutodiffError::Message(format!(
                "ParallelDelay::backward: parameter gradient shape {:?} does not match parameter shape {:?}",
                param_grad.shape(),
                param.shape()
            )));
        }

        let scale = -2.0 * std::f64::consts::PI / self.nfft as f64;
        let mut grad_input = ArrayD::zeros(IxDyn(input_shape));

        with_cached_delay_responses(
            &self.response_cache,
            &self.param,
            self.nfft,
            self.tau_min,
            |responses| {
                for ch in 0..n_channels {
                    let raw = param[ch];
                    let dtau_draw = raw_to_tau_derivative(raw);
                    let h = &responses[ch];

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
                        param_grad[ch] += accum.re * dtau_draw;

                        let mut input_grad_slice = grad_input.index_axis_mut(Axis(1), bin);
                        let mut input_grad_bin = input_grad_slice.index_axis_mut(Axis(1), ch);
                        for (destination, &gradient) in
                            input_grad_bin.iter_mut().zip(grad_bin.iter())
                        {
                            *destination += gradient * h_val.conj();
                        }
                    }
                }
            },
        )?;

        Ok(DiffTensor::from_array(grad_input))
    }

    fn input_channels(&self) -> usize {
        self.param.shape().first().copied().unwrap_or(0)
    }
    fn output_channels(&self) -> usize {
        self.param.shape().first().copied().unwrap_or(0)
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

#![allow(
    clippy::cast_precision_loss,
    reason = "FFT sizes are audio buffer lengths that fit exactly in f64 for practical values"
)]

use ndarray::{ArrayD, IxDyn};
use num_complex::Complex;
use realfft::RealFftPlanner;

use crate::error::AutodiffError;
use crate::module::DiffModule;
use crate::tensor::DiffTensor;

/// Convert an FFT size to `f64` for arithmetic.
#[inline]
const fn nfft_as_f64(nfft: usize) -> f64 {
    nfft as f64
}

/// Return whether a packed real-FFT bin has no negative-frequency partner.
#[inline]
const fn is_packed_endpoint(nfft: usize, bin: usize) -> bool {
    bin == 0 || (nfft.is_multiple_of(2) && bin == nfft / 2)
}

/// Weight a packed real-FFT bin before applying the unnormalised inverse FFT
/// to compute the adjoint of an unnormalised forward FFT.
#[inline]
const fn rfft_adjoint_weight(nfft: usize, bin: usize) -> f64 {
    if is_packed_endpoint(nfft, bin) {
        1.0
    } else {
        0.5
    }
}

/// Weight an unnormalised forward-FFT bin to compute the adjoint of the
/// normalised inverse real FFT.
#[inline]
fn irfft_adjoint_weight(nfft: usize, bin: usize) -> f64 {
    if is_packed_endpoint(nfft, bin) {
        1.0 / nfft_as_f64(nfft)
    } else {
        2.0 / nfft_as_f64(nfft)
    }
}

/// Decompose a tensor shape into `(batch, time, channels)`.
///
/// Rank-1 tensors are interpreted as `(1, time, 1)`. Higher-rank tensors are
/// interpreted as `(leading_dims..., time, channels)` where `time` is the
/// second-to-last axis and `channels` is the last axis.
fn shape_to_batch_time_channels(shape: &[usize]) -> Result<(usize, usize, usize), AutodiffError> {
    match shape.len() {
        0 => Err(AutodiffError::Message(
            "FFT: input tensor must have at least one dimension".to_string(),
        )),
        1 => Ok((1, shape[0], 1)),
        _ => {
            let channels = shape[shape.len() - 1];
            let time = shape[shape.len() - 2];
            let batch = shape[..shape.len() - 2].iter().product();
            Ok((batch, time, channels))
        }
    }
}

/// Build an output shape where the time axis has been replaced by `n_bins`.
fn output_shape_for(input_shape: &[usize], n_bins: usize) -> Vec<usize> {
    let mut output_shape = input_shape.to_vec();
    if output_shape.len() == 1 {
        output_shape[0] = n_bins;
    } else {
        let time_axis = output_shape.len() - 2;
        output_shape[time_axis] = n_bins;
    }
    output_shape
}

/// Real-to-complex FFT differentiable module.
///
/// Processes the second-to-last axis of a tensor as the time dimension. The
/// last axis is the channel dimension. Rank-1 input is treated as a single
/// channel.
#[derive(Debug, Clone)]
pub struct Fft {
    pub nfft: usize,
    pub channels: usize,
}

impl Fft {
    /// Create a new single-channel FFT module.
    #[must_use]
    pub const fn new(nfft: usize) -> Self {
        Self { nfft, channels: 1 }
    }

    /// Create a new FFT module for `channels` parallel channels.
    #[must_use]
    pub const fn with_channels(nfft: usize, channels: usize) -> Self {
        Self { nfft, channels }
    }

    const fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }
}

impl DiffModule<f64> for Fft {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        let (batch, time, channels) = shape_to_batch_time_channels(input_shape)?;
        if time != self.nfft {
            return Err(AutodiffError::Message(format!(
                "Fft: expected time dimension {}, got {}",
                self.nfft, time
            )));
        }
        if channels != self.channels {
            return Err(AutodiffError::Message(format!(
                "Fft: expected {} channels, got {}",
                self.channels, channels
            )));
        }

        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(self.nfft);

        let input_3d = input
            .data
            .view()
            .into_shape_with_order((batch, time, channels))
            .map_err(|e| AutodiffError::Message(format!("Fft: failed to reshape input: {e}")))?;
        let mut output = ArrayD::zeros(IxDyn(&[batch, self.n_bins(), channels]));

        for b in 0..batch {
            for c in 0..channels {
                let mut input_vec = vec![0.0; self.nfft];
                for t in 0..self.nfft {
                    input_vec[t] = input_3d[[b, t, c]].re;
                }
                let mut spectrum = r2c.make_output_vec();
                r2c.process(&mut input_vec, &mut spectrum)?;
                for (bin, value) in spectrum.iter().enumerate() {
                    output[[b, bin, c]] = *value;
                }
            }
        }

        let output = output
            .into_shape_with_order(IxDyn(&output_shape_for(input_shape, self.n_bins())))
            .map_err(|e| AutodiffError::Message(format!("Fft: failed to reshape output: {e}")))?;

        Ok(DiffTensor::from_array(output))
    }

    fn backward(
        &mut self,
        _input: &DiffTensor<f64>,
        _output: &DiffTensor<f64>,
        grad_output: &DiffTensor<f64>,
    ) -> Result<DiffTensor<f64>, AutodiffError> {
        let grad_shape = grad_output.data.shape();
        let (batch, n_bins, channels) = shape_to_batch_time_channels(grad_shape)?;
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "Fft::backward: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if channels != self.channels {
            return Err(AutodiffError::Message(format!(
                "Fft::backward: expected {} channels, got {}",
                self.channels, channels
            )));
        }

        let mut planner = RealFftPlanner::<f64>::new();
        let c2r = planner.plan_fft_inverse(self.nfft);

        let grad_3d = grad_output
            .data
            .view()
            .into_shape_with_order((batch, n_bins, channels))
            .map_err(|e| AutodiffError::Message(format!("Fft: failed to reshape grad: {e}")))?;
        let mut grad_input = ArrayD::zeros(IxDyn(&[batch, self.nfft, channels]));

        for b in 0..batch {
            for c in 0..channels {
                let mut grad_vec = vec![Complex::new(0.0, 0.0); self.n_bins()];
                for bin in 0..self.n_bins() {
                    let sample = grad_3d[[b, bin, c]];
                    let weight = rfft_adjoint_weight(self.nfft, bin);
                    grad_vec[bin] = if is_packed_endpoint(self.nfft, bin) {
                        Complex::new(sample.re, 0.0)
                    } else {
                        sample * weight
                    };
                }

                let mut grad_time = c2r.make_output_vec();
                c2r.process(&mut grad_vec, &mut grad_time)?;

                for t in 0..self.nfft {
                    grad_input[[b, t, c]] = Complex::new(grad_time[t], 0.0);
                }
            }
        }

        let grad_input = grad_input
            .into_shape_with_order(IxDyn(&output_shape_for(grad_shape, self.nfft)))
            .map_err(|e| {
                AutodiffError::Message(format!("Fft: failed to reshape grad_input: {e}"))
            })?;

        Ok(DiffTensor::from_array(grad_input))
    }

    fn input_channels(&self) -> usize {
        self.channels
    }

    fn output_channels(&self) -> usize {
        self.channels
    }

    fn n_bins(&self) -> usize {
        self.n_bins()
    }

    fn parameters(&self) -> Vec<&ArrayD<f64>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<f64>> {
        vec![]
    }

    fn gradients(&self) -> Vec<&ArrayD<f64>> {
        vec![]
    }

    fn zero_grad(&mut self) {}
}

/// Complex-to-real inverse FFT differentiable module.
///
/// Processes the second-to-last axis of a tensor as the frequency-axis. The
/// last axis is the channel dimension. Rank-1 input is treated as a single
/// channel.
#[derive(Debug, Clone)]
pub struct Ifft {
    pub nfft: usize,
    pub channels: usize,
}

impl Ifft {
    /// Create a new single-channel inverse FFT module.
    #[must_use]
    pub const fn new(nfft: usize) -> Self {
        Self { nfft, channels: 1 }
    }

    /// Create a new inverse FFT module for `channels` parallel channels.
    #[must_use]
    pub const fn with_channels(nfft: usize, channels: usize) -> Self {
        Self { nfft, channels }
    }

    const fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }
}

impl DiffModule<f64> for Ifft {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        let (batch, n_bins, channels) = shape_to_batch_time_channels(input_shape)?;
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "Ifft: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if channels != self.channels {
            return Err(AutodiffError::Message(format!(
                "Ifft: expected {} channels, got {}",
                self.channels, channels
            )));
        }

        let mut planner = RealFftPlanner::<f64>::new();
        let c2r = planner.plan_fft_inverse(self.nfft);

        let input_3d = input
            .data
            .view()
            .into_shape_with_order((batch, n_bins, channels))
            .map_err(|e| AutodiffError::Message(format!("Ifft: failed to reshape input: {e}")))?;
        let mut output = ArrayD::zeros(IxDyn(&[batch, self.nfft, channels]));
        let scale = nfft_as_f64(self.nfft);

        for b in 0..batch {
            for c in 0..channels {
                let mut input_vec = vec![Complex::new(0.0, 0.0); self.n_bins()];
                for bin in 0..self.n_bins() {
                    input_vec[bin] = input_3d[[b, bin, c]];
                }

                let mut time = c2r.make_output_vec();
                c2r.process(&mut input_vec, &mut time)?;

                for t in 0..self.nfft {
                    output[[b, t, c]] = Complex::new(time[t] / scale, 0.0);
                }
            }
        }

        let output = output
            .into_shape_with_order(IxDyn(&output_shape_for(input_shape, self.nfft)))
            .map_err(|e| AutodiffError::Message(format!("Ifft: failed to reshape output: {e}")))?;

        Ok(DiffTensor::from_array(output))
    }

    fn backward(
        &mut self,
        _input: &DiffTensor<f64>,
        _output: &DiffTensor<f64>,
        grad_output: &DiffTensor<f64>,
    ) -> Result<DiffTensor<f64>, AutodiffError> {
        let grad_shape = grad_output.data.shape();
        let (batch, time, channels) = shape_to_batch_time_channels(grad_shape)?;
        if time != self.nfft {
            return Err(AutodiffError::Message(format!(
                "Ifft::backward: expected time dimension {}, got {}",
                self.nfft, time
            )));
        }
        if channels != self.channels {
            return Err(AutodiffError::Message(format!(
                "Ifft::backward: expected {} channels, got {}",
                self.channels, channels
            )));
        }

        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(self.nfft);

        let grad_3d = grad_output
            .data
            .view()
            .into_shape_with_order((batch, time, channels))
            .map_err(|e| AutodiffError::Message(format!("Ifft: failed to reshape grad: {e}")))?;
        let mut grad_input = ArrayD::zeros(IxDyn(&[batch, self.n_bins(), channels]));

        for b in 0..batch {
            for c in 0..channels {
                let mut grad_vec = vec![0.0; self.nfft];
                for t in 0..self.nfft {
                    grad_vec[t] = grad_3d[[b, t, c]].re;
                }

                let mut spectrum = r2c.make_output_vec();
                r2c.process(&mut grad_vec, &mut spectrum)?;

                for (bin, sample) in spectrum.iter_mut().enumerate() {
                    let weight = irfft_adjoint_weight(self.nfft, bin);
                    grad_input[[b, bin, c]] = if is_packed_endpoint(self.nfft, bin) {
                        Complex::new(sample.re * weight, 0.0)
                    } else {
                        *sample * weight
                    };
                }
            }
        }

        let grad_input = grad_input
            .into_shape_with_order(IxDyn(&output_shape_for(grad_shape, self.n_bins())))
            .map_err(|e| {
                AutodiffError::Message(format!("Ifft: failed to reshape grad_input: {e}"))
            })?;

        Ok(DiffTensor::from_array(grad_input))
    }

    fn input_channels(&self) -> usize {
        self.channels
    }

    fn output_channels(&self) -> usize {
        self.channels
    }

    fn n_bins(&self) -> usize {
        self.n_bins()
    }

    fn parameters(&self) -> Vec<&ArrayD<f64>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<f64>> {
        vec![]
    }

    fn gradients(&self) -> Vec<&ArrayD<f64>> {
        vec![]
    }

    fn zero_grad(&mut self) {}
}

/// Real-to-complex FFT with an exponential anti-aliasing envelope.
#[derive(Debug, Clone)]
pub struct FftAntiAlias {
    pub nfft: usize,
    pub channels: usize,
    pub alias_decay_db: f64,
    pub gamma: f64,
    pub envelope: Vec<f64>,
}

impl FftAntiAlias {
    /// Create a new single-channel anti-aliased FFT module.
    ///
    /// The envelope decays by `alias_decay_db` dB across the FFT window.
    #[must_use]
    pub fn new(nfft: usize, alias_decay_db: f64) -> Self {
        Self::with_channels(nfft, 1, alias_decay_db)
    }

    /// Create a new anti-aliased FFT module for `channels` parallel channels.
    #[must_use]
    pub fn with_channels(nfft: usize, channels: usize, alias_decay_db: f64) -> Self {
        let gamma = 10_f64.powf(-alias_decay_db.abs() / (20.0 * nfft_as_f64(nfft)));

        let mut envelope = Vec::with_capacity(nfft);
        let mut value = 1.0;
        for _ in 0..nfft {
            envelope.push(value);
            value *= gamma;
        }

        Self {
            nfft,
            channels,
            alias_decay_db,
            gamma,
            envelope,
        }
    }

    const fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }
}

impl DiffModule<f64> for FftAntiAlias {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        let (batch, time, channels) = shape_to_batch_time_channels(input_shape)?;
        if time != self.nfft {
            return Err(AutodiffError::Message(format!(
                "FftAntiAlias: expected time dimension {}, got {}",
                self.nfft, time
            )));
        }
        if channels != self.channels {
            return Err(AutodiffError::Message(format!(
                "FftAntiAlias: expected {} channels, got {}",
                self.channels, channels
            )));
        }

        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(self.nfft);

        let input_3d = input
            .data
            .view()
            .into_shape_with_order((batch, time, channels))
            .map_err(|e| {
                AutodiffError::Message(format!("FftAntiAlias: failed to reshape input: {e}"))
            })?;
        let mut output = ArrayD::zeros(IxDyn(&[batch, self.n_bins(), channels]));

        for b in 0..batch {
            for c in 0..channels {
                let mut input_vec = vec![0.0; self.nfft];
                for t in 0..self.nfft {
                    input_vec[t] = input_3d[[b, t, c]].re * self.envelope[t];
                }
                let mut spectrum = r2c.make_output_vec();
                r2c.process(&mut input_vec, &mut spectrum)?;
                for (bin, value) in spectrum.iter().enumerate() {
                    output[[b, bin, c]] = *value;
                }
            }
        }

        let output = output
            .into_shape_with_order(IxDyn(&output_shape_for(input_shape, self.n_bins())))
            .map_err(|e| {
                AutodiffError::Message(format!("FftAntiAlias: failed to reshape output: {e}"))
            })?;

        Ok(DiffTensor::from_array(output))
    }

    fn backward(
        &mut self,
        _input: &DiffTensor<f64>,
        _output: &DiffTensor<f64>,
        grad_output: &DiffTensor<f64>,
    ) -> Result<DiffTensor<f64>, AutodiffError> {
        let grad_shape = grad_output.data.shape();
        let (batch, n_bins, channels) = shape_to_batch_time_channels(grad_shape)?;
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "FftAntiAlias::backward: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if channels != self.channels {
            return Err(AutodiffError::Message(format!(
                "FftAntiAlias::backward: expected {} channels, got {}",
                self.channels, channels
            )));
        }

        let mut planner = RealFftPlanner::<f64>::new();
        let c2r = planner.plan_fft_inverse(self.nfft);

        let grad_3d = grad_output
            .data
            .view()
            .into_shape_with_order((batch, n_bins, channels))
            .map_err(|e| {
                AutodiffError::Message(format!("FftAntiAlias: failed to reshape grad: {e}"))
            })?;
        let mut grad_input = ArrayD::zeros(IxDyn(&[batch, self.nfft, channels]));

        for b in 0..batch {
            for c in 0..channels {
                let mut grad_vec = vec![Complex::new(0.0, 0.0); self.n_bins()];
                for bin in 0..self.n_bins() {
                    let sample = grad_3d[[b, bin, c]];
                    let weight = rfft_adjoint_weight(self.nfft, bin);
                    grad_vec[bin] = if is_packed_endpoint(self.nfft, bin) {
                        Complex::new(sample.re, 0.0)
                    } else {
                        sample * weight
                    };
                }

                let mut grad_time = c2r.make_output_vec();
                c2r.process(&mut grad_vec, &mut grad_time)?;

                for t in 0..self.nfft {
                    grad_input[[b, t, c]] = Complex::new(grad_time[t] * self.envelope[t], 0.0);
                }
            }
        }

        let grad_input = grad_input
            .into_shape_with_order(IxDyn(&output_shape_for(grad_shape, self.nfft)))
            .map_err(|e| {
                AutodiffError::Message(format!("FftAntiAlias: failed to reshape grad_input: {e}"))
            })?;

        Ok(DiffTensor::from_array(grad_input))
    }

    fn input_channels(&self) -> usize {
        self.channels
    }

    fn output_channels(&self) -> usize {
        self.channels
    }

    fn n_bins(&self) -> usize {
        self.n_bins()
    }

    fn parameters(&self) -> Vec<&ArrayD<f64>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<f64>> {
        vec![]
    }

    fn gradients(&self) -> Vec<&ArrayD<f64>> {
        vec![]
    }

    fn zero_grad(&mut self) {}
}

/// Complex-to-real inverse FFT with an exponential anti-aliasing envelope.
#[derive(Debug, Clone)]
pub struct IfftAntiAlias {
    pub nfft: usize,
    pub channels: usize,
    pub alias_decay_db: f64,
    pub gamma: f64,
    pub envelope: Vec<f64>,
}

impl IfftAntiAlias {
    /// Create a new single-channel anti-aliased inverse FFT module.
    ///
    /// The envelope decays by `alias_decay_db` dB across the FFT window.
    #[must_use]
    pub fn new(nfft: usize, alias_decay_db: f64) -> Self {
        Self::with_channels(nfft, 1, alias_decay_db)
    }

    /// Create a new anti-aliased inverse FFT module for `channels` parallel channels.
    #[must_use]
    pub fn with_channels(nfft: usize, channels: usize, alias_decay_db: f64) -> Self {
        let gamma = 10_f64.powf(-alias_decay_db.abs() / (20.0 * nfft_as_f64(nfft)));

        let mut envelope = Vec::with_capacity(nfft);
        let mut value = 1.0;
        for _ in 0..nfft {
            envelope.push(value);
            value *= gamma;
        }

        Self {
            nfft,
            channels,
            alias_decay_db,
            gamma,
            envelope,
        }
    }

    const fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }
}

impl DiffModule<f64> for IfftAntiAlias {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        let (batch, n_bins, channels) = shape_to_batch_time_channels(input_shape)?;
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "IfftAntiAlias: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if channels != self.channels {
            return Err(AutodiffError::Message(format!(
                "IfftAntiAlias: expected {} channels, got {}",
                self.channels, channels
            )));
        }

        let mut planner = RealFftPlanner::<f64>::new();
        let c2r = planner.plan_fft_inverse(self.nfft);

        let input_3d = input
            .data
            .view()
            .into_shape_with_order((batch, n_bins, channels))
            .map_err(|e| {
                AutodiffError::Message(format!("IfftAntiAlias: failed to reshape input: {e}"))
            })?;
        let mut output = ArrayD::zeros(IxDyn(&[batch, self.nfft, channels]));
        let scale = nfft_as_f64(self.nfft);

        for b in 0..batch {
            for c in 0..channels {
                let mut input_vec = vec![Complex::new(0.0, 0.0); self.n_bins()];
                for bin in 0..self.n_bins() {
                    input_vec[bin] = input_3d[[b, bin, c]];
                }

                let mut time = c2r.make_output_vec();
                c2r.process(&mut input_vec, &mut time)?;

                for t in 0..self.nfft {
                    output[[b, t, c]] = Complex::new(time[t] * self.envelope[t] / scale, 0.0);
                }
            }
        }

        let output = output
            .into_shape_with_order(IxDyn(&output_shape_for(input_shape, self.nfft)))
            .map_err(|e| {
                AutodiffError::Message(format!("IfftAntiAlias: failed to reshape output: {e}"))
            })?;

        Ok(DiffTensor::from_array(output))
    }

    fn backward(
        &mut self,
        _input: &DiffTensor<f64>,
        _output: &DiffTensor<f64>,
        grad_output: &DiffTensor<f64>,
    ) -> Result<DiffTensor<f64>, AutodiffError> {
        let grad_shape = grad_output.data.shape();
        let (batch, time, channels) = shape_to_batch_time_channels(grad_shape)?;
        if time != self.nfft {
            return Err(AutodiffError::Message(format!(
                "IfftAntiAlias::backward: expected time dimension {}, got {}",
                self.nfft, time
            )));
        }
        if channels != self.channels {
            return Err(AutodiffError::Message(format!(
                "IfftAntiAlias::backward: expected {} channels, got {}",
                self.channels, channels
            )));
        }

        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(self.nfft);

        let grad_3d = grad_output
            .data
            .view()
            .into_shape_with_order((batch, time, channels))
            .map_err(|e| {
                AutodiffError::Message(format!("IfftAntiAlias: failed to reshape grad: {e}"))
            })?;
        let mut grad_input = ArrayD::zeros(IxDyn(&[batch, self.n_bins(), channels]));

        for b in 0..batch {
            for c in 0..channels {
                let mut grad_vec = vec![0.0; self.nfft];
                for t in 0..self.nfft {
                    grad_vec[t] = grad_3d[[b, t, c]].re * self.envelope[t];
                }

                let mut spectrum = r2c.make_output_vec();
                r2c.process(&mut grad_vec, &mut spectrum)?;

                for (bin, sample) in spectrum.iter_mut().enumerate() {
                    let weight = irfft_adjoint_weight(self.nfft, bin);
                    grad_input[[b, bin, c]] = if is_packed_endpoint(self.nfft, bin) {
                        Complex::new(sample.re * weight, 0.0)
                    } else {
                        *sample * weight
                    };
                }
            }
        }

        let grad_input = grad_input
            .into_shape_with_order(IxDyn(&output_shape_for(grad_shape, self.n_bins())))
            .map_err(|e| {
                AutodiffError::Message(format!("IfftAntiAlias: failed to reshape grad_input: {e}"))
            })?;

        Ok(DiffTensor::from_array(grad_input))
    }

    fn input_channels(&self) -> usize {
        self.channels
    }

    fn output_channels(&self) -> usize {
        self.channels
    }

    fn n_bins(&self) -> usize {
        self.n_bins()
    }

    fn parameters(&self) -> Vec<&ArrayD<f64>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<f64>> {
        vec![]
    }

    fn gradients(&self) -> Vec<&ArrayD<f64>> {
        vec![]
    }

    fn zero_grad(&mut self) {}
}

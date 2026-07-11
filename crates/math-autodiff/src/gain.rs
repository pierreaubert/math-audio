//! Simple differentiable gain modules.

#![allow(
    clippy::cast_precision_loss,
    reason = "nfft is an audio buffer length that fits exactly in f64 for practical values"
)]
#![allow(
    clippy::uninlined_format_args,
    reason = "format strings are clearer with explicit arguments in error messages"
)]

use ndarray::{ArrayD, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis, IxDyn};
use num_complex::Complex;

use crate::error::AutodiffError;
use crate::module::DiffModule;
use crate::tensor::DiffTensor;

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

fn view1<'a>(param: &'a ArrayD<f64>, name: &str) -> Result<ArrayView1<'a, f64>, AutodiffError> {
    let shape = param.shape();
    if shape.len() != 1 {
        return Err(AutodiffError::Message(format!(
            "{name}: expected 1-D parameter tensor, got shape {:?}",
            shape
        )));
    }
    let n = shape[0];
    param
        .view()
        .into_shape_with_order(n)
        .map_err(|e| AutodiffError::Message(format!("{name}: failed to reshape param: {e}")))
}

fn view1_mut<'a>(
    param: &'a mut ArrayD<f64>,
    name: &str,
) -> Result<ArrayViewMut1<'a, f64>, AutodiffError> {
    let shape = param.shape();
    if shape.len() != 1 {
        return Err(AutodiffError::Message(format!(
            "{name}: expected 1-D parameter gradient tensor, got shape {:?}",
            shape
        )));
    }
    let n = shape[0];
    param
        .view_mut()
        .into_shape_with_order(n)
        .map_err(|e| AutodiffError::Message(format!("{name}: failed to reshape param_grad: {e}")))
}

/// Matrix gain module: mixes `n_in` input channels into `n_out` output channels
/// with a frequency-independent real gain matrix.
#[derive(Debug, Clone)]
pub struct Gain {
    /// FFT length.
    pub nfft: usize,
    /// Raw gain parameters, shape `(n_out, n_in)`.
    pub param: ArrayD<f64>,
    /// Accumulated parameter gradients, same shape as `param`.
    pub param_grad: ArrayD<f64>,
}

impl Gain {
    /// Create a new gain module with zero-initialized parameters and gradients.
    ///
    /// # Errors
    ///
    /// Returns an error if `nfft` is zero.
    pub fn new(nfft: usize, n_out: usize, n_in: usize) -> Result<Self, AutodiffError> {
        if nfft == 0 {
            return Err(AutodiffError::Message(
                "Gain: nfft must be greater than 0".to_string(),
            ));
        }
        Ok(Self {
            nfft,
            param: ArrayD::zeros(IxDyn(&[n_out, n_in])),
            param_grad: ArrayD::zeros(IxDyn(&[n_out, n_in])),
        })
    }

    fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }
}

impl DiffModule<f64> for Gain {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let param = view2(&self.param, "Gain")?;
        let input_shape = input.data.shape();
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "Gain::forward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let n_bins = input_shape[1];
        let n_in = input_shape[2];
        let (n_out, n_in_stored) = param.dim();
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "Gain::forward: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if n_in != n_in_stored {
            return Err(AutodiffError::Message(format!(
                "Gain::forward: expected {} input channels, got {}",
                n_in_stored, n_in
            )));
        }

        let mut output_shape = input_shape.to_vec();
        output_shape[2] = n_out;
        let mut output = ArrayD::zeros(IxDyn(&output_shape));

        for out_ch in 0..n_out {
            for in_ch in 0..n_in {
                let h = Complex::new(param[[out_ch, in_ch]], 0.0);
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
        let n_bins_expected = self.n_bins();
        let param_shape = self.param.shape();
        if param_shape.len() != 2 {
            return Err(AutodiffError::Message(format!(
                "Gain::backward: expected 2-D parameter tensor, got shape {:?}",
                param_shape
            )));
        }
        let (n_out, n_in_stored) = (param_shape[0], param_shape[1]);
        let param = view2(&self.param, "Gain")?;
        let mut param_grad = view2_mut(&mut self.param_grad, "Gain")?;

        let input_shape = input.data.shape();
        let grad_shape = grad_output.data.shape();
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "Gain::backward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        if grad_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "Gain::backward: grad_output must have at least 3 dimensions, got {:?}",
                grad_shape
            )));
        }
        let n_bins = input_shape[1];
        let n_in = input_shape[2];
        if n_bins != n_bins_expected {
            return Err(AutodiffError::Message(format!(
                "Gain::backward: expected {} frequency bins, got {}",
                n_bins_expected, n_bins
            )));
        }
        if n_in != n_in_stored {
            return Err(AutodiffError::Message(format!(
                "Gain::backward: expected {} input channels, got {}",
                n_in_stored, n_in
            )));
        }
        if grad_shape[1] != n_bins || grad_shape[2] != n_out {
            return Err(AutodiffError::Message(format!(
                "Gain::backward: grad_output shape {:?} incompatible with (..., {}, {})",
                grad_shape, n_bins, n_out
            )));
        }

        // Accumulate parameter gradients.
        for out_ch in 0..n_out {
            let grad_slice = grad_output.data.index_axis(Axis(2), out_ch);
            for in_ch in 0..n_in {
                let input_slice = input.data.index_axis(Axis(2), in_ch);
                let prod = &grad_slice * &input_slice.mapv(|x| x.conj());
                param_grad[[out_ch, in_ch]] += prod.sum().re;
            }
        }

        // Compute dLoss/dInput.
        let mut grad_input = ArrayD::zeros(IxDyn(input_shape));
        for in_ch in 0..n_in {
            for out_ch in 0..n_out {
                let h = param[[out_ch, in_ch]];
                let grad_slice = grad_output.data.index_axis(Axis(2), out_ch);
                let mut input_grad_slice = grad_input.index_axis_mut(Axis(2), in_ch);
                input_grad_slice += &grad_slice.mapv(|x| x * h);
            }
        }

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

/// Parallel (per-channel) gain module: multiplies each channel by a scalar
/// frequency-independent gain.
#[derive(Debug, Clone)]
pub struct ParallelGain {
    /// FFT length.
    pub nfft: usize,
    /// Raw gain parameters, shape `(n_channels,)`.
    pub param: ArrayD<f64>,
    /// Accumulated parameter gradients, same shape as `param`.
    pub param_grad: ArrayD<f64>,
}

impl ParallelGain {
    /// Create a new parallel gain module with zero-initialized parameters and
    /// gradients.
    ///
    /// # Errors
    ///
    /// Returns an error if `nfft` is zero.
    pub fn new(nfft: usize, n_channels: usize) -> Result<Self, AutodiffError> {
        if nfft == 0 {
            return Err(AutodiffError::Message(
                "ParallelGain: nfft must be greater than 0".to_string(),
            ));
        }
        Ok(Self {
            nfft,
            param: ArrayD::zeros(IxDyn(&[n_channels])),
            param_grad: ArrayD::zeros(IxDyn(&[n_channels])),
        })
    }

    fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }
}

impl DiffModule<f64> for ParallelGain {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let param = view1(&self.param, "ParallelGain")?;
        let input_shape = input.data.shape();
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "ParallelGain::forward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let n_bins = input_shape[1];
        let n_channels = input_shape[2];
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "ParallelGain::forward: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if n_channels != param.len() {
            return Err(AutodiffError::Message(format!(
                "ParallelGain::forward: expected {} channels, got {}",
                param.len(),
                n_channels
            )));
        }

        let mut output = input.data.clone();
        for ch in 0..n_channels {
            let h = Complex::new(param[ch], 0.0);
            let mut slice = output.index_axis_mut(Axis(2), ch);
            slice.mapv_inplace(|x| x * h);
        }

        Ok(DiffTensor::from_array(output))
    }

    fn backward(
        &mut self,
        input: &DiffTensor<f64>,
        _output: &DiffTensor<f64>,
        grad_output: &DiffTensor<f64>,
    ) -> Result<DiffTensor<f64>, AutodiffError> {
        let n_bins_expected = self.n_bins();
        let param_shape = self.param.shape();
        if param_shape.len() != 1 {
            return Err(AutodiffError::Message(format!(
                "ParallelGain::backward: expected 1-D parameter tensor, got shape {:?}",
                param_shape
            )));
        }
        let n_channels_stored = param_shape[0];
        let param = view1(&self.param, "ParallelGain")?;
        let mut param_grad = view1_mut(&mut self.param_grad, "ParallelGain")?;

        let input_shape = input.data.shape();
        let grad_shape = grad_output.data.shape();
        if input_shape != grad_shape {
            return Err(AutodiffError::Message(format!(
                "ParallelGain::backward: input shape {:?} does not match grad_output shape {:?}",
                input_shape, grad_shape
            )));
        }
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "ParallelGain::backward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let n_bins = input_shape[1];
        let n_channels = input_shape[2];
        if n_bins != n_bins_expected {
            return Err(AutodiffError::Message(format!(
                "ParallelGain::backward: expected {} frequency bins, got {}",
                n_bins_expected, n_bins
            )));
        }
        if n_channels != n_channels_stored {
            return Err(AutodiffError::Message(format!(
                "ParallelGain::backward: expected {} channels, got {}",
                n_channels_stored, n_channels
            )));
        }

        // Accumulate parameter gradients.
        for ch in 0..n_channels {
            let grad_slice = grad_output.data.index_axis(Axis(2), ch);
            let input_slice = input.data.index_axis(Axis(2), ch);
            let prod = &grad_slice * &input_slice.mapv(|x| x.conj());
            param_grad[ch] += prod.sum().re;
        }

        // Compute dLoss/dInput.
        let mut grad_input = ArrayD::zeros(IxDyn(input_shape));
        for ch in 0..n_channels {
            let h = param[ch];
            let grad_slice = grad_output.data.index_axis(Axis(2), ch);
            let mut input_grad_slice = grad_input.index_axis_mut(Axis(2), ch);
            input_grad_slice += &grad_slice.mapv(|x| x * h);
        }

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

/// Element-wise magnitude module.
///
/// Forward returns `|x|` (real non-negative). Backward propagates through
/// complex values as `grad_input = grad_output * x / |x|` for `|x| > 0`, and
/// zero otherwise.
#[derive(Debug, Clone)]
pub struct Magnitude {
    /// FFT length.
    pub nfft: usize,
    /// Number of channels.
    pub n_channels: usize,
}

impl Magnitude {
    /// Create a new magnitude module.
    #[must_use]
    pub const fn new(nfft: usize, n_channels: usize) -> Self {
        Self { nfft, n_channels }
    }

    fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }
}

impl DiffModule<f64> for Magnitude {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "Magnitude::forward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let n_bins = input_shape[1];
        let n_channels = input_shape[2];
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "Magnitude::forward: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if n_channels != self.n_channels {
            return Err(AutodiffError::Message(format!(
                "Magnitude::forward: expected {} channels, got {}",
                self.n_channels, n_channels
            )));
        }

        let output = input.data.mapv(|x| {
            let norm = x.norm();
            if norm > 0.0 {
                Complex::new(norm, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            }
        });
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
                "Magnitude::backward: input shape {:?} does not match grad_output shape {:?}",
                input_shape, grad_shape
            )));
        }

        let grad_input = input.data.mapv(|x| {
            let norm = x.norm();
            if norm > 1e-12 {
                x / norm
            } else {
                Complex::new(0.0, 0.0)
            }
        });
        let real_grad_output = grad_output.data.mapv(|g| Complex::new(g.re, 0.0));
        let grad = grad_input * real_grad_output;
        Ok(DiffTensor::from_array(grad))
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

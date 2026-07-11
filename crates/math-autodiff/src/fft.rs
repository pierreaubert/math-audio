#![allow(
    clippy::cast_precision_loss,
    reason = "FFT sizes are audio buffer lengths that fit exactly in f64 for practical values"
)]

use ndarray::{ArrayD, IxDyn};
use num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::{
    cell::RefCell,
    collections::HashMap,
    fmt,
    sync::{Arc, OnceLock},
};

use crate::error::AutodiffError;
use crate::module::DiffModule;
use crate::tensor::DiffTensor;

/// Convert an FFT size to `f64` for arithmetic.
#[inline]
const fn nfft_as_f64(nfft: usize) -> f64 {
    nfft as f64
}

const fn validate_fft_config(nfft: usize, channels: usize) {
    assert!(nfft > 0, "FFT: nfft must be greater than 0");
    assert!(channels > 0, "FFT: channels must be greater than 0");
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

#[derive(Clone)]
struct FftPlans {
    forward: Arc<dyn RealToComplex<f64>>,
    inverse: Arc<dyn ComplexToReal<f64>>,
}

impl FftPlans {
    fn new(nfft: usize) -> Self {
        let mut planner = RealFftPlanner::<f64>::new();
        Self {
            forward: planner.plan_fft_forward(nfft),
            inverse: planner.plan_fft_inverse(nfft),
        }
    }
}

impl fmt::Debug for FftPlans {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("FftPlans")
    }
}

struct FftBuffers {
    real: Vec<f64>,
    complex: Vec<Complex<f64>>,
}

impl FftBuffers {
    fn new(nfft: usize) -> Self {
        Self {
            real: vec![0.0; nfft],
            complex: vec![Complex::default(); nfft / 2 + 1],
        }
    }
}

thread_local! {
    static FFT_BUFFER_CACHE: RefCell<HashMap<usize, FftBuffers>> = RefCell::new(HashMap::new());
}

fn with_fft_buffers<R>(
    nfft: usize,
    operation: impl FnOnce(&mut [f64], &mut [Complex<f64>]) -> R,
) -> R {
    FFT_BUFFER_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let buffers = cache.entry(nfft).or_insert_with(|| FftBuffers::new(nfft));
        operation(&mut buffers.real, &mut buffers.complex)
    })
}

fn copy_real_parts(source: &[Complex<f64>], destination: &mut [f64]) {
    debug_assert_eq!(source.len(), destination.len());

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: both slices have equal length. The kernel processes pairs
        // within bounds and handles the possible final element scalarly.
        unsafe { copy_real_parts_neon(source, destination) };
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    for (output, input) in destination.iter_mut().zip(source) {
        *output = input.re;
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
unsafe fn copy_real_parts_neon(source: &[Complex<f64>], destination: &mut [f64]) {
    use std::arch::aarch64::{vld2q_f64, vst1q_f64};

    let mut index = 0;
    while index + 2 <= source.len() {
        // SAFETY: the loop guard leaves two Complex<f64> (four f64 lanes) in
        // source and two f64 lanes in destination.
        unsafe {
            let deinterleaved = vld2q_f64(source.as_ptr().add(index).cast::<f64>());
            vst1q_f64(destination.as_mut_ptr().add(index), deinterleaved.0);
        }
        index += 2;
    }
    if index < source.len() {
        destination[index] = source[index].re;
    }
}

fn store_real_as_complex(source: &[f64], destination: &mut [Complex<f64>], scale: f64) {
    debug_assert_eq!(source.len(), destination.len());

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: both slices have equal length. The kernel processes pairs
        // within bounds and handles the possible final element scalarly.
        unsafe { store_real_as_complex_neon(source, destination, scale) };
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    for (output, &input) in destination.iter_mut().zip(source) {
        *output = Complex::new(input * scale, 0.0);
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
unsafe fn store_real_as_complex_neon(source: &[f64], destination: &mut [Complex<f64>], scale: f64) {
    use std::arch::aarch64::{float64x2x2_t, vdupq_n_f64, vld1q_f64, vmulq_n_f64, vst2q_f64};

    let zero = vdupq_n_f64(0.0);
    let mut index = 0;
    while index + 2 <= source.len() {
        // SAFETY: the loop guard leaves two f64 lanes in source and two
        // Complex<f64> (four f64 lanes) in destination.
        unsafe {
            let real = vmulq_n_f64(vld1q_f64(source.as_ptr().add(index)), scale);
            vst2q_f64(
                destination.as_mut_ptr().add(index).cast::<f64>(),
                float64x2x2_t(real, zero),
            );
        }
        index += 2;
    }
    if index < source.len() {
        destination[index] = Complex::new(source[index] * scale, 0.0);
    }
}

/// Real-to-complex FFT differentiable module.
///
/// Processes the second-to-last axis of a tensor as the time dimension. The
/// last axis is the channel dimension. Rank-1 input is treated as a single
/// channel. Only the real component of each time-domain sample is transformed;
/// the imaginary component is intentionally ignored.
#[derive(Debug, Clone)]
pub struct Fft {
    pub nfft: usize,
    pub channels: usize,
    plans: OnceLock<FftPlans>,
}

impl Fft {
    /// Create a new single-channel FFT module.
    #[must_use]
    pub const fn new(nfft: usize) -> Self {
        validate_fft_config(nfft, 1);
        Self {
            nfft,
            channels: 1,
            plans: OnceLock::new(),
        }
    }

    /// Create a new FFT module for `channels` parallel channels.
    #[must_use]
    pub const fn with_channels(nfft: usize, channels: usize) -> Self {
        validate_fft_config(nfft, channels);
        Self {
            nfft,
            channels,
            plans: OnceLock::new(),
        }
    }

    const fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }

    fn plans(&self) -> &FftPlans {
        self.plans.get_or_init(|| FftPlans::new(self.nfft))
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

        let r2c = &self.plans().forward;

        let input_3d = input
            .data
            .view()
            .into_shape_with_order((batch, time, channels))
            .map_err(|e| AutodiffError::Message(format!("Fft: failed to reshape input: {e}")))?;
        let mut output = ArrayD::zeros(IxDyn(&[batch, self.n_bins(), channels]));

        for b in 0..batch {
            for c in 0..channels {
                with_fft_buffers(self.nfft, |input_vec, spectrum| {
                    if channels == 1 {
                        let start = b * self.nfft;
                        let source = &input_3d
                            .as_slice()
                            .expect("reshaped FFT input must remain contiguous")
                            [start..start + self.nfft];
                        copy_real_parts(source, input_vec);
                    } else {
                        for t in 0..self.nfft {
                            input_vec[t] = input_3d[[b, t, c]].re;
                        }
                    }
                    r2c.process(input_vec, spectrum)?;
                    if channels == 1 {
                        let start = b * self.n_bins();
                        output
                            .as_slice_mut()
                            .expect("FFT output allocation must be contiguous")
                            [start..start + self.n_bins()]
                            .copy_from_slice(spectrum);
                    } else {
                        for (bin, value) in spectrum.iter().enumerate() {
                            output[[b, bin, c]] = *value;
                        }
                    }
                    Ok::<(), AutodiffError>(())
                })?;
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

        let c2r = &self.plans().inverse;

        let grad_3d = grad_output
            .data
            .view()
            .into_shape_with_order((batch, n_bins, channels))
            .map_err(|e| AutodiffError::Message(format!("Fft: failed to reshape grad: {e}")))?;
        let mut grad_input = ArrayD::zeros(IxDyn(&[batch, self.nfft, channels]));

        for b in 0..batch {
            for c in 0..channels {
                with_fft_buffers(self.nfft, |grad_time, grad_vec| {
                    for bin in 0..self.n_bins() {
                        let sample = grad_3d[[b, bin, c]];
                        let weight = rfft_adjoint_weight(self.nfft, bin);
                        grad_vec[bin] = if is_packed_endpoint(self.nfft, bin) {
                            Complex::new(sample.re, 0.0)
                        } else {
                            sample * weight
                        };
                    }
                    c2r.process(grad_vec, grad_time)?;
                    if channels == 1 {
                        let start = b * self.nfft;
                        let destination = &mut grad_input
                            .as_slice_mut()
                            .expect("FFT gradient allocation must be contiguous")
                            [start..start + self.nfft];
                        store_real_as_complex(grad_time, destination, 1.0);
                    } else {
                        for t in 0..self.nfft {
                            grad_input[[b, t, c]] = Complex::new(grad_time[t], 0.0);
                        }
                    }
                    Ok::<(), AutodiffError>(())
                })?;
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
    plans: OnceLock<FftPlans>,
}

impl Ifft {
    /// Create a new single-channel inverse FFT module.
    #[must_use]
    pub const fn new(nfft: usize) -> Self {
        validate_fft_config(nfft, 1);
        Self {
            nfft,
            channels: 1,
            plans: OnceLock::new(),
        }
    }

    /// Create a new inverse FFT module for `channels` parallel channels.
    #[must_use]
    pub const fn with_channels(nfft: usize, channels: usize) -> Self {
        validate_fft_config(nfft, channels);
        Self {
            nfft,
            channels,
            plans: OnceLock::new(),
        }
    }

    const fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }

    fn plans(&self) -> &FftPlans {
        self.plans.get_or_init(|| FftPlans::new(self.nfft))
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

        let c2r = &self.plans().inverse;

        let input_3d = input
            .data
            .view()
            .into_shape_with_order((batch, n_bins, channels))
            .map_err(|e| AutodiffError::Message(format!("Ifft: failed to reshape input: {e}")))?;
        let mut output = ArrayD::zeros(IxDyn(&[batch, self.nfft, channels]));
        let scale = nfft_as_f64(self.nfft);

        for b in 0..batch {
            for c in 0..channels {
                with_fft_buffers(self.nfft, |time, input_vec| {
                    if channels == 1 {
                        let start = b * self.n_bins();
                        input_vec.copy_from_slice(
                            &input_3d
                                .as_slice()
                                .expect("reshaped IFFT input must remain contiguous")
                                [start..start + self.n_bins()],
                        );
                    } else {
                        for bin in 0..self.n_bins() {
                            input_vec[bin] = input_3d[[b, bin, c]];
                        }
                    }
                    c2r.process(input_vec, time)?;
                    if channels == 1 {
                        let start = b * self.nfft;
                        let destination = &mut output
                            .as_slice_mut()
                            .expect("IFFT output allocation must be contiguous")
                            [start..start + self.nfft];
                        store_real_as_complex(time, destination, scale.recip());
                    } else {
                        for t in 0..self.nfft {
                            output[[b, t, c]] = Complex::new(time[t] / scale, 0.0);
                        }
                    }
                    Ok::<(), AutodiffError>(())
                })?;
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

        let r2c = &self.plans().forward;

        let grad_3d = grad_output
            .data
            .view()
            .into_shape_with_order((batch, time, channels))
            .map_err(|e| AutodiffError::Message(format!("Ifft: failed to reshape grad: {e}")))?;
        let mut grad_input = ArrayD::zeros(IxDyn(&[batch, self.n_bins(), channels]));

        for b in 0..batch {
            for c in 0..channels {
                with_fft_buffers(self.nfft, |grad_vec, spectrum| {
                    for t in 0..self.nfft {
                        grad_vec[t] = grad_3d[[b, t, c]].re;
                    }
                    r2c.process(grad_vec, spectrum)?;
                    for (bin, sample) in spectrum.iter().enumerate() {
                        let weight = irfft_adjoint_weight(self.nfft, bin);
                        grad_input[[b, bin, c]] = if is_packed_endpoint(self.nfft, bin) {
                            Complex::new(sample.re * weight, 0.0)
                        } else {
                            *sample * weight
                        };
                    }
                    Ok::<(), AutodiffError>(())
                })?;
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
///
/// Only the real component of each time-domain sample is transformed.
#[derive(Debug, Clone)]
pub struct FftAntiAlias {
    pub nfft: usize,
    pub channels: usize,
    pub alias_decay_db: f64,
    pub gamma: f64,
    pub envelope: Vec<f64>,
    plans: FftPlans,
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
    ///
    /// # Panics
    ///
    /// Panics if the FFT size or channel count is zero, or if the decay is not finite.
    #[must_use]
    pub fn with_channels(nfft: usize, channels: usize, alias_decay_db: f64) -> Self {
        validate_fft_config(nfft, channels);
        assert!(
            alias_decay_db.is_finite(),
            "FftAntiAlias: alias_decay_db must be finite"
        );
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
            plans: FftPlans::new(nfft),
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

        let r2c = &self.plans.forward;

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
                with_fft_buffers(self.nfft, |input_vec, spectrum| {
                    for t in 0..self.nfft {
                        input_vec[t] = input_3d[[b, t, c]].re * self.envelope[t];
                    }
                    r2c.process(input_vec, spectrum)?;
                    for (bin, value) in spectrum.iter().enumerate() {
                        output[[b, bin, c]] = *value;
                    }
                    Ok::<(), AutodiffError>(())
                })?;
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

        let c2r = &self.plans.inverse;

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
                with_fft_buffers(self.nfft, |grad_time, grad_vec| {
                    for bin in 0..self.n_bins() {
                        let sample = grad_3d[[b, bin, c]];
                        let weight = rfft_adjoint_weight(self.nfft, bin);
                        grad_vec[bin] = if is_packed_endpoint(self.nfft, bin) {
                            Complex::new(sample.re, 0.0)
                        } else {
                            sample * weight
                        };
                    }
                    c2r.process(grad_vec, grad_time)?;
                    for t in 0..self.nfft {
                        grad_input[[b, t, c]] = Complex::new(grad_time[t] * self.envelope[t], 0.0);
                    }
                    Ok::<(), AutodiffError>(())
                })?;
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
    plans: FftPlans,
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
    ///
    /// # Panics
    ///
    /// Panics if the FFT size or channel count is zero, or if the decay is not finite.
    #[must_use]
    pub fn with_channels(nfft: usize, channels: usize, alias_decay_db: f64) -> Self {
        validate_fft_config(nfft, channels);
        assert!(
            alias_decay_db.is_finite(),
            "IfftAntiAlias: alias_decay_db must be finite"
        );
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
            plans: FftPlans::new(nfft),
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

        let c2r = &self.plans.inverse;

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
                with_fft_buffers(self.nfft, |time, input_vec| {
                    for bin in 0..self.n_bins() {
                        input_vec[bin] = input_3d[[b, bin, c]];
                    }
                    c2r.process(input_vec, time)?;
                    for t in 0..self.nfft {
                        output[[b, t, c]] = Complex::new(time[t] * self.envelope[t] / scale, 0.0);
                    }
                    Ok::<(), AutodiffError>(())
                })?;
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

        let r2c = &self.plans.forward;

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
                with_fft_buffers(self.nfft, |grad_vec, spectrum| {
                    for t in 0..self.nfft {
                        grad_vec[t] = grad_3d[[b, t, c]].re * self.envelope[t];
                    }
                    r2c.process(grad_vec, spectrum)?;
                    for (bin, sample) in spectrum.iter().enumerate() {
                        let weight = irfft_adjoint_weight(self.nfft, bin);
                        grad_input[[b, bin, c]] = if is_packed_endpoint(self.nfft, bin) {
                            Complex::new(sample.re * weight, 0.0)
                        } else {
                            *sample * weight
                        };
                    }
                    Ok::<(), AutodiffError>(())
                })?;
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

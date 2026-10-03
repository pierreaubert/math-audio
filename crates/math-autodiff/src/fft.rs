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
    marker::PhantomData,
    sync::{Arc, Mutex, OnceLock},
};

use crate::error::AutodiffError;
use crate::module::{DiffModule, Scalar, fconst};
use crate::tensor::DiffTensor;

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
fn rfft_adjoint_weight<T: Scalar>(nfft: usize, bin: usize) -> T {
    if is_packed_endpoint(nfft, bin) {
        T::one()
    } else {
        fconst::<T>(0.5)
    }
}

/// Weight an unnormalised forward-FFT bin to compute the adjoint of the
/// normalised inverse real FFT.
#[inline]
fn irfft_adjoint_weight<T: Scalar>(nfft: usize, bin: usize) -> T {
    if is_packed_endpoint(nfft, bin) {
        T::one() / fconst::<T>(nfft as f64)
    } else {
        fconst::<T>(2.0) / fconst::<T>(nfft as f64)
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

/// Shared realfft plans. Public only to name the `FftScalar` return type.
#[doc(hidden)]
#[derive(Clone)]
pub struct FftPlans<T: realfft::FftNum> {
    forward: Arc<dyn RealToComplex<T>>,
    inverse: Arc<dyn ComplexToReal<T>>,
}

/// Scalar bound for FFT modules: differentiable scalar plus realfft support,
/// per-type plan/buffer caches, and copy kernels. Implemented for `f32`/`f64`.
pub trait FftScalar: Scalar + realfft::FftNum {
    /// Fetch shared FFT plans for `nfft`, building them once.
    fn shared_plans(nfft: usize) -> Arc<FftPlans<Self>>;
    /// Run `operation` with reusable real/complex/scratch buffers for `nfft`.
    fn with_fft_buffers<R>(
        nfft: usize,
        scratch_len: usize,
        operation: impl FnOnce(&mut [Self], &mut [Complex<Self>], &mut [Complex<Self>]) -> R,
    ) -> R;
    /// Copy the real parts of `source` into `destination`.
    fn copy_real_parts(source: &[Complex<Self>], destination: &mut [Self]);
    /// Store `source` as complex values scaled by `scale`.
    fn store_real_as_complex(source: &[Self], destination: &mut [Complex<Self>], scale: Self);
}

static FFT_PLAN_CACHE_F32: OnceLock<Mutex<HashMap<usize, Arc<FftPlans<f32>>>>> = OnceLock::new();
static FFT_PLAN_CACHE_F64: OnceLock<Mutex<HashMap<usize, Arc<FftPlans<f64>>>>> = OnceLock::new();

thread_local! {
    static FFT_BUFFER_CACHE_F32: RefCell<HashMap<usize, FftBuffers<f32>>> =
        RefCell::new(HashMap::new());
    static FFT_BUFFER_CACHE_F64: RefCell<HashMap<usize, FftBuffers<f64>>> =
        RefCell::new(HashMap::new());
}

impl FftScalar for f32 {
    fn shared_plans(nfft: usize) -> Arc<FftPlans<Self>> {
        let cache = FFT_PLAN_CACHE_F32.get_or_init(|| Mutex::new(HashMap::new()));
        let mut cache = cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cache
            .entry(nfft)
            .or_insert_with(|| Arc::new(FftPlans::new(nfft)))
            .clone()
    }

    fn with_fft_buffers<R>(
        nfft: usize,
        scratch_len: usize,
        operation: impl FnOnce(&mut [Self], &mut [Complex<Self>], &mut [Complex<Self>]) -> R,
    ) -> R {
        FFT_BUFFER_CACHE_F32.with(|cache| {
            let mut cache = cache.borrow_mut();
            let buffers = cache.entry(nfft).or_insert_with(|| FftBuffers::new(nfft));
            if buffers.scratch.len() < scratch_len {
                buffers.scratch.resize(scratch_len, Complex::new(0.0, 0.0));
            }
            operation(
                &mut buffers.real,
                &mut buffers.complex,
                &mut buffers.scratch[..scratch_len],
            )
        })
    }

    fn copy_real_parts(source: &[Complex<Self>], destination: &mut [Self]) {
        debug_assert_eq!(source.len(), destination.len());
        for (output, input) in destination.iter_mut().zip(source) {
            *output = input.re;
        }
    }

    fn store_real_as_complex(source: &[Self], destination: &mut [Complex<Self>], scale: Self) {
        debug_assert_eq!(source.len(), destination.len());
        for (output, &input) in destination.iter_mut().zip(source) {
            *output = Complex::new(input * scale, 0.0);
        }
    }
}

impl FftScalar for f64 {
    fn shared_plans(nfft: usize) -> Arc<FftPlans<Self>> {
        let cache = FFT_PLAN_CACHE_F64.get_or_init(|| Mutex::new(HashMap::new()));
        let mut cache = cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cache
            .entry(nfft)
            .or_insert_with(|| Arc::new(FftPlans::new(nfft)))
            .clone()
    }

    fn with_fft_buffers<R>(
        nfft: usize,
        scratch_len: usize,
        operation: impl FnOnce(&mut [Self], &mut [Complex<Self>], &mut [Complex<Self>]) -> R,
    ) -> R {
        FFT_BUFFER_CACHE_F64.with(|cache| {
            let mut cache = cache.borrow_mut();
            let buffers = cache.entry(nfft).or_insert_with(|| FftBuffers::new(nfft));
            if buffers.scratch.len() < scratch_len {
                buffers.scratch.resize(scratch_len, Complex::new(0.0, 0.0));
            }
            operation(
                &mut buffers.real,
                &mut buffers.complex,
                &mut buffers.scratch[..scratch_len],
            )
        })
    }

    fn copy_real_parts(source: &[Complex<Self>], destination: &mut [Self]) {
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

    fn store_real_as_complex(source: &[Self], destination: &mut [Complex<Self>], scale: Self) {
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
}

impl<T: realfft::FftNum> FftPlans<T> {
    fn new(nfft: usize) -> Self {
        let mut planner = RealFftPlanner::<T>::new();
        Self {
            forward: planner.plan_fft_forward(nfft),
            inverse: planner.plan_fft_inverse(nfft),
        }
    }
}

impl<T: realfft::FftNum> fmt::Debug for FftPlans<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("FftPlans")
    }
}

struct FftBuffers<T> {
    real: Vec<T>,
    complex: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
}

impl<T: Scalar> FftBuffers<T> {
    fn new(nfft: usize) -> Self {
        Self {
            real: vec![T::zero(); nfft],
            complex: vec![Complex::new(T::zero(), T::zero()); nfft / 2 + 1],
            scratch: Vec::new(),
        }
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
pub struct Fft<T = f64> {
    pub nfft: usize,
    pub channels: usize,
    _marker: PhantomData<T>,
}

impl<T> Fft<T> {
    /// Create a new single-channel FFT module.
    #[must_use]
    pub const fn new(nfft: usize) -> Self {
        validate_fft_config(nfft, 1);
        Self {
            nfft,
            channels: 1,
            _marker: PhantomData,
        }
    }

    /// Create a new FFT module for `channels` parallel channels.
    #[must_use]
    pub const fn with_channels(nfft: usize, channels: usize) -> Self {
        validate_fft_config(nfft, channels);
        Self {
            nfft,
            channels,
            _marker: PhantomData,
        }
    }

    const fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }
}

impl<T: FftScalar> DiffModule<T> for Fft<T> {
    fn forward(&self, input: &DiffTensor<T>) -> Result<DiffTensor<T>, AutodiffError> {
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

        let plans = T::shared_plans(self.nfft);
        let r2c = &plans.forward;

        let input_3d = input
            .data
            .as_standard_layout()
            .into_shape_with_order((batch, time, channels))
            .map_err(|e| AutodiffError::Message(format!("Fft: failed to reshape input: {e}")))?;
        let mut output = ArrayD::zeros(IxDyn(&[batch, self.n_bins(), channels]));

        for b in 0..batch {
            for c in 0..channels {
                T::with_fft_buffers(
                    self.nfft,
                    r2c.get_scratch_len(),
                    |input_vec, spectrum, scratch| {
                        if channels == 1 {
                            let start = b * self.nfft;
                            if let Some(source) = input_3d.as_slice() {
                                T::copy_real_parts(&source[start..start + self.nfft], input_vec);
                            } else {
                                for t in 0..self.nfft {
                                    input_vec[t] = input_3d[[b, t, 0]].re;
                                }
                            }
                        } else {
                            for t in 0..self.nfft {
                                input_vec[t] = input_3d[[b, t, c]].re;
                            }
                        }
                        r2c.process_with_scratch(input_vec, spectrum, scratch)?;
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
                    },
                )?;
            }
        }

        let output = output
            .into_shape_with_order(IxDyn(&output_shape_for(input_shape, self.n_bins())))
            .map_err(|e| AutodiffError::Message(format!("Fft: failed to reshape output: {e}")))?;

        Ok(DiffTensor::from_array(output))
    }

    fn backward(
        &mut self,
        _input: &DiffTensor<T>,
        _output: &DiffTensor<T>,
        grad_output: &DiffTensor<T>,
    ) -> Result<DiffTensor<T>, AutodiffError> {
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

        let plans = T::shared_plans(self.nfft);
        let c2r = &plans.inverse;

        let grad_3d = grad_output
            .data
            .as_standard_layout()
            .into_shape_with_order((batch, n_bins, channels))
            .map_err(|e| AutodiffError::Message(format!("Fft: failed to reshape grad: {e}")))?;
        let mut grad_input = ArrayD::zeros(IxDyn(&[batch, self.nfft, channels]));

        for b in 0..batch {
            for c in 0..channels {
                T::with_fft_buffers(
                    self.nfft,
                    c2r.get_scratch_len(),
                    |grad_time, grad_vec, scratch| {
                        if channels == 1
                            && let Some(data) = grad_3d.as_slice()
                        {
                            let start = b * self.n_bins();
                            for bin in 0..self.n_bins() {
                                let sample = data[start + bin];
                                let weight: T = rfft_adjoint_weight(self.nfft, bin);
                                grad_vec[bin] = if is_packed_endpoint(self.nfft, bin) {
                                    Complex::new(sample.re, T::zero())
                                } else {
                                    sample * weight
                                };
                            }
                        } else {
                            for bin in 0..self.n_bins() {
                                let sample = grad_3d[[b, bin, c]];
                                let weight: T = rfft_adjoint_weight(self.nfft, bin);
                                grad_vec[bin] = if is_packed_endpoint(self.nfft, bin) {
                                    Complex::new(sample.re, T::zero())
                                } else {
                                    sample * weight
                                };
                            }
                        }
                        c2r.process_with_scratch(grad_vec, grad_time, scratch)?;
                        if channels == 1 {
                            let start = b * self.nfft;
                            let destination = &mut grad_input
                                .as_slice_mut()
                                .expect("FFT gradient allocation must be contiguous")
                                [start..start + self.nfft];
                            T::store_real_as_complex(grad_time, destination, T::one());
                        } else {
                            for t in 0..self.nfft {
                                grad_input[[b, t, c]] = Complex::new(grad_time[t], T::zero());
                            }
                        }
                        Ok::<(), AutodiffError>(())
                    },
                )?;
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

    fn parameters(&self) -> Vec<&ArrayD<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<T>> {
        vec![]
    }

    fn gradients(&self) -> Vec<&ArrayD<T>> {
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
pub struct Ifft<T = f64> {
    pub nfft: usize,
    pub channels: usize,
    _marker: PhantomData<T>,
}

impl<T> Ifft<T> {
    /// Create a new single-channel inverse FFT module.
    #[must_use]
    pub const fn new(nfft: usize) -> Self {
        validate_fft_config(nfft, 1);
        Self {
            nfft,
            channels: 1,
            _marker: PhantomData,
        }
    }

    /// Create a new inverse FFT module for `channels` parallel channels.
    #[must_use]
    pub const fn with_channels(nfft: usize, channels: usize) -> Self {
        validate_fft_config(nfft, channels);
        Self {
            nfft,
            channels,
            _marker: PhantomData,
        }
    }

    const fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }
}

impl<T: FftScalar> DiffModule<T> for Ifft<T> {
    fn forward(&self, input: &DiffTensor<T>) -> Result<DiffTensor<T>, AutodiffError> {
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

        let plans = T::shared_plans(self.nfft);
        let c2r = &plans.inverse;

        let input_3d = input
            .data
            .as_standard_layout()
            .into_shape_with_order((batch, n_bins, channels))
            .map_err(|e| AutodiffError::Message(format!("Ifft: failed to reshape input: {e}")))?;
        let mut output = ArrayD::zeros(IxDyn(&[batch, self.nfft, channels]));
        let scale = fconst::<T>(self.nfft as f64);

        for b in 0..batch {
            for c in 0..channels {
                T::with_fft_buffers(
                    self.nfft,
                    c2r.get_scratch_len(),
                    |time, input_vec, scratch| {
                        if channels == 1 {
                            let start = b * self.n_bins();
                            if let Some(source) = input_3d.as_slice() {
                                input_vec.copy_from_slice(&source[start..start + self.n_bins()]);
                            } else {
                                for bin in 0..self.n_bins() {
                                    input_vec[bin] = input_3d[[b, bin, 0]];
                                }
                            }
                        } else {
                            for bin in 0..self.n_bins() {
                                input_vec[bin] = input_3d[[b, bin, c]];
                            }
                        }
                        c2r.process_with_scratch(input_vec, time, scratch)?;
                        if channels == 1 {
                            let start = b * self.nfft;
                            let destination = &mut output
                                .as_slice_mut()
                                .expect("IFFT output allocation must be contiguous")
                                [start..start + self.nfft];
                            T::store_real_as_complex(time, destination, scale.recip());
                        } else {
                            for t in 0..self.nfft {
                                output[[b, t, c]] = Complex::new(time[t] / scale, T::zero());
                            }
                        }
                        Ok::<(), AutodiffError>(())
                    },
                )?;
            }
        }

        let output = output
            .into_shape_with_order(IxDyn(&output_shape_for(input_shape, self.nfft)))
            .map_err(|e| AutodiffError::Message(format!("Ifft: failed to reshape output: {e}")))?;

        Ok(DiffTensor::from_array(output))
    }

    fn backward(
        &mut self,
        _input: &DiffTensor<T>,
        _output: &DiffTensor<T>,
        grad_output: &DiffTensor<T>,
    ) -> Result<DiffTensor<T>, AutodiffError> {
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

        let plans = T::shared_plans(self.nfft);
        let r2c = &plans.forward;

        let grad_3d = grad_output
            .data
            .as_standard_layout()
            .into_shape_with_order((batch, time, channels))
            .map_err(|e| AutodiffError::Message(format!("Ifft: failed to reshape grad: {e}")))?;
        let mut grad_input = ArrayD::zeros(IxDyn(&[batch, self.n_bins(), channels]));

        for b in 0..batch {
            for c in 0..channels {
                T::with_fft_buffers(
                    self.nfft,
                    r2c.get_scratch_len(),
                    |grad_vec, spectrum, scratch| {
                        if channels == 1
                            && let Some(data) = grad_3d.as_slice()
                        {
                            let start = b * self.nfft;
                            for t in 0..self.nfft {
                                grad_vec[t] = data[start + t].re;
                            }
                        } else {
                            for t in 0..self.nfft {
                                grad_vec[t] = grad_3d[[b, t, c]].re;
                            }
                        }
                        r2c.process_with_scratch(grad_vec, spectrum, scratch)?;
                        for (bin, sample) in spectrum.iter().enumerate() {
                            let weight: T = irfft_adjoint_weight(self.nfft, bin);
                            grad_input[[b, bin, c]] = if is_packed_endpoint(self.nfft, bin) {
                                Complex::new(sample.re * weight, T::zero())
                            } else {
                                *sample * weight
                            };
                        }
                        Ok::<(), AutodiffError>(())
                    },
                )?;
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

    fn parameters(&self) -> Vec<&ArrayD<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<T>> {
        vec![]
    }

    fn gradients(&self) -> Vec<&ArrayD<T>> {
        vec![]
    }

    fn zero_grad(&mut self) {}
}

/// Real-to-complex FFT with an exponential anti-aliasing envelope.
///
/// Only the real component of each time-domain sample is transformed.
#[derive(Debug, Clone)]
pub struct FftAntiAlias<T = f64> {
    pub nfft: usize,
    pub channels: usize,
    pub alias_decay_db: T,
    pub gamma: T,
    pub envelope: Vec<T>,
}

impl<T: FftScalar> FftAntiAlias<T> {
    /// Create a new single-channel anti-aliased FFT module.
    ///
    /// The envelope decays by `alias_decay_db` dB across the FFT window.
    #[must_use]
    pub fn new(nfft: usize, alias_decay_db: T) -> Self {
        Self::with_channels(nfft, 1, alias_decay_db)
    }

    /// Create a new anti-aliased FFT module for `channels` parallel channels.
    ///
    /// # Panics
    ///
    /// Panics if the FFT size or channel count is zero, or if the decay is not finite.
    #[must_use]
    pub fn with_channels(nfft: usize, channels: usize, alias_decay_db: T) -> Self {
        validate_fft_config(nfft, channels);
        assert!(
            alias_decay_db.is_finite(),
            "FftAntiAlias: alias_decay_db must be finite"
        );
        let gamma = fconst::<T>(10.0)
            .powf(-alias_decay_db.abs() / (fconst::<T>(20.0) * fconst::<T>(nfft as f64)));

        let mut envelope = Vec::with_capacity(nfft);
        let mut value = T::one();
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

impl<T: FftScalar> DiffModule<T> for FftAntiAlias<T> {
    fn forward(&self, input: &DiffTensor<T>) -> Result<DiffTensor<T>, AutodiffError> {
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

        let plans = T::shared_plans(self.nfft);
        let r2c = &plans.forward;

        let input_3d = input
            .data
            .as_standard_layout()
            .into_shape_with_order((batch, time, channels))
            .map_err(|e| {
                AutodiffError::Message(format!("FftAntiAlias: failed to reshape input: {e}"))
            })?;
        let mut output = ArrayD::zeros(IxDyn(&[batch, self.n_bins(), channels]));

        for b in 0..batch {
            for c in 0..channels {
                T::with_fft_buffers(
                    self.nfft,
                    r2c.get_scratch_len(),
                    |input_vec, spectrum, scratch| {
                        for t in 0..self.nfft {
                            input_vec[t] = input_3d[[b, t, c]].re * self.envelope[t];
                        }
                        r2c.process_with_scratch(input_vec, spectrum, scratch)?;
                        for (bin, value) in spectrum.iter().enumerate() {
                            output[[b, bin, c]] = *value;
                        }
                        Ok::<(), AutodiffError>(())
                    },
                )?;
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
        _input: &DiffTensor<T>,
        _output: &DiffTensor<T>,
        grad_output: &DiffTensor<T>,
    ) -> Result<DiffTensor<T>, AutodiffError> {
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

        let plans = T::shared_plans(self.nfft);
        let c2r = &plans.inverse;

        let grad_3d = grad_output
            .data
            .as_standard_layout()
            .into_shape_with_order((batch, n_bins, channels))
            .map_err(|e| {
                AutodiffError::Message(format!("FftAntiAlias: failed to reshape grad: {e}"))
            })?;
        let mut grad_input = ArrayD::zeros(IxDyn(&[batch, self.nfft, channels]));

        for b in 0..batch {
            for c in 0..channels {
                T::with_fft_buffers(
                    self.nfft,
                    c2r.get_scratch_len(),
                    |grad_time, grad_vec, scratch| {
                        if channels == 1
                            && let Some(data) = grad_3d.as_slice()
                        {
                            let start = b * self.n_bins();
                            for bin in 0..self.n_bins() {
                                let sample = data[start + bin];
                                let weight: T = rfft_adjoint_weight(self.nfft, bin);
                                grad_vec[bin] = if is_packed_endpoint(self.nfft, bin) {
                                    Complex::new(sample.re, T::zero())
                                } else {
                                    sample * weight
                                };
                            }
                        } else {
                            for bin in 0..self.n_bins() {
                                let sample = grad_3d[[b, bin, c]];
                                let weight: T = rfft_adjoint_weight(self.nfft, bin);
                                grad_vec[bin] = if is_packed_endpoint(self.nfft, bin) {
                                    Complex::new(sample.re, T::zero())
                                } else {
                                    sample * weight
                                };
                            }
                        }
                        c2r.process_with_scratch(grad_vec, grad_time, scratch)?;
                        for t in 0..self.nfft {
                            grad_input[[b, t, c]] =
                                Complex::new(grad_time[t] * self.envelope[t], T::zero());
                        }
                        Ok::<(), AutodiffError>(())
                    },
                )?;
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

    fn parameters(&self) -> Vec<&ArrayD<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<T>> {
        vec![]
    }

    fn gradients(&self) -> Vec<&ArrayD<T>> {
        vec![]
    }

    fn zero_grad(&mut self) {}
}

/// Complex-to-real inverse FFT with an exponential anti-aliasing envelope.
#[derive(Debug, Clone)]
pub struct IfftAntiAlias<T = f64> {
    pub nfft: usize,
    pub channels: usize,
    pub alias_decay_db: T,
    pub gamma: T,
    pub envelope: Vec<T>,
}

impl<T: FftScalar> IfftAntiAlias<T> {
    /// Create a new single-channel anti-aliased inverse FFT module.
    ///
    /// The envelope decays by `alias_decay_db` dB across the FFT window.
    #[must_use]
    pub fn new(nfft: usize, alias_decay_db: T) -> Self {
        Self::with_channels(nfft, 1, alias_decay_db)
    }

    /// Create a new anti-aliased inverse FFT module for `channels` parallel channels.
    ///
    /// # Panics
    ///
    /// Panics if the FFT size or channel count is zero, or if the decay is not finite.
    #[must_use]
    pub fn with_channels(nfft: usize, channels: usize, alias_decay_db: T) -> Self {
        validate_fft_config(nfft, channels);
        assert!(
            alias_decay_db.is_finite(),
            "IfftAntiAlias: alias_decay_db must be finite"
        );
        let gamma = fconst::<T>(10.0)
            .powf(-alias_decay_db.abs() / (fconst::<T>(20.0) * fconst::<T>(nfft as f64)));

        let mut envelope = Vec::with_capacity(nfft);
        let mut value = T::one();
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

impl<T: FftScalar> DiffModule<T> for IfftAntiAlias<T> {
    fn forward(&self, input: &DiffTensor<T>) -> Result<DiffTensor<T>, AutodiffError> {
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

        let plans = T::shared_plans(self.nfft);
        let c2r = &plans.inverse;

        let input_3d = input
            .data
            .as_standard_layout()
            .into_shape_with_order((batch, n_bins, channels))
            .map_err(|e| {
                AutodiffError::Message(format!("IfftAntiAlias: failed to reshape input: {e}"))
            })?;
        let mut output = ArrayD::zeros(IxDyn(&[batch, self.nfft, channels]));
        let scale = fconst::<T>(self.nfft as f64);

        for b in 0..batch {
            for c in 0..channels {
                T::with_fft_buffers(
                    self.nfft,
                    c2r.get_scratch_len(),
                    |time, input_vec, scratch| {
                        for bin in 0..self.n_bins() {
                            input_vec[bin] = input_3d[[b, bin, c]];
                        }
                        c2r.process_with_scratch(input_vec, time, scratch)?;
                        for t in 0..self.nfft {
                            output[[b, t, c]] =
                                Complex::new(time[t] * self.envelope[t] / scale, T::zero());
                        }
                        Ok::<(), AutodiffError>(())
                    },
                )?;
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
        _input: &DiffTensor<T>,
        _output: &DiffTensor<T>,
        grad_output: &DiffTensor<T>,
    ) -> Result<DiffTensor<T>, AutodiffError> {
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

        let plans = T::shared_plans(self.nfft);
        let r2c = &plans.forward;

        let grad_3d = grad_output
            .data
            .as_standard_layout()
            .into_shape_with_order((batch, time, channels))
            .map_err(|e| {
                AutodiffError::Message(format!("IfftAntiAlias: failed to reshape grad: {e}"))
            })?;
        let mut grad_input = ArrayD::zeros(IxDyn(&[batch, self.n_bins(), channels]));

        for b in 0..batch {
            for c in 0..channels {
                T::with_fft_buffers(
                    self.nfft,
                    r2c.get_scratch_len(),
                    |grad_vec, spectrum, scratch| {
                        if channels == 1
                            && let Some(data) = grad_3d.as_slice()
                        {
                            let start = b * self.nfft;
                            for t in 0..self.nfft {
                                grad_vec[t] = data[start + t].re * self.envelope[t];
                            }
                        } else {
                            for t in 0..self.nfft {
                                grad_vec[t] = grad_3d[[b, t, c]].re * self.envelope[t];
                            }
                        }
                        r2c.process_with_scratch(grad_vec, spectrum, scratch)?;
                        for (bin, sample) in spectrum.iter().enumerate() {
                            let weight: T = irfft_adjoint_weight(self.nfft, bin);
                            grad_input[[b, bin, c]] = if is_packed_endpoint(self.nfft, bin) {
                                Complex::new(sample.re * weight, T::zero())
                            } else {
                                *sample * weight
                            };
                        }
                        Ok::<(), AutodiffError>(())
                    },
                )?;
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

    fn parameters(&self) -> Vec<&ArrayD<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<T>> {
        vec![]
    }

    fn gradients(&self) -> Vec<&ArrayD<T>> {
        vec![]
    }

    fn zero_grad(&mut self) {}
}

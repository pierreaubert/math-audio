use ndarray::{ArrayD, IxDyn};
use num_complex::Complex;
use num_traits::Zero;

use crate::error::AutodiffError;
use crate::tensor::DiffTensor;

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Initial FNV-1a hash state for cache-validation fingerprints.
///
/// Fingerprints only guard internal caches against stale reuse; they are not
/// cryptographic hashes and must not be used where collision resistance
/// matters.
#[inline]
pub(crate) const fn fnv1a_init() -> u64 {
    FNV_OFFSET_BASIS
}

/// Fold one `u64` into an FNV-1a hash state.
#[inline]
pub(crate) fn fnv1a_step(hash: u64, value: u64) -> u64 {
    (hash ^ value).wrapping_mul(FNV_PRIME)
}

/// Cast an `f64` constant to generic float code. Infallible for `f32`/`f64`,
/// the only types modules are implemented for.
pub(crate) fn fconst<T: num_traits::NumCast>(x: f64) -> T {
    num_traits::NumCast::from(x).expect("audio constants fit f32/f64")
}

/// Cast a generic scalar to `f64` for cross-crate calls. Infallible for
/// `f32`/`f64`, the only types modules are implemented for.
pub(crate) fn to_f64<T: num_traits::NumCast>(x: T) -> f64 {
    num_traits::NumCast::from(x).expect("f32/f64 convert to f64")
}

/// Bit representation for cache-validation fingerprints over generic floats.
pub trait HashBits: Copy {
    /// Return the value's bit pattern widened to `u64`.
    fn hash_bits(self) -> u64;
}

impl HashBits for f32 {
    #[inline]
    fn hash_bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}

impl HashBits for f64 {
    #[inline]
    fn hash_bits(self) -> u64 {
        self.to_bits()
    }
}

/// Combined bound for differentiable-module scalar types: float arithmetic,
/// in-place ops, and fingerprint hashing. Implemented for `f32`/`f64`.
pub trait Scalar:
    num_traits::Float
    + num_traits::NumAssign
    + HashBits
    + std::iter::Sum
    + std::fmt::Display
    + std::fmt::Debug
{
}

impl<
    T: num_traits::Float
        + num_traits::NumAssign
        + HashBits
        + std::iter::Sum
        + std::fmt::Display
        + std::fmt::Debug,
> Scalar for T
{
}

/// Resize `tensor` to `shape` (zeroed) when its shape differs, otherwise zero
/// it in place. Used by `*_into` overrides to reuse caller-provided buffers.
pub(crate) fn reset_buffer<T>(tensor: &mut DiffTensor<T>, shape: &[usize])
where
    T: Clone + num_traits::Num,
{
    if tensor.data.shape() == shape {
        tensor.data.fill(Complex::zero());
    } else {
        tensor.data = ArrayD::from_elem(IxDyn(shape), Complex::zero());
    }
}

pub(crate) fn validate_spectral_gradient_shape(
    name: &str,
    input_shape: &[usize],
    grad_shape: &[usize],
    output_channels: usize,
) -> Result<(), AutodiffError> {
    if input_shape.len() < 3 || grad_shape.len() != input_shape.len() {
        return Err(AutodiffError::Message(format!(
            "{name}: input and grad_output must have the same rank of at least 3, got {input_shape:?} and {grad_shape:?}"
        )));
    }
    if grad_shape[2] != output_channels {
        return Err(AutodiffError::Message(format!(
            "{name}: expected {output_channels} output channels, got {}",
            grad_shape[2]
        )));
    }
    for axis in 0..input_shape.len() {
        if axis != 2 && input_shape[axis] != grad_shape[axis] {
            return Err(AutodiffError::Message(format!(
                "{name}: input shape {input_shape:?} and grad_output shape {grad_shape:?} differ at axis {axis}"
            )));
        }
    }
    Ok(())
}

/// A differentiable frequency-domain audio module.
pub trait DiffModule<T> {
    /// Forward pass: compute output spectrum from input spectrum.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying operation fails (for example, an FFT
    /// processing error).
    fn forward(&self, input: &DiffTensor<T>) -> Result<DiffTensor<T>, AutodiffError>;

    /// Accumulate gradients of the loss w.r.t. this module's parameters.
    /// `grad_output` is dLoss/dOutput.
    /// Returns dLoss/dInput.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying operation fails (for example, an FFT
    /// processing error).
    fn backward(
        &mut self,
        input: &DiffTensor<T>,
        output: &DiffTensor<T>,
        grad_output: &DiffTensor<T>,
    ) -> Result<DiffTensor<T>, AutodiffError>;

    /// Forward pass writing into a caller-provided buffer.
    ///
    /// `out` is resized (and zeroed) when its shape is wrong, so hot loops can
    /// reuse one allocation across calls. The default implementation allocates
    /// via [`forward`](Self::forward); override to reuse `out`'s storage.
    ///
    /// # Errors
    ///
    /// Same conditions as [`forward`](Self::forward).
    fn forward_into(
        &self,
        input: &DiffTensor<T>,
        out: &mut DiffTensor<T>,
    ) -> Result<(), AutodiffError> {
        *out = self.forward(input)?;
        Ok(())
    }

    /// Backward pass writing dLoss/dInput into a caller-provided buffer.
    ///
    /// `grad_input_out` is resized (and zeroed) when its shape is wrong, so
    /// hot loops can reuse one allocation across calls. The default
    /// implementation allocates via [`backward`](Self::backward); override to
    /// reuse `grad_input_out`'s storage.
    ///
    /// # Errors
    ///
    /// Same conditions as [`backward`](Self::backward).
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors backward plus the out-buffer"
    )]
    fn backward_into(
        &mut self,
        input: &DiffTensor<T>,
        output: &DiffTensor<T>,
        grad_output: &DiffTensor<T>,
        grad_input_out: &mut DiffTensor<T>,
    ) -> Result<(), AutodiffError> {
        *grad_input_out = self.backward(input, output, grad_output)?;
        Ok(())
    }

    /// Backward pass that only accumulates parameter gradients.
    ///
    /// Equivalent to [`backward`](Self::backward) except the dLoss/dInput
    /// tensor is neither computed nor returned, for callers that provably
    /// discard it. The default implementation calls `backward` and drops the
    /// result; override to skip the `grad_input` computation.
    ///
    /// # Errors
    ///
    /// Same conditions as [`backward`](Self::backward).
    fn backward_params_only(
        &mut self,
        input: &DiffTensor<T>,
        output: &DiffTensor<T>,
        grad_output: &DiffTensor<T>,
    ) -> Result<(), AutodiffError> {
        let _ = self.backward(input, output, grad_output)?;
        Ok(())
    }

    /// Number of input channels expected by this module.
    fn input_channels(&self) -> usize;

    /// Number of output channels produced by this module.
    fn output_channels(&self) -> usize;

    /// Number of FFT frequency bins (`nfft/2+1`).
    fn n_bins(&self) -> usize;

    /// Return references to this module's parameter tensors.
    fn parameters(&self) -> Vec<&ArrayD<T>>;

    /// Return mutable references to this module's parameter tensors.
    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<T>>;

    /// Return references to this module's accumulated parameter gradients.
    fn gradients(&self) -> Vec<&ArrayD<T>>;

    /// Zero all accumulated parameter gradients.
    fn zero_grad(&mut self);
}

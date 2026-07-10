use std::any::Any;

use crate::error::AutodiffError;
use crate::tensor::DiffTensor;

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

    /// Number of input channels expected by this module.
    fn input_channels(&self) -> usize;

    /// Number of output channels produced by this module.
    fn output_channels(&self) -> usize;

    /// Number of FFT frequency bins (`nfft/2+1`).
    fn n_bins(&self) -> usize;

    /// Return a reference to this module as `&dyn Any`.
    ///
    /// Enables downcasting from `Box<dyn DiffModule<f64>>` to concrete types
    /// for parameter inspection and gradient zeroing.
    fn as_any(&self) -> &dyn Any
    where
        T: 'static;

    /// Return a mutable reference to this module as `&mut dyn Any`.
    fn as_any_mut(&mut self) -> &mut dyn Any
    where
        T: 'static;
}

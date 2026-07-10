use ndarray::ArrayD;

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

    /// Return references to this module's parameter tensors.
    fn parameters(&self) -> Vec<&ArrayD<f64>>;

    /// Return mutable references to this module's parameter tensors.
    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<f64>>;

    /// Return references to this module's accumulated parameter gradients.
    fn gradients(&self) -> Vec<&ArrayD<f64>>;

    /// Zero all accumulated parameter gradients.
    fn zero_grad(&mut self);
}

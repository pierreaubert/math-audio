use crate::tensor::DiffTensor;

/// A differentiable frequency-domain audio module.
pub trait DiffModule<T> {
    /// Forward pass: compute output spectrum from input spectrum.
    fn forward(&self, input: &DiffTensor<T>) -> DiffTensor<T>;

    /// Accumulate gradients of the loss w.r.t. this module's parameters.
    /// `grad_output` is dLoss/dOutput.
    /// Returns dLoss/dInput.
    fn backward(
        &mut self,
        input: &DiffTensor<T>,
        output: &DiffTensor<T>,
        grad_output: &DiffTensor<T>,
    ) -> DiffTensor<T>;

    /// Number of input channels expected by this module.
    fn input_channels(&self) -> usize;

    /// Number of output channels produced by this module.
    fn output_channels(&self) -> usize;

    /// Number of FFT frequency bins (`nfft/2+1`).
    fn n_bins(&self) -> usize;
}

//! Optimizers for differentiable DSP parameters.

use ndarray::ArrayD;

use crate::error::AutodiffError;

/// Stochastic gradient descent optimizer with a constant learning rate.
#[derive(Debug, Clone)]
pub struct Sgd {
    /// Learning rate.
    pub lr: f64,
}

impl Sgd {
    /// Create a new SGD optimizer with the given learning rate.
    #[must_use]
    pub const fn new(lr: f64) -> Self {
        Self { lr }
    }

    /// Apply a single SGD update: `param -= lr * grad`.
    ///
    /// # Errors
    ///
    /// Returns an error if the parameter and gradient counts or shapes differ.
    pub fn step(
        &self,
        params: &mut [&mut ArrayD<f64>],
        grads: &[&ArrayD<f64>],
    ) -> Result<(), AutodiffError> {
        if params.len() != grads.len() {
            return Err(AutodiffError::Message(format!(
                "SGD: received {} parameters but {} gradients",
                params.len(),
                grads.len()
            )));
        }
        for (index, (param, grad)) in params.iter().zip(grads).enumerate() {
            if param.shape() != grad.shape() {
                return Err(AutodiffError::Message(format!(
                    "SGD: parameter {index} shape {:?} does not match gradient shape {:?}",
                    param.shape(),
                    grad.shape()
                )));
            }
        }
        for (p, g) in params.iter_mut().zip(grads) {
            **p -= &(*g * self.lr);
        }
        Ok(())
    }
}

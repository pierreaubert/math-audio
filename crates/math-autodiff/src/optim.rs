//! Optimizers for differentiable DSP parameters.

use ndarray::ArrayD;

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
    pub fn step(&self, params: &mut [ArrayD<f64>], grads: &[ArrayD<f64>]) {
        for (p, g) in params.iter_mut().zip(grads) {
            *p -= &(g * self.lr);
        }
    }
}

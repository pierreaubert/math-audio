use ndarray::{Array, ArrayD, Dimension};
use num_complex::Complex;

/// Gradient of a scalar loss with respect to a parameter tensor.
#[derive(Debug, Clone)]
pub struct Gradient<T = f64> {
    pub data: ArrayD<Complex<T>>,
}

impl<T: num_traits::Float> Gradient<T> {
    /// Create a gradient tensor filled with zeros.
    pub fn zeros<D: Dimension>(shape: D) -> Self {
        Self {
            data: Array::zeros(shape).into_dyn(),
        }
    }
}

/// Trait for modules/variables that expose differentiable parameters.
pub trait Parameters<T> {
    /// Return an immutable view of each parameter tensor.
    fn parameters(&self) -> Vec<&ArrayD<Complex<T>>>;

    /// Return a mutable view of each parameter tensor.
    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<Complex<T>>>;
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn zeros_creates_correct_shape_and_zero_filled() {
        let g = Gradient::<f64>::zeros(ndarray::Ix2(4, 5));
        assert_eq!(g.data.shape(), &[4, 5]);
        for c in g.data.iter() {
            assert_abs_diff_eq!(c.re, 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(c.im, 0.0, epsilon = 1e-12);
        }
    }
}

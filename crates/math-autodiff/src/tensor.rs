use ndarray::{Array, ArrayD, Dimension};
use num_complex::Complex;

/// Complex-valued tensor used for frequency-domain signals and responses.
/// Supports dynamic rank so that `(B, M, N)`, `(M, N, N)`, and `(M, N)`
/// can be represented uniformly.
#[derive(Debug, Clone)]
pub struct DiffTensor<T = f64> {
    pub data: ArrayD<Complex<T>>,
}

impl<T: num_traits::Float> DiffTensor<T> {
    /// Create a tensor filled with zeros.
    pub fn zeros<D: Dimension>(shape: D) -> Self {
        Self {
            data: Array::zeros(shape).into_dyn(),
        }
    }

    /// Create a tensor from an existing ndarray.
    pub fn from_array<D: Dimension>(data: Array<Complex<T>, D>) -> Self {
        Self {
            data: data.into_dyn(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn zeros_creates_correct_shape_and_zero_filled() {
        let t = DiffTensor::<f64>::zeros(ndarray::Ix2(2, 3));
        assert_eq!(t.data.shape(), &[2, 3]);
        for c in t.data.iter() {
            assert_abs_diff_eq!(c.re, 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(c.im, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn from_array_preserves_shape_and_values() {
        let values = vec![Complex::new(1.0, -1.0); 6];
        let arr = Array::from_shape_vec(ndarray::Ix2(2, 3), values).unwrap();
        let t = DiffTensor::<f64>::from_array(arr.clone());
        assert_eq!(t.data.shape(), &[2, 3]);
        for (actual, expected) in t.data.iter().zip(arr.iter()) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-12);
        }
    }
}

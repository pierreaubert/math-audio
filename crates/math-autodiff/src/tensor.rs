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

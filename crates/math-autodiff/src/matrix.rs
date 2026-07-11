//! Learnable frequency-domain matrix modules.

#![allow(
    clippy::cast_precision_loss,
    reason = "nfft is an audio buffer length that fits exactly in f64 for practical values"
)]
#![allow(
    clippy::uninlined_format_args,
    reason = "format strings are clearer with explicit arguments in error messages"
)]

use nalgebra::DMatrix;
use ndarray::{Array2, ArrayD, ArrayView2, ArrayViewMut2, Axis, IxDyn};
use num_complex::Complex;

use crate::error::AutodiffError;
use crate::module::DiffModule;
use crate::tensor::DiffTensor;

fn view2<'a>(param: &'a ArrayD<f64>, name: &str) -> Result<ArrayView2<'a, f64>, AutodiffError> {
    let shape = param.shape();
    if shape.len() != 2 {
        return Err(AutodiffError::Message(format!(
            "{name}: expected 2-D parameter tensor, got shape {:?}",
            shape
        )));
    }
    let (n_out, n_in) = (shape[0], shape[1]);
    param
        .view()
        .into_shape_with_order((n_out, n_in))
        .map_err(|e| AutodiffError::Message(format!("{name}: failed to reshape param: {e}")))
}

fn view2_mut<'a>(
    param: &'a mut ArrayD<f64>,
    name: &str,
) -> Result<ArrayViewMut2<'a, f64>, AutodiffError> {
    let shape = param.shape();
    if shape.len() != 2 {
        return Err(AutodiffError::Message(format!(
            "{name}: expected 2-D parameter gradient tensor, got shape {:?}",
            shape
        )));
    }
    let (n_out, n_in) = (shape[0], shape[1]);
    param
        .view_mut()
        .into_shape_with_order((n_out, n_in))
        .map_err(|e| AutodiffError::Message(format!("{name}: failed to reshape param_grad: {e}")))
}

fn ndarray2_to_dmatrix(mat: &Array2<f64>) -> DMatrix<f64> {
    let data: Vec<f64> = mat.iter().copied().collect();
    DMatrix::from_row_slice(mat.nrows(), mat.ncols(), &data)
}

fn dmatrix_to_ndarray2(mat: &DMatrix<f64>) -> Array2<f64> {
    let mut out = Array2::zeros((mat.nrows(), mat.ncols()));
    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            out[[i, j]] = mat[(i, j)];
        }
    }
    out
}

fn matrix_exp_skew_view(raw: &ArrayView2<f64>) -> Array2<f64> {
    let skew = raw.to_owned() - raw.t();
    let dm = ndarray2_to_dmatrix(&skew);
    let exp = dm.exp();
    dmatrix_to_ndarray2(&exp)
}

/// Compute the derivative of the scalar loss w.r.t. the real matrix `M`.
///
/// This is shared by both `Dense` and `Orthogonal` parameterizations: for
/// `Dense` it is the parameter gradient directly, while `Orthogonal` applies
/// the chain rule through the skew-symmetric exponential map.
fn compute_dl_dm(
    grad_output: &DiffTensor<f64>,
    input: &DiffTensor<f64>,
    n_out: usize,
    n_in: usize,
) -> Array2<f64> {
    let mut dl_dm = Array2::<f64>::zeros((n_out, n_in));
    for out_ch in 0..n_out {
        let grad_slice = grad_output.data.index_axis(Axis(2), out_ch);
        for in_ch in 0..n_in {
            let input_slice = input.data.index_axis(Axis(2), in_ch);
            let prod = &grad_slice * &input_slice.mapv(|x| x.conj());
            dl_dm[[out_ch, in_ch]] = prod.sum().re;
        }
    }
    dl_dm
}

/// Compute the matrix exponential of the skew-symmetric matrix `raw - raw.t()`.
///
/// The result is an orthogonal matrix.
#[must_use]
pub fn matrix_exp_skew(raw: &Array2<f64>) -> Array2<f64> {
    matrix_exp_skew_view(&raw.view())
}

/// Parameterization type for the `Matrix` module.
#[derive(Debug, Clone, Copy)]
pub enum MatrixType {
    /// Unconstrained real matrix.
    Dense,
    /// Orthogonal matrix parameterized as the exponential of a skew-symmetric
    /// matrix.
    Orthogonal,
}

/// Frequency-independent learnable matrix module.
#[derive(Debug, Clone)]
pub struct Matrix {
    /// FFT length.
    pub nfft: usize,
    /// Number of output channels.
    pub n_out: usize,
    /// Number of input channels.
    pub n_in: usize,
    /// Parameterization type.
    pub matrix_type: MatrixType,
    /// Raw parameters, shape `(n_out, n_in)`.
    pub param: ArrayD<f64>,
    /// Accumulated parameter gradients, same shape as `param`.
    pub param_grad: ArrayD<f64>,
}

impl Matrix {
    /// Create a new learnable matrix module.
    ///
    /// # Errors
    ///
    /// Returns an error if `nfft` is zero, or if `MatrixType::Orthogonal` is
    /// requested with non-square dimensions.
    pub fn new(
        nfft: usize,
        n_out: usize,
        n_in: usize,
        matrix_type: MatrixType,
    ) -> Result<Self, AutodiffError> {
        if nfft == 0 {
            return Err(AutodiffError::Message(
                "Matrix: nfft must be greater than 0".to_string(),
            ));
        }
        let param_shape = match matrix_type {
            MatrixType::Dense => vec![n_out, n_in],
            MatrixType::Orthogonal => {
                if n_out != n_in {
                    return Err(AutodiffError::Message(
                        "Matrix::Orthogonal requires square shape".to_string(),
                    ));
                }
                vec![n_out, n_in]
            }
        };
        Ok(Self {
            nfft,
            n_out,
            n_in,
            matrix_type,
            param: ArrayD::zeros(IxDyn(&param_shape)),
            param_grad: ArrayD::zeros(IxDyn(&param_shape)),
        })
    }

    fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }

    /// Build the real frequency-independent matrix `M`.
    ///
    /// # Errors
    ///
    /// Returns an error if the parameter tensor has an unexpected shape.
    pub fn build_matrix(&self) -> Result<Array2<Complex<f64>>, AutodiffError> {
        match self.matrix_type {
            MatrixType::Dense => {
                let v = view2(&self.param, "Matrix")?;
                Ok(v.mapv(|x| Complex::new(x, 0.0)))
            }
            MatrixType::Orthogonal => {
                let v = view2(&self.param, "Matrix")?;
                let orth = matrix_exp_skew_view(&v);
                Ok(orth.mapv(|x| Complex::new(x, 0.0)))
            }
        }
    }
}

impl DiffModule<f64> for Matrix {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "Matrix::forward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let n_bins = input_shape[1];
        let n_in = input_shape[2];
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "Matrix::forward: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if n_in != self.n_in {
            return Err(AutodiffError::Message(format!(
                "Matrix::forward: expected {} input channels, got {}",
                self.n_in, n_in
            )));
        }

        let m = self.build_matrix()?;
        let mut output_shape = input_shape.to_vec();
        output_shape[2] = self.n_out;
        let mut output = ArrayD::zeros(IxDyn(&output_shape));

        for out_ch in 0..self.n_out {
            for in_ch in 0..n_in {
                let h = m[[out_ch, in_ch]];
                let input_slice = input.data.index_axis(Axis(2), in_ch);
                let mut output_slice = output.index_axis_mut(Axis(2), out_ch);
                output_slice += &input_slice.mapv(|x| x * h);
            }
        }
        Ok(DiffTensor::from_array(output))
    }

    fn backward(
        &mut self,
        input: &DiffTensor<f64>,
        _output: &DiffTensor<f64>,
        grad_output: &DiffTensor<f64>,
    ) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        let grad_shape = grad_output.data.shape();
        if input_shape.len() < 3 || grad_shape.len() < 3 {
            return Err(AutodiffError::Message(
                "Matrix::backward: input and grad_output must have at least 3 dimensions"
                    .to_string(),
            ));
        }
        if input_shape.len() != grad_shape.len() {
            return Err(AutodiffError::Message(format!(
                "Matrix::backward: input and grad_output must have same rank, got {:?} and {:?}",
                input_shape, grad_shape
            )));
        }
        let n_bins = input_shape[1];
        if input_shape[2] != self.n_in {
            return Err(AutodiffError::Message(format!(
                "Matrix::backward: expected {} input channels, got {}",
                self.n_in, input_shape[2]
            )));
        }
        if grad_shape[1] != n_bins || grad_shape[2] != self.n_out {
            return Err(AutodiffError::Message(format!(
                "Matrix::backward: grad_output shape {:?} incompatible with (..., {}, {})",
                grad_shape, n_bins, self.n_out
            )));
        }
        for (dim, (&g, &inp)) in grad_shape[..grad_shape.len() - 2]
            .iter()
            .zip(input_shape[..input_shape.len() - 2].iter())
            .enumerate()
        {
            if g != inp {
                return Err(AutodiffError::Message(format!(
                    "Matrix::backward: batch dimension {} mismatch: grad_output {:?} vs input {:?}",
                    dim, grad_shape, input_shape
                )));
            }
        }

        let m = self.build_matrix()?;
        let mut grad_input = ArrayD::zeros(IxDyn(input_shape));

        // Dense gradient is straightforward.
        match self.matrix_type {
            MatrixType::Dense => {
                let dl_dm = compute_dl_dm(grad_output, input, self.n_out, self.n_in);
                let mut pg = view2_mut(&mut self.param_grad, "Matrix")?;
                pg += &dl_dm;
            }
            MatrixType::Orthogonal => {
                // Compute dL/dM (same shape as M).
                let dl_dm = compute_dl_dm(grad_output, input, self.n_out, self.n_in);
                // Numerical Jacobian of M w.r.t. raw parameters.
                let v = view2(&self.param, "Matrix")?;
                let eps = 1e-7;
                for i in 0..self.n_out {
                    for j in 0..self.n_in {
                        let mut v_plus = v.to_owned();
                        v_plus[[i, j]] += eps;
                        let m_plus = matrix_exp_skew_view(&v_plus.view());
                        let mut v_minus = v.to_owned();
                        v_minus[[i, j]] -= eps;
                        let m_minus = matrix_exp_skew_view(&v_minus.view());
                        let deriv = (&m_plus - &m_minus) / (2.0 * eps);
                        let grad = (&deriv * &dl_dm).sum();
                        self.param_grad[[i, j]] += grad;
                    }
                }
            }
        }

        for in_ch in 0..self.n_in {
            for out_ch in 0..self.n_out {
                let h = m[[out_ch, in_ch]].conj();
                let grad_slice = grad_output.data.index_axis(Axis(2), out_ch);
                let mut input_grad_slice = grad_input.index_axis_mut(Axis(2), in_ch);
                input_grad_slice += &grad_slice.mapv(|x| x * h);
            }
        }

        Ok(DiffTensor::from_array(grad_input))
    }

    fn input_channels(&self) -> usize {
        self.n_in
    }
    fn output_channels(&self) -> usize {
        self.n_out
    }
    fn n_bins(&self) -> usize {
        self.n_bins()
    }
    fn parameters(&self) -> Vec<&ArrayD<f64>> {
        vec![&self.param]
    }
    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<f64>> {
        vec![&mut self.param]
    }
    fn gradients(&self) -> Vec<&ArrayD<f64>> {
        vec![&self.param_grad]
    }
    fn zero_grad(&mut self) {
        self.param_grad.fill(0.0);
    }
}

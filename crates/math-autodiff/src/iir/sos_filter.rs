//! Generic differentiable cascade of second-order sections.

#![allow(
    clippy::cast_precision_loss,
    reason = "nfft is an audio buffer length that fits exactly in f64 for practical values"
)]
#![allow(
    clippy::similar_names,
    reason = "coefficient derivative names are intentionally paired (dh_db/dh_da)"
)]
#![allow(
    clippy::type_complexity,
    reason = "SOS coefficient tensors are inherently multi-dimensional"
)]
#![allow(
    clippy::uninlined_format_args,
    reason = "format strings are clearer with explicit arguments in error messages"
)]

use ndarray::{Array3, Array4, ArrayD, Axis, IxDyn};
use num_complex::Complex;

use crate::error::AutodiffError;
use crate::iir::response::{
    SosFrequencyBasis, sos_coefficient_vjp_with_basis, sos_frequency_response,
};
use crate::module::{DiffModule, validate_spectral_gradient_shape};
use crate::tensor::DiffTensor;

/// Split the packed SOS parameter tensor into separate b and a coefficient tensors.
///
/// Input shape: `(K, 6, N_out, N_in)` where the 6 slots are `[b0, b1, b2, a0, a1, a2]`.
/// Output shapes: `(K, 3, N_out, N_in)` for `b` and `(K, 3, N_out, N_in)` for `a`.
fn split_param(
    param: &ArrayD<f64>,
) -> Result<(Array4<Complex<f64>>, Array4<Complex<f64>>), AutodiffError> {
    let shape = param.shape();
    if shape.len() != 4 || shape[1] != 6 {
        return Err(AutodiffError::Message(format!(
            "SosFilter: expected param shape (K, 6, N_out, N_in), got {:?}",
            shape
        )));
    }
    let (k, n_out, n_in) = (shape[0], shape[2], shape[3]);
    let mut b = Array4::zeros((k, 3, n_out, n_in));
    let mut a = Array4::zeros((k, 3, n_out, n_in));
    for section in 0..k {
        for out_ch in 0..n_out {
            for in_ch in 0..n_in {
                for tap in 0..3 {
                    b[[section, tap, out_ch, in_ch]] =
                        Complex::new(param[[section, tap, out_ch, in_ch]], 0.0);
                    a[[section, tap, out_ch, in_ch]] =
                        Complex::new(param[[section, 3 + tap, out_ch, in_ch]], 0.0);
                }
            }
        }
    }
    Ok((b, a))
}

fn validate_stable_denominators(a: &Array4<Complex<f64>>) -> Result<(), AutodiffError> {
    for section in 0..a.dim().0 {
        for out_ch in 0..a.dim().2 {
            for in_ch in 0..a.dim().3 {
                let a0 = a[[section, 0, out_ch, in_ch]];
                let a1 = a[[section, 1, out_ch, in_ch]];
                let a2 = a[[section, 2, out_ch, in_ch]];
                if !a0.is_finite() || a0.norm() <= f64::EPSILON {
                    return Err(AutodiffError::Message(format!(
                        "SosFilter: denominator section {section} has an invalid leading coefficient"
                    )));
                }
                let discriminant = (a1 * a1 - Complex::from(4.0) * a0 * a2).sqrt();
                let denominator = Complex::from(2.0) * a0;
                let pole_a = (-a1 + discriminant) / denominator;
                let pole_b = (-a1 - discriminant) / denominator;
                if !pole_a.is_finite()
                    || !pole_b.is_finite()
                    || pole_a.norm() >= 1.0
                    || pole_b.norm() >= 1.0
                {
                    return Err(AutodiffError::Message(format!(
                        "SosFilter: denominator section {section} has an unstable pole"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Generic cascade of second-order sections with learnable coefficients.
///
/// The raw-coefficient interface intentionally exposes pole stability to the
/// optimizer, but forward and backward reject non-finite or unit-circle poles
/// with [`AutodiffError`] instead of producing NaNs.
#[derive(Debug, Clone)]
pub struct SosFilter {
    pub nfft: usize,
    pub n_sections: usize,
    pub n_out: usize,
    pub n_in: usize,
    pub alias_decay_db: f64,
    pub param: ArrayD<f64>,
    pub param_grad: ArrayD<f64>,
    work_h: Array3<Complex<f64>>,
    work_b_response: Array4<Complex<f64>>,
    work_a_response: Array4<Complex<f64>>,
    work_dl_dh: Array3<Complex<f64>>,
    work_grad_input: ArrayD<Complex<f64>>,
}

impl SosFilter {
    /// Create a finite zero-response SOS filter with unit denominator leading
    /// coefficients and zero-initialized gradients.
    ///
    /// # Errors
    ///
    /// Returns an error if `nfft` or `n_sections` is zero.
    pub fn new(
        nfft: usize,
        n_sections: usize,
        n_out: usize,
        n_in: usize,
        alias_decay_db: f64,
    ) -> Result<Self, AutodiffError> {
        if nfft == 0 {
            return Err(AutodiffError::Message(
                "SosFilter: nfft must be greater than 0".to_string(),
            ));
        }
        if n_sections == 0 {
            return Err(AutodiffError::Message(
                "SosFilter: n_sections must be greater than 0".to_string(),
            ));
        }
        if n_out == 0 {
            return Err(AutodiffError::Message(
                "SosFilter: n_out must be greater than 0".to_string(),
            ));
        }
        if n_in == 0 {
            return Err(AutodiffError::Message(
                "SosFilter: n_in must be greater than 0".to_string(),
            ));
        }
        if !alias_decay_db.is_finite() {
            return Err(AutodiffError::Message(
                "SosFilter: alias_decay_db must be finite".to_string(),
            ));
        }
        let mut param = ArrayD::zeros(IxDyn(&[n_sections, 6, n_out, n_in]));
        for section in 0..n_sections {
            for out_ch in 0..n_out {
                for in_ch in 0..n_in {
                    param[[section, 3, out_ch, in_ch]] = 1.0;
                }
            }
        }
        Ok(Self {
            nfft,
            n_sections,
            n_out,
            n_in,
            alias_decay_db,
            param,
            param_grad: ArrayD::zeros(IxDyn(&[n_sections, 6, n_out, n_in])),
            work_h: Array3::zeros((0, 0, 0)),
            work_b_response: Array4::zeros((0, 0, 0, 0)),
            work_a_response: Array4::zeros((0, 0, 0, 0)),
            work_dl_dh: Array3::zeros((0, 0, 0)),
            work_grad_input: ArrayD::zeros(IxDyn(&[])),
        })
    }

    fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }

    fn gamma(&self) -> [f64; 3] {
        let gamma = 10.0_f64.powf(-self.alias_decay_db.abs() / (20.0 * self.nfft as f64));
        [1.0, gamma, gamma * gamma]
    }
}

impl DiffModule<f64> for SosFilter {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "SosFilter::forward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let n_bins = input_shape[1];
        let n_in = input_shape[2];
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "SosFilter::forward: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if n_in != self.n_in {
            return Err(AutodiffError::Message(format!(
                "SosFilter::forward: expected {} input channels, got {}",
                self.n_in, n_in
            )));
        }

        let (b, a) = split_param(&self.param)?;
        validate_stable_denominators(&a)?;
        let gamma = self.gamma();
        let h = sos_frequency_response(&b, &a, self.nfft, Some(&gamma))?;

        let mut output_shape = input_shape.to_vec();
        output_shape[2] = self.n_out;
        let mut output = ArrayD::zeros(IxDyn(&output_shape));

        for out_ch in 0..self.n_out {
            for in_ch in 0..n_in {
                for bin in 0..n_bins {
                    let h_val = h[[bin, out_ch, in_ch]];
                    let input_slice = input.data.index_axis(Axis(1), bin);
                    let input_bin = input_slice.index_axis(Axis(1), in_ch);
                    let mut output_slice = output.index_axis_mut(Axis(1), bin);
                    let mut output_bin = output_slice.index_axis_mut(Axis(1), out_ch);
                    for (destination, &source) in output_bin.iter_mut().zip(input_bin.iter()) {
                        *destination += source * h_val;
                    }
                }
            }
        }

        Ok(DiffTensor::from_array(output))
    }

    #[allow(clippy::too_many_lines)]
    fn backward(
        &mut self,
        input: &DiffTensor<f64>,
        _output: &DiffTensor<f64>,
        grad_output: &DiffTensor<f64>,
    ) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        let grad_shape = grad_output.data.shape();
        validate_spectral_gradient_shape(
            "SosFilter::backward",
            input_shape,
            grad_shape,
            self.n_out,
        )?;
        let n_bins = input_shape[1];
        let n_in = input_shape[2];
        let n_out = grad_shape[2];
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "SosFilter::backward: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if n_out != self.n_out {
            return Err(AutodiffError::Message(format!(
                "SosFilter::backward: expected {} output channels, got {}",
                self.n_out, n_out
            )));
        }
        if n_in != self.n_in {
            return Err(AutodiffError::Message(format!(
                "SosFilter::backward: expected {} input channels, got {}",
                self.n_in, n_in
            )));
        }

        let (b, a) = split_param(&self.param)?;
        validate_stable_denominators(&a)?;
        let gamma = self.gamma();
        let basis = SosFrequencyBasis::new(self.nfft, &gamma);

        if self.work_dl_dh.dim() != (n_bins, n_out, n_in) {
            self.work_dl_dh = Array3::zeros((n_bins, n_out, n_in));
        }
        if self.work_h.dim() != (n_bins, n_out, n_in) {
            self.work_h = Array3::zeros((n_bins, n_out, n_in));
        }
        if self.work_b_response.dim() != (self.n_sections, n_bins, n_out, n_in) {
            self.work_b_response = Array4::zeros((self.n_sections, n_bins, n_out, n_in));
        }
        if self.work_a_response.dim() != (self.n_sections, n_bins, n_out, n_in) {
            self.work_a_response = Array4::zeros((self.n_sections, n_bins, n_out, n_in));
        }
        if self.work_grad_input.shape() != input_shape {
            self.work_grad_input = ArrayD::zeros(IxDyn(input_shape));
        }
        self.work_dl_dh.fill(Complex::default());
        self.work_grad_input.fill(Complex::default());

        // dL/dH[bin, out, in] = sum_b grad_output[b, bin, out] * conj(input[b, bin, in])
        for bin in 0..n_bins {
            for out_ch in 0..n_out {
                for in_ch in 0..n_in {
                    let grad_slice = grad_output.data.index_axis(Axis(1), bin);
                    let grad_bin = grad_slice.index_axis(Axis(1), out_ch);
                    let input_slice = input.data.index_axis(Axis(1), bin);
                    let input_bin = input_slice.index_axis(Axis(1), in_ch);
                    self.work_dl_dh[[bin, out_ch, in_ch]] = grad_bin
                        .iter()
                        .zip(input_bin.iter())
                        .map(|(g, x)| *g * x.conj())
                        .sum::<Complex<f64>>();
                }
            }
        }

        let (response_db, response_da) = sos_coefficient_vjp_with_basis(
            &b,
            &a,
            &basis,
            &self.work_dl_dh,
            &mut self.work_h,
            &mut self.work_b_response,
            &mut self.work_a_response,
        )?;

        {
            let mut param_grad = self
                .param_grad
                .view_mut()
                .into_shape_with_order((self.n_sections, 6, self.n_out, self.n_in))
                .map_err(|e| {
                    AutodiffError::Message(format!("SosFilter: failed to reshape param_grad: {e}"))
                })?;

            for section in 0..self.n_sections {
                for out_ch in 0..n_out {
                    for in_ch in 0..n_in {
                        for tap in 0..3 {
                            let accum_b = response_db[[section, tap, out_ch, in_ch]];
                            let accum_a = response_da[[section, tap, out_ch, in_ch]];
                            param_grad[[section, tap, out_ch, in_ch]] += accum_b;
                            param_grad[[section, 3 + tap, out_ch, in_ch]] += accum_a;
                        }
                    }
                }
            }
        }
        let h = &self.work_h;
        for in_ch in 0..n_in {
            for out_ch in 0..n_out {
                for bin in 0..n_bins {
                    let h_conj = h[[bin, out_ch, in_ch]].conj();
                    let grad_slice = grad_output.data.index_axis(Axis(1), bin);
                    let grad_bin = grad_slice.index_axis(Axis(1), out_ch);
                    let mut input_grad_slice = self.work_grad_input.index_axis_mut(Axis(1), bin);
                    let mut input_grad_bin = input_grad_slice.index_axis_mut(Axis(1), in_ch);
                    for (destination, &gradient) in input_grad_bin.iter_mut().zip(grad_bin.iter()) {
                        *destination += gradient * h_conj;
                    }
                }
            }
        }

        let grad_input = std::mem::replace(&mut self.work_grad_input, ArrayD::zeros(IxDyn(&[])));
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

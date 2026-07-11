//! Differentiable State Variable Filter (SVF) mapped to a single SOS section.

#![allow(
    clippy::cast_precision_loss,
    reason = "nfft is an audio buffer length that fits exactly in f64 for practical values"
)]
#![allow(
    non_snake_case,
    reason = "`R` is the damping coefficient naming used in the task brief and SVF literature"
)]
#![allow(
    clippy::similar_names,
    reason = "coefficient derivative names are intentionally paired (db_dparam/da_dparam, dl_db/dl_da)"
)]
#![allow(
    clippy::type_complexity,
    reason = "SOS coefficient Jacobians are inherently multi-dimensional"
)]
#![allow(
    clippy::uninlined_format_args,
    reason = "format strings are clearer with explicit arguments in error messages"
)]

use ndarray::{Array4, Array5, ArrayD, IxDyn};
use num_complex::Complex;
use std::f64::consts::PI;

use crate::error::AutodiffError;
use crate::iir::sos_filter::SosFilter;
use crate::module::{DiffModule, validate_spectral_gradient_shape};
use crate::tensor::DiffTensor;

/// SVF filter type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvfType {
    Lowpass,
    Highpass,
    Bandpass,
    Notch,
    Peak,
    Lowshelf,
    Highshelf,
    Allpass,
}

impl SvfType {
    /// Number of physical parameters for this filter type.
    #[must_use]
    pub const fn n_params(self) -> usize {
        match self {
            Self::Peak | Self::Lowshelf | Self::Highshelf => 3,
            _ => 2,
        }
    }
}

/// Coefficients and their physical parameter derivatives for a single
/// SVF-mapped biquad section.
#[derive(Debug, Clone, Copy)]
struct SvfCoeffs {
    b: [f64; 3],
    a: [f64; 3],
    /// `db_dparam[tap][param]` w.r.t. physical parameters (`fc`/`R`/`gain_db`).
    db_dparam: [[f64; 3]; 3],
    /// `da_dparam[tap][param]` w.r.t. physical parameters.
    da_dparam: [[f64; 3]; 3],
}

impl SvfCoeffs {
    fn zeros() -> Self {
        Self {
            b: [0.0; 3],
            a: [0.0; 3],
            db_dparam: [[0.0; 3]; 3],
            da_dparam: [[0.0; 3]; 3],
        }
    }
}

/// Clamp cutoff frequency to the valid open interval `(0, fs/2)`.
#[inline]
fn clamp_fc(fc: f64, fs: f64) -> f64 {
    fc.clamp(1.0, fs * 0.499)
}

/// Clamp damping `R` to a small positive value to avoid division by zero.
#[inline]
fn clamp_r(R: f64) -> f64 {
    R.max(1e-6)
}

/// Convert dB gain to the SVF shelving/peak factor `a = 10^(gain_db/40)`.
#[inline]
fn gain_factor(gain_db: f64) -> f64 {
    10.0_f64.powf(gain_db / 40.0)
}

/// Compute normalized biquad coefficients `(b, a)` that realize the same
/// frequency response as `math-iir-fir::SvfFilter::process`.
///
/// `R` is the damping coefficient (reciprocal of the SVF quality factor `q`).
fn svf_coefficients(
    fc: f64,
    R: f64,
    gain_db: f64,
    fs: f64,
    filter_type: SvfType,
) -> ([f64; 3], [f64; 3]) {
    let fc = clamp_fc(fc, fs);
    let R = clamp_r(R);
    let g = (PI * fc / fs).tan();
    let k = R;
    let a = gain_factor(gain_db);

    // Base numerator polynomials for HP, BP, LP in powers of z^-1.
    let hp = [1.0, -2.0, 1.0];
    let bp = [g, 0.0, -g];
    let lp = [g * g, 2.0 * g * g, g * g];

    // Common denominator (unnormalized).
    let a0 = 1.0 + k * g + g * g;
    let a1_un = -2.0 + 2.0 * g * g;
    let a2_un = 1.0 - k * g + g * g;

    // Mix coefficients for H = mix_hp*HP + mix_bp*BP + mix_lp*LP.
    // Derived from the SVF state-space identity `input = HP + k*BP + LP`.
    let (b_un, denom_un): ([f64; 3], [f64; 3]) = match filter_type {
        SvfType::Lowpass => (lp, [a0, a1_un, a2_un]),
        SvfType::Highpass => (hp, [a0, a1_un, a2_un]),
        SvfType::Bandpass => (bp, [a0, a1_un, a2_un]),
        SvfType::Notch => {
            let b = [hp[0] + lp[0], hp[1] + lp[1], hp[2] + lp[2]];
            (b, [a0, a1_un, a2_un])
        }
        SvfType::Allpass => {
            let b = [
                hp[0] - k * bp[0] + lp[0],
                hp[1] - k * bp[1] + lp[1],
                hp[2] - k * bp[2] + lp[2],
            ];
            (b, [a0, a1_un, a2_un])
        }
        SvfType::Peak => {
            let k_peak = k / a;
            let mix_bp = k * a;
            let b = [
                hp[0] + mix_bp * bp[0] + lp[0],
                hp[1] + mix_bp * bp[1] + lp[1],
                hp[2] + mix_bp * bp[2] + lp[2],
            ];
            let a0_peak = 1.0 + k_peak * g + g * g;
            let a1_peak = -2.0 + 2.0 * g * g;
            let a2_peak = 1.0 - k_peak * g + g * g;
            (b, [a0_peak, a1_peak, a2_peak])
        }
        SvfType::Lowshelf => {
            let g_prime = g * a.sqrt();
            let a2 = a * a;
            let hp_p = [1.0, -2.0, 1.0];
            let bp_p = [g_prime, 0.0, -g_prime];
            let lp_p = [
                g_prime * g_prime,
                2.0 * g_prime * g_prime,
                g_prime * g_prime,
            ];
            // H = HP' + k*a^2*BP' + a^2*LP'.
            let b = [
                hp_p[0] + k * a2 * bp_p[0] + a2 * lp_p[0],
                hp_p[1] + k * a2 * bp_p[1] + a2 * lp_p[1],
                hp_p[2] + k * a2 * bp_p[2] + a2 * lp_p[2],
            ];
            let a0_s = 1.0 + k * g_prime + g_prime * g_prime;
            let a1_s = -2.0 + 2.0 * g_prime * g_prime;
            let a2_s = 1.0 - k * g_prime + g_prime * g_prime;
            (b, [a0_s, a1_s, a2_s])
        }
        SvfType::Highshelf => {
            let g_prime = g / a.sqrt();
            let a2 = a * a;
            let hp_p = [1.0, -2.0, 1.0];
            let bp_p = [g_prime, 0.0, -g_prime];
            let lp_p = [
                g_prime * g_prime,
                2.0 * g_prime * g_prime,
                g_prime * g_prime,
            ];
            // H = a^2*HP' + k*a*(a + 1 - a^2)*BP' + LP'.
            let mix_bp = k * a * (a + 1.0 - a2);
            let b = [
                a2 * hp_p[0] + mix_bp * bp_p[0] + lp_p[0],
                a2 * hp_p[1] + mix_bp * bp_p[1] + lp_p[1],
                a2 * hp_p[2] + mix_bp * bp_p[2] + lp_p[2],
            ];
            let a0_s = 1.0 + k * g_prime + g_prime * g_prime;
            let a1_s = -2.0 + 2.0 * g_prime * g_prime;
            let a2_s = 1.0 - k * g_prime + g_prime * g_prime;
            (b, [a0_s, a1_s, a2_s])
        }
    };

    let a0 = denom_un[0];
    (
        [b_un[0] / a0, b_un[1] / a0, b_un[2] / a0],
        [1.0, denom_un[1] / a0, denom_un[2] / a0],
    )
}

/// Compute coefficients and their physical parameter derivatives using central
/// finite differences.
fn svf_coefficients_with_gradients(
    fc: f64,
    R: f64,
    gain_db: f64,
    fs: f64,
    filter_type: SvfType,
) -> SvfCoeffs {
    let n_params = filter_type.n_params();
    let (b, a) = svf_coefficients(fc, R, gain_db, fs, filter_type);
    let mut coeffs = SvfCoeffs::zeros();
    coeffs.b = b;
    coeffs.a = a;

    for p in 0..n_params {
        let value = match p {
            0 => fc,
            1 => R,
            2 => gain_db,
            _ => unreachable!(),
        };
        let eps = f64::EPSILON.cbrt() * value.abs().max(1.0);
        let (fc_plus, R_plus, gain_plus) = match p {
            0 => (fc + eps, R, gain_db),
            1 => (fc, R + eps, gain_db),
            2 => (fc, R, gain_db + eps),
            _ => unreachable!(),
        };
        let (fc_minus, R_minus, gain_minus) = match p {
            0 => (fc - eps, R, gain_db),
            1 => (fc, R - eps, gain_db),
            2 => (fc, R, gain_db - eps),
            _ => unreachable!(),
        };

        let (b_plus, a_plus) = svf_coefficients(fc_plus, R_plus, gain_plus, fs, filter_type);
        let (b_minus, a_minus) = svf_coefficients(fc_minus, R_minus, gain_minus, fs, filter_type);

        for tap in 0..3 {
            coeffs.db_dparam[tap][p] = (b_plus[tap] - b_minus[tap]) / (2.0 * eps);
            coeffs.da_dparam[tap][p] = (a_plus[tap] - a_minus[tap]) / (2.0 * eps);
        }
    }

    coeffs
}

/// Differentiable State Variable Filter mapped to a single SOS section.
#[derive(Debug, Clone)]
pub struct SvFilter {
    pub nfft: usize,
    pub fs: f64,
    pub filter_type: SvfType,
    pub n_out: usize,
    pub n_in: usize,
    pub alias_decay_db: f64,
    /// Physical parameters, shape `(1, P, N_out, N_in)` where `P` is 2 or 3.
    pub param: ArrayD<f64>,
    /// Accumulated parameter gradients, same shape as `param`.
    pub param_grad: ArrayD<f64>,
    inner: SosFilter,
}

impl SvFilter {
    /// Create a new SVF filter module with trainable interior defaults and zero gradients.
    ///
    /// # Errors
    ///
    /// Returns an error if `nfft`, `n_out`, or `n_in` is zero, or if `fs` is not
    /// finite and positive.
    pub fn new(
        nfft: usize,
        fs: f64,
        n_out: usize,
        n_in: usize,
        filter_type: SvfType,
        alias_decay_db: f64,
    ) -> Result<Self, AutodiffError> {
        if nfft == 0 {
            return Err(AutodiffError::Message(
                "SvFilter: nfft must be greater than 0".to_string(),
            ));
        }
        if fs <= 0.0 || !fs.is_finite() {
            return Err(AutodiffError::Message(
                "SvFilter: fs must be finite and greater than 0".to_string(),
            ));
        }
        if fs * 0.499 <= 1.0 {
            return Err(AutodiffError::Message(
                "SvFilter: fs is too small for a valid cutoff interval".to_string(),
            ));
        }
        if n_out == 0 {
            return Err(AutodiffError::Message(
                "SvFilter: n_out must be greater than 0".to_string(),
            ));
        }
        if n_in == 0 {
            return Err(AutodiffError::Message(
                "SvFilter: n_in must be greater than 0".to_string(),
            ));
        }
        let n_params = filter_type.n_params();
        let mut param = ArrayD::zeros(IxDyn(&[1, n_params, n_out, n_in]));
        let default_fc = 1_000.0_f64.min((1.0 + fs * 0.499) * 0.5);
        for out_ch in 0..n_out {
            for in_ch in 0..n_in {
                param[[0, 0, out_ch, in_ch]] = default_fc;
                param[[0, 1, out_ch, in_ch]] = std::f64::consts::SQRT_2;
            }
        }
        let param_grad = ArrayD::zeros(IxDyn(&[1, n_params, n_out, n_in]));
        let inner = SosFilter::new(nfft, 1, n_out, n_in, alias_decay_db)?;

        Ok(Self {
            nfft,
            fs,
            filter_type,
            n_out,
            n_in,
            alias_decay_db,
            param,
            param_grad,
            inner,
        })
    }

    fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }

    /// Return the normalized biquad coefficients `(b, a)` for the first
    /// input/output channel pair.
    ///
    /// This is a convenience accessor for tests and debugging; the module's
    /// `forward`/`backward` handle multi-channel tensors internally.
    #[must_use]
    pub fn coefficients(&self) -> ([f64; 3], [f64; 3]) {
        let fc = self.param[[0, 0, 0, 0]];
        let R = self.param[[0, 1, 0, 0]];
        let gain_db = if self.filter_type.n_params() > 2 {
            self.param[[0, 2, 0, 0]]
        } else {
            0.0
        };
        svf_coefficients(fc, R, gain_db, self.fs, self.filter_type)
    }

    /// Build complex coefficients and physical parameter derivatives for the
    /// current parameters.
    fn build_coeffs_and_grads(
        &self,
    ) -> Result<
        (
            Array4<Complex<f64>>,
            Array4<Complex<f64>>,
            Array5<f64>,
            Array5<f64>,
        ),
        AutodiffError,
    > {
        let n_params = self.filter_type.n_params();
        let param_view = self
            .param
            .view()
            .into_shape_with_order((1, n_params, self.n_out, self.n_in))
            .map_err(|e| {
                AutodiffError::Message(format!("SvFilter: failed to reshape param: {e}"))
            })?;

        let mut b = Array4::zeros((1, 3, self.n_out, self.n_in));
        let mut a = Array4::zeros((1, 3, self.n_out, self.n_in));
        let mut db_dparam = Array5::zeros((1, 3, n_params, self.n_out, self.n_in));
        let mut da_dparam = Array5::zeros((1, 3, n_params, self.n_out, self.n_in));

        for out_ch in 0..self.n_out {
            for in_ch in 0..self.n_in {
                let fc = param_view[[0, 0, out_ch, in_ch]];
                let R = param_view[[0, 1, out_ch, in_ch]];
                let gain_db = if n_params > 2 {
                    param_view[[0, 2, out_ch, in_ch]]
                } else {
                    0.0
                };
                let coeffs =
                    svf_coefficients_with_gradients(fc, R, gain_db, self.fs, self.filter_type);

                for tap in 0..3 {
                    b[[0, tap, out_ch, in_ch]] = Complex::new(coeffs.b[tap], 0.0);
                    a[[0, tap, out_ch, in_ch]] = Complex::new(coeffs.a[tap], 0.0);
                    for p in 0..n_params {
                        db_dparam[[0, tap, p, out_ch, in_ch]] = coeffs.db_dparam[tap][p];
                        da_dparam[[0, tap, p, out_ch, in_ch]] = coeffs.da_dparam[tap][p];
                    }
                }
            }
        }

        Ok((b, a, db_dparam, da_dparam))
    }
}

impl DiffModule<f64> for SvFilter {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "SvFilter::forward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let n_bins = input_shape[1];
        let n_in = input_shape[2];
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "SvFilter::forward: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if n_in != self.n_in {
            return Err(AutodiffError::Message(format!(
                "SvFilter::forward: expected {} input channels, got {}",
                self.n_in, n_in
            )));
        }

        // The inner filter must track the current physical parameters. We use a
        // fresh copy so that `forward` remains callable on a shared reference.
        let mut inner = self.inner.clone();
        let n_params = self.filter_type.n_params();
        let param_view = self
            .param
            .view()
            .into_shape_with_order((1, n_params, self.n_out, self.n_in))
            .map_err(|e| {
                AutodiffError::Message(format!("SvFilter::forward: failed to reshape param: {e}"))
            })?;
        for out_ch in 0..self.n_out {
            for in_ch in 0..self.n_in {
                let fc = param_view[[0, 0, out_ch, in_ch]];
                let R = param_view[[0, 1, out_ch, in_ch]];
                let gain_db = if n_params > 2 {
                    param_view[[0, 2, out_ch, in_ch]]
                } else {
                    0.0
                };
                let (b, a) = svf_coefficients(fc, R, gain_db, self.fs, self.filter_type);
                for tap in 0..3 {
                    inner.param[[0, tap, out_ch, in_ch]] = b[tap];
                    inner.param[[0, 3 + tap, out_ch, in_ch]] = a[tap];
                }
            }
        }

        inner.forward(input)
    }

    fn backward(
        &mut self,
        input: &DiffTensor<f64>,
        output: &DiffTensor<f64>,
        grad_output: &DiffTensor<f64>,
    ) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        let grad_shape = grad_output.data.shape();
        let output_shape = output.data.shape();
        validate_spectral_gradient_shape(
            "SvFilter::backward",
            input_shape,
            grad_shape,
            self.n_out,
        )?;
        if grad_shape != output_shape {
            return Err(AutodiffError::Message(format!(
                "SvFilter::backward: grad_output shape {:?} does not match output shape {:?}",
                grad_shape, output_shape
            )));
        }
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "SvFilter::backward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let n_bins = input_shape[1];
        let n_in = input_shape[2];
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "SvFilter::backward: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if n_in != self.n_in {
            return Err(AutodiffError::Message(format!(
                "SvFilter::backward: expected {} input channels, got {}",
                self.n_in, n_in
            )));
        }

        let n_params = self.filter_type.n_params();
        let (b, a, db_dparam, da_dparam) = self.build_coeffs_and_grads()?;

        self.inner.zero_grad();

        // Synchronise the inner SOS coefficients with the current parameters.
        for out_ch in 0..self.n_out {
            for in_ch in 0..self.n_in {
                for tap in 0..3 {
                    self.inner.param[[0, tap, out_ch, in_ch]] = b[[0, tap, out_ch, in_ch]].re;
                    self.inner.param[[0, 3 + tap, out_ch, in_ch]] = a[[0, tap, out_ch, in_ch]].re;
                }
            }
        }

        let grad_input = self.inner.backward(input, output, grad_output)?;

        // Map inner SOS coefficient gradients back to physical parameter gradients.
        let inner_grad = self
            .inner
            .param_grad
            .view()
            .into_shape_with_order((1, 6, self.n_out, self.n_in))
            .map_err(|e| {
                AutodiffError::Message(format!(
                    "SvFilter::backward: failed to reshape inner param_grad: {e}"
                ))
            })?;
        let mut param_grad = self
            .param_grad
            .view_mut()
            .into_shape_with_order((1, n_params, self.n_out, self.n_in))
            .map_err(|e| {
                AutodiffError::Message(format!(
                    "SvFilter::backward: failed to reshape param_grad: {e}"
                ))
            })?;

        for out_ch in 0..self.n_out {
            for in_ch in 0..self.n_in {
                for p in 0..n_params {
                    let mut accum = 0.0;
                    for tap in 0..3 {
                        let dl_db = inner_grad[[0, tap, out_ch, in_ch]];
                        let dl_da = inner_grad[[0, 3 + tap, out_ch, in_ch]];
                        accum += dl_db * db_dparam[[0, tap, p, out_ch, in_ch]];
                        accum += dl_da * da_dparam[[0, tap, p, out_ch, in_ch]];
                    }
                    param_grad[[0, p, out_ch, in_ch]] += accum;
                }
            }
        }

        Ok(grad_input)
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
        self.inner.zero_grad();
    }
}

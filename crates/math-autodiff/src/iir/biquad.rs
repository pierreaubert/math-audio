//! Differentiable RBJ biquad filter modules.

#![allow(
    clippy::cast_precision_loss,
    reason = "nfft is an audio buffer length that fits exactly in f64 for practical values"
)]
#![allow(
    clippy::manual_midpoint,
    reason = "RBJ formulas are documented with the (a+b)/2 form"
)]
#![allow(
    clippy::similar_names,
    reason = "coefficient derivative names are intentionally paired (b/a, raw/gain/norm)"
)]
#![allow(
    clippy::type_complexity,
    reason = "SOS coefficient Jacobians are inherently multi-dimensional"
)]
#![allow(
    clippy::uninlined_format_args,
    reason = "format strings are clearer with explicit arguments in error messages"
)]

use ndarray::{
    Array2, Array3, Array4, Array5, ArrayD, ArrayView3, ArrayView4, ArrayViewMut3, ArrayViewMut4,
    Axis, IxDyn,
};
use math_audio_iir_fir::BiquadFilterType;
use num_complex::Complex;
use std::f64::consts::{LN_10, PI, SQRT_2};

use crate::error::AutodiffError;
use crate::iir::response::{
    sos_frequency_response, sos_frequency_response_jacobian,
    sos_frequency_response_jacobian_parallel, sos_frequency_response_parallel,
};
use crate::module::DiffModule;
use crate::tensor::DiffTensor;

/// Sigmoid activation mapping raw parameters to the `(0, 1)` interval.
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Derivative of [`sigmoid`] expressed as a function of its output.
#[inline]
fn sigmoid_derivative_from_output(s: f64) -> f64 {
    s * (1.0 - s)
}

/// Map a raw gain parameter to a dB value clamped to `[-60, 60]`.
fn raw_gain_to_db(gain_raw: f64) -> f64 {
    let abs_gain = gain_raw.abs();
    let gain_db = if abs_gain > 0.0 {
        20.0 * abs_gain.log10()
    } else {
        f64::NEG_INFINITY
    };
    gain_db.clamp(-60.0, 60.0)
}

/// Derivative of [`raw_gain_to_db`] w.r.t. the raw gain parameter.
fn raw_gain_to_db_derivative(gain_raw: f64) -> f64 {
    let abs_gain = gain_raw.abs();
    let gain_db = if abs_gain > 0.0 {
        20.0 * abs_gain.log10()
    } else {
        f64::NEG_INFINITY
    };
    if gain_db <= -60.0 || gain_db >= 60.0 {
        0.0
    } else {
        20.0 / (gain_raw * LN_10)
    }
}

/// Convert a dB gain to a linear gain and its derivative.
fn db_to_linear_with_derivative(gain_db: f64) -> (f64, f64) {
    let gain_lin = 10.0_f64.powf(gain_db / 20.0);
    let derivative = gain_lin * LN_10 / 20.0;
    (gain_lin, derivative)
}

/// Number of tunable parameters for a given filter type.
const fn n_params_for(filter_type: BiquadFilterType) -> usize {
    if matches!(filter_type, BiquadFilterType::Bandpass) {
        3
    } else {
        2
    }
}

/// Coefficients and their parameter derivatives for a single biquad section.
#[derive(Debug, Clone, Copy)]
struct SectionCoeffs {
    b: [f64; 3],
    a: [f64; 3],
    /// `db_dparam[tap][param]` w.r.t. physical parameters (`fc`/`gain` or
    /// `fc1`/`fc2`/`gain`).
    db_dparam: [[f64; 3]; 3],
    /// `da_dparam[tap][param]` w.r.t. physical parameters.
    da_dparam: [[f64; 3]; 3],
}

impl SectionCoeffs {
    /// Allocate a zeroed coefficient set.
    fn zeros() -> Self {
        Self {
            b: [0.0; 3],
            a: [0.0; 3],
            db_dparam: [[0.0; 3]; 3],
            da_dparam: [[0.0; 3]; 3],
        }
    }
}

/// Compute normalized RBJ lowpass or highpass coefficients and physical
/// parameter gradients.
fn compute_lowpass_highpass(fc: f64, gain_db: f64, fs: f64, highpass: bool) -> SectionCoeffs {
    let omega = 2.0 * PI * fc / fs;
    let sn = omega.sin();
    let cs = omega.cos();
    let q = 1.0 / SQRT_2;
    let alpha = sn / (2.0 * q);

    let (b0, b1, b2): (f64, f64, f64);
    if highpass {
        b0 = (1.0 + cs) / 2.0;
        b1 = -(1.0 + cs);
        b2 = (1.0 + cs) / 2.0;
    } else {
        b0 = (1.0 - cs) / 2.0;
        b1 = 1.0 - cs;
        b2 = (1.0 - cs) / 2.0;
    }
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cs;
    let a2 = 1.0 - alpha;

    let domega_dfc = 2.0 * PI / fs;
    let dcs_dfc = -sn * domega_dfc;
    let dsn_dfc = cs * domega_dfc;
    let dalpha_dfc = dsn_dfc / (2.0 * q);

    let (db0_dfc, db1_dfc, db2_dfc): (f64, f64, f64);
    if highpass {
        db0_dfc = dcs_dfc / 2.0;
        db1_dfc = -dcs_dfc;
        db2_dfc = dcs_dfc / 2.0;
    } else {
        db0_dfc = -dcs_dfc / 2.0;
        db1_dfc = -dcs_dfc;
        db2_dfc = -dcs_dfc / 2.0;
    }
    let da0_dfc = dalpha_dfc;
    let da1_dfc = -2.0 * dcs_dfc;
    let da2_dfc = -dalpha_dfc;

    let (gain_lin, dgain_lin_dgain_db) = db_to_linear_with_derivative(gain_db);

    // Apply gain to numerator.
    let b0_g = b0 * gain_lin;
    let b1_g = b1 * gain_lin;
    let b2_g = b2 * gain_lin;

    // Normalize by a0.
    let a0_sq = a0 * a0;
    let b0_n = b0_g / a0;
    let b1_n = b1_g / a0;
    let b2_n = b2_g / a0;
    let a1_n = a1 / a0;
    let a2_n = a2 / a0;

    // Gradients after gain application.
    let db0_g_dfc = db0_dfc * gain_lin;
    let db1_g_dfc = db1_dfc * gain_lin;
    let db2_g_dfc = db2_dfc * gain_lin;
    let da0_g_dfc = da0_dfc;
    let da1_g_dfc = da1_dfc;
    let da2_g_dfc = da2_dfc;

    let db0_g_dgain = b0 * dgain_lin_dgain_db;
    let db1_g_dgain = b1 * dgain_lin_dgain_db;
    let db2_g_dgain = b2 * dgain_lin_dgain_db;

    // Gradients after normalization.
    let db0_n_dfc = (db0_g_dfc * a0 - b0_g * da0_g_dfc) / a0_sq;
    let db1_n_dfc = (db1_g_dfc * a0 - b1_g * da0_g_dfc) / a0_sq;
    let db2_n_dfc = (db2_g_dfc * a0 - b2_g * da0_g_dfc) / a0_sq;
    let da1_n_dfc = (da1_g_dfc * a0 - a1 * da0_g_dfc) / a0_sq;
    let da2_n_dfc = (da2_g_dfc * a0 - a2 * da0_g_dfc) / a0_sq;

    let db0_n_dgain = db0_g_dgain / a0;
    let db1_n_dgain = db1_g_dgain / a0;
    let db2_n_dgain = db2_g_dgain / a0;

    let mut coeffs = SectionCoeffs::zeros();
    coeffs.b = [b0_n, b1_n, b2_n];
    coeffs.a = [1.0, a1_n, a2_n];
    coeffs.db_dparam = [
        [db0_n_dfc, db0_n_dgain, 0.0],
        [db1_n_dfc, db1_n_dgain, 0.0],
        [db2_n_dfc, db2_n_dgain, 0.0],
    ];
    coeffs.da_dparam = [
        [0.0, 0.0, 0.0],
        [da1_n_dfc, 0.0, 0.0],
        [da2_n_dfc, 0.0, 0.0],
    ];
    coeffs
}

/// Compute normalized RBJ bandpass coefficients and physical parameter
/// gradients.
#[allow(clippy::similar_names)]
fn compute_bandpass(fc1: f64, fc2: f64, gain_db: f64, fs: f64) -> SectionCoeffs {
    let omega1 = 2.0 * PI * fc1 / fs;
    let omega2 = 2.0 * PI * fc2 / fs;
    let omega_c = (omega1 + omega2) / 2.0;
    let bw = (fc2 / fc1).log2();

    let sn_c = omega_c.sin();
    let cs_c = omega_c.cos();
    let c = 2.0_f64.ln() / 2.0;
    let alpha = sn_c * (c * bw * omega_c / sn_c).sinh();

    let b0 = alpha;
    let b1 = 0.0;
    let b2 = -alpha;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cs_c;
    let a2 = 1.0 - alpha;

    let (gain_lin, dgain_lin_dgain_db) = db_to_linear_with_derivative(gain_db);

    // Apply gain to numerator.
    let b0_g = b0 * gain_lin;
    let b1_g = b1 * gain_lin;
    let b2_g = b2 * gain_lin;

    // Normalize by a0.
    let a0_sq = a0 * a0;
    let b0_n = b0_g / a0;
    let b1_n = b1_g / a0;
    let b2_n = b2_g / a0;
    let a1_n = a1 / a0;
    let a2_n = a2 / a0;

    // Derivatives of omega_c and BW w.r.t. physical cutoffs.
    let domega_c_dfc1 = PI / fs;
    let domega_c_dfc2 = PI / fs;
    let ln2 = 2.0_f64.ln();
    let dbw_dfc1 = -1.0 / (fc1 * ln2);
    let dbw_dfc2 = 1.0 / (fc2 * ln2);

    // Derivative of alpha = sin(omega_c) * sinh(c * bw * omega_c / sin(omega_c)).
    let u = c * bw * omega_c / sn_c;
    let du_dfc = |domega_c_dfc: f64, dbw_dfc: f64| {
        let d_omega_over_sin = domega_c_dfc * (sn_c - omega_c * cs_c) / (sn_c * sn_c);
        dbw_dfc * omega_c / sn_c + bw * d_omega_over_sin
    };
    let du_dfc1 = du_dfc(domega_c_dfc1, dbw_dfc1);
    let du_dfc2 = du_dfc(domega_c_dfc2, dbw_dfc2);

    let dalpha_dfc = |domega_c_dfc: f64, du_dfc: f64| {
        cs_c * domega_c_dfc * u.sinh() + sn_c * u.cosh() * c * du_dfc
    };
    let dalpha_dfc1 = dalpha_dfc(domega_c_dfc1, du_dfc1);
    let dalpha_dfc2 = dalpha_dfc(domega_c_dfc2, du_dfc2);

    // Raw derivatives of unnormalized coefficients.
    let db0_dfc1 = dalpha_dfc1;
    let db2_dfc1 = -dalpha_dfc1;
    let da0_dfc1 = dalpha_dfc1;
    let da1_dfc1 = 2.0 * sn_c * domega_c_dfc1;
    let da2_dfc1 = -dalpha_dfc1;

    let db0_dfc2 = dalpha_dfc2;
    let db2_dfc2 = -dalpha_dfc2;
    let da0_dfc2 = dalpha_dfc2;
    let da1_dfc2 = 2.0 * sn_c * domega_c_dfc2;
    let da2_dfc2 = -dalpha_dfc2;

    // After gain.
    let db0_g_dfc1 = db0_dfc1 * gain_lin;
    let db2_g_dfc1 = db2_dfc1 * gain_lin;
    let da0_g_dfc1 = da0_dfc1;
    let da1_g_dfc1 = da1_dfc1;
    let da2_g_dfc1 = da2_dfc1;

    let db0_g_dfc2 = db0_dfc2 * gain_lin;
    let db2_g_dfc2 = db2_dfc2 * gain_lin;
    let da0_g_dfc2 = da0_dfc2;
    let da1_g_dfc2 = da1_dfc2;
    let da2_g_dfc2 = da2_dfc2;

    let db0_g_dgain = b0 * dgain_lin_dgain_db;
    let db2_g_dgain = -b0 * dgain_lin_dgain_db;

    // After normalization.
    let db0_n_dfc1 = (db0_g_dfc1 * a0 - b0_g * da0_g_dfc1) / a0_sq;
    let db2_n_dfc1 = (db2_g_dfc1 * a0 - b2_g * da0_g_dfc1) / a0_sq;
    let da1_n_dfc1 = (da1_g_dfc1 * a0 - a1 * da0_g_dfc1) / a0_sq;
    let da2_n_dfc1 = (da2_g_dfc1 * a0 - a2 * da0_g_dfc1) / a0_sq;

    let db0_n_dfc2 = (db0_g_dfc2 * a0 - b0_g * da0_g_dfc2) / a0_sq;
    let db2_n_dfc2 = (db2_g_dfc2 * a0 - b2_g * da0_g_dfc2) / a0_sq;
    let da1_n_dfc2 = (da1_g_dfc2 * a0 - a1 * da0_g_dfc2) / a0_sq;
    let da2_n_dfc2 = (da2_g_dfc2 * a0 - a2 * da0_g_dfc2) / a0_sq;

    let db0_n_dgain = db0_g_dgain / a0;
    let db2_n_dgain = db2_g_dgain / a0;

    let mut coeffs = SectionCoeffs::zeros();
    coeffs.b = [b0_n, b1_n, b2_n];
    coeffs.a = [1.0, a1_n, a2_n];
    coeffs.db_dparam = [
        [db0_n_dfc1, db0_n_dfc2, db0_n_dgain],
        [0.0, 0.0, 0.0],
        [db2_n_dfc1, db2_n_dfc2, db2_n_dgain],
    ];
    coeffs.da_dparam = [
        [0.0, 0.0, 0.0],
        [da1_n_dfc1, da1_n_dfc2, 0.0],
        [da2_n_dfc1, da2_n_dfc2, 0.0],
    ];
    coeffs
}

fn biquad_param_view(param: &ArrayD<f64>) -> Result<ArrayView4<'_, f64>, AutodiffError> {
    let shape = param.shape();
    if shape.len() != 4 {
        return Err(AutodiffError::Message(format!(
            "Biquad: expected 4-D parameter tensor, got shape {:?}",
            shape
        )));
    }
    let (n_sections, n_params, n_out, n_in) = (shape[0], shape[1], shape[2], shape[3]);
    param
        .view()
        .into_shape_with_order((n_sections, n_params, n_out, n_in))
        .map_err(|e| AutodiffError::Message(format!("Biquad: failed to reshape param: {e}")))
}

fn biquad_param_grad_view_mut(
    param_grad: &mut ArrayD<f64>,
) -> Result<ArrayViewMut4<'_, f64>, AutodiffError> {
    let shape = param_grad.shape();
    if shape.len() != 4 {
        return Err(AutodiffError::Message(format!(
            "Biquad: expected 4-D parameter gradient tensor, got shape {:?}",
            shape
        )));
    }
    let (n_sections, n_params, n_out, n_in) = (shape[0], shape[1], shape[2], shape[3]);
    param_grad
        .view_mut()
        .into_shape_with_order((n_sections, n_params, n_out, n_in))
        .map_err(|e| AutodiffError::Message(format!("Biquad: failed to reshape param_grad: {e}")))
}

fn parallel_biquad_param_view(param: &ArrayD<f64>) -> Result<ArrayView3<'_, f64>, AutodiffError> {
    let shape = param.shape();
    if shape.len() != 3 {
        return Err(AutodiffError::Message(format!(
            "ParallelBiquad: expected 3-D parameter tensor, got shape {:?}",
            shape
        )));
    }
    let (n_sections, n_params, n_channels) = (shape[0], shape[1], shape[2]);
    param
        .view()
        .into_shape_with_order((n_sections, n_params, n_channels))
        .map_err(|e| {
            AutodiffError::Message(format!("ParallelBiquad: failed to reshape param: {e}"))
        })
}

fn parallel_biquad_param_grad_view_mut(
    param_grad: &mut ArrayD<f64>,
) -> Result<ArrayViewMut3<'_, f64>, AutodiffError> {
    let shape = param_grad.shape();
    if shape.len() != 3 {
        return Err(AutodiffError::Message(format!(
            "ParallelBiquad: expected 3-D parameter gradient tensor, got shape {:?}",
            shape
        )));
    }
    let (n_sections, n_params, n_channels) = (shape[0], shape[1], shape[2]);
    param_grad
        .view_mut()
        .into_shape_with_order((n_sections, n_params, n_channels))
        .map_err(|e| {
            AutodiffError::Message(format!(
                "ParallelBiquad: failed to reshape param_grad: {e}"
            ))
        })
}

/// Differentiable RBJ biquad with arbitrary input/output channel coupling.
#[derive(Debug, Clone)]
pub struct Biquad {
    /// FFT length.
    pub nfft: usize,
    /// Sample rate in Hz.
    pub fs: f64,
    /// Number of cascaded SOS sections.
    pub n_sections: usize,
    /// Filter type (lowpass, highpass, or bandpass).
    pub filter_type: BiquadFilterType,
    /// Raw parameters, shape `(n_sections, P, n_out, n_in)`.
    pub param: ArrayD<f64>,
    /// Accumulated parameter gradients, same shape as `param`.
    pub param_grad: ArrayD<f64>,
    /// Anti-aliasing decay in dB.
    pub alias_decay_db: f64,
}

impl Biquad {
    /// Create a new biquad module with zero-initialized parameters and gradients.
    ///
    /// # Errors
    ///
    /// Returns an error if `nfft` is zero.
    pub fn new(
        nfft: usize,
        fs: f64,
        n_sections: usize,
        filter_type: BiquadFilterType,
        n_out: usize,
        n_in: usize,
        alias_decay_db: f64,
    ) -> Result<Self, AutodiffError> {
        if nfft == 0 {
            return Err(AutodiffError::Message(
                "Biquad: nfft must be greater than 0".to_string(),
            ));
        }
        let n_params = n_params_for(filter_type);
        Ok(Self {
            nfft,
            fs,
            n_sections,
            filter_type,
            param: ArrayD::zeros(IxDyn(&[n_sections, n_params, n_out, n_in])),
            param_grad: ArrayD::zeros(IxDyn(&[n_sections, n_params, n_out, n_in])),
            alias_decay_db,
        })
    }

    fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }

    /// Build the anti-aliasing envelope `[gamma^0, gamma^1, gamma^2]`.
    fn gamma(&self) -> [f64; 3] {
        let gamma = 10.0_f64.powf(-self.alias_decay_db.abs() / (20.0 * self.nfft as f64));
        [1.0, gamma, gamma * gamma]
    }

    /// Map raw parameters to normalized coefficients and parameter gradients.
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
        let param = biquad_param_view(&self.param)?;
        let (n_sections, n_params, n_out, n_in) = param.dim();
        let mut b = Array4::zeros((n_sections, 3, n_out, n_in));
        let mut a = Array4::zeros((n_sections, 3, n_out, n_in));
        let mut db_dparam = Array5::zeros((n_sections, 3, n_params, n_out, n_in));
        let mut da_dparam = Array5::zeros((n_sections, 3, n_params, n_out, n_in));

        let half_fs = self.fs / 2.0;

        for section in 0..n_sections {
            for out_ch in 0..n_out {
                for in_ch in 0..n_in {
                    let coeffs = match self.filter_type {
                        BiquadFilterType::Lowpass | BiquadFilterType::Highpass => {
                            let fc_raw = param[[section, 0, out_ch, in_ch]];
                            let gain_raw = param[[section, 1, out_ch, in_ch]];
                            let fc_norm = sigmoid(fc_raw);
                            let fc = fc_norm * half_fs;
                            let gain_db = raw_gain_to_db(gain_raw);
                            let mut c = compute_lowpass_highpass(
                                fc,
                                gain_db,
                                self.fs,
                                self.filter_type == BiquadFilterType::Highpass,
                            );
                            let dfc_dfc_raw = sigmoid_derivative_from_output(fc_norm) * half_fs;
                            let dgain_db_dgain_raw = raw_gain_to_db_derivative(gain_raw);
                            for tap in 0..3 {
                                c.db_dparam[tap][0] *= dfc_dfc_raw;
                                c.da_dparam[tap][0] *= dfc_dfc_raw;
                                c.db_dparam[tap][1] *= dgain_db_dgain_raw;
                            }
                            c
                        }
                        BiquadFilterType::Bandpass => {
                            let fc1_raw = param[[section, 0, out_ch, in_ch]];
                            let fc2_raw = param[[section, 1, out_ch, in_ch]];
                            let gain_raw = param[[section, 2, out_ch, in_ch]];
                            let fc1_norm = sigmoid(fc1_raw);
                            let fc2_norm = sigmoid(fc2_raw);
                            let (fc_low_norm, fc_high_norm, swapped) = if fc1_norm <= fc2_norm {
                                (fc1_norm, fc2_norm, false)
                            } else {
                                (fc2_norm, fc1_norm, true)
                            };
                            let fc1 = fc_low_norm * half_fs;
                            let fc2 = fc_high_norm * half_fs;
                            let gain_db = raw_gain_to_db(gain_raw);
                            let mut c = compute_bandpass(fc1, fc2, gain_db, self.fs);

                            let dfc1_norm_dfc1_raw = sigmoid_derivative_from_output(fc1_norm);
                            let dfc2_norm_dfc2_raw = sigmoid_derivative_from_output(fc2_norm);
                            let dfc1_norm_dfc2_raw = if swapped {
                                sigmoid_derivative_from_output(fc2_norm)
                            } else {
                                0.0
                            };
                            let dfc2_norm_dfc1_raw = if swapped {
                                sigmoid_derivative_from_output(fc1_norm)
                            } else {
                                0.0
                            };
                            let dgain_db_dgain_raw = raw_gain_to_db_derivative(gain_raw);

                            for tap in 0..3 {
                                let db_df1 = c.db_dparam[tap][0];
                                let db_df2 = c.db_dparam[tap][1];
                                let db_dg = c.db_dparam[tap][2];
                                c.db_dparam[tap][0] = db_df1 * dfc1_norm_dfc1_raw * half_fs
                                    + db_df2 * dfc2_norm_dfc1_raw * half_fs;
                                c.db_dparam[tap][1] = db_df1 * dfc1_norm_dfc2_raw * half_fs
                                    + db_df2 * dfc2_norm_dfc2_raw * half_fs;
                                c.db_dparam[tap][2] = db_dg * dgain_db_dgain_raw;

                                let da_df1 = c.da_dparam[tap][0];
                                let da_df2 = c.da_dparam[tap][1];
                                c.da_dparam[tap][0] = da_df1 * dfc1_norm_dfc1_raw * half_fs
                                    + da_df2 * dfc2_norm_dfc1_raw * half_fs;
                                c.da_dparam[tap][1] = da_df1 * dfc1_norm_dfc2_raw * half_fs
                                    + da_df2 * dfc2_norm_dfc2_raw * half_fs;
                            }
                            c
                        }
                        _ => {
                            return Err(AutodiffError::Message(format!(
                                "Biquad: unsupported filter type {:?}",
                                self.filter_type
                            )));
                        }
                    };

                    for tap in 0..3 {
                        b[[section, tap, out_ch, in_ch]] = Complex::new(coeffs.b[tap], 0.0);
                        a[[section, tap, out_ch, in_ch]] = Complex::new(coeffs.a[tap], 0.0);
                        for param_idx in 0..n_params {
                            db_dparam[[section, tap, param_idx, out_ch, in_ch]] =
                                coeffs.db_dparam[tap][param_idx];
                            da_dparam[[section, tap, param_idx, out_ch, in_ch]] =
                                coeffs.da_dparam[tap][param_idx];
                        }
                    }
                }
            }
        }

        Ok((b, a, db_dparam, da_dparam))
    }
}

impl DiffModule<f64> for Biquad {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "Biquad::forward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let n_bins = input_shape[1];
        let n_in = input_shape[2];
        let param_shape = self.param.shape();
        if param_shape.len() != 4 {
            return Err(AutodiffError::Message(format!(
                "Biquad::forward: expected 4-D parameter tensor, got shape {:?}",
                param_shape
            )));
        }
        let n_out_stored = param_shape[2];
        let n_in_stored = param_shape[3];
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "Biquad::forward: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if n_in != n_in_stored {
            return Err(AutodiffError::Message(format!(
                "Biquad::forward: expected {} input channels, got {}",
                n_in_stored, n_in
            )));
        }

        let (b, a, _, _) = self.build_coeffs_and_grads()?;
        let h = sos_frequency_response(&b, &a, self.nfft, Some(&self.gamma()))?;

        let mut output_shape: Vec<usize> = input_shape.to_vec();
        output_shape[2] = n_out_stored;
        let mut output = ArrayD::zeros(IxDyn(&output_shape));

        for bin in 0..n_bins {
            for in_ch in 0..n_in {
                let input_axis2 = input.data.index_axis(Axis(2), in_ch);
                let input_bin = input_axis2.index_axis(Axis(1), bin);
                for out_ch in 0..n_out_stored {
                    let h_val = h[[bin, out_ch, in_ch]];
                    let mut output_axis2 = output.index_axis_mut(Axis(1), bin);
                    let mut output_bin = output_axis2.index_axis_mut(Axis(1), out_ch);
                    output_bin += &input_bin.mapv(|x| x * h_val);
                }
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
        if input_shape != grad_shape {
            return Err(AutodiffError::Message(format!(
                "Biquad::backward: input shape {:?} does not match grad_output shape {:?}",
                input_shape, grad_shape
            )));
        }
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "Biquad::backward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let n_bins = input_shape[1];
        let n_in = input_shape[2];
        let param_shape = self.param.shape();
        if param_shape.len() != 4 {
            return Err(AutodiffError::Message(format!(
                "Biquad::backward: expected 4-D parameter tensor, got shape {:?}",
                param_shape
            )));
        }
        let n_sections = param_shape[0];
        let n_params = param_shape[1];
        let n_out = param_shape[2];
        let n_in_stored = param_shape[3];
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "Biquad::backward: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if n_in != n_in_stored {
            return Err(AutodiffError::Message(format!(
                "Biquad::backward: expected {} input channels, got {}",
                n_in_stored, n_in
            )));
        }

        let (b, a, db_dparam, da_dparam) = self.build_coeffs_and_grads()?;
        let gamma = self.gamma();
        let h = sos_frequency_response(&b, &a, self.nfft, Some(&gamma))?;
        let (dh_db, dh_da) = sos_frequency_response_jacobian(&b, &a, self.nfft, Some(&gamma))?;

        // Compute dLoss/dH using real parts (MVP assumption: real time-domain signals).
        let mut dl_dh = Array3::zeros((n_bins, n_out, n_in));
        for out_ch in 0..n_out {
            let grad_axis2 = grad_output.data.index_axis(Axis(2), out_ch);
            for in_ch in 0..n_in {
                let input_axis2 = input.data.index_axis(Axis(2), in_ch);
                for bin in 0..n_bins {
                    let grad_bin = grad_axis2.index_axis(Axis(1), bin);
                    let input_bin = input_axis2.index_axis(Axis(1), bin);
                    let prod = &grad_bin * &input_bin;
                    dl_dh[[bin, out_ch, in_ch]] = prod.sum();
                }
            }
        }

        // Accumulate parameter gradients. Non-finite Jacobian terms occur at
        // frequencies where the numerator/denominator response is exactly zero
        // (e.g. lowpass at Nyquist); their contribution is skipped because the
        // loss gradient at those bins is also zero for practical targets.
        let mut param_grad = biquad_param_grad_view_mut(&mut self.param_grad)?;
        for section in 0..n_sections {
            for out_ch in 0..n_out {
                for in_ch in 0..n_in {
                    for param_idx in 0..n_params {
                        let mut accum = 0.0;
                        for tap in 0..3 {
                            let db_dp = db_dparam[[section, tap, param_idx, out_ch, in_ch]];
                            let da_dp = da_dparam[[section, tap, param_idx, out_ch, in_ch]];
                            for bin in 0..n_bins {
                                let term = dh_db[[bin, section, tap, out_ch, in_ch]] * db_dp
                                    + dh_da[[bin, section, tap, out_ch, in_ch]] * da_dp;
                                if term.is_finite() {
                                    accum += (dl_dh[[bin, out_ch, in_ch]].conj() * term).re;
                                }
                            }
                        }
                        param_grad[[section, param_idx, out_ch, in_ch]] += accum;
                    }
                }
            }
        }

        // Compute dLoss/dInput.
        let mut grad_input = ArrayD::zeros(IxDyn(input_shape));
        for in_ch in 0..n_in {
            for out_ch in 0..n_out {
                for bin in 0..n_bins {
                    let h_conj = h[[bin, out_ch, in_ch]].conj();
                    let grad_axis2 = grad_output.data.index_axis(Axis(2), out_ch);
                    let grad_bin = grad_axis2.index_axis(Axis(1), bin);
                    let mut input_axis2 = grad_input.index_axis_mut(Axis(2), in_ch);
                    let mut input_bin = input_axis2.index_axis_mut(Axis(1), bin);
                    input_bin += &grad_bin.mapv(|x| x * h_conj);
                }
            }
        }

        Ok(DiffTensor::from_array(grad_input))
    }

    fn input_channels(&self) -> usize {
        self.param.shape().get(3).copied().unwrap_or(0)
    }

    fn output_channels(&self) -> usize {
        self.param.shape().get(2).copied().unwrap_or(0)
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

/// Differentiable RBJ biquad with a diagonal (per-channel) frequency response.
#[derive(Debug, Clone)]
pub struct ParallelBiquad {
    /// FFT length.
    pub nfft: usize,
    /// Sample rate in Hz.
    pub fs: f64,
    /// Number of cascaded SOS sections.
    pub n_sections: usize,
    /// Filter type (lowpass, highpass, or bandpass).
    pub filter_type: BiquadFilterType,
    /// Raw parameters, shape `(n_sections, P, N)`.
    pub param: ArrayD<f64>,
    /// Accumulated parameter gradients, same shape as `param`.
    pub param_grad: ArrayD<f64>,
    /// Anti-aliasing decay in dB.
    pub alias_decay_db: f64,
}

impl ParallelBiquad {
    /// Create a new parallel biquad module with zero-initialized parameters and
    /// gradients.
    ///
    /// # Errors
    ///
    /// Returns an error if `nfft` is zero.
    pub fn new(
        nfft: usize,
        fs: f64,
        n_sections: usize,
        filter_type: BiquadFilterType,
        n_channels: usize,
        alias_decay_db: f64,
    ) -> Result<Self, AutodiffError> {
        if nfft == 0 {
            return Err(AutodiffError::Message(
                "ParallelBiquad: nfft must be greater than 0".to_string(),
            ));
        }
        let n_params = n_params_for(filter_type);
        Ok(Self {
            nfft,
            fs,
            n_sections,
            filter_type,
            param: ArrayD::zeros(IxDyn(&[n_sections, n_params, n_channels])),
            param_grad: ArrayD::zeros(IxDyn(&[n_sections, n_params, n_channels])),
            alias_decay_db,
        })
    }

    fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }

    /// Build the anti-aliasing envelope `[gamma^0, gamma^1, gamma^2]`.
    fn gamma(&self) -> [f64; 3] {
        let gamma = 10.0_f64.powf(-self.alias_decay_db.abs() / (20.0 * self.nfft as f64));
        [1.0, gamma, gamma * gamma]
    }

    /// Map raw parameters to normalized coefficients and parameter gradients.
    fn build_coeffs_and_grads(
        &self,
    ) -> Result<
        (
            Array3<Complex<f64>>,
            Array3<Complex<f64>>,
            Array4<f64>,
            Array4<f64>,
        ),
        AutodiffError,
    > {
        let param = parallel_biquad_param_view(&self.param)?;
        let (n_sections, n_params, n_channels) = param.dim();
        let mut b = Array3::zeros((n_sections, 3, n_channels));
        let mut a = Array3::zeros((n_sections, 3, n_channels));
        let mut db_dparam = Array4::zeros((n_sections, 3, n_params, n_channels));
        let mut da_dparam = Array4::zeros((n_sections, 3, n_params, n_channels));

        let half_fs = self.fs / 2.0;

        for section in 0..n_sections {
            for ch in 0..n_channels {
                let coeffs = match self.filter_type {
                    BiquadFilterType::Lowpass | BiquadFilterType::Highpass => {
                        let fc_raw = param[[section, 0, ch]];
                        let gain_raw = param[[section, 1, ch]];
                        let fc_norm = sigmoid(fc_raw);
                        let fc = fc_norm * half_fs;
                        let gain_db = raw_gain_to_db(gain_raw);
                        let mut c = compute_lowpass_highpass(
                            fc,
                            gain_db,
                            self.fs,
                            self.filter_type == BiquadFilterType::Highpass,
                        );
                        let dfc_dfc_raw = sigmoid_derivative_from_output(fc_norm) * half_fs;
                        let dgain_db_dgain_raw = raw_gain_to_db_derivative(gain_raw);
                        for tap in 0..3 {
                            c.db_dparam[tap][0] *= dfc_dfc_raw;
                            c.da_dparam[tap][0] *= dfc_dfc_raw;
                            c.db_dparam[tap][1] *= dgain_db_dgain_raw;
                        }
                        c
                    }
                    BiquadFilterType::Bandpass => {
                        let fc1_raw = param[[section, 0, ch]];
                        let fc2_raw = param[[section, 1, ch]];
                        let gain_raw = param[[section, 2, ch]];
                        let fc1_norm = sigmoid(fc1_raw);
                        let fc2_norm = sigmoid(fc2_raw);
                        let (fc_low_norm, fc_high_norm, swapped) = if fc1_norm <= fc2_norm {
                            (fc1_norm, fc2_norm, false)
                        } else {
                            (fc2_norm, fc1_norm, true)
                        };
                        let fc1 = fc_low_norm * half_fs;
                        let fc2 = fc_high_norm * half_fs;
                        let gain_db = raw_gain_to_db(gain_raw);
                        let mut c = compute_bandpass(fc1, fc2, gain_db, self.fs);

                        let dfc1_norm_dfc1_raw = sigmoid_derivative_from_output(fc1_norm);
                        let dfc2_norm_dfc2_raw = sigmoid_derivative_from_output(fc2_norm);
                        let dfc1_norm_dfc2_raw = if swapped {
                            sigmoid_derivative_from_output(fc2_norm)
                        } else {
                            0.0
                        };
                        let dfc2_norm_dfc1_raw = if swapped {
                            sigmoid_derivative_from_output(fc1_norm)
                        } else {
                            0.0
                        };
                        let dgain_db_dgain_raw = raw_gain_to_db_derivative(gain_raw);

                        for tap in 0..3 {
                            let db_df1 = c.db_dparam[tap][0];
                            let db_df2 = c.db_dparam[tap][1];
                            let db_dg = c.db_dparam[tap][2];
                            c.db_dparam[tap][0] = db_df1 * dfc1_norm_dfc1_raw * half_fs
                                + db_df2 * dfc2_norm_dfc1_raw * half_fs;
                            c.db_dparam[tap][1] = db_df1 * dfc1_norm_dfc2_raw * half_fs
                                + db_df2 * dfc2_norm_dfc2_raw * half_fs;
                            c.db_dparam[tap][2] = db_dg * dgain_db_dgain_raw;

                            let da_df1 = c.da_dparam[tap][0];
                            let da_df2 = c.da_dparam[tap][1];
                            c.da_dparam[tap][0] = da_df1 * dfc1_norm_dfc1_raw * half_fs
                                + da_df2 * dfc2_norm_dfc1_raw * half_fs;
                            c.da_dparam[tap][1] = da_df1 * dfc1_norm_dfc2_raw * half_fs
                                + da_df2 * dfc2_norm_dfc2_raw * half_fs;
                        }
                        c
                    }
                    _ => {
                        return Err(AutodiffError::Message(format!(
                            "ParallelBiquad: unsupported filter type {:?}",
                            self.filter_type
                        )));
                    }
                };

                for tap in 0..3 {
                    b[[section, tap, ch]] = Complex::new(coeffs.b[tap], 0.0);
                    a[[section, tap, ch]] = Complex::new(coeffs.a[tap], 0.0);
                    for param_idx in 0..n_params {
                        db_dparam[[section, tap, param_idx, ch]] =
                            coeffs.db_dparam[tap][param_idx];
                        da_dparam[[section, tap, param_idx, ch]] =
                            coeffs.da_dparam[tap][param_idx];
                    }
                }
            }
        }

        Ok((b, a, db_dparam, da_dparam))
    }
}

impl DiffModule<f64> for ParallelBiquad {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "ParallelBiquad::forward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let n_bins = input_shape[1];
        let n_channels = input_shape[2];
        let param_shape = self.param.shape();
        if param_shape.len() != 3 {
            return Err(AutodiffError::Message(format!(
                "ParallelBiquad::forward: expected 3-D parameter tensor, got shape {:?}",
                param_shape
            )));
        }
        let n_channels_stored = param_shape[2];
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "ParallelBiquad::forward: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if n_channels != n_channels_stored {
            return Err(AutodiffError::Message(format!(
                "ParallelBiquad::forward: expected {} channels, got {}",
                n_channels_stored, n_channels
            )));
        }

        let (b, a, _, _) = self.build_coeffs_and_grads()?;
        let h = sos_frequency_response_parallel(&b, &a, self.nfft, Some(&self.gamma()))?;

        let mut output = ArrayD::zeros(IxDyn(input_shape));
        for ch in 0..n_channels {
            for bin in 0..n_bins {
                let h_val = h[[bin, ch]];
                let input_axis2 = input.data.index_axis(Axis(2), ch);
                let input_bin = input_axis2.index_axis(Axis(1), bin);
                let mut output_axis2 = output.index_axis_mut(Axis(2), ch);
                let mut output_bin = output_axis2.index_axis_mut(Axis(1), bin);
                output_bin += &input_bin.mapv(|x| x * h_val);
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
        if input_shape != grad_shape {
            return Err(AutodiffError::Message(format!(
                "ParallelBiquad::backward: input shape {:?} does not match grad_output shape {:?}",
                input_shape, grad_shape
            )));
        }
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "ParallelBiquad::backward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let n_bins = input_shape[1];
        let n_channels = input_shape[2];
        let param_shape = self.param.shape();
        if param_shape.len() != 3 {
            return Err(AutodiffError::Message(format!(
                "ParallelBiquad::backward: expected 3-D parameter tensor, got shape {:?}",
                param_shape
            )));
        }
        let n_sections = param_shape[0];
        let n_params = param_shape[1];
        let n_channels_stored = param_shape[2];
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "ParallelBiquad::backward: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if n_channels != n_channels_stored {
            return Err(AutodiffError::Message(format!(
                "ParallelBiquad::backward: expected {} channels, got {}",
                n_channels_stored, n_channels
            )));
        }

        let (b, a, db_dparam, da_dparam) = self.build_coeffs_and_grads()?;
        let gamma = self.gamma();
        let h = sos_frequency_response_parallel(&b, &a, self.nfft, Some(&gamma))?;
        let (dh_db, dh_da) =
            sos_frequency_response_jacobian_parallel(&b, &a, self.nfft, Some(&gamma))?;

        // Compute dLoss/dH using real parts.
        let mut dl_dh = Array2::zeros((n_bins, n_channels));
        for ch in 0..n_channels {
            let grad_axis2 = grad_output.data.index_axis(Axis(2), ch);
            let input_axis2 = input.data.index_axis(Axis(2), ch);
            for bin in 0..n_bins {
                let grad_bin = grad_axis2.index_axis(Axis(1), bin);
                let input_bin = input_axis2.index_axis(Axis(1), bin);
                dl_dh[[bin, ch]] = (&grad_bin * &input_bin).sum();
            }
        }

        // Accumulate parameter gradients. See `Biquad::backward` for the
        // non-finite Jacobian handling rationale.
        let mut param_grad = parallel_biquad_param_grad_view_mut(&mut self.param_grad)?;
        for section in 0..n_sections {
            for ch in 0..n_channels {
                for param_idx in 0..n_params {
                    let mut accum = 0.0;
                    for tap in 0..3 {
                        let db_dp = db_dparam[[section, tap, param_idx, ch]];
                        let da_dp = da_dparam[[section, tap, param_idx, ch]];
                        for bin in 0..n_bins {
                            let term = dh_db[[bin, section, tap, ch]] * db_dp
                                + dh_da[[bin, section, tap, ch]] * da_dp;
                            if term.is_finite() {
                                accum += (dl_dh[[bin, ch]].conj() * term).re;
                            }
                        }
                    }
                    param_grad[[section, param_idx, ch]] += accum;
                }
            }
        }

        // Compute dLoss/dInput.
        let mut grad_input = ArrayD::zeros(IxDyn(input_shape));
        for ch in 0..n_channels {
            for bin in 0..n_bins {
                let h_conj = h[[bin, ch]].conj();
                let grad_axis2 = grad_output.data.index_axis(Axis(2), ch);
                let grad_bin = grad_axis2.index_axis(Axis(1), bin);
                let mut input_axis2 = grad_input.index_axis_mut(Axis(2), ch);
                let mut input_bin = input_axis2.index_axis_mut(Axis(1), bin);
                input_bin += &grad_bin.mapv(|x| x * h_conj);
            }
        }

        Ok(DiffTensor::from_array(grad_input))
    }

    fn input_channels(&self) -> usize {
        self.param.shape().get(2).copied().unwrap_or(0)
    }

    fn output_channels(&self) -> usize {
        self.param.shape().get(2).copied().unwrap_or(0)
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

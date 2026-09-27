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

use math_audio_iir_fir::BiquadFilterType;
use ndarray::{
    Array3, Array4, Array5, ArrayD, ArrayView3, ArrayView4, ArrayViewMut3, ArrayViewMut4, Axis,
    IxDyn,
};
use num_complex::Complex;
use std::f64::consts::{PI, SQRT_2};

use crate::error::AutodiffError;
use crate::iir::response::BasisCache;
use crate::iir::response::{
    SosFrequencyBasis, sos_coefficient_vjp_with_basis, sos_frequency_response_parallel,
};
use crate::module::{
    DiffModule, Scalar, fconst, fnv1a_init, fnv1a_step, validate_spectral_gradient_shape,
};
use crate::tensor::DiffTensor;

/// Sigmoid activation mapping raw parameters to the `(0, 1)` interval.
#[inline]
fn sigmoid<T: Scalar>(x: T) -> T {
    let eps = fconst::<T>(1e-6);
    let s = T::one() / (T::one() + (-x).exp());
    s.max(eps).min(T::one() - eps)
}

/// Derivative of [`sigmoid`] expressed as a function of its output.
#[inline]
fn sigmoid_derivative_from_output<T: Scalar>(s: T) -> T {
    let eps = fconst::<T>(1e-6);
    if s <= eps || s >= T::one() - eps {
        T::zero()
    } else {
        s * (T::one() - s)
    }
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
struct SectionCoeffs<T> {
    b: [T; 3],
    a: [T; 3],
    /// `db_dparam[tap][param]` w.r.t. physical parameters (`fc`/`gain` or
    /// `fc1`/`fc2`/`gain`).
    db_dparam: [[T; 3]; 3],
    /// `da_dparam[tap][param]` w.r.t. physical parameters.
    da_dparam: [[T; 3]; 3],
}

impl<T: Scalar> SectionCoeffs<T> {
    /// Allocate a zeroed coefficient set.
    fn zeros() -> Self {
        Self {
            b: [T::zero(); 3],
            a: [T::zero(); 3],
            db_dparam: [[T::zero(); 3]; 3],
            da_dparam: [[T::zero(); 3]; 3],
        }
    }
}

/// Compute normalized RBJ lowpass or highpass coefficients without gradients.
fn compute_lowpass_highpass_coeffs<T: Scalar>(
    fc: T,
    gain: T,
    fs: T,
    highpass: bool,
) -> ([T; 3], [T; 3]) {
    let omega = fconst::<T>(2.0) * fconst::<T>(PI) * fc / fs;
    let sn = omega.sin();
    let cs = omega.cos();
    let q = T::one() / fconst::<T>(SQRT_2);
    let alpha = sn / (fconst::<T>(2.0) * q);

    let (b0, b1, b2): (T, T, T);
    if highpass {
        b0 = (T::one() + cs) / fconst::<T>(2.0);
        b1 = -(T::one() + cs);
        b2 = (T::one() + cs) / fconst::<T>(2.0);
    } else {
        b0 = (T::one() - cs) / fconst::<T>(2.0);
        b1 = T::one() - cs;
        b2 = (T::one() - cs) / fconst::<T>(2.0);
    }
    let a0 = T::one() + alpha;
    let a1 = -fconst::<T>(2.0) * cs;
    let a2 = T::one() - alpha;

    let inv_a0 = T::one() / a0;
    (
        [b0 * gain * inv_a0, b1 * gain * inv_a0, b2 * gain * inv_a0],
        [T::one(), a1 * inv_a0, a2 * inv_a0],
    )
}

/// Compute normalized RBJ bandpass coefficients without gradients.
#[allow(clippy::similar_names)]
fn compute_bandpass_coeffs<T: Scalar>(fc1: T, fc2: T, gain: T, fs: T) -> ([T; 3], [T; 3]) {
    let omega1 = fconst::<T>(2.0) * fconst::<T>(PI) * fc1 / fs;
    let omega2 = fconst::<T>(2.0) * fconst::<T>(PI) * fc2 / fs;
    let omega_c = (omega1 + omega2) / fconst::<T>(2.0);
    let bw = (fc2 / fc1).log2();

    let sn_c = omega_c.sin();
    let cs_c = omega_c.cos();
    let c = fconst::<T>(2.0).ln() / fconst::<T>(2.0);
    let alpha = sn_c * (c * bw * omega_c / sn_c).sinh();

    let b0 = alpha;
    let b2 = -alpha;
    let a0 = T::one() + alpha;
    let a1 = -fconst::<T>(2.0) * cs_c;
    let a2 = T::one() - alpha;

    let inv_a0 = T::one() / a0;
    (
        [b0 * gain * inv_a0, T::zero(), b2 * gain * inv_a0],
        [T::one(), a1 * inv_a0, a2 * inv_a0],
    )
}

/// Compute normalized RBJ lowpass or highpass coefficients and physical
/// parameter gradients.
fn compute_lowpass_highpass<T: Scalar>(fc: T, gain: T, fs: T, highpass: bool) -> SectionCoeffs<T> {
    let omega = fconst::<T>(2.0) * fconst::<T>(PI) * fc / fs;
    let sn = omega.sin();
    let cs = omega.cos();
    let q = T::one() / fconst::<T>(SQRT_2);
    let alpha = sn / (fconst::<T>(2.0) * q);

    let (b0, b1, b2): (T, T, T);
    if highpass {
        b0 = (T::one() + cs) / fconst::<T>(2.0);
        b1 = -(T::one() + cs);
        b2 = (T::one() + cs) / fconst::<T>(2.0);
    } else {
        b0 = (T::one() - cs) / fconst::<T>(2.0);
        b1 = T::one() - cs;
        b2 = (T::one() - cs) / fconst::<T>(2.0);
    }
    let a0 = T::one() + alpha;
    let a1 = -fconst::<T>(2.0) * cs;
    let a2 = T::one() - alpha;

    let domega_dfc = fconst::<T>(2.0) * fconst::<T>(PI) / fs;
    let dcs_dfc = -sn * domega_dfc;
    let dsn_dfc = cs * domega_dfc;
    let dalpha_dfc = dsn_dfc / (fconst::<T>(2.0) * q);

    let (db0_dfc, db1_dfc, db2_dfc): (T, T, T);
    if highpass {
        db0_dfc = dcs_dfc / fconst::<T>(2.0);
        db1_dfc = -dcs_dfc;
        db2_dfc = dcs_dfc / fconst::<T>(2.0);
    } else {
        db0_dfc = -dcs_dfc / fconst::<T>(2.0);
        db1_dfc = -dcs_dfc;
        db2_dfc = -dcs_dfc / fconst::<T>(2.0);
    }
    let da0_dfc = dalpha_dfc;
    let da1_dfc = -fconst::<T>(2.0) * dcs_dfc;
    let da2_dfc = -dalpha_dfc;

    // Apply gain to numerator.
    let b0_g = b0 * gain;
    let b1_g = b1 * gain;
    let b2_g = b2 * gain;

    // Normalize by a0.
    let a0_sq = a0 * a0;
    let b0_n = b0_g / a0;
    let b1_n = b1_g / a0;
    let b2_n = b2_g / a0;
    let a1_n = a1 / a0;
    let a2_n = a2 / a0;

    // Gradients after gain application.
    let db0_g_dfc = db0_dfc * gain;
    let db1_g_dfc = db1_dfc * gain;
    let db2_g_dfc = db2_dfc * gain;
    let da0_g_dfc = da0_dfc;
    let da1_g_dfc = da1_dfc;
    let da2_g_dfc = da2_dfc;

    let db0_g_dgain = b0;
    let db1_g_dgain = b1;
    let db2_g_dgain = b2;

    // Gradients after normalization.
    let db0_n_dfc = (db0_g_dfc * a0 - b0_g * da0_g_dfc) / a0_sq;
    let db1_n_dfc = (db1_g_dfc * a0 - b1_g * da0_g_dfc) / a0_sq;
    let db2_n_dfc = (db2_g_dfc * a0 - b2_g * da0_g_dfc) / a0_sq;
    let da1_n_dfc = (da1_g_dfc * a0 - a1 * da0_g_dfc) / a0_sq;
    let da2_n_dfc = (da2_g_dfc * a0 - a2 * da0_g_dfc) / a0_sq;

    let db0_n_dgain = db0_g_dgain / a0;
    let db1_n_dgain = db1_g_dgain / a0;
    let db2_n_dgain = db2_g_dgain / a0;

    let mut coeffs = SectionCoeffs::<T>::zeros();
    coeffs.b = [b0_n, b1_n, b2_n];
    coeffs.a = [T::one(), a1_n, a2_n];
    coeffs.db_dparam = [
        [db0_n_dfc, db0_n_dgain, T::zero()],
        [db1_n_dfc, db1_n_dgain, T::zero()],
        [db2_n_dfc, db2_n_dgain, T::zero()],
    ];
    coeffs.da_dparam = [
        [T::zero(), T::zero(), T::zero()],
        [da1_n_dfc, T::zero(), T::zero()],
        [da2_n_dfc, T::zero(), T::zero()],
    ];
    coeffs
}

/// Compute normalized RBJ bandpass coefficients and physical parameter
/// gradients.
#[allow(clippy::similar_names)]
fn compute_bandpass<T: Scalar>(fc1: T, fc2: T, gain: T, fs: T) -> SectionCoeffs<T> {
    let omega1 = fconst::<T>(2.0) * fconst::<T>(PI) * fc1 / fs;
    let omega2 = fconst::<T>(2.0) * fconst::<T>(PI) * fc2 / fs;
    let omega_c = (omega1 + omega2) / fconst::<T>(2.0);
    let bw = (fc2 / fc1).log2();

    let sn_c = omega_c.sin();
    let cs_c = omega_c.cos();
    let c = fconst::<T>(2.0).ln() / fconst::<T>(2.0);
    let alpha = sn_c * (c * bw * omega_c / sn_c).sinh();

    let b0 = alpha;
    let b1 = T::zero();
    let b2 = -alpha;
    let a0 = T::one() + alpha;
    let a1 = -fconst::<T>(2.0) * cs_c;
    let a2 = T::one() - alpha;

    // Apply gain to numerator.
    let b0_g = b0 * gain;
    let b1_g = b1 * gain;
    let b2_g = b2 * gain;

    // Normalize by a0.
    let a0_sq = a0 * a0;
    let b0_n = b0_g / a0;
    let b1_n = b1_g / a0;
    let b2_n = b2_g / a0;
    let a1_n = a1 / a0;
    let a2_n = a2 / a0;

    // Derivatives of omega_c and BW w.r.t. physical cutoffs.
    let domega_c_dfc1 = fconst::<T>(PI) / fs;
    let domega_c_dfc2 = fconst::<T>(PI) / fs;
    let ln2 = fconst::<T>(2.0).ln();
    let dbw_dfc1 = -T::one() / (fc1 * ln2);
    let dbw_dfc2 = T::one() / (fc2 * ln2);

    // Derivative of alpha = sin(omega_c) * sinh(c * bw * omega_c / sin(omega_c)).
    let u = c * bw * omega_c / sn_c;
    let du_dfc = |domega_c_dfc: T, dbw_dfc: T| {
        let d_omega_over_sin = domega_c_dfc * (sn_c - omega_c * cs_c) / (sn_c * sn_c);
        dbw_dfc * omega_c / sn_c + bw * d_omega_over_sin
    };
    let du_dfc1 = du_dfc(domega_c_dfc1, dbw_dfc1);
    let du_dfc2 = du_dfc(domega_c_dfc2, dbw_dfc2);

    let dalpha_dfc =
        |domega_c_dfc: T, du_dfc: T| cs_c * domega_c_dfc * u.sinh() + sn_c * u.cosh() * c * du_dfc;
    let dalpha_dfc1 = dalpha_dfc(domega_c_dfc1, du_dfc1);
    let dalpha_dfc2 = dalpha_dfc(domega_c_dfc2, du_dfc2);

    // Raw derivatives of unnormalized coefficients.
    let db0_dfc1 = dalpha_dfc1;
    let db2_dfc1 = -dalpha_dfc1;
    let da0_dfc1 = dalpha_dfc1;
    let da1_dfc1 = fconst::<T>(2.0) * sn_c * domega_c_dfc1;
    let da2_dfc1 = -dalpha_dfc1;

    let db0_dfc2 = dalpha_dfc2;
    let db2_dfc2 = -dalpha_dfc2;
    let da0_dfc2 = dalpha_dfc2;
    let da1_dfc2 = fconst::<T>(2.0) * sn_c * domega_c_dfc2;
    let da2_dfc2 = -dalpha_dfc2;

    // After gain.
    let db0_g_dfc1 = db0_dfc1 * gain;
    let db2_g_dfc1 = db2_dfc1 * gain;
    let da0_g_dfc1 = da0_dfc1;
    let da1_g_dfc1 = da1_dfc1;
    let da2_g_dfc1 = da2_dfc1;

    let db0_g_dfc2 = db0_dfc2 * gain;
    let db2_g_dfc2 = db2_dfc2 * gain;
    let da0_g_dfc2 = da0_dfc2;
    let da1_g_dfc2 = da1_dfc2;
    let da2_g_dfc2 = da2_dfc2;

    let db0_g_dgain = b0;
    let db2_g_dgain = -b0;

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

    let mut coeffs = SectionCoeffs::<T>::zeros();
    coeffs.b = [b0_n, b1_n, b2_n];
    coeffs.a = [T::one(), a1_n, a2_n];
    coeffs.db_dparam = [
        [db0_n_dfc1, db0_n_dfc2, db0_n_dgain],
        [T::zero(), T::zero(), T::zero()],
        [db2_n_dfc1, db2_n_dfc2, db2_n_dgain],
    ];
    coeffs.da_dparam = [
        [T::zero(), T::zero(), T::zero()],
        [da1_n_dfc1, da1_n_dfc2, T::zero()],
        [da2_n_dfc1, da2_n_dfc2, T::zero()],
    ];
    coeffs
}

fn biquad_param_view<T: Scalar>(param: &ArrayD<T>) -> Result<ArrayView4<'_, T>, AutodiffError> {
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

fn biquad_param_grad_view_mut<T: Scalar>(
    param_grad: &mut ArrayD<T>,
) -> Result<ArrayViewMut4<'_, T>, AutodiffError> {
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

fn parallel_biquad_param_view<T: Scalar>(
    param: &ArrayD<T>,
) -> Result<ArrayView3<'_, T>, AutodiffError> {
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

fn parallel_biquad_param_grad_view_mut<T: Scalar>(
    param_grad: &mut ArrayD<T>,
) -> Result<ArrayViewMut3<'_, T>, AutodiffError> {
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
            AutodiffError::Message(format!("ParallelBiquad: failed to reshape param_grad: {e}"))
        })
}

/// Cached coefficient set for a [`Biquad`] module.
#[derive(Debug, Clone)]
struct BiquadCoeffCache<T> {
    b: Array4<Complex<T>>,
    a: Array4<Complex<T>>,
    db_dparam: Array5<T>,
    da_dparam: Array5<T>,
}

#[derive(Debug, Clone)]
struct ParallelBiquadCoeffCache<T> {
    b: Array4<Complex<T>>,
    a: Array4<Complex<T>>,
    db_dparam: Array4<T>,
    da_dparam: Array4<T>,
}

/// Fast FNV-1a hash over parameter values for cache invalidation.
fn hash_param<T: Scalar>(param: &ArrayD<T>) -> u64 {
    let mut h = fnv1a_init();
    if let Some(slice) = param.as_slice() {
        for &v in slice {
            h = fnv1a_step(h, v.hash_bits());
        }
    } else {
        for &v in param {
            h = fnv1a_step(h, v.hash_bits());
        }
    }
    h
}

/// Differentiable RBJ biquad with arbitrary input/output channel coupling.
#[derive(Debug, Clone)]
pub struct Biquad<T = f64> {
    /// FFT length.
    pub nfft: usize,
    /// Sample rate in Hz.
    pub fs: T,
    /// Number of cascaded SOS sections.
    pub n_sections: usize,
    /// Filter type (lowpass, highpass, or bandpass).
    pub filter_type: BiquadFilterType,
    /// Raw parameters, shape `(n_sections, P, n_out, n_in)`.
    pub param: ArrayD<T>,
    /// Accumulated parameter gradients, same shape as `param`.
    pub param_grad: ArrayD<T>,
    /// Anti-aliasing decay in dB.
    pub alias_decay_db: T,
    /// Coefficient cache keyed by parameter hash.
    coeff_cache: Option<BiquadCoeffCache<T>>,
    /// Hash of parameters used to build `coeff_cache`.
    param_hash: u64,
    /// Reusable working buffers for the backward pass.
    work_h: Array3<Complex<T>>,
    work_b_response: Array4<Complex<T>>,
    work_a_response: Array4<Complex<T>>,
    work_dl_dh: Array3<Complex<T>>,
    work_grad_input: ArrayD<Complex<T>>,
}

impl<T: BasisCache> Biquad<T> {
    /// Create a new biquad module with trainable unity gain and zero gradients.
    ///
    /// # Errors
    ///
    /// Returns an error if `nfft` is zero.
    pub fn new(
        nfft: usize,
        fs: T,
        n_sections: usize,
        filter_type: BiquadFilterType,
        n_out: usize,
        n_in: usize,
        alias_decay_db: T,
    ) -> Result<Self, AutodiffError> {
        if nfft == 0 {
            return Err(AutodiffError::Message(
                "Biquad: nfft must be greater than 0".to_string(),
            ));
        }
        if fs <= T::zero() || !fs.is_finite() {
            return Err(AutodiffError::Message(
                "Biquad: fs must be positive and finite".to_string(),
            ));
        }
        if n_sections == 0 || n_out == 0 || n_in == 0 {
            return Err(AutodiffError::Message(
                "Biquad: section and channel counts must be greater than 0".to_string(),
            ));
        }
        if !alias_decay_db.is_finite() {
            return Err(AutodiffError::Message(
                "Biquad: alias_decay_db must be finite".to_string(),
            ));
        }
        if !matches!(
            filter_type,
            BiquadFilterType::Lowpass | BiquadFilterType::Highpass | BiquadFilterType::Bandpass
        ) {
            return Err(AutodiffError::Message(format!(
                "Biquad: unsupported filter type {filter_type:?}"
            )));
        }
        let n_params = n_params_for(filter_type);
        let mut param = ArrayD::zeros(IxDyn(&[n_sections, n_params, n_out, n_in]));
        let gain_index = n_params - 1;
        for section in 0..n_sections {
            for out_ch in 0..n_out {
                for in_ch in 0..n_in {
                    param[[section, gain_index, out_ch, in_ch]] = T::one();
                    if filter_type == BiquadFilterType::Bandpass {
                        param[[section, 0, out_ch, in_ch]] = -fconst::<T>(3.0).ln();
                    }
                }
            }
        }
        Ok(Self {
            nfft,
            fs,
            n_sections,
            filter_type,
            param,
            param_grad: ArrayD::zeros(IxDyn(&[n_sections, n_params, n_out, n_in])),
            alias_decay_db,
            coeff_cache: None,
            param_hash: 0,
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

    /// Build the anti-aliasing envelope `[gamma^0, gamma^1, gamma^2]`.
    fn gamma(&self) -> [T; 3] {
        let gamma = fconst::<T>(10.0)
            .powf(-self.alias_decay_db.abs() / (fconst::<T>(20.0) * fconst::<T>(self.nfft as f64)));
        [T::one(), gamma, gamma * gamma]
    }

    /// Return a reference to the cached coefficients, recomputing only when
    /// `self.param` has changed since the last call.
    fn ensure_coeffs_cached(&mut self) -> Result<&BiquadCoeffCache<T>, AutodiffError> {
        let hash = hash_param(&self.param);
        if self.coeff_cache.is_none() || self.param_hash != hash {
            let (b, a, db_dparam, da_dparam) = self.build_coeffs_and_grads()?;
            self.param_hash = hash;
            self.coeff_cache = Some(BiquadCoeffCache::<T> {
                b,
                a,
                db_dparam,
                da_dparam,
            });
        }
        Ok(self.coeff_cache.as_ref().expect("cache just populated"))
    }

    /// Map raw parameters to normalized coefficients and parameter gradients.
    fn build_coeffs_and_grads(
        &self,
    ) -> Result<(Array4<Complex<T>>, Array4<Complex<T>>, Array5<T>, Array5<T>), AutodiffError> {
        let param = biquad_param_view(&self.param)?;
        let (n_sections, n_params, n_out, n_in) = param.dim();
        let mut b = Array4::zeros((n_sections, 3, n_out, n_in));
        let mut a = Array4::zeros((n_sections, 3, n_out, n_in));
        let mut db_dparam = Array5::zeros((n_sections, 3, n_params, n_out, n_in));
        let mut da_dparam = Array5::zeros((n_sections, 3, n_params, n_out, n_in));

        let half_fs = self.fs / fconst::<T>(2.0);

        for section in 0..n_sections {
            for out_ch in 0..n_out {
                for in_ch in 0..n_in {
                    let coeffs = match self.filter_type {
                        BiquadFilterType::Lowpass | BiquadFilterType::Highpass => {
                            let fc_raw = param[[section, 0, out_ch, in_ch]];
                            let gain_raw = param[[section, 1, out_ch, in_ch]];
                            let fc_norm = sigmoid(fc_raw);
                            let fc = fc_norm * half_fs;
                            let mut c = compute_lowpass_highpass(
                                fc,
                                gain_raw,
                                self.fs,
                                self.filter_type == BiquadFilterType::Highpass,
                            );
                            let dfc_dfc_raw = sigmoid_derivative_from_output(fc_norm) * half_fs;
                            for tap in 0..3 {
                                c.db_dparam[tap][0] *= dfc_dfc_raw;
                                c.da_dparam[tap][0] *= dfc_dfc_raw;
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
                            let mut c = compute_bandpass(fc1, fc2, gain_raw, self.fs);

                            let dfc1_raw = sigmoid_derivative_from_output(fc1_norm) * half_fs;
                            let dfc2_raw = sigmoid_derivative_from_output(fc2_norm) * half_fs;

                            for tap in 0..3 {
                                let db_df1 = c.db_dparam[tap][0];
                                let db_df2 = c.db_dparam[tap][1];
                                let db_dg = c.db_dparam[tap][2];
                                c.db_dparam[tap][0] =
                                    if swapped { db_df2 } else { db_df1 } * dfc1_raw;
                                c.db_dparam[tap][1] =
                                    if swapped { db_df1 } else { db_df2 } * dfc2_raw;
                                c.db_dparam[tap][2] = db_dg;

                                let da_df1 = c.da_dparam[tap][0];
                                let da_df2 = c.da_dparam[tap][1];
                                c.da_dparam[tap][0] =
                                    if swapped { da_df2 } else { da_df1 } * dfc1_raw;
                                c.da_dparam[tap][1] =
                                    if swapped { da_df1 } else { da_df2 } * dfc2_raw;
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
                        b[[section, tap, out_ch, in_ch]] = Complex::new(coeffs.b[tap], T::zero());
                        a[[section, tap, out_ch, in_ch]] = Complex::new(coeffs.a[tap], T::zero());
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

impl<T: BasisCache> DiffModule<T> for Biquad<T> {
    #[allow(clippy::too_many_lines)]
    fn forward(&self, input: &DiffTensor<T>) -> Result<DiffTensor<T>, AutodiffError> {
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

        // Build the frequency response directly from raw parameters, avoiding
        // gradient coefficient arrays and complex b/a buffers.  Numerator and
        // denominator products are accumulated across sections so only one
        // complex division per frequency bin is required.
        let param = biquad_param_view(&self.param)?;
        let n_sections = param.dim().0;
        let n_out = n_out_stored;

        let gamma = self.gamma();
        let basis = SosFrequencyBasis::new(self.nfft, &gamma);
        let mut num_acc = Array3::from_elem((n_bins, n_out, n_in), Complex::from(T::one()));
        let mut den_acc = Array3::from_elem((n_bins, n_out, n_in), Complex::from(T::one()));
        // Both accumulators are freshly allocated C-contiguous arrays and the
        // cached basis is C-contiguous by construction, so flat slices exist.
        let basis_slice = basis.response.as_slice().expect("SOS basis is contiguous");
        let num_slice = num_acc
            .as_slice_mut()
            .expect("numerator accumulator is contiguous");
        let den_slice = den_acc
            .as_slice_mut()
            .expect("denominator accumulator is contiguous");
        let half_fs = self.fs / fconst::<T>(2.0);

        for section in 0..n_sections {
            for out_ch in 0..n_out {
                for in_ch in 0..n_in {
                    let (b, a) = match self.filter_type {
                        BiquadFilterType::Lowpass | BiquadFilterType::Highpass => {
                            let fc_raw = param[[section, 0, out_ch, in_ch]];
                            let gain_raw = param[[section, 1, out_ch, in_ch]];
                            let fc_norm = sigmoid(fc_raw);
                            let fc = fc_norm * half_fs;
                            compute_lowpass_highpass_coeffs(
                                fc,
                                gain_raw,
                                self.fs,
                                self.filter_type == BiquadFilterType::Highpass,
                            )
                        }
                        BiquadFilterType::Bandpass => {
                            let fc1_raw = param[[section, 0, out_ch, in_ch]];
                            let fc2_raw = param[[section, 1, out_ch, in_ch]];
                            let gain_raw = param[[section, 2, out_ch, in_ch]];
                            let fc1_norm = sigmoid(fc1_raw);
                            let fc2_norm = sigmoid(fc2_raw);
                            let (fc_low_norm, fc_high_norm) = if fc1_norm <= fc2_norm {
                                (fc1_norm, fc2_norm)
                            } else {
                                (fc2_norm, fc1_norm)
                            };
                            let fc1 = fc_low_norm * half_fs;
                            let fc2 = fc_high_norm * half_fs;
                            compute_bandpass_coeffs(fc1, fc2, gain_raw, self.fs)
                        }
                        _ => {
                            return Err(AutodiffError::Message(format!(
                                "Biquad::forward: unsupported filter type {:?}",
                                self.filter_type
                            )));
                        }
                    };

                    for bin in 0..n_bins {
                        let z1 = basis_slice[n_bins + bin];
                        let z2 = basis_slice[2 * n_bins + bin];
                        let numerator = Complex::new(
                            b[0] + b[1] * z1.re + b[2] * z2.re,
                            b[1] * z1.im + b[2] * z2.im,
                        );
                        let denominator = Complex::new(
                            a[0] + a[1] * z1.re + a[2] * z2.re,
                            a[1] * z1.im + a[2] * z2.im,
                        );
                        let acc_idx = (bin * n_out + out_ch) * n_in + in_ch;
                        num_slice[acc_idx] *= numerator;
                        den_slice[acc_idx] *= denominator;
                    }
                }
            }
        }

        let mut output_shape: Vec<usize> = input_shape.to_vec();
        output_shape[2] = n_out;
        let mut output = ArrayD::zeros(IxDyn(&output_shape));

        if input_shape.len() == 3
            && let Some(input_data) = input.data.as_slice()
            && let Some(output_data) = output.as_slice_mut()
        {
            // Contiguous fast path: flat indexing, one division per bin.
            let batch = input_shape[0];
            for out_ch in 0..n_out {
                for in_ch in 0..n_in {
                    for bin in 0..n_bins {
                        let acc_idx = (bin * n_out + out_ch) * n_in + in_ch;
                        let h_val = num_slice[acc_idx] / den_slice[acc_idx];
                        for batch_index in 0..batch {
                            let frame = batch_index * n_bins + bin;
                            output_data[frame * n_out + out_ch] +=
                                input_data[frame * n_in + in_ch] * h_val;
                        }
                    }
                }
            }
            return Ok(DiffTensor::from_array(output));
        }

        for bin in 0..n_bins {
            for in_ch in 0..n_in {
                let input_axis2 = input.data.index_axis(Axis(2), in_ch);
                let input_bin = input_axis2.index_axis(Axis(1), bin);
                for out_ch in 0..n_out {
                    let acc_idx = (bin * n_out + out_ch) * n_in + in_ch;
                    let h_val = num_slice[acc_idx] / den_slice[acc_idx];
                    let mut output_axis2 = output.index_axis_mut(Axis(1), bin);
                    let mut output_bin = output_axis2.index_axis_mut(Axis(1), out_ch);
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
        input: &DiffTensor<T>,
        output: &DiffTensor<T>,
        grad_output: &DiffTensor<T>,
    ) -> Result<DiffTensor<T>, AutodiffError> {
        let input_shape = input.data.shape();
        let grad_shape = grad_output.data.shape();
        let output_shape = output.data.shape();
        if grad_shape != output_shape {
            return Err(AutodiffError::Message(format!(
                "Biquad::backward: grad_output shape {:?} does not match output shape {:?}",
                grad_shape, output_shape
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
        validate_spectral_gradient_shape("Biquad::backward", input_shape, grad_shape, n_out)?;
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

        let nfft = self.nfft;
        let gamma = self.gamma();

        self.ensure_coeffs_cached()?;
        let cache = self.coeff_cache.take().ok_or_else(|| {
            AutodiffError::Message("Biquad: coefficient cache missing".to_string())
        })?;
        let result = (|| {
            let b = &cache.b;
            let a = &cache.a;
            let db_dparam = &cache.db_dparam;
            let da_dparam = &cache.da_dparam;

            // Resize reusable working buffers when shapes change.
            if self.work_dl_dh.dim() != (n_bins, n_out, n_in) {
                self.work_dl_dh = Array3::zeros((n_bins, n_out, n_in));
            }
            if self.work_h.dim() != (n_bins, n_out, n_in) {
                self.work_h = Array3::zeros((n_bins, n_out, n_in));
            }
            if self.work_b_response.dim() != (n_sections, n_bins, n_out, n_in) {
                self.work_b_response = Array4::zeros((n_sections, n_bins, n_out, n_in));
            }
            if self.work_a_response.dim() != (n_sections, n_bins, n_out, n_in) {
                self.work_a_response = Array4::zeros((n_sections, n_bins, n_out, n_in));
            }
            if self.work_grad_input.shape() != input_shape {
                self.work_grad_input = ArrayD::zeros(IxDyn(input_shape));
            }
            self.work_grad_input
                .fill(Complex::new(T::zero(), T::zero()));

            // Compute dLoss/dH using real parts (MVP assumption: real time-domain signals).
            self.work_dl_dh.fill(Complex::new(T::zero(), T::zero()));
            if input_shape.len() == 3
                && grad_shape.len() == 3
                && let Some(grad_data) = grad_output.data.as_slice()
                && let Some(input_data) = input.data.as_slice()
                && let Some(dl_dh_data) = self.work_dl_dh.as_slice_mut()
            {
                // Contiguous fast path: flat indexing, no per-bin view creation.
                let batch = input_shape[0];
                for out_ch in 0..n_out {
                    for in_ch in 0..n_in {
                        for bin in 0..n_bins {
                            let mut accum = Complex::new(T::zero(), T::zero());
                            for batch_index in 0..batch {
                                let frame = batch_index * n_bins + bin;
                                accum += grad_data[frame * n_out + out_ch]
                                    * input_data[frame * n_in + in_ch].conj();
                            }
                            dl_dh_data[(bin * n_out + out_ch) * n_in + in_ch] = accum;
                        }
                    }
                }
            } else {
                for out_ch in 0..n_out {
                    let grad_axis2 = grad_output.data.index_axis(Axis(2), out_ch);
                    for in_ch in 0..n_in {
                        let input_axis2 = input.data.index_axis(Axis(2), in_ch);
                        for bin in 0..n_bins {
                            let grad_bin = grad_axis2.index_axis(Axis(1), bin);
                            let input_bin = input_axis2.index_axis(Axis(1), bin);
                            self.work_dl_dh[[bin, out_ch, in_ch]] = grad_bin
                                .iter()
                                .zip(input_bin.iter())
                                .map(|(grad, input)| *grad * input.conj())
                                .sum::<Complex<T>>();
                        }
                    }
                }
            }

            let basis = SosFrequencyBasis::new(nfft, &gamma);
            let (response_db, response_da) = sos_coefficient_vjp_with_basis(
                b,
                a,
                &basis,
                &self.work_dl_dh,
                &mut self.work_h,
                &mut self.work_b_response,
                &mut self.work_a_response,
            )?;

            // Accumulate parameter gradients.
            {
                let mut param_grad = biquad_param_grad_view_mut(&mut self.param_grad)?;
                for section in 0..n_sections {
                    for out_ch in 0..n_out {
                        for in_ch in 0..n_in {
                            for param_idx in 0..n_params {
                                let mut accum = T::zero();
                                for tap in 0..3 {
                                    accum += response_db[[section, tap, out_ch, in_ch]]
                                        * db_dparam[[section, tap, param_idx, out_ch, in_ch]]
                                        + response_da[[section, tap, out_ch, in_ch]]
                                            * da_dparam[[section, tap, param_idx, out_ch, in_ch]];
                                }
                                param_grad[[section, param_idx, out_ch, in_ch]] += accum;
                            }
                        }
                    }
                }
            }

            // Compute dLoss/dInput into the reusable buffer.
            if input_shape.len() == 3
                && grad_shape.len() == 3
                && let Some(h_data) = self.work_h.as_slice()
                && let Some(grad_data) = grad_output.data.as_slice()
                && let Some(grad_input_data) = self.work_grad_input.as_slice_mut()
            {
                // Contiguous fast path: flat indexing, no per-bin view creation.
                let batch = input_shape[0];
                for in_ch in 0..n_in {
                    for out_ch in 0..n_out {
                        for bin in 0..n_bins {
                            let h_conj = h_data[(bin * n_out + out_ch) * n_in + in_ch].conj();
                            for batch_index in 0..batch {
                                let frame = batch_index * n_bins + bin;
                                grad_input_data[frame * n_in + in_ch] +=
                                    grad_data[frame * n_out + out_ch] * h_conj;
                            }
                        }
                    }
                }
            } else {
                for in_ch in 0..n_in {
                    for out_ch in 0..n_out {
                        for bin in 0..n_bins {
                            let h_conj = self.work_h[[bin, out_ch, in_ch]].conj();
                            let grad_axis2 = grad_output.data.index_axis(Axis(2), out_ch);
                            let grad_bin = grad_axis2.index_axis(Axis(1), bin);
                            let mut input_axis2 =
                                self.work_grad_input.index_axis_mut(Axis(2), in_ch);
                            let mut input_bin = input_axis2.index_axis_mut(Axis(1), bin);
                            for (destination, &gradient) in
                                input_bin.iter_mut().zip(grad_bin.iter())
                            {
                                *destination += gradient * h_conj;
                            }
                        }
                    }
                }
            }
            let grad_input =
                std::mem::replace(&mut self.work_grad_input, ArrayD::zeros(IxDyn(&[])));

            Ok(DiffTensor::from_array(grad_input))
        })();
        self.coeff_cache = Some(cache);
        result
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

    fn parameters(&self) -> Vec<&ArrayD<T>> {
        vec![&self.param]
    }

    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<T>> {
        vec![&mut self.param]
    }

    fn gradients(&self) -> Vec<&ArrayD<T>> {
        vec![&self.param_grad]
    }

    fn zero_grad(&mut self) {
        self.param_grad.fill(T::zero());
    }
}

/// Differentiable RBJ biquad with a diagonal (per-channel) frequency response.
#[derive(Debug, Clone)]
pub struct ParallelBiquad<T = f64> {
    /// FFT length.
    pub nfft: usize,
    /// Sample rate in Hz.
    pub fs: T,
    /// Number of cascaded SOS sections.
    pub n_sections: usize,
    /// Filter type (lowpass, highpass, or bandpass).
    pub filter_type: BiquadFilterType,
    /// Raw parameters, shape `(n_sections, P, N)`.
    pub param: ArrayD<T>,
    /// Accumulated parameter gradients, same shape as `param`.
    pub param_grad: ArrayD<T>,
    /// Anti-aliasing decay in dB.
    pub alias_decay_db: T,
    coeff_cache: Option<ParallelBiquadCoeffCache<T>>,
    param_hash: u64,
    work_h: ndarray::Array3<Complex<T>>,
    work_b_response: ndarray::Array4<Complex<T>>,
    work_a_response: ndarray::Array4<Complex<T>>,
    work_dl_dh: ndarray::Array3<Complex<T>>,
}

impl<T: BasisCache> ParallelBiquad<T> {
    /// Create a new parallel biquad module with trainable unity gain and zero
    /// gradients.
    ///
    /// # Errors
    ///
    /// Returns an error if `nfft` is zero.
    pub fn new(
        nfft: usize,
        fs: T,
        n_sections: usize,
        filter_type: BiquadFilterType,
        n_channels: usize,
        alias_decay_db: T,
    ) -> Result<Self, AutodiffError> {
        if nfft == 0 {
            return Err(AutodiffError::Message(
                "ParallelBiquad: nfft must be greater than 0".to_string(),
            ));
        }
        if fs <= T::zero() || !fs.is_finite() {
            return Err(AutodiffError::Message(
                "ParallelBiquad: fs must be positive and finite".to_string(),
            ));
        }
        if n_sections == 0 || n_channels == 0 {
            return Err(AutodiffError::Message(
                "ParallelBiquad: section and channel counts must be greater than 0".to_string(),
            ));
        }
        if !alias_decay_db.is_finite() {
            return Err(AutodiffError::Message(
                "ParallelBiquad: alias_decay_db must be finite".to_string(),
            ));
        }
        if !matches!(
            filter_type,
            BiquadFilterType::Lowpass | BiquadFilterType::Highpass | BiquadFilterType::Bandpass
        ) {
            return Err(AutodiffError::Message(format!(
                "ParallelBiquad: unsupported filter type {filter_type:?}"
            )));
        }
        let n_params = n_params_for(filter_type);
        let mut param = ArrayD::zeros(IxDyn(&[n_sections, n_params, n_channels]));
        let gain_index = n_params - 1;
        for section in 0..n_sections {
            for ch in 0..n_channels {
                param[[section, gain_index, ch]] = T::one();
                if filter_type == BiquadFilterType::Bandpass {
                    param[[section, 0, ch]] = -fconst::<T>(3.0).ln();
                }
            }
        }
        Ok(Self {
            nfft,
            fs,
            n_sections,
            filter_type,
            param,
            param_grad: ArrayD::zeros(IxDyn(&[n_sections, n_params, n_channels])),
            alias_decay_db,
            coeff_cache: None,
            param_hash: 0,
            work_h: ndarray::Array3::zeros((0, 0, 0)),
            work_b_response: ndarray::Array4::zeros((0, 0, 0, 0)),
            work_a_response: ndarray::Array4::zeros((0, 0, 0, 0)),
            work_dl_dh: ndarray::Array3::zeros((0, 0, 0)),
        })
    }

    fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }

    /// Build the anti-aliasing envelope `[gamma^0, gamma^1, gamma^2]`.
    fn gamma(&self) -> [T; 3] {
        let gamma = fconst::<T>(10.0)
            .powf(-self.alias_decay_db.abs() / (fconst::<T>(20.0) * fconst::<T>(self.nfft as f64)));
        [T::one(), gamma, gamma * gamma]
    }

    fn ensure_coeffs_cached(&mut self) -> Result<&ParallelBiquadCoeffCache<T>, AutodiffError> {
        let hash = hash_param(&self.param);
        if self.coeff_cache.is_none() || self.param_hash != hash {
            let (b, a, db_dparam, da_dparam) = self.build_coeffs_and_grads()?;
            let b_dim = b.dim();
            let a_dim = a.dim();
            let b = b
                .into_shape_with_order((b_dim.0, 3, b_dim.2, 1))
                .map_err(|e| {
                    AutodiffError::Message(format!("ParallelBiquad: failed to reshape b: {e}"))
                })?;
            let a = a
                .into_shape_with_order((a_dim.0, 3, a_dim.2, 1))
                .map_err(|e| {
                    AutodiffError::Message(format!("ParallelBiquad: failed to reshape a: {e}"))
                })?;
            self.param_hash = hash;
            self.coeff_cache = Some(ParallelBiquadCoeffCache::<T> {
                b,
                a,
                db_dparam,
                da_dparam,
            });
        }
        Ok(self.coeff_cache.as_ref().expect("cache just populated"))
    }

    /// Map raw parameters to normalized coefficients and parameter gradients.
    fn build_coeffs_and_grads(
        &self,
    ) -> Result<(Array3<Complex<T>>, Array3<Complex<T>>, Array4<T>, Array4<T>), AutodiffError> {
        let param = parallel_biquad_param_view(&self.param)?;
        let (n_sections, n_params, n_channels) = param.dim();
        let mut b = Array3::zeros((n_sections, 3, n_channels));
        let mut a = Array3::zeros((n_sections, 3, n_channels));
        let mut db_dparam = Array4::zeros((n_sections, 3, n_params, n_channels));
        let mut da_dparam = Array4::zeros((n_sections, 3, n_params, n_channels));

        let half_fs = self.fs / fconst::<T>(2.0);

        for section in 0..n_sections {
            for ch in 0..n_channels {
                let coeffs = match self.filter_type {
                    BiquadFilterType::Lowpass | BiquadFilterType::Highpass => {
                        let fc_raw = param[[section, 0, ch]];
                        let gain_raw = param[[section, 1, ch]];
                        let fc_norm = sigmoid(fc_raw);
                        let fc = fc_norm * half_fs;
                        let mut c = compute_lowpass_highpass(
                            fc,
                            gain_raw,
                            self.fs,
                            self.filter_type == BiquadFilterType::Highpass,
                        );
                        let dfc_dfc_raw = sigmoid_derivative_from_output(fc_norm) * half_fs;
                        for tap in 0..3 {
                            c.db_dparam[tap][0] *= dfc_dfc_raw;
                            c.da_dparam[tap][0] *= dfc_dfc_raw;
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
                        let mut c = compute_bandpass(fc1, fc2, gain_raw, self.fs);

                        let dfc1_raw = sigmoid_derivative_from_output(fc1_norm) * half_fs;
                        let dfc2_raw = sigmoid_derivative_from_output(fc2_norm) * half_fs;

                        for tap in 0..3 {
                            let db_df1 = c.db_dparam[tap][0];
                            let db_df2 = c.db_dparam[tap][1];
                            let db_dg = c.db_dparam[tap][2];
                            c.db_dparam[tap][0] = if swapped { db_df2 } else { db_df1 } * dfc1_raw;
                            c.db_dparam[tap][1] = if swapped { db_df1 } else { db_df2 } * dfc2_raw;
                            c.db_dparam[tap][2] = db_dg;

                            let da_df1 = c.da_dparam[tap][0];
                            let da_df2 = c.da_dparam[tap][1];
                            c.da_dparam[tap][0] = if swapped { da_df2 } else { da_df1 } * dfc1_raw;
                            c.da_dparam[tap][1] = if swapped { da_df1 } else { da_df2 } * dfc2_raw;
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
                    b[[section, tap, ch]] = Complex::new(coeffs.b[tap], T::zero());
                    a[[section, tap, ch]] = Complex::new(coeffs.a[tap], T::zero());
                    for param_idx in 0..n_params {
                        db_dparam[[section, tap, param_idx, ch]] = coeffs.db_dparam[tap][param_idx];
                        da_dparam[[section, tap, param_idx, ch]] = coeffs.da_dparam[tap][param_idx];
                    }
                }
            }
        }

        Ok((b, a, db_dparam, da_dparam))
    }
}

impl<T: BasisCache> DiffModule<T> for ParallelBiquad<T> {
    fn forward(&self, input: &DiffTensor<T>) -> Result<DiffTensor<T>, AutodiffError> {
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
        if input_shape.len() == 3
            && let Some(h_data) = h.as_slice()
            && let Some(input_data) = input.data.as_slice()
            && let Some(output_data) = output.as_slice_mut()
        {
            // Contiguous fast path: flat indexing, no per-bin view creation.
            let batch = input_shape[0];
            for ch in 0..n_channels {
                for bin in 0..n_bins {
                    let h_val = h_data[bin * n_channels + ch];
                    for batch_index in 0..batch {
                        let frame = batch_index * n_bins + bin;
                        output_data[frame * n_channels + ch] +=
                            input_data[frame * n_channels + ch] * h_val;
                    }
                }
            }
            return Ok(DiffTensor::from_array(output));
        }
        for ch in 0..n_channels {
            for bin in 0..n_bins {
                let h_val = h[[bin, ch]];
                let input_axis2 = input.data.index_axis(Axis(2), ch);
                let input_bin = input_axis2.index_axis(Axis(1), bin);
                let mut output_axis2 = output.index_axis_mut(Axis(2), ch);
                let mut output_bin = output_axis2.index_axis_mut(Axis(1), bin);
                for (destination, &source) in output_bin.iter_mut().zip(input_bin.iter()) {
                    *destination += source * h_val;
                }
            }
        }

        Ok(DiffTensor::from_array(output))
    }

    #[allow(clippy::too_many_lines)]
    fn backward(
        &mut self,
        input: &DiffTensor<T>,
        output: &DiffTensor<T>,
        grad_output: &DiffTensor<T>,
    ) -> Result<DiffTensor<T>, AutodiffError> {
        let input_shape = input.data.shape();
        let grad_shape = grad_output.data.shape();
        let output_shape = output.data.shape();
        if grad_shape != output_shape {
            return Err(AutodiffError::Message(format!(
                "ParallelBiquad::backward: grad_output shape {:?} does not match output shape {:?}",
                grad_shape, output_shape
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
        validate_spectral_gradient_shape(
            "ParallelBiquad::backward",
            input_shape,
            grad_shape,
            n_channels_stored,
        )?;
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

        self.ensure_coeffs_cached()?;
        let cache = self.coeff_cache.take().ok_or_else(|| {
            AutodiffError::Message("ParallelBiquad: coefficient cache missing".to_string())
        })?;
        let result = (|| {
            let b = &cache.b;
            let a = &cache.a;
            let db_dparam = &cache.db_dparam;
            let da_dparam = &cache.da_dparam;
            let gamma = self.gamma();
            let basis = SosFrequencyBasis::new(self.nfft, &gamma);

            if self.work_dl_dh.dim() != (n_bins, n_channels, 1) {
                self.work_dl_dh = Array3::zeros((n_bins, n_channels, 1));
            }
            if self.work_h.dim() != (n_bins, n_channels, 1) {
                self.work_h = Array3::zeros((n_bins, n_channels, 1));
            }
            if self.work_b_response.dim() != (n_sections, n_bins, n_channels, 1) {
                self.work_b_response = Array4::zeros((n_sections, n_bins, n_channels, 1));
            }
            if self.work_a_response.dim() != (n_sections, n_bins, n_channels, 1) {
                self.work_a_response = Array4::zeros((n_sections, n_bins, n_channels, 1));
            }
            self.work_dl_dh.fill(Complex::new(T::zero(), T::zero()));
            if input_shape.len() == 3
                && grad_shape.len() == 3
                && let Some(grad_data) = grad_output.data.as_slice()
                && let Some(input_data) = input.data.as_slice()
                && let Some(dl_dh_data) = self.work_dl_dh.as_slice_mut()
            {
                // Contiguous fast path: flat indexing, no per-bin view creation.
                let batch = input_shape[0];
                for ch in 0..n_channels {
                    for bin in 0..n_bins {
                        let mut accum = Complex::new(T::zero(), T::zero());
                        for batch_index in 0..batch {
                            let frame = batch_index * n_bins + bin;
                            accum += grad_data[frame * n_channels + ch]
                                * input_data[frame * n_channels + ch].conj();
                        }
                        dl_dh_data[bin * n_channels + ch] = accum;
                    }
                }
            } else {
                for ch in 0..n_channels {
                    let grad_axis2 = grad_output.data.index_axis(Axis(2), ch);
                    let input_axis2 = input.data.index_axis(Axis(2), ch);
                    for bin in 0..n_bins {
                        let grad_bin = grad_axis2.index_axis(Axis(1), bin);
                        let input_bin = input_axis2.index_axis(Axis(1), bin);
                        self.work_dl_dh[[bin, ch, 0]] = grad_bin
                            .iter()
                            .zip(input_bin.iter())
                            .map(|(grad, input)| *grad * input.conj())
                            .sum::<Complex<T>>();
                    }
                }
            }

            let (response_db, response_da) = sos_coefficient_vjp_with_basis(
                b,
                a,
                &basis,
                &self.work_dl_dh,
                &mut self.work_h,
                &mut self.work_b_response,
                &mut self.work_a_response,
            )?;

            {
                let mut param_grad = parallel_biquad_param_grad_view_mut(&mut self.param_grad)?;
                for section in 0..n_sections {
                    for ch in 0..n_channels {
                        for param_idx in 0..n_params {
                            let mut accum = T::zero();
                            for tap in 0..3 {
                                accum += response_db[[section, tap, ch, 0]]
                                    * db_dparam[[section, tap, param_idx, ch]]
                                    + response_da[[section, tap, ch, 0]]
                                        * da_dparam[[section, tap, param_idx, ch]];
                            }
                            param_grad[[section, param_idx, ch]] += accum;
                        }
                    }
                }
            }

            let mut grad_input = ArrayD::zeros(IxDyn(input_shape));
            if input_shape.len() == 3
                && grad_shape.len() == 3
                && let Some(h_data) = self.work_h.as_slice()
                && let Some(grad_data) = grad_output.data.as_slice()
                && let Some(grad_input_data) = grad_input.as_slice_mut()
            {
                // Contiguous fast path: flat indexing, no per-bin view creation.
                let batch = input_shape[0];
                for ch in 0..n_channels {
                    for bin in 0..n_bins {
                        let h_conj = h_data[bin * n_channels + ch].conj();
                        for batch_index in 0..batch {
                            let frame = batch_index * n_bins + bin;
                            grad_input_data[frame * n_channels + ch] +=
                                grad_data[frame * n_channels + ch] * h_conj;
                        }
                    }
                }
            } else {
                for ch in 0..n_channels {
                    for bin in 0..n_bins {
                        let h_conj = self.work_h[[bin, ch, 0]].conj();
                        let grad_axis2 = grad_output.data.index_axis(Axis(2), ch);
                        let grad_bin = grad_axis2.index_axis(Axis(1), bin);
                        let mut input_axis2 = grad_input.index_axis_mut(Axis(2), ch);
                        let mut input_bin = input_axis2.index_axis_mut(Axis(1), bin);
                        for (destination, &gradient) in input_bin.iter_mut().zip(grad_bin.iter()) {
                            *destination += gradient * h_conj;
                        }
                    }
                }
            }

            Ok(DiffTensor::from_array(grad_input))
        })();
        self.coeff_cache = Some(cache);
        result
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

    fn parameters(&self) -> Vec<&ArrayD<T>> {
        vec![&self.param]
    }

    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<T>> {
        vec![&mut self.param]
    }

    fn gradients(&self) -> Vec<&ArrayD<T>> {
        vec![&self.param_grad]
    }

    fn zero_grad(&mut self) {
        self.param_grad.fill(T::zero());
    }
}

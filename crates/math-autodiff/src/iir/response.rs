use ndarray::{Array2, Array3, Array4, Array5, ArrayBase, Axis, Data, Ix4};
use num_complex::Complex;
use rustfft::FftPlanner;

use crate::error::AutodiffError;

const DEFAULT_GAMMA: [f64; 3] = [1.0, 1.0, 1.0];

/// Response of a cascade of SOS sections.
#[derive(Debug, Clone)]
pub struct SosResponse {
    /// H(f) shape (M, `N_out`, `N_in`)
    pub h: Array3<Complex<f64>>,
    /// dH/d(b_{k,t}) shape (M, K, 3, `N_out`, `N_in`)
    pub dh_db: Array5<Complex<f64>>,
    /// dH/d(a_{k,t}) shape (M, K, 3, `N_out`, `N_in`)
    pub dh_da: Array5<Complex<f64>>,
}

/// Validate common SOS inputs and resolve the gamma envelope.
fn resolve_gamma(gamma: Option<&[f64; 3]>) -> [f64; 3] {
    gamma.copied().unwrap_or(DEFAULT_GAMMA)
}

/// Compute H(f) and its analytical Jacobian w.r.t. b/a coefficients.
///
/// # Arguments
/// * `b` - numerator coefficients, shape `(K, 3, N_out, N_in)`.
/// * `a` - denominator coefficients, same shape.
/// * `nfft` - FFT length.
/// * `gamma` - anti-alias envelope `[gamma^0, gamma^1, gamma^2]`.
///
/// # Panics
///
/// Panics if `b` and `a` have different shapes.
#[must_use]
pub fn sos_response(
    b: &Array4<Complex<f64>>,
    a: &Array4<Complex<f64>>,
    nfft: usize,
    gamma: &[f64; 3],
) -> SosResponse {
    assert_eq!(
        b.dim(),
        a.dim(),
        "sos_response: b and a must have the same shape"
    );
    let b_view = b.view();
    let a_view = a.view();
    sos_response_impl(&b_view, &a_view, nfft, gamma)
}

fn sos_response_impl<S>(
    b: &ArrayBase<S, Ix4>,
    a: &ArrayBase<S, Ix4>,
    nfft: usize,
    gamma: &[f64; 3],
) -> SosResponse
where
    S: Data<Elem = Complex<f64>>,
{
    let (n_sections, _, n_out, n_in) = b.dim();
    let n_bins = nfft / 2 + 1;

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(nfft);

    // Precompute FFT(gamma[t] * e_t) for each tap t. Shape (3, n_bins).
    let mut fft_envelope = Array2::zeros((3, n_bins));
    for tap in 0..3 {
        let mut buf = vec![Complex::<f64>::default(); nfft];
        buf[tap] = Complex::from(gamma[tap]);
        fft.process(&mut buf);
        for bin in 0..n_bins {
            fft_envelope[[tap, bin]] = buf[bin];
        }
    }

    // Compute B_k and A_k for every section, and accumulate H = prod_k B_k/A_k.
    let mut h = Array3::from_elem((n_bins, n_out, n_in), Complex::from(1.0));
    let mut b_response = Array4::zeros((n_sections, n_bins, n_out, n_in));
    let mut a_response = Array4::zeros((n_sections, n_bins, n_out, n_in));

    for section in 0..n_sections {
        // Numerator B_section(f).
        for out_ch in 0..n_out {
            for in_ch in 0..n_in {
                let mut buf = vec![Complex::<f64>::default(); nfft];
                for tap in 0..3 {
                    buf[tap] = b[[section, tap, out_ch, in_ch]] * gamma[tap];
                }
                fft.process(&mut buf);
                for bin in 0..n_bins {
                    let val = buf[bin];
                    b_response[[section, bin, out_ch, in_ch]] = val;
                    h[[bin, out_ch, in_ch]] *= val;
                }
            }
        }

        // Denominator A_section(f).
        for out_ch in 0..n_out {
            for in_ch in 0..n_in {
                let mut buf = vec![Complex::<f64>::default(); nfft];
                for tap in 0..3 {
                    buf[tap] = a[[section, tap, out_ch, in_ch]] * gamma[tap];
                }
                fft.process(&mut buf);
                for bin in 0..n_bins {
                    let val = buf[bin];
                    a_response[[section, bin, out_ch, in_ch]] = val;
                    h[[bin, out_ch, in_ch]] /= val;
                }
            }
        }
    }

    // Analytical Jacobians.
    let mut jacobian_b = Array5::zeros((n_bins, n_sections, 3, n_out, n_in));
    let mut jacobian_a = Array5::zeros((n_bins, n_sections, 3, n_out, n_in));

    for section in 0..n_sections {
        for tap in 0..3 {
            for bin in 0..n_bins {
                let envelope_bin = fft_envelope[[tap, bin]];
                for out_ch in 0..n_out {
                    for in_ch in 0..n_in {
                        let b_bin = b_response[[section, bin, out_ch, in_ch]];
                        let a_bin = a_response[[section, bin, out_ch, in_ch]];
                        let mut other_sections = Complex::new(1.0, 0.0);
                        for other in 0..n_sections {
                            if other != section {
                                other_sections *= b_response[[other, bin, out_ch, in_ch]]
                                    / a_response[[other, bin, out_ch, in_ch]];
                            }
                        }

                        jacobian_b[[bin, section, tap, out_ch, in_ch]] =
                            other_sections * envelope_bin / a_bin;
                        jacobian_a[[bin, section, tap, out_ch, in_ch]] =
                            -other_sections * b_bin * envelope_bin / (a_bin * a_bin);
                    }
                }
            }
        }
    }

    SosResponse {
        h,
        dh_db: jacobian_b,
        dh_da: jacobian_a,
    }
}

/// Compute the complex frequency response of a cascade of second-order
/// sections.
///
/// Coefficients have shape `(K, 3, N_out, N_in)` where `K` is the number of
/// SOS sections, `3` is the number of taps, and `N_out`/`N_in` are the output
/// and input channel counts. The returned response has shape
/// `(M, N_out, N_in)` with `M = nfft / 2 + 1`.
///
/// If `gamma` is `None`, an identity envelope `[1.0, 1.0, 1.0]` is used.
///
/// # Errors
///
/// Returns `AutodiffError` if `b` and `a` have different shapes or if `nfft`
/// is zero.
pub fn sos_frequency_response(
    b: &Array4<Complex<f64>>,
    a: &Array4<Complex<f64>>,
    nfft: usize,
    gamma: Option<&[f64; 3]>,
) -> Result<Array3<Complex<f64>>, AutodiffError> {
    validate_sos_4d_inputs(b, a, nfft)?;
    let gamma = resolve_gamma(gamma);
    let b_view = b.view();
    let a_view = a.view();
    let resp = sos_response_impl(&b_view, &a_view, nfft, &gamma);
    Ok(resp.h)
}

/// Compute the analytical Jacobian of the SOS frequency response w.r.t. the
/// numerator and denominator coefficients.
///
/// Coefficients have shape `(K, 3, N_out, N_in)`. The returned Jacobians have
/// shape `(M, K, 3, N_out, N_in)` with `M = nfft / 2 + 1`.
///
/// If `gamma` is `None`, an identity envelope `[1.0, 1.0, 1.0]` is used.
///
/// # Errors
///
/// Returns `AutodiffError` if `b` and `a` have different shapes or if `nfft`
/// is zero.
#[allow(clippy::type_complexity)]
pub fn sos_frequency_response_jacobian(
    b: &Array4<Complex<f64>>,
    a: &Array4<Complex<f64>>,
    nfft: usize,
    gamma: Option<&[f64; 3]>,
) -> Result<(Array5<Complex<f64>>, Array5<Complex<f64>>), AutodiffError> {
    validate_sos_4d_inputs(b, a, nfft)?;
    let gamma = resolve_gamma(gamma);
    let b_view = b.view();
    let a_view = a.view();
    let resp = sos_response_impl(&b_view, &a_view, nfft, &gamma);
    Ok((resp.dh_db, resp.dh_da))
}

/// Compute the complex frequency response of a cascade of second-order
/// sections with a flat channel layout.
///
/// Coefficients have shape `(K, 3, N)` where `K` is the number of SOS sections,
/// `3` is the number of taps, and `N` is the number of channels. The returned
/// response has shape `(M, N)` with `M = nfft / 2 + 1`.
///
/// This is a convenience wrapper that internally reshapes the 3-D input to
/// `(K, 3, N, 1)` so that each channel is treated as an independent output.
///
/// If `gamma` is `None`, an identity envelope `[1.0, 1.0, 1.0]` is used.
///
/// # Errors
///
/// Returns `AutodiffError` if `b` and `a` have different shapes, if the
/// second axis is not `3`, or if `nfft` is zero.
pub fn sos_frequency_response_parallel(
    b: &Array3<Complex<f64>>,
    a: &Array3<Complex<f64>>,
    nfft: usize,
    gamma: Option<&[f64; 3]>,
) -> Result<Array2<Complex<f64>>, AutodiffError> {
    validate_sos_3d_inputs(b, a, nfft)?;
    let gamma = resolve_gamma(gamma);
    let (n_sections, _, n_channels) = b.dim();
    let b_view = b.view();
    let a_view = a.view();
    let b4 = b_view
        .to_shape((n_sections, 3, n_channels, 1))
        .map_err(|e| AutodiffError::Message(e.to_string()))?;
    let a4 = a_view
        .to_shape((n_sections, 3, n_channels, 1))
        .map_err(|e| AutodiffError::Message(e.to_string()))?;
    let resp = sos_response_impl(&b4, &a4, nfft, &gamma);
    Ok(resp.h.index_axis(Axis(2), 0).to_owned())
}

/// Compute the analytical Jacobian of the SOS frequency response for a flat
/// channel layout.
///
/// Coefficients have shape `(K, 3, N)`. The returned Jacobians have shape
/// `(M, K, 3, N)` with `M = nfft / 2 + 1`.
///
/// If `gamma` is `None`, an identity envelope `[1.0, 1.0, 1.0]` is used.
///
/// # Errors
///
/// Returns `AutodiffError` if `b` and `a` have different shapes, if the
/// second axis is not `3`, or if `nfft` is zero.
#[allow(clippy::type_complexity)]
pub fn sos_frequency_response_jacobian_parallel(
    b: &Array3<Complex<f64>>,
    a: &Array3<Complex<f64>>,
    nfft: usize,
    gamma: Option<&[f64; 3]>,
) -> Result<(Array4<Complex<f64>>, Array4<Complex<f64>>), AutodiffError> {
    validate_sos_3d_inputs(b, a, nfft)?;
    let gamma = resolve_gamma(gamma);
    let (n_sections, _, n_channels) = b.dim();
    let b_view = b.view();
    let a_view = a.view();
    let b4 = b_view
        .to_shape((n_sections, 3, n_channels, 1))
        .map_err(|e| AutodiffError::Message(e.to_string()))?;
    let a4 = a_view
        .to_shape((n_sections, 3, n_channels, 1))
        .map_err(|e| AutodiffError::Message(e.to_string()))?;
    let resp = sos_response_impl(&b4, &a4, nfft, &gamma);
    Ok((
        resp.dh_db.index_axis(Axis(4), 0).to_owned(),
        resp.dh_da.index_axis(Axis(4), 0).to_owned(),
    ))
}

fn validate_sos_4d_inputs(
    b: &Array4<Complex<f64>>,
    a: &Array4<Complex<f64>>,
    nfft: usize,
) -> Result<(), AutodiffError> {
    if b.dim() != a.dim() {
        return Err(AutodiffError::Message(format!(
            "sos_frequency_response: b and a must have the same shape, got {:?} and {:?}",
            b.dim(),
            a.dim()
        )));
    }
    let (_, n_taps, _, _) = b.dim();
    if n_taps != 3 {
        return Err(AutodiffError::Message(format!(
            "sos_frequency_response: second axis must be 3, got {n_taps}"
        )));
    }
    if nfft == 0 {
        return Err(AutodiffError::Message(
            "sos_frequency_response: nfft must be greater than 0".to_string(),
        ));
    }
    Ok(())
}

fn validate_sos_3d_inputs(
    b: &Array3<Complex<f64>>,
    a: &Array3<Complex<f64>>,
    nfft: usize,
) -> Result<(), AutodiffError> {
    if b.dim() != a.dim() {
        return Err(AutodiffError::Message(format!(
            "sos_frequency_response_parallel: b and a must have the same shape, got {:?} and {:?}",
            b.dim(),
            a.dim()
        )));
    }
    let (_, n_taps, _) = b.dim();
    if n_taps != 3 {
        return Err(AutodiffError::Message(format!(
            "sos_frequency_response_parallel: second axis must be 3, got {n_taps}"
        )));
    }
    if nfft == 0 {
        return Err(AutodiffError::Message(
            "sos_frequency_response_parallel: nfft must be greater than 0".to_string(),
        ));
    }
    Ok(())
}

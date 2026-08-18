use ndarray::{Array2, Array3, Array4, Array5, ArrayBase, Axis, Data, Ix4};
use num_complex::Complex;
use std::{cell::RefCell, collections::HashMap, sync::Arc};

use crate::error::AutodiffError;

const DEFAULT_GAMMA: [f64; 3] = [1.0, 1.0, 1.0];
const SOS_BIN_CHUNK: usize = 8;
type SosBasisKey = (usize, [u64; 3]);
type SosBasisCache = HashMap<SosBasisKey, Arc<Array2<Complex<f64>>>>;

thread_local! {
    static SOS_BASIS_CACHE: RefCell<SosBasisCache> = RefCell::new(HashMap::new());
}

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

#[allow(
    clippy::cast_precision_loss,
    reason = "FFT indices and practical audio FFT sizes are exactly representable as f64"
)]
fn tap_frequency_response(nfft: usize, gamma: &[f64; 3]) -> Array2<Complex<f64>> {
    let n_bins = nfft / 2 + 1;
    let mut response = Array2::zeros((3, n_bins));
    for bin in 0..n_bins {
        let angle = -std::f64::consts::TAU * bin as f64 / nfft as f64;
        let z1 = Complex::new(angle.cos(), angle.sin());
        response[[0, bin]] = Complex::from(gamma[0]);
        response[[1, bin]] = z1 * gamma[1];
        response[[2, bin]] = z1 * z1 * gamma[2];
    }
    response
}

/// Parameter-independent frequency basis for three-tap SOS sections.
#[derive(Debug, Clone)]
pub(crate) struct SosFrequencyBasis {
    nfft: usize,
    pub(crate) response: Arc<Array2<Complex<f64>>>,
}

impl SosFrequencyBasis {
    pub(crate) fn new(nfft: usize, gamma: &[f64; 3]) -> Self {
        let key = (nfft, gamma.map(f64::to_bits));
        let response = SOS_BASIS_CACHE.with(|cache| {
            cache
                .borrow_mut()
                .entry(key)
                .or_insert_with(|| Arc::new(tap_frequency_response(nfft, gamma)))
                .clone()
        });
        Self { nfft, response }
    }

    fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }
}

fn sos_frequency_response_impl<S>(
    b: &ArrayBase<S, Ix4>,
    a: &ArrayBase<S, Ix4>,
    basis: &SosFrequencyBasis,
) -> Array3<Complex<f64>>
where
    S: Data<Elem = Complex<f64>>,
{
    let (n_sections, _, n_out, n_in) = b.dim();
    let n_bins = basis.n_bins();
    let tap_response = &basis.response;
    let mut h = Array3::from_elem((n_bins, n_out, n_in), Complex::from(1.0));

    // Process bins in small chunks so that the per-bin H values can live in
    // registers across all SOS sections, and the per-bin basis values are
    // loaded only once per chunk.
    let n_chunked = n_bins / SOS_BIN_CHUNK * SOS_BIN_CHUNK;

    for bin_start in (0..n_chunked).step_by(SOS_BIN_CHUNK) {
        let mut basis0_chunk = [Complex::from(0.0); SOS_BIN_CHUNK];
        let mut basis1_chunk = [Complex::from(0.0); SOS_BIN_CHUNK];
        let mut basis2_chunk = [Complex::from(0.0); SOS_BIN_CHUNK];
        for i in 0..SOS_BIN_CHUNK {
            let bin = bin_start + i;
            basis0_chunk[i] = tap_response[[0, bin]];
            basis1_chunk[i] = tap_response[[1, bin]];
            basis2_chunk[i] = tap_response[[2, bin]];
        }

        for out_ch in 0..n_out {
            for in_ch in 0..n_in {
                let mut h_chunk = [Complex::from(1.0); SOS_BIN_CHUNK];
                for section in 0..n_sections {
                    let b0 = b[[section, 0, out_ch, in_ch]];
                    let b1 = b[[section, 1, out_ch, in_ch]];
                    let b2 = b[[section, 2, out_ch, in_ch]];
                    let a0 = a[[section, 0, out_ch, in_ch]];
                    let a1 = a[[section, 1, out_ch, in_ch]];
                    let a2 = a[[section, 2, out_ch, in_ch]];

                    for i in 0..SOS_BIN_CHUNK {
                        let basis0 = basis0_chunk[i];
                        let basis1 = basis1_chunk[i];
                        let basis2 = basis2_chunk[i];
                        let numerator = b0 * basis0 + b1 * basis1 + b2 * basis2;
                        let denominator = a0 * basis0 + a1 * basis1 + a2 * basis2;
                        h_chunk[i] *= numerator / denominator;
                    }
                }
                for i in 0..SOS_BIN_CHUNK {
                    h[[bin_start + i, out_ch, in_ch]] = h_chunk[i];
                }
            }
        }
    }

    // Tail bins (fewer than SOS_BIN_CHUNK).
    for bin in n_chunked..n_bins {
        let basis0 = tap_response[[0, bin]];
        let basis1 = tap_response[[1, bin]];
        let basis2 = tap_response[[2, bin]];
        for out_ch in 0..n_out {
            for in_ch in 0..n_in {
                let mut h_val = Complex::from(1.0);
                for section in 0..n_sections {
                    let b0 = b[[section, 0, out_ch, in_ch]];
                    let b1 = b[[section, 1, out_ch, in_ch]];
                    let b2 = b[[section, 2, out_ch, in_ch]];
                    let a0 = a[[section, 0, out_ch, in_ch]];
                    let a1 = a[[section, 1, out_ch, in_ch]];
                    let a2 = a[[section, 2, out_ch, in_ch]];
                    let numerator = b0 * basis0 + b1 * basis1 + b2 * basis2;
                    let denominator = a0 * basis0 + a1 * basis1 + a2 * basis2;
                    h_val *= numerator / denominator;
                }
                h[[bin, out_ch, in_ch]] = h_val;
            }
        }
    }

    h
}

/// Compute H(f) and its analytical Jacobian w.r.t. b/a coefficients.
///
/// # Arguments
/// * `b` - numerator coefficients, shape `(K, 3, N_out, N_in)`.
/// * `a` - denominator coefficients, same shape.
/// * `nfft` - FFT length.
/// * `gamma` - anti-alias envelope `[gamma^0, gamma^1, gamma^2]`.
///
/// # Errors
///
/// Returns an error if the coefficient tensors have incompatible shapes, the
/// tap axis is not length three, or `nfft` is zero.
pub fn sos_response(
    b: &Array4<Complex<f64>>,
    a: &Array4<Complex<f64>>,
    nfft: usize,
    gamma: &[f64; 3],
) -> Result<SosResponse, AutodiffError> {
    validate_sos_4d_inputs(b, a, nfft)?;
    let basis = SosFrequencyBasis::new(nfft, gamma);
    Ok(sos_response_with_basis(b, a, &basis))
}

pub(crate) fn sos_response_with_basis(
    b: &Array4<Complex<f64>>,
    a: &Array4<Complex<f64>>,
    basis: &SosFrequencyBasis,
) -> SosResponse {
    let b_view = b.view();
    let a_view = a.view();
    sos_response_impl(&b_view, &a_view, basis)
}

#[allow(clippy::too_many_lines)]
// Stride names follow ndarray's axis ordering (section/bin/out/in); the
// `bin` and `in` labels are single-character apart, which triggers
// `similar_names`, but renaming them would obscure the axis they index.
#[allow(clippy::similar_names)]
pub(crate) fn sos_coefficient_vjp_with_basis(
    b: &Array4<Complex<f64>>,
    a: &Array4<Complex<f64>>,
    basis: &SosFrequencyBasis,
    dl_dh: &Array3<Complex<f64>>,
    h: &mut Array3<Complex<f64>>,
    b_response: &mut Array4<Complex<f64>>,
    a_response: &mut Array4<Complex<f64>>,
) -> Result<(Array4<f64>, Array4<f64>), AutodiffError> {
    if b.dim() != a.dim() {
        return Err(AutodiffError::Message(format!(
            "sos_coefficient_vjp: b and a must have the same shape, got {:?} and {:?}",
            b.dim(),
            a.dim()
        )));
    }
    let (n_sections, _, n_out, n_in) = b.dim();
    if b.shape()[1] != 3 {
        return Err(AutodiffError::Message(format!(
            "sos_coefficient_vjp: tap axis must be 3, got {}",
            b.shape()[1]
        )));
    }
    let n_bins = basis.n_bins();
    let expected_h = (n_bins, n_out, n_in);
    let expected_response = (n_sections, n_bins, n_out, n_in);
    if dl_dh.dim() != expected_h || h.dim() != expected_h {
        return Err(AutodiffError::Message(format!(
            "sos_coefficient_vjp: expected response-gradient shapes {:?}, got {:?} and {:?}",
            expected_h,
            dl_dh.dim(),
            h.dim()
        )));
    }
    if b_response.dim() != expected_response || a_response.dim() != expected_response {
        return Err(AutodiffError::Message(format!(
            "sos_coefficient_vjp: expected coefficient-response shapes {:?}, got {:?} and {:?}",
            expected_response,
            b_response.dim(),
            a_response.dim()
        )));
    }

    h.fill(Complex::from(1.0));
    let mut has_zero_b = false;

    // Precompute raw pointers and strides for all arrays used in both passes.
    // `b` and `a` share the shape `(K, 3, N_out, N_in)`; `b_response` and
    // `a_response` share the shape `(K, M, N_out, N_in)`.
    let b_ptr = b.as_ptr();
    let a_ptr = a.as_ptr();
    let b_resp_ptr = b_response.as_mut_ptr();
    let a_resp_ptr = a_response.as_mut_ptr();
    let h_ptr = h.as_mut_ptr();
    let dl_ptr = dl_dh.as_ptr();
    let basis_ptr = basis.response.as_ptr();

    let b_strides = b.strides();
    let b_sec_stride = b_strides[0].cast_unsigned();
    let b_tap_stride = b_strides[1].cast_unsigned();
    let b_out_stride = b_strides[2].cast_unsigned();
    let b_in_stride = b_strides[3].cast_unsigned();
    let a_strides = a.strides();
    let a_sec_stride = a_strides[0].cast_unsigned();
    let a_tap_stride = a_strides[1].cast_unsigned();
    let a_out_stride = a_strides[2].cast_unsigned();
    let a_in_stride = a_strides[3].cast_unsigned();

    let resp_strides = b_response.strides();
    let resp_sec_stride = resp_strides[0].cast_unsigned();
    let resp_bin_stride = resp_strides[1].cast_unsigned();
    let resp_out_stride = resp_strides[2].cast_unsigned();
    let resp_in_stride = resp_strides[3].cast_unsigned();
    let a_resp_strides = a_response.strides();
    let a_resp_sec_stride = a_resp_strides[0].cast_unsigned();
    let a_resp_bin_stride = a_resp_strides[1].cast_unsigned();
    let a_resp_out_stride = a_resp_strides[2].cast_unsigned();
    let a_resp_in_stride = a_resp_strides[3].cast_unsigned();

    let h_strides = h.strides();
    let h_bin_stride = h_strides[0].cast_unsigned();
    let h_out_stride = h_strides[1].cast_unsigned();
    let h_in_stride = h_strides[2].cast_unsigned();

    let dl_strides = dl_dh.strides();
    let dl_bin_stride = dl_strides[0].cast_unsigned();
    let dl_out_stride = dl_strides[1].cast_unsigned();
    let dl_in_stride = dl_strides[2].cast_unsigned();

    // First pass: compute B_k and A_k together and accumulate H = prod_k B_k/A_k.
    // Process bins in chunks so the per-bin H values stay in registers across
    // all SOS sections, and each bin's basis values are loaded only once.
    let n_chunked = n_bins / SOS_BIN_CHUNK * SOS_BIN_CHUNK;
    for bin_start in (0..n_chunked).step_by(SOS_BIN_CHUNK) {
        let mut basis0_chunk = [Complex::from(0.0); SOS_BIN_CHUNK];
        let mut basis1_chunk = [Complex::from(0.0); SOS_BIN_CHUNK];
        let mut basis2_chunk = [Complex::from(0.0); SOS_BIN_CHUNK];
        for i in 0..SOS_BIN_CHUNK {
            let bin = bin_start + i;
            // SAFETY: bin_start + i < n_chunked <= n_bins, so all offsets are in bounds.
            unsafe {
                basis0_chunk[i] = *basis_ptr.add(bin);
                basis1_chunk[i] = *basis_ptr.add(n_bins + bin);
                basis2_chunk[i] = *basis_ptr.add(2 * n_bins + bin);
            }
        }
        for out_ch in 0..n_out {
            for in_ch in 0..n_in {
                // Accumulate numerator and denominator products separately so
                // only one complex division per bin is needed at the end.
                let mut num_chunk = [Complex::from(1.0); SOS_BIN_CHUNK];
                let mut den_chunk = [Complex::from(1.0); SOS_BIN_CHUNK];
                let h_base = out_ch * h_out_stride + in_ch * h_in_stride;
                for section in 0..n_sections {
                    let coeff_base =
                        section * b_sec_stride + out_ch * b_out_stride + in_ch * b_in_stride;
                    let a_coeff_base =
                        section * a_sec_stride + out_ch * a_out_stride + in_ch * a_in_stride;
                    let resp_base = section * resp_sec_stride
                        + out_ch * resp_out_stride
                        + in_ch * resp_in_stride;
                    let a_resp_base = section * a_resp_sec_stride
                        + out_ch * a_resp_out_stride
                        + in_ch * a_resp_in_stride;
                    // SAFETY: coeff_base is built from bounded section/out/in and
                    // the three tap offsets (0/1/2) are inside the tap axis of length 3.
                    let (b0, b1, b2, a0, a1, a2) = unsafe {
                        (
                            *b_ptr.add(coeff_base),
                            *b_ptr.add(coeff_base + b_tap_stride),
                            *b_ptr.add(coeff_base + 2 * b_tap_stride),
                            *a_ptr.add(a_coeff_base),
                            *a_ptr.add(a_coeff_base + a_tap_stride),
                            *a_ptr.add(a_coeff_base + 2 * a_tap_stride),
                        )
                    };
                    for i in 0..SOS_BIN_CHUNK {
                        let basis0 = basis0_chunk[i];
                        let basis1 = basis1_chunk[i];
                        let basis2 = basis2_chunk[i];
                        let numerator = b0 * basis0 + b1 * basis1 + b2 * basis2;
                        let denominator = a0 * basis0 + a1 * basis1 + a2 * basis2;
                        let bin = bin_start + i;
                        // SAFETY: resp_base + bin*resp_bin_stride stays inside the
                        // bin dimension of length n_bins for this section/channel.
                        unsafe {
                            *b_resp_ptr.add(resp_base + bin * resp_bin_stride) = numerator;
                            *a_resp_ptr.add(a_resp_base + bin * a_resp_bin_stride) = denominator;
                        }
                        num_chunk[i] *= numerator;
                        den_chunk[i] *= denominator;
                        has_zero_b |= numerator == Complex::default();
                    }
                }
                // SAFETY: h_base + (bin_start+i)*h_bin_stride stays inside h.
                unsafe {
                    for i in 0..SOS_BIN_CHUNK {
                        let bin = bin_start + i;
                        *h_ptr.add(h_base + bin * h_bin_stride) = num_chunk[i] / den_chunk[i];
                    }
                }
            }
        }
    }
    for bin in n_chunked..n_bins {
        // SAFETY: bin is in [n_chunked, n_bins), so all offsets are in bounds.
        let (basis0, basis1, basis2) = unsafe {
            (
                *basis_ptr.add(bin),
                *basis_ptr.add(n_bins + bin),
                *basis_ptr.add(2 * n_bins + bin),
            )
        };
        for out_ch in 0..n_out {
            for in_ch in 0..n_in {
                let mut num_val = Complex::from(1.0);
                let mut den_val = Complex::from(1.0);
                let h_base = out_ch * h_out_stride + in_ch * h_in_stride;
                for section in 0..n_sections {
                    let coeff_base =
                        section * b_sec_stride + out_ch * b_out_stride + in_ch * b_in_stride;
                    let a_coeff_base =
                        section * a_sec_stride + out_ch * a_out_stride + in_ch * a_in_stride;
                    let resp_base = section * resp_sec_stride
                        + out_ch * resp_out_stride
                        + in_ch * resp_in_stride;
                    let a_resp_base = section * a_resp_sec_stride
                        + out_ch * a_resp_out_stride
                        + in_ch * a_resp_in_stride;
                    // SAFETY: same bounded coeff_base/tap offsets as above.
                    let (b0, b1, b2, a0, a1, a2) = unsafe {
                        (
                            *b_ptr.add(coeff_base),
                            *b_ptr.add(coeff_base + b_tap_stride),
                            *b_ptr.add(coeff_base + 2 * b_tap_stride),
                            *a_ptr.add(a_coeff_base),
                            *a_ptr.add(a_coeff_base + a_tap_stride),
                            *a_ptr.add(a_coeff_base + 2 * a_tap_stride),
                        )
                    };
                    let numerator = b0 * basis0 + b1 * basis1 + b2 * basis2;
                    let denominator = a0 * basis0 + a1 * basis1 + a2 * basis2;
                    // SAFETY: resp_base + bin*resp_bin_stride is in bounds.
                    unsafe {
                        *b_resp_ptr.add(resp_base + bin * resp_bin_stride) = numerator;
                        *a_resp_ptr.add(a_resp_base + bin * a_resp_bin_stride) = denominator;
                    }
                    num_val *= numerator;
                    den_val *= denominator;
                    has_zero_b |= numerator == Complex::default();
                }
                // SAFETY: h_base + bin*h_bin_stride is in bounds.
                unsafe {
                    *h_ptr.add(h_base + bin * h_bin_stride) = num_val / den_val;
                }
            }
        }
    }

    let mut db = Array4::zeros((n_sections, 3, n_out, n_in));
    let mut da = Array4::zeros((n_sections, 3, n_out, n_in));

    if has_zero_b {
        // Rare path: a section numerator is exactly zero somewhere, so we
        // must compute other_sections explicitly to avoid 0/0 NaNs.
        for section in 0..n_sections {
            for tap in 0..3 {
                for out_ch in 0..n_out {
                    for in_ch in 0..n_in {
                        let mut numerator_gradient = 0.0;
                        let mut denominator_gradient = 0.0;
                        for bin in 0..n_bins {
                            let b_bin = b_response[[section, bin, out_ch, in_ch]];
                            let a_bin = a_response[[section, bin, out_ch, in_ch]];
                            let mut other_sections = Complex::new(1.0, 0.0);
                            for other in 0..n_sections {
                                if other != section {
                                    other_sections *= b_response[[other, bin, out_ch, in_ch]]
                                        / a_response[[other, bin, out_ch, in_ch]];
                                }
                            }
                            let frequency_term = basis.response[[tap, bin]];
                            let derivative_b = other_sections * frequency_term / a_bin;
                            let derivative_a =
                                -other_sections * b_bin * frequency_term / (a_bin * a_bin);
                            let loss_gradient = dl_dh[[bin, out_ch, in_ch]].conj();
                            numerator_gradient += (loss_gradient * derivative_b).re;
                            denominator_gradient += (loss_gradient * derivative_a).re;
                        }
                        db[[section, tap, out_ch, in_ch]] = numerator_gradient;
                        da[[section, tap, out_ch, in_ch]] = denominator_gradient;
                    }
                }
            }
        }
    } else {
        // Common path: every section numerator is non-zero, so h*A_s/B_s is
        // safe and avoids the per-section product. Reduce the six per-bin
        // complex divisions to two reciprocals, use real-only accumulation for
        // the real-valued tap-0 basis, and walk the bin lanes with raw pointers
        // following the arrays' natural strides.
        for section in 0..n_sections {
            for out_ch in 0..n_out {
                for in_ch in 0..n_in {
                    let mut g0_num = 0.0;
                    let mut g0_den = 0.0;
                    let mut g1_num = 0.0;
                    let mut g1_den = 0.0;
                    let mut g2_num = 0.0;
                    let mut g2_den = 0.0;

                    let b_base = section * resp_sec_stride
                        + out_ch * resp_out_stride
                        + in_ch * resp_in_stride;
                    let a_base = section * a_resp_sec_stride
                        + out_ch * a_resp_out_stride
                        + in_ch * a_resp_in_stride;
                    let h_base = out_ch * h_out_stride + in_ch * h_in_stride;
                    let dl_base = out_ch * dl_out_stride + in_ch * dl_in_stride;

                    // SAFETY: all offsets are derived from the arrays' own
                    // strides and stay within the allocated shape (section,
                    // out_ch, in_ch are bounded by their respective dimensions
                    // and bin is bounded by n_bins).
                    unsafe {
                        for bin in 0..n_bins {
                            let b_bin = *b_resp_ptr.add(b_base + bin * resp_bin_stride);
                            let a_bin = *a_resp_ptr.add(a_base + bin * a_resp_bin_stride);
                            let h_bin = *h_ptr.add(h_base + bin * h_bin_stride);
                            let loss_gradient = (*dl_ptr.add(dl_base + bin * dl_bin_stride)).conj();

                            // One reciprocal per coefficient instead of
                            // dividing each tap contribution separately.
                            let inv_b = 1.0 / b_bin;
                            let inv_a = 1.0 / a_bin;
                            let cb = loss_gradient * h_bin * inv_b;
                            let ca = -loss_gradient * h_bin * inv_a;

                            // Tap 0 basis is real-valued; avoid a complex multiply.
                            let freq0_re = (*basis_ptr.add(bin)).re;
                            g0_num += cb.re * freq0_re;
                            g0_den += ca.re * freq0_re;

                            let freq1 = *basis_ptr.add(n_bins + bin);
                            g1_num += (cb * freq1).re;
                            g1_den += (ca * freq1).re;

                            let freq2 = *basis_ptr.add(2 * n_bins + bin);
                            g2_num += (cb * freq2).re;
                            g2_den += (ca * freq2).re;
                        }
                    }

                    db[[section, 0, out_ch, in_ch]] = g0_num;
                    da[[section, 0, out_ch, in_ch]] = g0_den;
                    db[[section, 1, out_ch, in_ch]] = g1_num;
                    da[[section, 1, out_ch, in_ch]] = g1_den;
                    db[[section, 2, out_ch, in_ch]] = g2_num;
                    da[[section, 2, out_ch, in_ch]] = g2_den;
                }
            }
        }
    }

    Ok((db, da))
}

fn sos_response_impl<S>(
    b: &ArrayBase<S, Ix4>,
    a: &ArrayBase<S, Ix4>,
    basis: &SosFrequencyBasis,
) -> SosResponse
where
    S: Data<Elem = Complex<f64>>,
{
    let (n_sections, _, n_out, n_in) = b.dim();
    let n_bins = basis.n_bins();
    let fft_envelope = &basis.response;

    // Compute B_k and A_k for every section, and accumulate H = prod_k B_k/A_k.
    let mut h = Array3::from_elem((n_bins, n_out, n_in), Complex::from(1.0));
    let mut b_response = Array4::zeros((n_sections, n_bins, n_out, n_in));
    let mut a_response = Array4::zeros((n_sections, n_bins, n_out, n_in));

    // Compute B_k and A_k for every section in one pass, and accumulate
    // H = prod_k B_k/A_k.
    for section in 0..n_sections {
        for bin in 0..n_bins {
            let basis0 = fft_envelope[[0, bin]];
            let basis1 = fft_envelope[[1, bin]];
            let basis2 = fft_envelope[[2, bin]];
            for out_ch in 0..n_out {
                for in_ch in 0..n_in {
                    let b0 = b[[section, 0, out_ch, in_ch]];
                    let b1 = b[[section, 1, out_ch, in_ch]];
                    let b2 = b[[section, 2, out_ch, in_ch]];
                    let a0 = a[[section, 0, out_ch, in_ch]];
                    let a1 = a[[section, 1, out_ch, in_ch]];
                    let a2 = a[[section, 2, out_ch, in_ch]];
                    let numerator = b0 * basis0 + b1 * basis1 + b2 * basis2;
                    let denominator = a0 * basis0 + a1 * basis1 + a2 * basis2;
                    b_response[[section, bin, out_ch, in_ch]] = numerator;
                    a_response[[section, bin, out_ch, in_ch]] = denominator;
                    h[[bin, out_ch, in_ch]] *= numerator / denominator;
                }
            }
        }
    }

    // Precompute the per-(section, bin, channel) product over the other
    // sections so the Jacobian loops avoid repeated inner "other" loops.
    let one = Complex::new(1.0, 0.0);
    let mut other_response = Array4::zeros((n_sections, n_bins, n_out, n_in));
    for section in 0..n_sections {
        for bin in 0..n_bins {
            for out_ch in 0..n_out {
                for in_ch in 0..n_in {
                    let mut other_sections = one;
                    for other in 0..n_sections {
                        if other != section {
                            other_sections *= b_response[[other, bin, out_ch, in_ch]]
                                / a_response[[other, bin, out_ch, in_ch]];
                        }
                    }
                    other_response[[section, bin, out_ch, in_ch]] = other_sections;
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
                        let other_sections = other_response[[section, bin, out_ch, in_ch]];

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
    let basis = SosFrequencyBasis::new(nfft, &gamma);
    Ok(sos_frequency_response_with_basis(b, a, &basis))
}

pub(crate) fn sos_frequency_response_with_basis(
    b: &Array4<Complex<f64>>,
    a: &Array4<Complex<f64>>,
    basis: &SosFrequencyBasis,
) -> Array3<Complex<f64>> {
    let b_view = b.view();
    let a_view = a.view();
    sos_frequency_response_impl(&b_view, &a_view, basis)
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
    let basis = SosFrequencyBasis::new(nfft, &gamma);
    let b_view = b.view();
    let a_view = a.view();
    let resp = sos_response_impl(&b_view, &a_view, &basis);
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
    let basis = SosFrequencyBasis::new(nfft, &gamma);
    let resp = sos_response_impl(&b4, &a4, &basis);
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
    let basis = SosFrequencyBasis::new(nfft, &gamma);
    let resp = sos_response_impl(&b4, &a4, &basis);
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

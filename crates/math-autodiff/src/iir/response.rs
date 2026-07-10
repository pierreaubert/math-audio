use ndarray::{Array2, Array3, Array4, Array5};
use num_complex::Complex;
use rustfft::FftPlanner;

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
/// Panics if `nfft` is zero, because a zero-length FFT is not meaningful.
#[must_use]
pub fn sos_response(
    b: &Array4<Complex<f64>>,
    a: &Array4<Complex<f64>>,
    nfft: usize,
    gamma: &[f64; 3],
) -> SosResponse {
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
                        let h_bin = h[[bin, out_ch, in_ch]];
                        let b_bin = b_response[[section, bin, out_ch, in_ch]];
                        let a_bin = a_response[[section, bin, out_ch, in_ch]];

                        jacobian_b[[bin, section, tap, out_ch, in_ch]] =
                            h_bin / b_bin * envelope_bin;
                        jacobian_a[[bin, section, tap, out_ch, in_ch]] =
                            -(h_bin / a_bin) * envelope_bin;
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

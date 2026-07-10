use approx::assert_abs_diff_eq;
use math_audio_autodiff::iir::response::sos_response;
use ndarray::Array4;
use num_complex::Complex;

#[test]
fn sos_response_dc_and_nyquist_gains_are_correct() {
    let nfft = 512;
    let gamma = [1.0, 0.95, 0.9];

    // Single SOS section: b = [1, 0, 0], a = [1, -0.5, 0.25]
    let mut b = Array4::zeros((1, 3, 1, 1));
    let mut a = Array4::zeros((1, 3, 1, 1));
    b[[0, 0, 0, 0]] = Complex::from(1.0);
    a[[0, 0, 0, 0]] = Complex::from(1.0);
    a[[0, 1, 0, 0]] = Complex::from(-0.5);
    a[[0, 2, 0, 0]] = Complex::from(0.25);

    let resp = sos_response(&b, &a, nfft, &gamma);

    let expected_dc = Complex::from(1.0) / Complex::from(1.0 - 0.5 * gamma[1] + 0.25 * gamma[2]);
    let expected_nyquist = Complex::from(1.0)
        / Complex::from(1.0 + 0.5 * gamma[1] + 0.25 * gamma[2]);

    assert_abs_diff_eq!(resp.h[[0, 0, 0]].re, expected_dc.re, epsilon = 1e-12);
    assert_abs_diff_eq!(resp.h[[0, 0, 0]].im, expected_dc.im, epsilon = 1e-12);

    let nyquist_bin = nfft / 2;
    assert_abs_diff_eq!(
        resp.h[[nyquist_bin, 0, 0]].re,
        expected_nyquist.re,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        resp.h[[nyquist_bin, 0, 0]].im,
        expected_nyquist.im,
        epsilon = 1e-12
    );
}

fn build_two_section_coeffs() -> (Array4<Complex<f64>>, Array4<Complex<f64>>) {
    let mut b = Array4::zeros((2, 3, 1, 1));
    let mut a = Array4::zeros((2, 3, 1, 1));

    b[[0, 0, 0, 0]] = Complex::from(1.0);
    a[[0, 0, 0, 0]] = Complex::from(1.0);
    a[[0, 1, 0, 0]] = Complex::from(-0.5);
    a[[0, 2, 0, 0]] = Complex::from(0.25);

    b[[1, 0, 0, 0]] = Complex::from(1.0);
    a[[1, 0, 0, 0]] = Complex::from(1.0);
    a[[1, 1, 0, 0]] = Complex::from(-0.3);
    a[[1, 2, 0, 0]] = Complex::from(0.1);

    (b, a)
}

#[test]
fn sos_response_jacobian_matches_finite_diff_for_b() {
    let nfft = 256;
    let gamma = [1.0, 0.98, 0.96];
    let epsilon = 1e-8;

    let (b, a) = build_two_section_coeffs();
    let resp = sos_response(&b, &a, nfft, &gamma);

    for section in 0..2 {
        for tap in 0..3 {
            let mut b_perturbed = b.clone();
            b_perturbed[[section, tap, 0, 0]].re += epsilon;
            let resp_perturbed = sos_response(&b_perturbed, &a, nfft, &gamma);

            let expected = (&resp_perturbed.h - &resp.h) / Complex::from(epsilon);

            for bin in 0..(nfft / 2 + 1) {
                let actual = resp.dh_db[[bin, section, tap, 0, 0]];
                assert_abs_diff_eq!(actual.re, expected[[bin, 0, 0]].re, epsilon = 1e-6);
                assert_abs_diff_eq!(actual.im, expected[[bin, 0, 0]].im, epsilon = 1e-6);
            }
        }
    }
}

#[test]
fn sos_response_jacobian_matches_finite_diff_for_a() {
    let nfft = 256;
    let gamma = [1.0, 0.98, 0.96];
    let epsilon = 1e-8;

    let (b, a) = build_two_section_coeffs();
    let resp = sos_response(&b, &a, nfft, &gamma);

    for section in 0..2 {
        for tap in 0..3 {
            let mut a_perturbed = a.clone();
            a_perturbed[[section, tap, 0, 0]].re += epsilon;
            let resp_perturbed = sos_response(&b, &a_perturbed, nfft, &gamma);

            let expected = (&resp_perturbed.h - &resp.h) / Complex::from(epsilon);

            for bin in 0..(nfft / 2 + 1) {
                let actual = resp.dh_da[[bin, section, tap, 0, 0]];
                assert_abs_diff_eq!(actual.re, expected[[bin, 0, 0]].re, epsilon = 1e-6);
                assert_abs_diff_eq!(actual.im, expected[[bin, 0, 0]].im, epsilon = 1e-6);
            }
        }
    }
}

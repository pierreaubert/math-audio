// The full-Jacobian API under test is deprecated but must keep working.
#![allow(deprecated)]

use approx::assert_abs_diff_eq;
use math_audio_autodiff::iir::response::{
    sos_frequency_response, sos_frequency_response_jacobian,
    sos_frequency_response_jacobian_parallel, sos_frequency_response_parallel, sos_response,
};
use ndarray::{Array3, Array4};
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

    let resp = sos_response(&b, &a, nfft, &gamma).unwrap();
    let h = sos_frequency_response(&b, &a, nfft, Some(&gamma)).unwrap();

    let expected_dc = Complex::from(1.0) / Complex::from(1.0 - 0.5 * gamma[1] + 0.25 * gamma[2]);
    let expected_nyquist =
        Complex::from(1.0) / Complex::from(1.0 + 0.5 * gamma[1] + 0.25 * gamma[2]);

    assert_abs_diff_eq!(resp.h[[0, 0, 0]].re, expected_dc.re, epsilon = 1e-12);
    assert_abs_diff_eq!(resp.h[[0, 0, 0]].im, expected_dc.im, epsilon = 1e-12);

    assert_abs_diff_eq!(h[[0, 0, 0]].re, expected_dc.re, epsilon = 1e-12);
    assert_abs_diff_eq!(h[[0, 0, 0]].im, expected_dc.im, epsilon = 1e-12);

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

    assert_abs_diff_eq!(
        h[[nyquist_bin, 0, 0]].re,
        expected_nyquist.re,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        h[[nyquist_bin, 0, 0]].im,
        expected_nyquist.im,
        epsilon = 1e-12
    );
}

#[test]
fn sos_frequency_response_uses_identity_gamma_when_none() {
    let nfft = 512;

    let mut b = Array4::zeros((1, 3, 1, 1));
    let mut a = Array4::zeros((1, 3, 1, 1));
    b[[0, 0, 0, 0]] = Complex::from(1.0);
    a[[0, 0, 0, 0]] = Complex::from(1.0);
    a[[0, 1, 0, 0]] = Complex::from(-0.5);
    a[[0, 2, 0, 0]] = Complex::from(0.25);

    let h_default = sos_frequency_response(&b, &a, nfft, None).unwrap();
    let h_identity = sos_frequency_response(&b, &a, nfft, Some(&[1.0, 1.0, 1.0])).unwrap();

    assert_abs_diff_eq!(
        h_default[[0, 0, 0]].re,
        h_identity[[0, 0, 0]].re,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        h_default[[0, 0, 0]].im,
        h_identity[[0, 0, 0]].im,
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
    let resp = sos_response(&b, &a, nfft, &gamma).unwrap();
    let (dh_db, _) = sos_frequency_response_jacobian(&b, &a, nfft, Some(&gamma)).unwrap();

    for section in 0..2 {
        for tap in 0..3 {
            let mut b_perturbed = b.clone();
            b_perturbed[[section, tap, 0, 0]].re += epsilon;
            let resp_perturbed = sos_response(&b_perturbed, &a, nfft, &gamma).unwrap();
            let h_perturbed = sos_frequency_response(&b_perturbed, &a, nfft, Some(&gamma)).unwrap();

            let expected = (&resp_perturbed.h - &resp.h) / Complex::from(epsilon);
            let expected_public = (&h_perturbed - &resp.h) / Complex::from(epsilon);

            for bin in 0..(nfft / 2 + 1) {
                let actual = resp.dh_db[[bin, section, tap, 0, 0]];
                let actual_public = dh_db[[bin, section, tap, 0, 0]];
                assert_abs_diff_eq!(actual.re, expected[[bin, 0, 0]].re, epsilon = 1e-6);
                assert_abs_diff_eq!(actual.im, expected[[bin, 0, 0]].im, epsilon = 1e-6);
                assert_abs_diff_eq!(
                    actual_public.re,
                    expected_public[[bin, 0, 0]].re,
                    epsilon = 1e-6
                );
                assert_abs_diff_eq!(
                    actual_public.im,
                    expected_public[[bin, 0, 0]].im,
                    epsilon = 1e-6
                );
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
    let resp = sos_response(&b, &a, nfft, &gamma).unwrap();
    let (_, dh_da) = sos_frequency_response_jacobian(&b, &a, nfft, Some(&gamma)).unwrap();

    for section in 0..2 {
        for tap in 0..3 {
            let mut a_perturbed = a.clone();
            a_perturbed[[section, tap, 0, 0]].re += epsilon;
            let resp_perturbed = sos_response(&b, &a_perturbed, nfft, &gamma).unwrap();
            let h_perturbed = sos_frequency_response(&b, &a_perturbed, nfft, Some(&gamma)).unwrap();

            let expected = (&resp_perturbed.h - &resp.h) / Complex::from(epsilon);
            let expected_public = (&h_perturbed - &resp.h) / Complex::from(epsilon);

            for bin in 0..(nfft / 2 + 1) {
                let actual = resp.dh_da[[bin, section, tap, 0, 0]];
                let actual_public = dh_da[[bin, section, tap, 0, 0]];
                assert_abs_diff_eq!(actual.re, expected[[bin, 0, 0]].re, epsilon = 1e-6);
                assert_abs_diff_eq!(actual.im, expected[[bin, 0, 0]].im, epsilon = 1e-6);
                assert_abs_diff_eq!(
                    actual_public.re,
                    expected_public[[bin, 0, 0]].re,
                    epsilon = 1e-6
                );
                assert_abs_diff_eq!(
                    actual_public.im,
                    expected_public[[bin, 0, 0]].im,
                    epsilon = 1e-6
                );
            }
        }
    }
}

fn build_two_section_coeffs_3d() -> (Array3<Complex<f64>>, Array3<Complex<f64>>) {
    let mut b = Array3::zeros((2, 3, 2));
    let mut a = Array3::zeros((2, 3, 2));

    // Channel 0.
    b[[0, 0, 0]] = Complex::from(1.0);
    a[[0, 0, 0]] = Complex::from(1.0);
    a[[0, 1, 0]] = Complex::from(-0.5);
    a[[0, 2, 0]] = Complex::from(0.25);

    b[[1, 0, 0]] = Complex::from(1.0);
    a[[1, 0, 0]] = Complex::from(1.0);
    a[[1, 1, 0]] = Complex::from(-0.3);
    a[[1, 2, 0]] = Complex::from(0.1);

    // Channel 1: same coefficients for easy verification.
    b[[0, 0, 1]] = Complex::from(1.0);
    a[[0, 0, 1]] = Complex::from(1.0);
    a[[0, 1, 1]] = Complex::from(-0.5);
    a[[0, 2, 1]] = Complex::from(0.25);

    b[[1, 0, 1]] = Complex::from(1.0);
    a[[1, 0, 1]] = Complex::from(1.0);
    a[[1, 1, 1]] = Complex::from(-0.3);
    a[[1, 2, 1]] = Complex::from(0.1);

    (b, a)
}

#[test]
fn sos_frequency_response_parallel_matches_4d_equivalent() {
    let nfft = 256;
    let gamma = [1.0, 0.98, 0.96];

    let (b3, a3) = build_two_section_coeffs_3d();
    let h3 = sos_frequency_response_parallel(&b3, &a3, nfft, Some(&gamma)).unwrap();

    // Reshape to 4-D (K, 3, N, 1) and compare.
    let (k, three, n) = b3.dim();
    let b4 = b3.clone().into_shape_with_order((k, three, n, 1)).unwrap();
    let a4 = a3.clone().into_shape_with_order((k, three, n, 1)).unwrap();
    let h4 = sos_frequency_response(&b4, &a4, nfft, Some(&gamma)).unwrap();

    for bin in 0..(nfft / 2 + 1) {
        for ch in 0..n {
            assert_abs_diff_eq!(h3[[bin, ch]].re, h4[[bin, ch, 0]].re, epsilon = 1e-12);
            assert_abs_diff_eq!(h3[[bin, ch]].im, h4[[bin, ch, 0]].im, epsilon = 1e-12);
        }
    }
}

#[test]
fn sos_frequency_response_jacobian_parallel_matches_finite_diff() {
    let nfft = 256;
    let gamma = [1.0, 0.98, 0.96];
    let epsilon = 1e-8;

    let (b, a) = build_two_section_coeffs_3d();
    let h = sos_frequency_response_parallel(&b, &a, nfft, Some(&gamma)).unwrap();
    let (dh_db, dh_da) =
        sos_frequency_response_jacobian_parallel(&b, &a, nfft, Some(&gamma)).unwrap();

    for section in 0..2 {
        for tap in 0..3 {
            for ch in 0..2 {
                let mut b_perturbed = b.clone();
                b_perturbed[[section, tap, ch]].re += epsilon;
                let h_perturbed =
                    sos_frequency_response_parallel(&b_perturbed, &a, nfft, Some(&gamma)).unwrap();
                let expected = (&h_perturbed - &h) / Complex::from(epsilon);

                for bin in 0..(nfft / 2 + 1) {
                    let actual = dh_db[[bin, section, tap, ch]];
                    assert_abs_diff_eq!(actual.re, expected[[bin, ch]].re, epsilon = 1e-6);
                    assert_abs_diff_eq!(actual.im, expected[[bin, ch]].im, epsilon = 1e-6);
                }

                let mut a_perturbed = a.clone();
                a_perturbed[[section, tap, ch]].re += epsilon;
                let h_perturbed =
                    sos_frequency_response_parallel(&b, &a_perturbed, nfft, Some(&gamma)).unwrap();
                let expected = (&h_perturbed - &h) / Complex::from(epsilon);

                for bin in 0..(nfft / 2 + 1) {
                    let actual = dh_da[[bin, section, tap, ch]];
                    assert_abs_diff_eq!(actual.re, expected[[bin, ch]].re, epsilon = 1e-6);
                    assert_abs_diff_eq!(actual.im, expected[[bin, ch]].im, epsilon = 1e-6);
                }
            }
        }
    }
}

#[test]
fn sos_frequency_response_errors_on_mismatched_shapes() {
    let b = Array4::zeros((1, 3, 1, 1));
    let a = Array4::zeros((1, 3, 2, 1));
    let err = sos_frequency_response(&b, &a, 64, None).unwrap_err();
    assert!(err.to_string().contains("same shape"));
}

#[test]
fn sos_frequency_response_errors_on_zero_nfft() {
    let b = Array4::zeros((1, 3, 1, 1));
    let a = Array4::zeros((1, 3, 1, 1));
    let err = sos_frequency_response(&b, &a, 0, None).unwrap_err();
    assert!(err.to_string().contains("nfft"));
}

#[test]
fn sos_frequency_response_errors_on_bad_tap_axis() {
    let b = Array4::zeros((1, 4, 1, 1));
    let a = Array4::zeros((1, 4, 1, 1));
    let err = sos_frequency_response(&b, &a, 64, None).unwrap_err();
    assert!(err.to_string().contains("second axis must be 3"));
}

#[test]
fn sos_jacobian_is_finite_at_zero_numerator() {
    let nfft = 64;
    let b = Array4::<Complex<f64>>::zeros((1, 3, 1, 1));
    let mut a = Array4::<Complex<f64>>::zeros((1, 3, 1, 1));
    a[[0, 0, 0, 0]] = Complex::new(1.0, 0.0);

    let (dh_db, dh_da) = sos_frequency_response_jacobian(&b, &a, nfft, None).unwrap();
    for value in &dh_db {
        assert!(value.is_finite(), "non-finite numerator Jacobian: {value}");
    }
    for value in &dh_da {
        assert!(
            value.is_finite(),
            "non-finite denominator Jacobian: {value}"
        );
    }

    // With A(z)=1 and B(z)=0, dH/db0 is exactly one at every bin.
    for bin in 0..=nfft / 2 {
        assert_abs_diff_eq!(dh_db[[bin, 0, 0, 0, 0]].re, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dh_db[[bin, 0, 0, 0, 0]].im, 0.0, epsilon = 1e-12);
    }
}

#[test]
fn sos_frequency_response_parallel_errors_on_bad_tap_axis() {
    let b = Array3::zeros((1, 4, 1));
    let a = Array3::zeros((1, 4, 1));
    let err = sos_frequency_response_parallel(&b, &a, 64, None).unwrap_err();
    assert!(err.to_string().contains("second axis must be 3"));
}

#[test]
fn sos_response_returns_errors_instead_of_panicking_on_invalid_inputs() {
    let b = Array4::zeros((1, 3, 1, 1));
    let mismatched_a = Array4::zeros((1, 3, 2, 1));
    let err = sos_response(&b, &mismatched_a, 64, &[1.0, 1.0, 1.0]).unwrap_err();
    assert!(err.to_string().contains("same shape"));

    let bad_taps = Array4::zeros((1, 4, 1, 1));
    let err = sos_response(&bad_taps, &bad_taps, 64, &[1.0, 1.0, 1.0]).unwrap_err();
    assert!(err.to_string().contains("second axis must be 3"));
}

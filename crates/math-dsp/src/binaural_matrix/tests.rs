use super::direct::direct_peak_sample;
use super::direct::suppress_log_sweep_harmonic_residues;
use super::misc::align_ir_to_reference_peak;
use super::misc::deconvolve_sweep_to_ir;
use super::misc::fdw_complex_half_spectrum;
use super::misc::half_spectrum_to_fir;
use super::solve::solve_minimax_regularized_inverse_bin;
use super::solve::solve_regularized_inverse_bin;
use super::solve::solve_weighted_regularized_inverse_bin;
use super::transfer_matrix_bin::TransferMatrixBin;
use nalgebra::DMatrix;
use num_complex::Complex64;

#[test]
fn inverse_solves_identity_for_well_conditioned_2x2() {
    let h = TransferMatrixBin::new(
        2,
        2,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.2, 0.0),
            Complex64::new(0.15, 0.0),
            Complex64::new(0.9, 0.0),
        ],
    );
    let target = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];

    let solved =
        solve_regularized_inverse_bin(std::slice::from_ref(&h), &target, 1e-9, None).unwrap();
    let f = DMatrix::from_row_slice(2, 2, &solved.values);
    let delivered = h.as_matrix() * f;

    assert!((delivered[(0, 0)].re - 1.0).abs() < 1e-6);
    assert!(delivered[(0, 1)].norm() < 1e-6);
    assert!(delivered[(1, 0)].norm() < 1e-6);
    assert!((delivered[(1, 1)].re - 1.0).abs() < 1e-6);
}

#[test]
fn inverse_limits_large_gains() {
    let h = TransferMatrixBin::new(
        2,
        2,
        vec![
            Complex64::new(0.001, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.001, 0.0),
        ],
    );
    let target = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];

    let solved = solve_regularized_inverse_bin(&[h], &target, 1e-12, Some(6.0)).unwrap();
    let max_mag = solved.values.iter().map(|v| v.norm()).fold(0.0, f64::max);
    assert!(max_mag <= 10.0_f64.powf(6.0 / 20.0) + 1e-9);
}

#[test]
fn half_spectrum_identity_yields_delayed_impulse() {
    let spectrum = vec![Complex64::new(1.0, 0.0); 9];
    let fir = half_spectrum_to_fir(&spectrum, 16, 4.0).unwrap();
    let peak = fir
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();
    assert_eq!(peak, 4);
    assert!((fir[peak] - 1.0).abs() < 1e-9);
}

#[test]
fn minimax_reduces_or_matches_worst_position_error() {
    let target = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let positions = vec![
        TransferMatrixBin::new(
            2,
            2,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.2, 0.0),
                Complex64::new(0.2, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        ),
        TransferMatrixBin::new(
            2,
            2,
            vec![
                Complex64::new(0.7, 0.0),
                Complex64::new(0.45, 0.0),
                Complex64::new(0.35, 0.0),
                Complex64::new(0.8, 0.0),
            ],
        ),
    ];
    let average = solve_regularized_inverse_bin(&positions, &target, 0.01, Some(12.0)).unwrap();
    let minimax =
        solve_minimax_regularized_inverse_bin(&positions, &target, 0.01, Some(12.0), 8).unwrap();
    assert!(minimax.worst_position_error <= average.worst_position_error + 1e-9);
}

#[test]
fn deconvolution_alignment_and_harmonic_suppression_are_stable() {
    let mut reference = vec![0.0; 64];
    reference[0] = 1.0;
    let mut recording = vec![0.0; 64];
    recording[7] = 0.5;
    let ir = deconvolve_sweep_to_ir(&recording, &reference, 64).unwrap();
    assert_eq!(direct_peak_sample(&ir), 7);
    let aligned = align_ir_to_reference_peak(&ir, 7);
    assert_eq!(direct_peak_sample(&aligned), 0);

    let mut residue = vec![1.0; 128];
    suppress_log_sweep_harmonic_residues(&mut residue, 48_000.0, 1.0, 20.0, 20_000.0, 3, 1.0);
    assert!(residue.contains(&0.0));
}

#[test]
fn harmonic_suppression_tracks_delayed_direct_peak() {
    let sample_rate = 1_000.0_f64;
    let duration = 1.0_f64;
    let start_hz = 10.0_f64;
    let end_hz = 1_000.0_f64;
    let harmonic = 2usize;
    let len = 512usize;
    let direct = 123usize;
    let offset = (duration * (harmonic as f64).ln() / (end_hz / start_hz).ln() * sample_rate)
        .round() as usize;
    let residue = (direct + len - offset % len) % len;

    let mut ir = vec![0.0; len];
    ir[direct] = 1.0;
    ir[residue] = 0.5;
    suppress_log_sweep_harmonic_residues(
        &mut ir,
        sample_rate,
        duration,
        start_hz,
        end_hz,
        harmonic,
        0.0,
    );

    assert_eq!(ir[residue], 0.0);
    assert_eq!(ir[direct], 1.0);
}

#[test]
fn fdw_complex_half_spectrum_returns_fft_bins() {
    let mut ir = vec![0.0; 128];
    ir[8] = 1.0;
    let spectrum = fdw_complex_half_spectrum(&ir, 48_000.0, 128, 8, 8.0, 3.0, 30.0).unwrap();
    assert_eq!(spectrum.len(), 65);
    assert!(spectrum[1].norm() > 0.0);
}

/// Regression test: the DC bin must be computed from the same Hann-windowed
/// segment around `direct_sample` as every other bin, not from the unwindowed
/// sum of the full IR (which includes reverb-tail energy the neighboring
/// bins exclude).
#[test]
fn fdw_complex_half_spectrum_dc_bin_excludes_reverb_tail() {
    let sample_rate = 48_000.0;
    let fft_size = 256;
    let direct = 100usize;
    let mut ir = vec![0.01f64; 4096]; // dense reverb tail
    ir[direct] = 1.0; // direct sound

    let spectrum =
        fdw_complex_half_spectrum(&ir, sample_rate, fft_size, direct, 8.0, 3.0, 30.0).unwrap();

    // Expected DC: Hann-windowed segment around the direct sound. At 0 Hz the
    // window length clamp yields `max_window_ms` and the phase term is 1.
    let half = (30.0 / 1000.0 * sample_rate * 0.5).round() as isize;
    let mut expected = 0.0f64;
    for delta in -half..=half {
        let idx = direct as isize + delta;
        if idx < 0 || idx >= ir.len() as isize {
            continue;
        }
        let x = delta as f64 / half as f64;
        let w = 0.5 + 0.5 * (std::f64::consts::PI * x).cos();
        expected += ir[idx as usize] * w;
    }

    // Sanity: the unwindowed full-IR sum (old behavior) must be dominated by
    // the tail so this test can actually distinguish the two.
    let full_sum: f64 = ir.iter().sum();
    assert!(
        full_sum > 4.0 * expected,
        "test setup: tail should dominate the full sum ({full_sum} vs {expected})"
    );

    assert!(
        (spectrum[0].re - expected).abs() < 1e-9 * expected.abs().max(1.0),
        "DC bin should be the windowed sum {expected}, got {}",
        spectrum[0].re
    );
    assert!(spectrum[0].im.abs() < 1e-9, "DC bin must be real");
}

#[test]
fn weighted_inverse_rejects_empty_positions() {
    let target = vec![Complex64::new(1.0, 0.0); 4];
    let err = solve_weighted_regularized_inverse_bin(&[], &[1.0], &target, 1e-6, None).unwrap_err();
    assert!(err.contains("at least one"));
}

#[test]
fn weighted_inverse_rejects_mismatched_weights() {
    let h = TransferMatrixBin::new(2, 2, vec![Complex64::new(1.0, 0.0); 4]);
    let target = vec![Complex64::new(1.0, 0.0); 4];
    let err =
        solve_weighted_regularized_inverse_bin(&[h], &[1.0, 2.0], &target, 1e-6, None).unwrap_err();
    assert!(err.contains("weights len"));
}

#[test]
fn weighted_inverse_rejects_negative_beta() {
    let h = TransferMatrixBin::new(2, 2, vec![Complex64::new(1.0, 0.0); 4]);
    let target = vec![Complex64::new(1.0, 0.0); 4];
    let err =
        solve_weighted_regularized_inverse_bin(&[h], &[1.0], &target, -1.0, None).unwrap_err();
    assert!(err.contains("beta"));
}

#[test]
fn weighted_inverse_rejects_nan_beta() {
    let h = TransferMatrixBin::new(2, 2, vec![Complex64::new(1.0, 0.0); 4]);
    let target = vec![Complex64::new(1.0, 0.0); 4];
    let err =
        solve_weighted_regularized_inverse_bin(&[h], &[1.0], &target, f64::NAN, None).unwrap_err();
    assert!(err.contains("beta"));
}

#[test]
fn weighted_inverse_rejects_bad_target_size() {
    let h = TransferMatrixBin::new(2, 2, vec![Complex64::new(1.0, 0.0); 4]);
    let target = vec![Complex64::new(1.0, 0.0); 3];
    let err =
        solve_weighted_regularized_inverse_bin(&[h], &[1.0], &target, 1e-6, None).unwrap_err();
    assert!(err.contains("target"));
}

#[test]
fn weighted_inverse_rejects_negative_weight() {
    let h = TransferMatrixBin::new(2, 2, vec![Complex64::new(1.0, 0.0); 4]);
    let target = vec![Complex64::new(1.0, 0.0); 4];
    let err =
        solve_weighted_regularized_inverse_bin(&[h], &[-1.0], &target, 1e-6, None).unwrap_err();
    assert!(err.contains("weights"));
}

#[test]
fn weighted_inverse_identity_with_weights() {
    let h = TransferMatrixBin::new(
        2,
        2,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    );
    let target = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let solved = solve_weighted_regularized_inverse_bin(
        std::slice::from_ref(&h),
        &[2.0],
        &target,
        1e-9,
        None,
    )
    .unwrap();
    let f = DMatrix::from_row_slice(2, 2, &solved.values);
    let delivered = h.as_matrix() * f;
    assert!((delivered[(0, 0)].re - 1.0).abs() < 1e-6);
    assert!(delivered[(0, 1)].norm() < 1e-6);
    assert!(delivered[(1, 0)].norm() < 1e-6);
    assert!((delivered[(1, 1)].re - 1.0).abs() < 1e-6);
}

#[test]
fn weighted_inverse_gain_clamping() {
    let h = TransferMatrixBin::new(
        2,
        2,
        vec![
            Complex64::new(0.001, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.001, 0.0),
        ],
    );
    let target = vec![Complex64::new(1.0, 0.0); 4];
    let solved =
        solve_weighted_regularized_inverse_bin(&[h], &[1.0], &target, 1e-12, Some(6.0)).unwrap();
    let max_mag = solved.values.iter().map(|v| v.norm()).fold(0.0, f64::max);
    assert!(max_mag <= 10.0_f64.powf(6.0 / 20.0) + 1e-9);
}

#[test]
fn weighted_inverse_unweighted_matches_regularized() {
    let h = TransferMatrixBin::new(
        2,
        2,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.2, 0.0),
            Complex64::new(0.15, 0.0),
            Complex64::new(0.9, 0.0),
        ],
    );
    let target = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let weighted = solve_weighted_regularized_inverse_bin(
        std::slice::from_ref(&h),
        &[1.0],
        &target,
        1e-9,
        None,
    )
    .unwrap();
    let regularized = solve_regularized_inverse_bin(&[h], &target, 1e-9, None).unwrap();
    for (a, b) in weighted.values.iter().zip(regularized.values.iter()) {
        assert!(
            (a - b).norm() < 1e-9,
            "weighted and regularized should match when weight=1"
        );
    }
}

/// Characterization test: the solver now uses an LU factorization
/// (`normal.lu().solve(&rhs)`) instead of an explicit inverse
/// (`normal.try_inverse() * rhs`). Pin agreement with the old formulation
/// (reimplemented inline below) to a tight tolerance.
#[test]
fn lu_solve_matches_explicit_inverse_reference() {
    let target = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let positions = vec![
        TransferMatrixBin::new(
            2,
            3,
            vec![
                Complex64::new(1.0, 0.1),
                Complex64::new(0.2, -0.3),
                Complex64::new(0.4, 0.2),
                Complex64::new(0.15, 0.05),
                Complex64::new(0.9, -0.1),
                Complex64::new(-0.2, 0.35),
            ],
        ),
        TransferMatrixBin::new(
            2,
            3,
            vec![
                Complex64::new(0.7, -0.2),
                Complex64::new(0.45, 0.1),
                Complex64::new(0.1, 0.4),
                Complex64::new(0.35, 0.25),
                Complex64::new(0.8, 0.0),
                Complex64::new(0.05, -0.15),
            ],
        ),
    ];
    let beta = 1e-3;
    let solved = solve_regularized_inverse_bin(&positions, &target, beta, None).unwrap();

    // Reference: old explicit-inverse formulation.
    let speakers = 3usize;
    let ears = 2usize;
    let target_matrix = DMatrix::from_row_slice(ears, ears, &target);
    let mut normal = DMatrix::<Complex64>::zeros(speakers, speakers);
    let mut rhs = DMatrix::<Complex64>::zeros(speakers, ears);
    for matrix in &positions {
        let h = matrix.as_matrix();
        let h_h = h.adjoint();
        normal += &h_h * &h;
        rhs += h_h * &target_matrix;
    }
    for idx in 0..speakers {
        normal[(idx, idx)] += Complex64::new(beta, 0.0);
    }
    let reference = normal.try_inverse().unwrap() * rhs;

    assert_eq!(solved.values.len(), speakers * ears);
    for row in 0..speakers {
        for col in 0..ears {
            let got = solved.values[row * ears + col];
            let expected = reference[(row, col)];
            assert!(
                (got - expected).norm() < 1e-12,
                "LU solve should match explicit inverse: got {got}, expected {expected}"
            );
        }
    }
}

//! Wolfram cross-check: unnormalized FFT peak of an integer-cycle tone.
//!
//! Oracle: `wolfram/fft_peak.wls` (same samples, `{1,-1}` DFT). The peak
//! bin reads exactly N/2; off-peak bins read ~0 (f32 sample rounding
//! bounds the floor). Tolerance 1e-6 relative on the peak.

use math_qa::{QaResult, assert_close, assert_close_abs, emit_result, provenance, reference};
use rustfft::{FftPlanner, num_complex::Complex};

const CASE: &str = "fft_peak";
const CASE_ID: &str = "math-qa.fft-peak.v1";
const TOL_PEAK: f64 = 1e-6;
const TOL_FLOOR_ABS: f64 = 0.05;

#[test]
fn wolfram_fft_peak() {
    let Some(ref_json) = reference(CASE, "fft_peak.wls") else {
        return;
    };
    assert_eq!(ref_json["case"], CASE_ID);
    let n: usize = serde_json::from_value(ref_json["n_samples"].clone()).unwrap();
    let expected_peak: f64 = serde_json::from_value(ref_json["peak_mag"].clone()).unwrap();
    let off_bins: Vec<usize> = serde_json::from_value(ref_json["off_bins"].clone()).unwrap();
    let off_mags: Vec<f64> = serde_json::from_value(ref_json["off_mags"].clone()).unwrap();

    // Same samples as the oracle: sin(2 pi 1000 n / 48000) in f32.
    let sr = 48000.0f64;
    let mut buf: Vec<Complex<f32>> = (0..n)
        .map(|i| {
            Complex::new(
                (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / sr).sin() as f32,
                0.0,
            )
        })
        .collect();
    FftPlanner::new().plan_fft_forward(n).process(&mut buf);

    let peak = buf[1000].norm() as f64;
    assert_close(peak, expected_peak, TOL_PEAK, "peak magnitude");
    assert_close(peak, n as f64 / 2.0, TOL_PEAK, "peak == N/2");
    for (bin, expected) in off_bins.iter().zip(off_mags.iter()) {
        let mag = buf[*bin].norm() as f64;
        assert_close_abs(
            mag,
            *expected,
            TOL_FLOOR_ABS,
            &format!("off-peak bin {bin}"),
        );
    }
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: (peak - expected_peak).abs() / expected_peak,
        tolerance: TOL_PEAK,
        provenance: provenance(),
    });
}

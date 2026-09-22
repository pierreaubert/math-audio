//! Wolfram cross-check: Schroeder decay and T30 of a noise-free
//! exponential IR.
//!
//! Oracle: `wolfram/schroeder_t60.wls` (exact backward integration +
//! least-squares fit over -5..-35 dB). Tolerances: T60 1e-4 relative
//! (fit-window edge quantization), curve points 1e-6 dB absolute.

use math_qa::{QaResult, assert_close, assert_close_abs, emit_result, provenance, reference};
use math_rir::schroeder_curve;

const CASE: &str = "schroeder_t60";
const CASE_ID: &str = "math-qa.schroeder-t60.v1";
const TOL_T60: f64 = 1e-4;
/// Absolute dB tolerance for curve points (already in dB, cross zero;
/// f32 IR rounding dominates at ~1e-9 dB).
const TOL_CURVE_DB: f64 = 1e-6;

#[test]
fn wolfram_schroeder_t60() {
    let Some(ref_json) = reference(CASE, "schroeder_t60.wls") else {
        return;
    };
    assert_eq!(ref_json["case"], CASE_ID);
    let sr: f64 = serde_json::from_value(ref_json["sample_rate_hz"].clone()).unwrap();
    let expected_t60: f64 = serde_json::from_value(ref_json["t60_s"].clone()).unwrap();
    let probe_times: Vec<f64> = serde_json::from_value(ref_json["probe_times_s"].clone()).unwrap();
    let probe_db: Vec<f64> = serde_json::from_value(ref_json["probe_db"].clone()).unwrap();

    // Same samples as the oracle: h[n] = exp(-alpha n / sr), alpha for
    // 60 dB decay in the planted T60.
    let planted: f64 = serde_json::from_value(ref_json["planted_t60_s"].clone()).unwrap();
    let alpha = 3.0 * std::f64::consts::LN_10 / planted;
    let n = (2.0 * sr) as usize;
    let rir: Vec<f32> = (0..n)
        .map(|i| (-alpha * i as f64 / sr).exp() as f32)
        .collect();

    let curve = schroeder_curve(&rir, sr);
    let (slope, _, _) = curve
        .fit_db_range(-5.0, -35.0)
        .expect("exact decay must fit");
    let t60 = -60.0 / slope;
    assert_close(t60, expected_t60, TOL_T60, "T30");

    let mut max_err = (t60 - expected_t60).abs() / expected_t60;
    for (t, expected) in probe_times.iter().zip(probe_db.iter()) {
        // Curve values are already in dB (and cross zero), so the
        // comparison is absolute, not relative.
        let idx = (*t * sr).round() as usize;
        let actual = curve.samples[idx.min(curve.samples.len() - 1)];
        let err = (actual - *expected).abs();
        assert_close_abs(actual, *expected, TOL_CURVE_DB, &format!("decay at t={t}s"));
        max_err = max_err.max(err);
    }
    // Cross-check through the public ISO entry point as well.
    let metrics = math_rir::analyze_iso3382(&rir, sr);
    assert_close(metrics.t30_s, expected_t60, TOL_T60, "ISO T30");
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_err,
        tolerance: TOL_T60,
        provenance: provenance(),
    });
}

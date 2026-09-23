//! Wolfram cross-check: dry-plus-bank Kautz correction response.
//!
//! Oracle: `wolfram/kautz_correction.wls` (direct z-domain summation of the
//! ordered allpass-coupled basis — independent of the Rust recurrence).
//! Compares `correction_response` at every grid point and rate (tolerance
//! 1e-9 complex relative), plus a streamed-impulse DTFT against the oracle
//! values (independent code path: recurrence stream plus naive DFT, never
//! the closed-form evaluator as its own oracle).

use math_audio_iir_fir::KautzCorrection;
use math_qa::{QaResult, complex_rel_error, emit_result, provenance, reference};
use num_complex::Complex64;

const CASE: &str = "kautz_correction";
const CASE_ID: &str = "math-qa.kautz-correction.v1";

#[test]
fn wolfram_kautz_correction() {
    let Some(ref_json) = reference(CASE, "kautz_correction.wls") else {
        return;
    };
    assert_eq!(ref_json["case"], CASE_ID);
    let mut max_err = 0.0f64;
    let mut max_tol = 0.0f64;

    let blocks: Vec<serde_json::Value> =
        serde_json::from_value(ref_json["blocks"].clone()).unwrap();
    assert!(!blocks.is_empty());
    for block in &blocks {
        let sr: f64 = serde_json::from_value(block["sample_rate_hz"].clone()).unwrap();
        let pairs: Vec<[f64; 2]> = serde_json::from_value(block["modes"].clone()).unwrap();
        let modes: Vec<(f64, f64)> = pairs.iter().map(|&[f, q]| (f, q)).collect();
        let gains: Vec<f64> = serde_json::from_value(block["gains"].clone()).unwrap();
        let mut bank = KautzCorrection::<f64>::new(&modes, sr).expect("oracle bank builds");
        bank.set_gains(&gains).expect("oracle gains apply");
        let points: Vec<serde_json::Value> =
            serde_json::from_value(block["points"].clone()).unwrap();
        for point in &points {
            let freq: f64 = serde_json::from_value(point["freq_hz"].clone()).unwrap();
            let tol: f64 = serde_json::from_value(point["tolerance"].clone()).unwrap();
            max_tol = max_tol.max(tol);
            let re: f64 = serde_json::from_value(point["re"].clone()).unwrap();
            let im: f64 = serde_json::from_value(point["im"].clone()).unwrap();
            let expected = Complex64::new(re, im);
            let got = bank.correction_response(freq);
            let got64 = Complex64::new(got.re, got.im);
            let err = complex_rel_error(got64, expected);
            assert!(
                err <= tol,
                "H({freq} Hz) at {sr} Hz: rust={got64:?} expected={expected:?} rel_err={err:.3e}"
            );
            max_err = max_err.max(err);
        }
    }

    // Streamed impulse DTFT vs oracle values (48 kHz two-section block):
    // recurrence stream plus naive DFT against the engine's formula.
    let block = &blocks[1];
    let sr: f64 = serde_json::from_value(block["sample_rate_hz"].clone()).unwrap();
    assert_eq!(sr, 48_000.0);
    let pairs: Vec<[f64; 2]> = serde_json::from_value(block["modes"].clone()).unwrap();
    let modes: Vec<(f64, f64)> = pairs.iter().map(|&[f, q]| (f, q)).collect();
    let gains: Vec<f64> = serde_json::from_value(block["gains"].clone()).unwrap();
    let mut bank = KautzCorrection::<f64>::new(&modes, sr).expect("bank builds");
    bank.set_gains(&gains).expect("gains apply");
    // Slowest pole (80 Hz, Q=6): r ~ 0.99913; 60k samples bury the tail.
    let n = 60_000usize;
    let mut recorded = Vec::with_capacity(n);
    for i in 0..n {
        recorded.push(bank.process(if i == 0 { 1.0 } else { 0.0 }));
    }
    let points: Vec<serde_json::Value> = serde_json::from_value(block["points"].clone()).unwrap();
    for point in points.iter().step_by(6) {
        let freq: f64 = serde_json::from_value(point["freq_hz"].clone()).unwrap();
        let tol: f64 = serde_json::from_value(point["tolerance"].clone()).unwrap();
        let omega = 2.0 * std::f64::consts::PI * freq / sr;
        let mut acc = Complex64::new(0.0, 0.0);
        for (k, &x) in recorded.iter().enumerate() {
            acc += Complex64::from_polar(x, -omega * k as f64);
        }
        let expected = Complex64::new(
            serde_json::from_value(point["re"].clone()).unwrap(),
            serde_json::from_value(point["im"].clone()).unwrap(),
        );
        let err = complex_rel_error(acc, expected);
        assert!(
            err <= tol * 1000.0,
            "impulse DTFT at {freq} Hz: rel_err={err:.3e}"
        );
        max_err = max_err.max(err / 1000.0);
    }

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_err,
        tolerance: max_tol,
        provenance: provenance(),
    });
}

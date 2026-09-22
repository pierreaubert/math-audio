//! Wolfram cross-check: M2 fixed-centre third-octave SPL of two sines.
//!
//! Oracle: `wolfram/third_octave_spl.wls` (peak band 1 kHz at 0 dB,
//! 2 kHz band at exactly 20log10(1/2), Hann-sidelobe neighbor ceiling).
//! Absolute dB tolerances: peak-band identity is exact, the second band
//! allows 0.3 dB of FFT leakage.

use math_audio_dsp::rir_early_late::third_octave_spl;
use math_qa::{QaResult, assert_close_abs, emit_result, provenance, reference};

const CASE: &str = "third_octave_spl";
const CASE_ID: &str = "math-qa.third-octave-spl.v1";
const TOL_SECOND_DB: f64 = 0.3;

#[test]
fn wolfram_third_octave_spl() {
    let Some(ref_json) = reference(CASE, "third_octave_spl.wls") else {
        return;
    };
    assert_eq!(ref_json["case"], CASE_ID);
    let peak_band: f64 = serde_json::from_value(ref_json["peak_band_hz"].clone()).unwrap();
    let second_band: f64 = serde_json::from_value(ref_json["second_band_hz"].clone()).unwrap();
    let second_db: f64 = serde_json::from_value(ref_json["second_band_db"].clone()).unwrap();
    let ceiling_db: f64 = serde_json::from_value(ref_json["neighbor_ceiling_db"].clone()).unwrap();

    // Same segment as the oracle: 1 kHz (amp 1) + 2 kHz (amp 1/2).
    let sr = 48000.0;
    let n = (0.1 * sr) as usize;
    let seg: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / sr;
            ((2.0 * std::f64::consts::PI * 1000.0 * t).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * 2000.0 * t).sin()) as f32
        })
        .collect();
    let spl = third_octave_spl(&seg, &[], sr);

    let peak = spl
        .iter()
        .max_by(|a, b| a.early_db.total_cmp(&b.early_db))
        .unwrap();
    assert!(
        (peak.centre_hz - peak_band).abs() < 1e-9,
        "peak band = {} Hz (expected {peak_band})",
        peak.centre_hz
    );
    let second = spl
        .iter()
        .find(|b| (b.centre_hz - second_band).abs() < 1e-9)
        .unwrap();
    assert_close_abs(second.early_db, second_db, TOL_SECOND_DB, "2 kHz band");
    for b in &spl {
        if (b.centre_hz - peak_band).abs() > 1e-9 && (b.centre_hz - second_band).abs() > 1e-9 {
            assert!(
                b.early_db <= ceiling_db,
                "band {} Hz leaks at {} dB (ceiling {ceiling_db})",
                b.centre_hz,
                b.early_db
            );
        }
    }
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: (second.early_db - second_db).abs() / second_db.abs(),
        tolerance: TOL_SECOND_DB,
        provenance: provenance(),
    });
}

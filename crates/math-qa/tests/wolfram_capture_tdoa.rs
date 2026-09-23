//! Wolfram cross-check: C1 chirp TDOA, C2 clock skew, C4 uncertainty.
//!
//! Oracle: `wolfram/capture_tdoa.wls` (schema 2: one C1/C4 block per
//! sample rate across the 48 kHz family, the 44.1 kHz family and
//! cheap-device rates). C1 compares the Rust FFT cross-correlation
//! against the engine's direct-summation correlation (independent method,
//! own chirp, own Fourier-shift delay application); C2/C4 compare exact
//! arithmetic evaluations. C2 is pure sample-domain arithmetic with no
//! rate input, so it keeps a single entry by design.

use math_audio_dsp::capture_tdoa::{
    ClockSkew, TdoaConfig, TdoaEstimate, estimate_chirp_tdoa, estimate_clock_skew,
    post_correction_uncertainty_us,
};
use math_audio_dsp::signals::gen_log_sweep;
use math_qa::{QaResult, assert_close, assert_close_abs, emit_result, provenance, reference};

const CASE: &str = "capture_tdoa";
const CASE_ID: &str = "math-qa.capture-tdoa.v1";

/// Exact fractional delay via windowed-sinc FIR (same family as the
/// in-crate C1 oracle, rewritten here so the QA test does not share code
/// with the implementation under test).
fn fractional_delay(input: &[f32], delay: f64) -> Vec<f32> {
    const TAPS: usize = 127;
    const CENTER: i64 = 63;
    let int = delay.floor() as i64;
    let frac = delay - delay.floor();
    let sinc = |x: f64| {
        if x.abs() < 1e-12 {
            1.0
        } else {
            (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
        }
    };
    let mut taps = vec![0.0f64; TAPS];
    for (i, tap) in taps.iter_mut().enumerate() {
        let u = 2.0 * i as f64 / (TAPS - 1) as f64 - 1.0;
        *tap =
            sinc(i as f64 - CENTER as f64 - frac) * 0.5 * (1.0 + (std::f64::consts::PI * u).cos());
    }
    let gain: f64 = taps.iter().sum();
    (0..input.len())
        .map(|n| {
            taps.iter()
                .enumerate()
                .map(|(i, h)| {
                    let m = n as i64 - int - (i as i64 - CENTER);
                    h / gain
                        * if m < 0 || m >= input.len() as i64 {
                            0.0
                        } else {
                            input[m as usize] as f64
                        }
                })
                .sum::<f64>() as f32
        })
        .collect()
}

fn config_for(sr: f64, band_lo: f64, band_hi: f64) -> TdoaConfig {
    TdoaConfig {
        sample_rate_hz: sr,
        band_lo_hz: band_lo,
        band_hi_hz: band_hi,
        ..TdoaConfig::default()
    }
}

#[test]
fn wolfram_capture_tdoa() {
    let Some(ref_json) = reference(CASE, "capture_tdoa.wls") else {
        return;
    };
    assert_eq!(ref_json["case"], CASE_ID);
    let mut max_err = 0.0f64;
    let mut max_tol = 0.0f64;

    // C1: per-rate chirp band, own delay application; the engine
    // estimated the same true delays independently at the same rate.
    let c1: Vec<serde_json::Value> =
        serde_json::from_value(ref_json["c1_entries"].clone()).unwrap();
    assert!(!c1.is_empty());
    for entry in &c1 {
        let sr: f64 = serde_json::from_value(entry["sample_rate_hz"].clone()).unwrap();
        let band_lo: f64 = serde_json::from_value(entry["band_lo_hz"].clone()).unwrap();
        let band_hi: f64 = serde_json::from_value(entry["band_hi_hz"].clone()).unwrap();
        let config = config_for(sr, band_lo, band_hi);
        let reference = gen_log_sweep((sr / 96.0) as f32, (sr / 4.0) as f32, 0.9, sr as u32, 0.25);
        let delay: f64 = serde_json::from_value(entry["true_delay_samples"].clone()).unwrap();
        let tol: f64 = serde_json::from_value(entry["offset_tolerance"].clone()).unwrap();
        let conf_tol: f64 =
            serde_json::from_value(entry["confidence_tolerance_db"].clone()).unwrap();
        max_tol = max_tol.max(tol);
        // The FIR oracle handles negative delays natively (advance with
        // zero fill, same as the in-crate C1 oracle); no padding needed.
        let recorded = fractional_delay(&reference, delay);
        let est = estimate_chirp_tdoa(&reference, &recorded, &config);
        assert!(est.valid, "C1 invalid at {sr} Hz delay {delay}");
        let expected: f64 = serde_json::from_value(entry["offset_samples"].clone()).unwrap();
        let err = (est.offset_samples - expected).abs();
        assert_close_abs(
            est.offset_samples,
            expected,
            tol,
            &format!("C1 offset at {sr} Hz delay {delay}"),
        );
        max_err = max_err.max(err / tol);
        let expected_conf: f64 = serde_json::from_value(entry["confidence_db"].clone()).unwrap();
        assert_close_abs(
            est.confidence_db,
            expected_conf,
            conf_tol,
            &format!("C1 confidence at {sr} Hz delay {delay}"),
        );
    }

    // C2: skew arithmetic from the oracle's chirp pair (rate-free).
    let c2: Vec<serde_json::Value> =
        serde_json::from_value(ref_json["c2_entries"].clone()).unwrap();
    for entry in &c2 {
        let d1: f64 = serde_json::from_value(entry["offset_samples"].clone()).unwrap();
        let ppm: f64 = serde_json::from_value(entry["skew_ppm"].clone()).unwrap();
        let tol: f64 = serde_json::from_value(entry["tolerance"].clone()).unwrap();
        max_tol = max_tol.max(tol);
        let spacing = 480_000.0;
        let d2 = d1 + ppm * spacing / 1e6;
        let mk = |o: f64| TdoaEstimate {
            offset_samples: o,
            confidence_db: 20.0,
            valid: true,
        };
        let skew = estimate_clock_skew(&mk(d1), &mk(d2), spacing);
        assert!(skew.valid);
        let err = math_qa::rel_error(skew.skew_ppm, ppm);
        assert_close(skew.skew_ppm, ppm, tol, "C2 skew_ppm");
        max_err = max_err.max(err);
    }

    // C4: bound evaluation against the oracle's formula value, per rate.
    let c4: Vec<serde_json::Value> =
        serde_json::from_value(ref_json["c4_entries"].clone()).unwrap();
    assert!(!c4.is_empty());
    for entry in &c4 {
        let sr: f64 = serde_json::from_value(entry["sample_rate_hz"].clone()).unwrap();
        let band_lo: f64 = serde_json::from_value(entry["band_lo_hz"].clone()).unwrap();
        let band_hi: f64 = serde_json::from_value(entry["band_hi_hz"].clone()).unwrap();
        let spacing: f64 = serde_json::from_value(entry["chirp_spacing_samples"].clone()).unwrap();
        let eval: f64 = serde_json::from_value(entry["eval_time_samples"].clone()).unwrap();
        let expected: f64 = serde_json::from_value(entry["uncertainty_us"].clone()).unwrap();
        let tol: f64 = serde_json::from_value(entry["tolerance"].clone()).unwrap();
        let config = config_for(sr, band_lo, band_hi);
        let mk = |conf: f64| TdoaEstimate {
            offset_samples: 0.0,
            confidence_db: conf,
            valid: true,
        };
        let skew = ClockSkew {
            offset_samples: 0.0,
            skew_ppm: 5.0,
            valid: true,
            low_confidence: false,
            inconsistent_spacing: false,
        };
        let got =
            post_correction_uncertainty_us(&mk(20.0), &mk(14.0), &skew, spacing, eval, &config);
        let err = math_qa::rel_error(got, expected);
        assert_close(got, expected, tol, &format!("C4 uncertainty_us at {sr} Hz"));
        max_err = max_err.max(err);
    }
    // C4 invalid flag → infinite, never zero.
    let bad = TdoaEstimate {
        offset_samples: 0.0,
        confidence_db: 20.0,
        valid: false,
    };
    let skew = ClockSkew {
        offset_samples: 0.0,
        skew_ppm: 0.0,
        valid: false,
        low_confidence: true,
        inconsistent_spacing: false,
    };
    let config = TdoaConfig::default();
    let inf = post_correction_uncertainty_us(&bad, &bad, &skew, 480_000.0, 0.0, &config);
    assert!(inf.is_infinite() && inf > 0.0);

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_err,
        tolerance: max_tol,
        provenance: provenance(),
    });
}

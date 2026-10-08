//! Wolfram cross-check: indirect (Schroeder-MTF) STI from synthetic IRs.
//!
//! Oracle: `wolfram/sti.wls` (closed-form IEC 60268-16:2020: analytic
//! exponential-decay MTF, Annex A TI mapping with +/-15 dB rails, MTI
//! means, Table A.1 alpha/beta combination). The Rust side runs
//! `analyze_sti` on a unit impulse (ideal MTF = 1) and on decaying
//! seven-carrier IRs whose band envelopes match the oracle up to
//! octave-filter transients and carrier ripple. Per-entry absolute
//! tolerances come from the oracle; all quantities live in [0, 1].

use math_qa::{QaResult, assert_close_abs, emit_result, provenance, reference};
use math_rir::sti::{
    STI_ALPHA, STI_BETA, STI_MODULATION_FREQUENCIES_HZ, STI_OCTAVE_CENTERS_HZ, analyze_sti,
};

const CASE: &str = "sti";
const CASE_ID: &str = "math-qa.sti.v1";
/// IEC constants are decimal literals on both sides; 1e-15 absorbs a
/// last-ulp decimal-to-binary difference while still catching typos.
const TOL_CONST: f64 = 1e-15;

fn build_impulse(len: usize, at: usize) -> Vec<f32> {
    let mut ir = vec![0.0f32; len];
    ir[at] = 1.0;
    ir
}

fn build_exponential(sr: f64, rt60: f64, duration_s: f64) -> Vec<f32> {
    let decay = 6.0 * std::f64::consts::LN_10 / rt60;
    (0..(sr * duration_s) as usize)
        .map(|i| {
            let t = i as f64 / sr;
            let carriers: f64 = STI_OCTAVE_CENTERS_HZ
                .iter()
                .map(|f| (std::f64::consts::TAU * f * t).sin())
                .sum();
            (carriers * (-decay * t / 2.0).exp()) as f32
        })
        .collect()
}

#[test]
fn wolfram_sti() {
    let Some(ref_json) = reference(CASE, "sti.wls") else {
        return;
    };
    assert_eq!(ref_json["case"], CASE_ID);

    // The IEC Table A.1 constants must match the oracle.
    let oracle_mod: Vec<f64> =
        serde_json::from_value(ref_json["modulation_frequencies_hz"].clone()).unwrap();
    let oracle_centers: Vec<f64> =
        serde_json::from_value(ref_json["octave_centers_hz"].clone()).unwrap();
    let oracle_alpha: Vec<f64> = serde_json::from_value(ref_json["alpha"].clone()).unwrap();
    let oracle_beta: Vec<f64> = serde_json::from_value(ref_json["beta"].clone()).unwrap();
    for (actual, expected) in STI_MODULATION_FREQUENCIES_HZ.iter().zip(oracle_mod.iter()) {
        assert_close_abs(*actual, *expected, TOL_CONST, "modulation frequency");
    }
    for (actual, expected) in STI_OCTAVE_CENTERS_HZ.iter().zip(oracle_centers.iter()) {
        assert_close_abs(*actual, *expected, TOL_CONST, "octave center");
    }
    for (actual, expected) in STI_ALPHA.iter().zip(oracle_alpha.iter()) {
        assert_close_abs(*actual, *expected, TOL_CONST, "alpha weight");
    }
    for (actual, expected) in STI_BETA.iter().zip(oracle_beta.iter()) {
        assert_close_abs(*actual, *expected, TOL_CONST, "beta weight");
    }

    let sr: f64 = serde_json::from_value(ref_json["sample_rate_hz"].clone()).unwrap();
    let entries: Vec<serde_json::Value> =
        serde_json::from_value(ref_json["entries"].clone()).unwrap();
    let mut max_err = 0.0f64;
    let mut max_tol = 0.0f64;
    for entry in &entries {
        let name: String = serde_json::from_value(entry["name"].clone()).unwrap();
        let tol_mtf: f64 = serde_json::from_value(entry["tolerance_mtf"].clone()).unwrap();
        let tol_sti: f64 = serde_json::from_value(entry["tolerance_sti"].clone()).unwrap();
        max_tol = max_tol.max(tol_mtf).max(tol_sti);
        let expected_mtf: Vec<Vec<f64>> = serde_json::from_value(entry["mtf"].clone()).unwrap();
        let expected_mti: Vec<f64> = serde_json::from_value(entry["mti"].clone()).unwrap();
        let expected_sti: f64 = serde_json::from_value(entry["sti"].clone()).unwrap();

        let ir = match entry["rt60_s"].as_f64() {
            Some(rt60) => {
                let duration_s: f64 =
                    serde_json::from_value(ref_json["ir_duration_s"][name.clone()].clone())
                        .unwrap();
                build_exponential(sr, rt60, duration_s)
            }
            None => build_impulse((sr * 1.0) as usize, 2400),
        };
        let result =
            analyze_sti(&ir, sr).unwrap_or_else(|e| panic!("{name}: analyze_sti failed: {e:?}"));

        let mut entry_err = 0.0f64;
        for (row, expected_row) in result.modulation_transfer.iter().zip(expected_mtf.iter()) {
            for (actual, expected) in row.iter().zip(expected_row.iter()) {
                let err = (actual - expected).abs();
                assert_close_abs(*actual, *expected, tol_mtf, &format!("{name} MTF"));
                entry_err = entry_err.max(err);
            }
        }
        for (actual, expected) in result.mti.iter().zip(expected_mti.iter()) {
            let err = (actual - expected).abs();
            assert_close_abs(*actual, *expected, tol_sti, &format!("{name} MTI"));
            entry_err = entry_err.max(err);
        }
        let err = (result.sti - expected_sti).abs();
        assert_close_abs(result.sti, expected_sti, tol_sti, &format!("{name} STI"));
        entry_err = entry_err.max(err);
        max_err = max_err.max(entry_err);
    }
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_err,
        tolerance: max_tol,
        provenance: provenance(),
    });
}

//! Wolfram cross-check: M1 comb quantities from two-delta IRs.
//!
//! Oracle: `wolfram/comb_first_dip.wls` (pure formulas: dip = 1/2dt,
//! ripple = 20log10((1+g)/(1-g)), path = c dt). The Rust picker must
//! reproduce them from bandpassed unit deltas. Tolerance 1e-6 relative.

use math_qa::{QaResult, assert_close, emit_result, provenance, reference};
use math_rir::report::{ReflectionTableConfig, early_reflection_table};

const CASE: &str = "comb_first_dip";
const CASE_ID: &str = "math-qa.comb-first-dip.v1";

#[test]
fn wolfram_comb_first_dip() {
    let Some(ref_json) = reference(CASE, "comb_first_dip.wls") else {
        return;
    };
    assert_eq!(ref_json["case"], CASE_ID);
    let entries: Vec<serde_json::Value> =
        serde_json::from_value(ref_json["entries"].clone()).unwrap();

    let sr = 48000.0;
    let cfg = ReflectionTableConfig::default();
    let mut max_err = 0.0f64;
    let mut max_tol = 0.0f64;
    for entry in &entries {
        let delay_s: f64 = serde_json::from_value(entry["delay_s"].clone()).unwrap();
        // Per-entry tolerance from the oracle (ring overlap at 2 ms).
        let tol: f64 = serde_json::from_value(entry["tolerance"].clone()).unwrap();
        max_tol = max_tol.max(tol);
        let delay_samples = (delay_s * sr).round() as usize;
        let n = 480 + delay_samples + (20.0 * sr / 1000.0) as usize;
        let mut rir = vec![0.0f32; n];
        rir[480] = 1.0;
        rir[480 + delay_samples] = 0.5;

        let table = early_reflection_table(&rir, sr, &cfg);
        assert_eq!(table.reflections.len(), 1);
        let r = &table.reflections[0];
        for (actual, key) in [
            (r.delay_ms, "delay_ms"),
            (r.gain_db, "gain_db"),
            (r.path_difference_m, "path_difference_m"),
            (r.first_dip_hz, "first_dip_hz"),
            (r.comb_ripple_db, "comb_ripple_db"),
        ] {
            let expected: f64 = serde_json::from_value(entry[key].clone()).unwrap();
            let err = math_qa::rel_error(actual, expected);
            assert_close(actual, expected, tol, &format!("{key} at dt={delay_s}s"));
            max_err = max_err.max(err);
        }
    }
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_err,
        tolerance: max_tol,
        provenance: provenance(),
    });
}

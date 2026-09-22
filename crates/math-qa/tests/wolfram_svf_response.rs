//! Wolfram cross-check: TPT-SVF lowpass complex response.
//!
//! Oracle: `wolfram/svf_response.wls` (bilinear-discretized analog
//! prototype — the exact mapping TPT implements — evaluated by an
//! independent CAS). Tolerance 1e-6 relative on complex values.

use math_audio_iir_fir::{SvfFilter, SvfFilterType};
use math_qa::{QaResult, complex_rel_error, emit_result, provenance, reference};
use num_complex::Complex64;

const CASE: &str = "svf_response";
const CASE_ID: &str = "math-qa.svf-response.v1";
const TOL: f64 = 1e-6;

#[test]
fn wolfram_svf_response() {
    let Some(ref_json) = reference(CASE, "svf_response.wls") else {
        return;
    };
    assert_eq!(ref_json["case"], CASE_ID);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let pairs: Vec<[f64; 2]> = serde_json::from_value(ref_json["response_re_im"].clone()).unwrap();

    let svf = SvfFilter::new(
        SvfFilterType::Lowpass,
        1000.0,
        48000.0,
        std::f64::consts::FRAC_1_SQRT_2,
        0.0,
    );
    let mut max_err = 0.0f64;
    for (f, pair) in freqs.iter().zip(pairs.iter()) {
        let rust = svf.response_at(*f);
        let expected = Complex64::new(pair[0], pair[1]);
        let err = complex_rel_error(rust, expected);
        assert!(
            err <= TOL,
            "H({f} Hz): rust={rust:?} expected={expected:?} rel_err={err:.3e} tol={TOL:.1e}"
        );
        max_err = max_err.max(err);
    }
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_err,
        tolerance: TOL,
        provenance: provenance(),
    });
}

//! Wolfram cross-check: RBJ lowpass biquad complex response.
//!
//! Oracle: `wolfram/biquad_response.wls` (RBJ cookbook, independent CAS
//! evaluation). Tolerance 1e-9 relative on the complex values.

use math_audio_dsp::biquad_complex_response;
use math_audio_iir_fir::{Biquad, BiquadFilterType};
use math_qa::{QaResult, complex_rel_error, emit_result, provenance, reference};
use num_complex::Complex64;

const CASE: &str = "biquad_response";
const CASE_ID: &str = "math-qa.biquad-response.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_biquad_response() {
    let Some(ref_json) = reference(CASE, "biquad_response.wls") else {
        return;
    };
    assert_eq!(ref_json["case"], CASE_ID);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let pairs: Vec<[f64; 2]> = serde_json::from_value(ref_json["response_re_im"].clone()).unwrap();

    let biquad = Biquad::new(
        BiquadFilterType::Lowpass,
        1000.0,
        48000.0,
        std::f64::consts::FRAC_1_SQRT_2,
        0.0,
    );
    let mut max_err = 0.0f64;
    for (f, pair) in freqs.iter().zip(pairs.iter()) {
        let rust = biquad_complex_response(&biquad, *f);
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

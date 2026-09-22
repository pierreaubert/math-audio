//! Wolfram cross-check: optimisation test-function values.
//!
//! Oracle: `wolfram/test_functions.wls` (textbook closed forms).
//! Exact agreement expected; tolerance 1e-12 relative (1e-9 absolute
//! at the known zeros).

use math_audio_test_functions::{ackley, rastrigin, rosenbrock, sphere};
use math_qa::{QaResult, assert_close, assert_close_abs, emit_result, provenance, reference};
use ndarray::Array1;

const CASE: &str = "test_functions";
const CASE_ID: &str = "math-qa.test-functions.v1";
const TOL_REL: f64 = 1e-12;
const TOL_ZERO_ABS: f64 = 1e-9;

#[test]
fn wolfram_test_functions() {
    let Some(ref_json) = reference(CASE, "test_functions.wls") else {
        return;
    };
    assert_eq!(ref_json["case"], CASE_ID);
    let values: serde_json::Value = ref_json["values"].clone();

    let cases: [(&str, f64, f64); 6] = [
        (
            "sphere_123",
            sphere(&Array1::from(vec![1.0, 2.0, 3.0])),
            14.0,
        ),
        (
            "rosenbrock_11",
            rosenbrock(&Array1::from(vec![1.0, 1.0])),
            0.0,
        ),
        (
            "rosenbrock_m11",
            rosenbrock(&Array1::from(vec![-1.0, 1.0])),
            4.0,
        ),
        (
            "rastrigin_half",
            rastrigin(&Array1::from(vec![0.5, -0.5])),
            40.5,
        ),
        ("ackley_00", ackley(&Array1::from(vec![0.0, 0.0])), 0.0),
        (
            "ackley_11",
            ackley(&Array1::from(vec![1.0, 1.0])),
            20.0 - 20.0 * (-0.2f64).exp(),
        ),
    ];
    let mut max_err: f64 = 0.0;
    for (key, rust, hand) in cases {
        let expected: f64 = serde_json::from_value(values[key].clone()).unwrap();
        // Oracle must agree with the hand-derived value first.
        assert_close_abs(expected, hand, TOL_ZERO_ABS, &format!("oracle {key}"));
        if hand == 0.0 {
            assert_close_abs(rust, expected, TOL_ZERO_ABS, key);
        } else {
            assert_close(rust, expected, TOL_REL, key);
            max_err = max_err.max(math_qa::rel_error(rust, expected));
        }
    }
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_err,
        tolerance: TOL_REL,
        provenance: provenance(),
    });
}

//! Wolfram cross-check: C5 DOA, C6 geometry, C3 resampler.
//!
//! Oracle: `wolfram/capture_array.wls`. C5 compares the Rust Cramer LS
//! solver against the engine's LinearSolve; C6 compares residuals and
//! scale calibration; C3 compares the Rust polyphase table against the
//! engine's direct textbook windowed-sinc interpolation of a closed-form
//! two-tone signal.

use math_audio_dsp::capture_array::{
    MicArray, PairDelay, calibrate_geometry_scale, check_geometry, estimate_doa_ls,
};
use math_audio_dsp::capture_resample::resample_to_common_clock;
use math_audio_dsp::capture_tdoa::ClockSkew;
use math_qa::{QaResult, assert_close, assert_close_abs, emit_result, provenance, reference};

const CASE: &str = "capture_array";
const CASE_ID: &str = "math-qa.capture-array.v1";

fn test_array() -> MicArray {
    MicArray::new(
        vec![
            [0.0, 0.0, 0.0],
            [0.12, 0.0, 0.0],
            [0.0, 0.12, 0.0],
            [0.0, 0.0, 0.12],
        ],
        48_000.0,
    )
    .expect("test array")
}

fn exact_pairs(array: &MicArray, dir: &[f64; 3]) -> Vec<PairDelay> {
    // Plane-wave delays from the documented convention (mic closer to the
    // source leads): delay_b_a = −((pb − pa)·dir)/c·sr.
    let mut pairs = Vec::new();
    for a in 0..array.len() {
        for b in (a + 1)..array.len() {
            let pa = array.position(a).unwrap();
            let pb = array.position(b).unwrap();
            let dot =
                (pb[0] - pa[0]) * dir[0] + (pb[1] - pa[1]) * dir[1] + (pb[2] - pa[2]) * dir[2];
            pairs.push(PairDelay {
                a,
                b,
                delay_samples: -dot / 343.0 * 48_000.0,
            });
        }
    }
    pairs
}

fn dir_of(az_deg: f64, el_deg: f64) -> [f64; 3] {
    let (az, el) = (az_deg.to_radians(), el_deg.to_radians());
    [el.cos() * az.cos(), el.cos() * az.sin(), el.sin()]
}

#[test]
fn wolfram_capture_array() {
    let Some(ref_json) = reference(CASE, "capture_array.wls") else {
        return;
    };
    assert_eq!(ref_json["case"], CASE_ID);
    let mut max_err = 0.0f64;
    let mut max_tol = 0.0f64;
    let array = test_array();

    // C5: engine solved the same exact-delay LS problem independently.
    let c5: Vec<serde_json::Value> =
        serde_json::from_value(ref_json["c5_entries"].clone()).unwrap();
    for entry in &c5 {
        let az: f64 = serde_json::from_value(entry["azimuth_deg"].clone()).unwrap();
        let el: f64 = serde_json::from_value(entry["elevation_deg"].clone()).unwrap();
        let tol: f64 = serde_json::from_value(entry["angle_tolerance_deg"].clone()).unwrap();
        let res_tol: f64 = serde_json::from_value(entry["residual_tolerance"].clone()).unwrap();
        max_tol = max_tol.max(tol);
        let pairs = exact_pairs(&array, &dir_of(az, el));
        let est = estimate_doa_ls(&array, &pairs).expect("ls solves");
        let exp_az: f64 = serde_json::from_value(entry["est_azimuth_deg"].clone()).unwrap();
        let exp_el: f64 = serde_json::from_value(entry["est_elevation_deg"].clone()).unwrap();
        let exp_res: f64 = serde_json::from_value(entry["rms_residual_samples"].clone()).unwrap();
        for (actual, expected, what) in [
            (est.azimuth_deg, exp_az, "C5 azimuth"),
            (est.elevation_deg, exp_el, "C5 elevation"),
        ] {
            let err = (actual - expected).abs();
            assert_close_abs(actual, expected, tol, &format!("{what} at az={az} el={el}"));
            max_err = max_err.max(err / tol);
        }
        assert_close_abs(est.rms_residual_samples, exp_res, res_tol, "C5 residual");
    }

    // C6: scale calibration against the engine value.
    let c6 = &ref_json["c6"];
    let big_positions: Vec<[f64; 3]> = (0..array.len())
        .map(|i| {
            let p = array.position(i).unwrap();
            [p[0] * 1.03, p[1] * 1.03, p[2] * 1.03]
        })
        .collect();
    let big = MicArray::new(big_positions, 48_000.0).unwrap();
    let dir = dir_of(30.0, 10.0);
    let pairs = exact_pairs(&big, &dir);
    let (scale, rms) = calibrate_geometry_scale(&array, &dir, &pairs).expect("calibrates");
    let exp_scale: f64 = serde_json::from_value(c6["scale_factor"].clone()).unwrap();
    let tol: f64 = serde_json::from_value(c6["scale_tolerance"].clone()).unwrap();
    let err = math_qa::rel_error(scale, exp_scale);
    assert_close(scale, exp_scale, tol, "C6 scale");
    max_err = max_err.max(err);
    let exp_rms: f64 = serde_json::from_value(c6["scale_rms_samples"].clone()).unwrap();
    assert_close_abs(rms, exp_rms, 1e-9, "C6 scale rms");
    // Declared geometry vs scaled measurements (worst residual 0.43 per
    // the oracle) must fail loudly at tol 0.1 — and exact pairs must pass.
    let exact = exact_pairs(&array, &dir);
    assert!(check_geometry(&array, &dir, &exact, 0.1).is_ok());
    let err = check_geometry(&array, &dir, &pairs, 0.1).expect_err("must fail loudly");
    assert!(err.contains("re-measure tape"), "{err}");

    // C3: closed-form two-tone signal, engine interpolated directly.
    let sr = 48_000.0;
    let tone = |n: f64| {
        (2.0 * std::f64::consts::PI * 440.0 * n / sr).sin()
            + 0.5 * (2.0 * std::f64::consts::PI * 2900.0 * n / sr + 1.0).sin()
    };
    let mic: Vec<f32> = (0..2000).map(|n| tone(n as f64) as f32).collect();
    let c3: Vec<serde_json::Value> =
        serde_json::from_value(ref_json["c3_entries"].clone()).unwrap();
    for entry in &c3 {
        let m: usize = serde_json::from_value(entry["index"].clone()).unwrap();
        let frac: f64 = serde_json::from_value(entry["offset_samples"].clone()).unwrap();
        let expected: f64 = serde_json::from_value(entry["value"].clone()).unwrap();
        let tol: f64 = serde_json::from_value(entry["tolerance_abs"].clone()).unwrap();
        max_tol = max_tol.max(tol);
        let skew = ClockSkew {
            offset_samples: frac,
            skew_ppm: 0.0,
            valid: true,
            low_confidence: false,
            inconsistent_spacing: false,
        };
        let out = resample_to_common_clock(&mic, &skew, mic.len()).expect("resamples");
        let err = (out[m] as f64 - expected).abs();
        assert_close_abs(
            out[m] as f64,
            expected,
            tol,
            &format!("C3 value at {m}+{frac}"),
        );
        max_err = max_err.max(err / tol);
    }

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_err,
        tolerance: max_tol,
        provenance: provenance(),
    });
}

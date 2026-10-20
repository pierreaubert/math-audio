//! Independent state-equivalence and factored matched-response regressions.

// Rust guideline compliant 2026-02-21
use super::*;
use num_complex::Complex;
use std::f64::consts::PI;

// The original implicit recurrence is an independent state-update reference.
fn implicit_step(f: &mut WarpedBiquad<f64>, x: f64) -> f64 {
    fn ap(x: f64, s: &mut f64, l: f64) -> f64 {
        let z = x + l * *s;
        let out = *s - l * z;
        *s = z;
        out
    }
    let l = f.lambda;
    let xd1 = ap(x, &mut f.ap_x1_s, l);
    let xd2 = ap(xd1, &mut f.ap_x2_s, l);
    let y = (f.b0 * x + f.b1 * xd1 + f.b2 * xd2
        - (1.0 - l * l) * ((f.a1 - f.a2 * l) * f.s_y1 + f.a2 * f.s_y2))
        / (1.0 - f.a1 * l + f.a2 * l * l);
    let yd1 = ap(y, &mut f.s_y1, l);
    ap(yd1, &mut f.s_y2, l);
    y
}

#[test]
fn retained_histories_map_to_the_same_implicit_future_after_retuning() {
    let mut maximum = 0.0_f64;
    let mut cases = 0;
    for kind in [
        BiquadFilterType::Peak,
        BiquadFilterType::Lowpass,
        BiquadFilterType::Highpass,
        BiquadFilterType::Bandpass,
        BiquadFilterType::Notch,
        BiquadFilterType::AllPass,
        BiquadFilterType::Lowshelf,
        BiquadFilterType::Highshelf,
        BiquadFilterType::LowshelfOrf,
        BiquadFilterType::HighshelfOrf,
        BiquadFilterType::PeakMatched,
    ] {
        for l in [-0.37, 0.0, 0.37, 0.766] {
            for freq in [20.0, 1000.0, 20000.0] {
                let mut f = WarpedBiquad::<f64>::new(kind, 1000.0, 48000.0, 0.7, 6.0, 0.37);
                for n in 0..257 {
                    f.process(0.1 * (n as f64 * 0.13).sin());
                }
                let history = [f.ap_x1_s, f.ap_x2_s, f.s_y1, f.s_y2];
                f.update_params(kind, freq, 96000.0, 0.7, -6.0, l);
                assert_eq!(history, [f.ap_x1_s, f.ap_x2_s, f.s_y1, f.s_y2]);
                let mut reference = f.clone();
                for n in 0..4096 {
                    let x = if n < 97 {
                        0.1 * (n as f64 * 0.17).sin()
                    } else {
                        0.0
                    };
                    maximum = maximum.max((f.process(x) - implicit_step(&mut reference, x)).abs());
                }
                cases += 1;
            }
        }
    }
    println!("stateful warped retunes={cases}, maximum implicit-reference error={maximum}");
    assert_eq!(cases, 132);
    assert!(maximum < 2e-7);
}

#[test]
fn old_serialized_allpass_histories_restore_without_reset() {
    let mut f = WarpedBiquad::<f64>::new(BiquadFilterType::Peak, 1000.0, 48000.0, 2.0, 6.0, 0.37);
    for n in 0..257 {
        f.process(0.1 * (n as f64 * 0.13).sin());
    }
    let mut json = serde_json::to_value(&f).unwrap();
    assert!(json.as_object_mut().unwrap().remove("physical").is_some());
    let mut old: WarpedBiquad<f64> = serde_json::from_value(json).unwrap();
    assert!(old.physical.is_none());
    let mut maximum = 0.0_f64;
    for n in 0..2048 {
        let x = if n == 7 { 0.3 } else { 0.0 };
        maximum = maximum.max((old.process(x) - f.process(x)).abs());
    }
    println!("old serialized history restore maximum error={maximum}");
    assert!(maximum < 2e-10);
}

// Evaluate the matched prototype in a rotated factored form, which retains
// tiny differences without forming nearly cancelling polynomial coefficients.
fn matched_reference(freq: f64, rate: f64, q: f64, gain: f64, l: f64, probe: f64) -> Complex<f64> {
    let scale = (1.0 + l) / (1.0 - l);
    let center = 2.0 * ((PI * freq / rate).tan() * scale).atan();
    let v = 2.0 * ((PI * probe / rate).tan() * scale).atan();
    let sigma = center / (2.0 * q);
    let r = (-sigma).exp();
    let u = -(-sigma).exp_m1();
    let t = u * (1.0 + r);
    let g = 10.0_f64.powf(gain.abs() / 20.0);
    let d = (g * g * t * t + (g * g - 1.0) * (u * u * center.cos() / center.sin()).powi(2)).sqrt();
    let real = u * u + 4.0 * r * (center / 2.0).sin().powi(2)
        - 2.0 * (1.0 + r * r) * (v / 2.0).sin().powi(2);
    let h = Complex::new(real, d * v.sin()) / Complex::new(real, t * v.sin());
    if gain < 0.0 { 1.0 / h } else { h }
}

#[test]
fn matched_physical_transfer_matches_independent_factored_prototype_at_lambda_extremes() {
    let mut maximum = 0.0_f64;
    let mut minimum = f64::INFINITY;
    let mut cases = 0;
    for rate in [44100.0, 48000.0, 96000.0, 192000.0] {
        for freq in [20.0, 1000.0, 20000.0] {
            for q in [0.1, 20.0] {
                for gain in [-24.0, 24.0] {
                    for l in [-0.9999, -0.37, 0.0, 0.37, 0.9999, 0.766] {
                        let f = WarpedBiquad::<f64>::new(
                            BiquadFilterType::PeakMatched,
                            freq,
                            rate,
                            q,
                            gain,
                            l,
                        );
                        let [b0, b1, b2, a1, a2] = f.physical.as_ref().unwrap().c;
                        for margin in [
                            1.0 + a1 + a2,
                            1.0 - a1 + a2,
                            1.0 - a2,
                            b0 + b1 + b2,
                            b0 - b1 + b2,
                            b0 - b2,
                        ] {
                            assert!(
                                margin > 0.0,
                                "physical pole/zero stability {rate}/{freq}/{q}/{gain}/{l}"
                            );
                            minimum = minimum.min(margin);
                        }
                        for probe in [
                            0.0,
                            freq * 0.5,
                            freq,
                            (freq * 1.5).min(rate * 0.45),
                            rate * 0.499,
                        ] {
                            let expected = matched_reference(freq, rate, q, gain, l, probe);
                            let actual = f.complex_response(probe);
                            let error = (actual - expected).norm() / (1.0 + expected.norm());
                            maximum = maximum.max(error);
                        }
                        cases += 1;
                    }
                }
            }
        }
    }
    println!(
        "matched warped cases={cases}, max normalized complex error={maximum}, min Jury={minimum}"
    );
    assert_eq!(cases, 288);
    assert!(maximum < 2e-7);
}

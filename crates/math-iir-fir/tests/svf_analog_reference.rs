//! Independent analog-polynomial references for SVF processing and analysis.

// Rust guideline compliant 2026-02-21
use math_audio_iir_fir::{SvfFilter, SvfFilterType};
use num_complex::Complex;
use std::f64::consts::PI;

fn analog(kind: SvfFilterType, q: f64, gain: f64) -> ([f64; 3], [f64; 3]) {
    let a = 10.0_f64.powf(gain / 40.0);
    match kind {
        SvfFilterType::Lowpass => ([1.0, 0.0, 0.0], [1.0, 1.0 / q, 1.0]),
        SvfFilterType::Highpass => ([0.0, 0.0, 1.0], [1.0, 1.0 / q, 1.0]),
        SvfFilterType::Bandpass => ([0.0, 1.0, 0.0], [1.0, 1.0 / q, 1.0]),
        SvfFilterType::Notch => ([1.0, 0.0, 1.0], [1.0, 1.0 / q, 1.0]),
        SvfFilterType::Allpass => ([1.0, -1.0 / q, 1.0], [1.0, 1.0 / q, 1.0]),
        SvfFilterType::Peak => ([1.0, a / q, 1.0], [1.0, 1.0 / (a * q), 1.0]),
        SvfFilterType::Lowshelf => ([a * a, a * a.sqrt() / q, a], [1.0, a.sqrt() / q, a]),
        SvfFilterType::Highshelf => ([a, a * a.sqrt() / q, a * a], [a, a.sqrt() / q, 1.0]),
    }
}
fn transform(p: [f64; 3], k: f64) -> [f64; 3] {
    let [c, b, a] = p;
    [
        c * k * k + b * k + a,
        2.0 * (c * k * k - a),
        c * k * k - b * k + a,
    ]
}
fn evaluate(p: [f64; 3], s: Complex<f64>) -> Complex<f64> {
    p[0] + p[1] * s + p[2] * s * s
}

#[test]
fn svf_output_and_complex_response_match_analog_prototypes_at_extremes() {
    let mut maximum_impulse = 0.0_f64;
    let mut maximum_response = 0.0_f64;
    let mut maximum_f32_impulse = 0.0_f64;
    let mut cases = 0;
    for rate in [44_100.0, 48_000.0, 96_000.0, 192_000.0] {
        for kind in [
            SvfFilterType::Lowpass,
            SvfFilterType::Highpass,
            SvfFilterType::Bandpass,
            SvfFilterType::Notch,
            SvfFilterType::Allpass,
            SvfFilterType::Peak,
            SvfFilterType::Lowshelf,
            SvfFilterType::Highshelf,
        ] {
            for freq in [20.0, 1000.0, 20000.0] {
                for q in [
                    0.1,
                    if kind == SvfFilterType::Notch {
                        40.0
                    } else {
                        20.0
                    },
                ] {
                    for gain in [-24.0, 24.0] {
                        let mut filter = SvfFilter::<f64>::new(kind, freq, rate, q, gain);
                        let mut narrow = SvfFilter::<f32>::new(
                            kind,
                            freq as f32,
                            rate as f32,
                            q as f32,
                            gain as f32,
                        );
                        let (num, den) = analog(kind, q, gain);
                        let k = (PI * freq / rate).tan();
                        let b = transform(num, k);
                        let a = transform(den, k);
                        let c = [
                            b[0] / a[0],
                            b[1] / a[0],
                            b[2] / a[0],
                            a[1] / a[0],
                            a[2] / a[0],
                        ];
                        assert!(
                            1.0 + c[3] + c[4] > 0.0 && 1.0 - c[3] + c[4] > 0.0 && 1.0 - c[4] > 0.0
                        );
                        let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
                        for n in 0..4096 {
                            let x = if n == 0 { 1.0 } else { 0.0 };
                            let y = c[0] * x + c[1] * x1 + c[2] * x2 - c[3] * y1 - c[4] * y2;
                            x2 = x1;
                            x1 = x;
                            y2 = y1;
                            y1 = y;
                            let actual = filter.process(x);
                            let actual_f32 = f64::from(narrow.process(x as f32));
                            assert!(actual.is_finite() && actual_f32.is_finite());
                            maximum_impulse = maximum_impulse.max((actual - y).abs());
                            maximum_f32_impulse = maximum_f32_impulse.max((actual_f32 - y).abs());
                        }
                        for probe in [
                            0.0,
                            freq * 0.5,
                            freq,
                            (freq * 1.5).min(rate * 0.45),
                            rate * 0.499,
                        ] {
                            let s = Complex::new(0.0, (PI * probe / rate).tan() / k);
                            let expected = evaluate(num, s) / evaluate(den, s);
                            let actual = filter.response_at(probe);
                            let error = (actual - expected).norm() / (1.0 + expected.norm());
                            maximum_response = maximum_response.max(error);
                        }
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 384);
    println!(
        "SVF analog cases={cases}, f64 impulse max={maximum_impulse}, normalized complex max={maximum_response}, f32 impulse max={maximum_f32_impulse}"
    );
    assert!(maximum_impulse < 2e-8, "f64 impulse bound");
    assert!(maximum_response < 2e-10, "complex response bound");
    assert!(maximum_f32_impulse < 2e-3, "f32 impulse bound");
}

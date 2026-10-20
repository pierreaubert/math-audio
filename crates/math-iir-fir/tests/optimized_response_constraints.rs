//! Independent response constraints for native optimized biquads.

// Rust guideline compliant 2026-02-21
use math_audio_iir_fir::{Biquad, BiquadCoefficients, BiquadFilterType};
use num_complex::Complex;
use std::f64::consts::PI;

fn response(c: BiquadCoefficients, omega: f64) -> Complex<f64> {
    let z = Complex::from_polar(1.0, -omega);
    (c.b0 + c.b1 * z + c.b2 * z * z) / (1.0 + c.a1 * z + c.a2 * z * z)
}

fn jury(c: BiquadCoefficients) -> f64 {
    [
        1.0 + c.a1 + c.a2,
        1.0 - c.a1 + c.a2,
        1.0 - c.a2,
        c.b0 + c.b1 + c.b2,
        c.b0 - c.b1 + c.b2,
        c.b0 - c.b2,
    ]
    .into_iter()
    .fold(f64::INFINITY, f64::min)
}

#[test]
fn optimized_filters_reach_gain_with_minimum_phase_and_reciprocal_cuts() {
    let mut max_gain_error = 0.0_f64;
    let mut max_reciprocal_error = 0.0_f64;
    let mut minimum_jury = f64::INFINITY;
    let mut cases = 0;
    for kind in [
        BiquadFilterType::PeakMatched,
        BiquadFilterType::LowshelfOrf,
        BiquadFilterType::HighshelfOrf,
    ] {
        for rate in [44_100.0, 48_000.0, 96_000.0, 192_000.0] {
            for freq in [20.0, 1000.0, 20000.0] {
                for q in [0.1, 20.0] {
                    let boost = Biquad::<f64>::new(kind, freq, rate, q, 24.0).coefficients();
                    let cut = Biquad::<f64>::new(kind, freq, rate, q, -24.0).coefficients();
                    minimum_jury = minimum_jury.min(jury(boost)).min(jury(cut));
                    let w = 2.0 * PI * freq / rate;
                    for (gain, c) in [(24.0, boost), (-24.0, cut)] {
                        let target_db = if kind == BiquadFilterType::PeakMatched {
                            gain
                        } else {
                            gain / 2.0
                        };
                        let measured = 20.0 * response(c, w).norm().log10();
                        max_gain_error = max_gain_error.max((measured - target_db).abs());
                        let dc_db = if kind == BiquadFilterType::LowshelfOrf {
                            gain
                        } else {
                            0.0
                        };
                        let nyquist_db = if kind == BiquadFilterType::HighshelfOrf {
                            gain
                        } else {
                            0.0
                        };
                        for (probe, db) in [(0.0, dc_db), (PI, nyquist_db)] {
                            max_gain_error = max_gain_error
                                .max((20.0 * response(c, probe).norm().log10() - db).abs());
                        }
                    }
                    for probe in [0.0, w * 0.5, w, (w * 1.5).min(PI * 0.99), PI] {
                        max_reciprocal_error = max_reciprocal_error
                            .max((response(boost, probe) * response(cut, probe) - 1.0).norm());
                    }
                    cases += 1;
                }
            }
        }
    }
    println!(
        "optimized cases={cases}, gain error dB={max_gain_error}, reciprocal complex error={max_reciprocal_error}, minimum Jury margin={minimum_jury}"
    );
    assert_eq!(cases, 72);
    assert!(max_gain_error < 2e-6, "requested gain must be reached");
    assert!(
        minimum_jury > 0.0,
        "poles and zeros must be strictly inside the unit circle"
    );
    assert!(
        max_reciprocal_error < 2e-7,
        "cuts must invert boosts, including phase"
    );
}

#[test]
fn optimized_zero_gain_is_identity_and_small_cuts_approach_it() {
    for kind in [
        BiquadFilterType::PeakMatched,
        BiquadFilterType::LowshelfOrf,
        BiquadFilterType::HighshelfOrf,
    ] {
        for gain in [-1e-6, 0.0, 1e-6] {
            let c = Biquad::<f64>::new(kind, 1000.0, 48000.0, 0.1, gain).coefficients();
            for probe in [10.0, 500.0, 1000.0, 2000.0, 20000.0] {
                assert!(
                    (response(c, 2.0 * PI * probe / 48000.0) - 1.0).norm() < 2e-6,
                    "{kind:?}/{gain}/{probe} must approach identity from both gain signs"
                );
            }
        }
    }
}

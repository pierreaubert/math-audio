//! Decay-fit regressions from the October 2026 implementation audit.

// Rust guideline compliant 2026-02-21
use math_rir::metrics::DecayCurve;

fn exponential(sample_rate: f64, duration: f64, t60: f64, gain: f64, noise: f64) -> Vec<f32> {
    (0..(sample_rate * duration) as usize)
        .map(|n| {
            let uniform = ((n as u64 * 1_103_515_245 + 12_345) & 0x7fff) as f64 / 32768.0 - 0.5;
            (gain
                * ((-3.0 * std::f64::consts::LN_10 * n as f64 / (sample_rate * t60)).exp()
                    + noise * uniform)) as f32
        })
        .collect()
}

#[test]
fn audit_t30_rejects_truncated_decay_without_headroom() {
    for sample_rate in [8000.0, 48000.0, 96000.0] {
        for duration in [0.4, 0.6] {
            for gain in [1.0, 1e-12] {
                let input = exponential(sample_rate, duration, 1.0, gain, 0.0);
                let curve = DecayCurve::from_rir(&input, sample_rate, 0, None);
                assert!(
                    curve.fit_db_range(-5.0, -35.0).is_none(),
                    "unsupported T30 accepted at fs={sample_rate}, duration={duration}, gain={gain}"
                );
            }
        }
    }
}

#[test]
fn audit_t30_retains_accurate_supported_fits() {
    for sample_rate in [8000.0, 48000.0, 96000.0] {
        for (duration, noise) in [(1.0, 0.0), (2.0, 1e-5)] {
            let input = exponential(sample_rate, duration, 1.0, 1.0, noise);
            let curve = DecayCurve::from_rir(&input, sample_rate, 0, None);
            let (slope, _, r2) = curve.fit_db_range(-5.0, -35.0).expect("adequate headroom");
            assert!((-60.0 / slope - 1.0).abs() < 0.03);
            assert!(r2 > 0.99);
        }
    }
}

#[test]
fn audit_t30_rejects_high_noise_despite_long_record() {
    let input = exponential(48000.0, 2.0, 1.0, 1.0, 0.03);
    let curve = DecayCurve::from_rir(&input, 48000.0, 0, None);
    assert!(curve.fit_db_range(-5.0, -35.0).is_none());
}

#[test]
fn bounded_band_pool_preserves_results_and_band_order() {
    use math_rir::bands::{
        BandAnalysisContext, BandWidth, analyze_iso3382_bands, analyze_iso3382_bands_with_pool,
    };
    let input = exponential(48000.0, 1.0, 0.4, 1.0, 1e-6);
    let bands = [125.0, 250.0, 1000.0, 4000.0];
    let expected = analyze_iso3382_bands(&input, 48000.0, &bands, BandWidth::Octave, 4);
    for size in [1, 2, 4] {
        let mut contexts: Vec<_> = (0..size).map(|_| BandAnalysisContext::new()).collect();
        let actual = analyze_iso3382_bands_with_pool(
            &input,
            48000.0,
            &bands,
            BandWidth::Octave,
            4,
            &mut contexts,
        );
        assert_eq!(actual.len(), expected.len());
        for ((fc, metric), (expected_fc, expected_metric)) in actual.iter().zip(&expected) {
            assert_eq!(fc, expected_fc);
            assert_eq!(metric.c80_db.to_bits(), expected_metric.c80_db.to_bits());
            assert_eq!(metric.t30_s.to_bits(), expected_metric.t30_s.to_bits());
        }
    }
}

#[test]
fn borrowed_decay_workspace_clears_invalid_results_without_mutating_owned_curves() {
    use math_rir::metrics::DecayWorkspace;
    let input = exponential(48000.0, 1.0, 0.4, 1.0, 0.0);
    let mut workspace = DecayWorkspace::new();
    let owned =
        DecayCurve::from_rir_with_workspace(&input, 48000.0, 0, Some(input.len()), &mut workspace);
    assert!(!owned.samples.is_empty());
    let snapshot = owned.samples.clone();
    for (signal, rate, start, cutoff) in [
        (input.as_slice(), 0.0, 0, None),
        (input.as_slice(), f64::NAN, 0, None),
        (&[][..], 48000.0, 0, None),
        (input.as_slice(), 48000.0, input.len(), None),
        (input.as_slice(), 48000.0, 0, Some(0)),
    ] {
        let curve = workspace.compute(signal, rate, start, cutoff);
        assert!(
            curve.samples.is_empty(),
            "stale curve returned after invalid input"
        );
        assert!(curve.fit_db_range(-5.0, -35.0).is_none());
        assert_eq!(owned.samples, snapshot);
    }
}

#[test]
#[should_panic(expected = "workspace pool cannot be empty")]
fn nonempty_band_request_requires_a_workspace_pool() {
    use math_rir::bands::{BandWidth, analyze_iso3382_bands_with_pool};
    analyze_iso3382_bands_with_pool(
        &[1.0, 0.0],
        48000.0,
        &[1000.0],
        BandWidth::Octave,
        4,
        &mut [],
    );
}

#[test]
fn empty_band_request_accepts_an_empty_pool() {
    use math_rir::bands::{BandWidth, analyze_iso3382_bands_with_pool};
    assert!(
        analyze_iso3382_bands_with_pool(&[], 48000.0, &[], BandWidth::Octave, 4, &mut [])
            .is_empty()
    );
}

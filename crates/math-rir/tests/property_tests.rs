// ============================================================================
// Property-Based Tests for math-rir
// ============================================================================
//
// Invariants checked:
//   - SSIR segmentation of a synthetic decaying impulse response returns
//     finite, non-empty regions.
//   - ISO 3382 metrics (T20, T30, C50, C80, D50) are invariant to amplitude
//     scaling of the IR.
//   - ISO 3382 metrics are finite for finite synthetic IRs.

use math_rir::{SsirConfig, analyze_rir, analyze_iso3382};
use proptest::prelude::*;

const SAMPLE_RATE: f64 = 48_000.0;
const DURATION_S: f64 = 0.6;

fn synthetic_rir(rt60_s: f64, noise_amp: f64, direct_amp: f32) -> Vec<f32> {
    let n = (SAMPLE_RATE * DURATION_S) as usize;
    let mut ir = vec![0.0_f32; n];
    if n == 0 {
        return ir;
    }

    // Direct sound at the first sample.
    ir[0] = direct_amp;

    // Energy decays by 60 dB over rt60_s; amplitude envelope is the square root.
    for i in 1..n {
        let t = i as f64 / SAMPLE_RATE;
        let amplitude = 10.0_f64.powf(-3.0 * t / rt60_s);
        // Deterministic pseudo-random noise in [-0.5, 0.5].
        let u = ((i.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fff) as f64
            / 32768.0;
        let noise = noise_amp * (u - 0.5);
        ir[i] = (amplitude + noise) as f32;
    }
    ir
}

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    (a - b).abs() <= tol.max(tol * a.abs())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// SSIR segmentation returns finite, non-empty regions for a synthetic
    /// exponentially decaying IR with additive noise.
    #[test]
    fn ssir_segmentation_returns_finite_nonempty_regions(
        rt60_s in 0.15f64..0.6f64,
        noise_amp in 1e-6f64..1e-3f64,
        direct_amp in 0.5f32..1.0f32,
    ) {
        let rir = synthetic_rir(rt60_s, noise_amp, direct_amp);
        let result = analyze_rir(&rir, &SsirConfig::new(SAMPLE_RATE));

        prop_assert!(
            !result.segments.is_empty(),
            "SSIR returned no segments"
        );
        for seg in &result.segments {
            prop_assert!(
                seg.len() > 0,
                "SSIR returned an empty segment"
            );
            prop_assert!(
                seg.onset_sample < seg.end_sample,
                "invalid segment bounds: onset {} >= end {}",
                seg.onset_sample,
                seg.end_sample
            );
            prop_assert!(
                seg.toa_sample >= seg.onset_sample && seg.toa_sample < seg.end_sample,
                "TOA {} outside segment [{}, {})",
                seg.toa_sample,
                seg.onset_sample,
                seg.end_sample
            );
            prop_assert!(
                seg.peak_energy.is_finite() && seg.peak_energy >= 0.0,
                "non-finite or negative peak energy {}",
                seg.peak_energy
            );
        }
        prop_assert!(
            result.mixing_time_samples < rir.len(),
            "mixing time {} out of bounds",
            result.mixing_time_samples
        );
    }

    /// ISO 3382 metrics are invariant to amplitude scaling of the IR.
    #[test]
    fn iso3382_metrics_are_amplitude_scaling_invariant(
        rt60_s in 0.15f64..0.6f64,
        noise_amp in 1e-6f64..1e-3f64,
        scale in 0.1f64..10.0f64,
    ) {
        let rir = synthetic_rir(rt60_s, noise_amp, 1.0);
        let scaled: Vec<f32> = rir.iter().map(|&s| (s as f64 * scale) as f32).collect();

        let m1 = analyze_iso3382(&rir, SAMPLE_RATE);
        let m2 = analyze_iso3382(&scaled, SAMPLE_RATE);

        let tol = 1e-6;
        prop_assert!(
            approx_eq(m1.t20_s, m2.t20_s, tol),
            "T20 not amplitude-invariant: {} vs {}",
            m1.t20_s,
            m2.t20_s
        );
        prop_assert!(
            approx_eq(m1.t30_s, m2.t30_s, tol),
            "T30 not amplitude-invariant: {} vs {}",
            m1.t30_s,
            m2.t30_s
        );
        prop_assert!(
            approx_eq(m1.c50_db, m2.c50_db, tol),
            "C50 not amplitude-invariant: {} vs {}",
            m1.c50_db,
            m2.c50_db
        );
        prop_assert!(
            approx_eq(m1.c80_db, m2.c80_db, tol),
            "C80 not amplitude-invariant: {} vs {}",
            m1.c80_db,
            m2.c80_db
        );
        prop_assert!(
            approx_eq(m1.d50, m2.d50, tol),
            "D50 not amplitude-invariant: {} vs {}",
            m1.d50,
            m2.d50
        );
    }

    /// ISO 3382 metrics produce finite values for finite synthetic IRs.
    #[test]
    fn iso3382_metrics_are_finite(
        rt60_s in 0.15f64..0.6f64,
        noise_amp in 1e-6f64..1e-3f64,
    ) {
        let rir = synthetic_rir(rt60_s, noise_amp, 1.0);
        let m = analyze_iso3382(&rir, SAMPLE_RATE);

        prop_assert!(m.t20_s.is_finite(), "T20 is not finite: {}", m.t20_s);
        prop_assert!(m.t30_s.is_finite(), "T30 is not finite: {}", m.t30_s);
        prop_assert!(m.c50_db.is_finite(), "C50 is not finite: {}", m.c50_db);
        prop_assert!(m.c80_db.is_finite(), "C80 is not finite: {}", m.c80_db);
        prop_assert!(m.d50.is_finite(), "D50 is not finite: {}", m.d50);
    }
}

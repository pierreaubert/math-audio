use math_audio_iir_fir::{FirDesignConfig, generate_kirkeby_correction};
// Insert into math-iir-fir/src/fir_design/tests.rs (uses its existing imports).
#[test]
fn kirkeby_phase_is_invariant_to_common_spl_reference() {
    let frequencies: Vec<f64> = (1..=2400).map(|i| i as f64 * 10.0).collect();
    let phase = vec![0.0; frequencies.len()];
    let config = FirDesignConfig {
        n_taps: 2048,
        sample_rate: 48000.0,
        min_freq: 20.0,
        max_freq: 1000.0,
        correct_excess_phase: true,
        phase_smoothing_octaves: 0.0,
        pre_ringing: None,
        ..Default::default()
    };
    let design = |level: f64| {
        let magnitude = vec![level; frequencies.len()];
        generate_kirkeby_correction(&frequencies, &magnitude, Some(&phase), &magnitude, &config)
    };
    let reference = design(0.0);
    for level in [-40.0, 80.0] {
        let shifted = design(level);
        assert!(reference.iter().chain(&shifted).all(|v| v.is_finite()));
        let error = reference
            .iter()
            .zip(&shifted)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            error < 1e-10,
            "common SPL reference {level} dB changed coefficients by {error}"
        );
    }
}

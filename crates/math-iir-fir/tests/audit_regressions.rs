//! Regression cases from the October 2026 implementation audit.

// Rust guideline compliant 2026-02-21
use math_audio_iir_fir::Fir;

#[test]
fn audit_small_asymmetric_f32_taps_survive_sample_and_mixed_block_processing() {
    for scale in [1.0_f32, 1e-7, 1e-15, 1e-25] {
        let coefficients = vec![0.0, scale, -scale * 0.25, scale * 0.5];
        let mut filter = Fir::new_custom(coefficients.clone(), 48000.0);
        let mut output = vec![filter.process(1.0)];
        let mut block = [0.0; 2];
        filter.process_block(&mut block);
        output.extend(block);
        output.push(filter.process(0.0));
        for (&actual, &expected) in output.iter().zip(&coefficients) {
            assert!((actual - expected).abs() <= scale * 1e-6, "{output:?}");
        }
        filter.reset();
        let repeated: Vec<_> = [1.0, 0.0, 0.0, 0.0]
            .into_iter()
            .map(|sample| filter.process(sample))
            .collect();
        for (&actual, &expected) in repeated.iter().zip(&coefficients) {
            assert!((actual - expected).abs() <= scale * 1e-6, "{repeated:?}");
        }
    }
}

#[test]
fn audit_small_asymmetric_f64_taps_match_direct_convolution() {
    for scale in [1.0_f64, 1e-17, 1e-100] {
        let mut filter = Fir::new_custom(vec![0.0, scale], 48000.0);
        assert_eq!(filter.process(1.0), 0.0);
        assert_eq!(filter.process(0.0), scale);
    }
}

//! Allocation contracts for reusable public DSP APIs.

// Rust guideline compliant 2026-02-21
use math_audio_dsp::analysis::SpectrumAnalyzer;
use math_audio_dsp::capture_resample::{CommonClockResampler, resample_to_common_clock};
use math_audio_dsp::capture_tdoa::ClockSkew;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region};

#[global_allocator]
static GLOBAL: &stats_alloc::StatsAlloc<std::alloc::System> = &INSTRUMENTED_SYSTEM;

#[test]
fn prepared_public_analysis_and_resampling_allocate_nothing() {
    let signal: Vec<_> = (0..48000).map(|n| (n as f32 * 0.137).sin()).collect();
    let expected = SpectrumAnalyzer::new(4096)
        .welch(&signal, 48000, 0.5)
        .unwrap();
    let mut analyzer = SpectrumAnalyzer::new(4096);
    let mut frequencies = vec![0.0; 2048];
    let mut magnitudes = vec![0.0; 2048];
    let mut phases = vec![0.0; 2048];
    let region = Region::new(GLOBAL);
    for _ in 0..3 {
        analyzer
            .welch_into(
                &signal,
                48000,
                0.5,
                &mut frequencies,
                &mut magnitudes,
                &mut phases,
            )
            .unwrap();
    }
    let stats = region.change();
    assert_eq!(stats.allocations + stats.reallocations, 0, "{stats:?}");
    assert_eq!((frequencies, magnitudes, phases), expected);

    let skew = ClockSkew {
        offset_samples: 0.25,
        skew_ppm: 5.0,
        valid: true,
        low_confidence: false,
        inconsistent_spacing: false,
    };
    let plan = CommonClockResampler::new(&skew).unwrap();
    let expected = resample_to_common_clock(&signal, &skew, signal.len()).unwrap();
    let mut output = vec![0.0; signal.len()];
    let region = Region::new(GLOBAL);
    for _ in 0..3 {
        plan.process_into(&signal, skew.offset_samples, &mut output)
            .unwrap();
    }
    let stats = region.change();
    assert_eq!(stats.allocations + stats.reallocations, 0, "{stats:?}");
    assert_eq!(output, expected);
    assert!(plan.process_into(&signal, f64::NAN, &mut output).is_none());
}

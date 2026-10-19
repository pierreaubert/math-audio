//! Regression cases from the October 2026 implementation audit.

// Rust guideline compliant 2026-02-21
use math_audio_dsp::analysis::compute_clarity_broadband;
use math_audio_dsp::ebur128::{EbuR128, Mode};
use math_audio_dsp::stft::DualWindowStft;
use math_audio_dsp::true_peak::TruePeakDetector;

fn tone(sample_rate: usize, amplitude: f32) -> Vec<f32> {
    (0..sample_rate)
        .map(|n| {
            amplitude
                * (std::f64::consts::TAU * 1000.0 * n as f64 / sample_rate as f64).sin() as f32
        })
        .collect()
}

#[test]
fn audit_integrated_loudness_retains_audio_after_an_hour_of_silence() {
    let mut meter = EbuR128::new(1, 8000, Mode::I).unwrap();
    meter.add_frames_f32(&tone(8000, 0.1)).unwrap();
    let silence = vec![0.0; 8000];
    // Settle the transition before recording the accepted program blocks.
    meter.add_frames_f32(&silence).unwrap();
    let before = meter.loudness_global().unwrap();
    let accepted = meter.gating_blocks_above_absolute_gate();
    for _ in 0..3602 {
        meter.add_frames_f32(&silence).unwrap();
    }
    assert_eq!(meter.gating_blocks_above_absolute_gate(), accepted);
    assert_eq!(meter.loudness_global().unwrap(), before);
    assert!(before.is_finite());
    meter.add_frames_f32(&tone(8000, 0.05)).unwrap();
    assert!(meter.gating_blocks_above_absolute_gate().len() > accepted.len());
}

#[test]
fn audit_relative_gate_never_readmits_below_absolute_gate() {
    let mut meter = EbuR128::new(1, 48000, Mode::I).unwrap();
    for amplitude in [0.0008, 0.00030] {
        let signal = tone(48000, amplitude);
        for _ in 0..10 {
            meter.add_frames_f32(&signal).unwrap();
        }
    }
    let above_absolute = meter.gating_blocks_above_absolute_gate();
    let mean = above_absolute.iter().sum::<f64>() / above_absolute.len() as f64;
    let accepted: Vec<_> = above_absolute
        .into_iter()
        .filter(|&energy| energy > mean * 0.1)
        .collect();
    let energy = accepted.iter().sum::<f64>();
    let expected = -0.691 + 10.0 * (energy / accepted.len() as f64).log10();
    assert!((meter.loudness_global().unwrap() - expected).abs() < 1e-10);
    let (count, actual_energy) = meter.gating_block_count_and_energy().unwrap();
    assert_eq!(count, accepted.len() as u64);
    assert!((actual_energy - energy).abs() < energy * 1e-12);
}

#[test]
fn audit_true_peak_matches_known_bandlimited_sines_across_frequency_and_phase() {
    for frequency in [0.03125_f64, 0.125, 0.25, 0.375] {
        for phase in [0.0_f64, 0.25, 0.5, 0.75] {
            let mut detector = TruePeakDetector::new();
            let mut peak = 0.0_f32;
            for n in 0..4096 {
                let sample = (std::f64::consts::TAU * frequency * n as f64
                    + std::f64::consts::PI * phase)
                    .sin() as f32;
                let window = detector.process_linear(sample);
                // Exclude the finite record's turn-on transient from the sine oracle.
                if n > 256 {
                    peak = peak.max(window);
                }
            }
            let error_db = 20.0 * peak.log10();
            assert!(
                error_db.abs() < 0.6,
                "frequency={frequency}, phase={phase}: {error_db} dB"
            );
        }
    }
}

#[test]
fn audit_clarity_is_invariant_to_gain_and_leading_silence() {
    let expected = 10.0 * 4.0_f32.log10();
    for gain in [1.0_f32, 1e-8, 1e-25, 1e20] {
        for delay in [0, 4800] {
            let mut impulse = vec![0.0; 48000 + delay];
            impulse[delay] = gain;
            impulse[delay + 4800] = gain * 0.5;
            let (c50, c80) = compute_clarity_broadband(&impulse, 48000.0);
            assert!(
                (c50 - expected).abs() < 1e-5,
                "gain={gain} delay={delay}: {c50}"
            );
            assert!(
                (c80 - expected).abs() < 1e-5,
                "gain={gain} delay={delay}: {c80}"
            );
        }
    }
}

#[test]
fn clarity_arrival_threshold_survives_subnormal_amplitudes() {
    let mut input = vec![0.0; 48000];
    input[0] = 1e-44;
    input[4800] = 5e-45;
    let expected = compute_clarity_broadband(&input, 48000.0);
    let mut delayed = vec![0.0; 4800];
    delayed.extend(input);
    assert_eq!(compute_clarity_broadband(&delayed, 48000.0), expected);
}

#[test]
fn audit_stft_identity_preserves_all_startup_and_tail_phases_after_reset() {
    for (analysis, synthesis, hop) in [(128, 32, 16), (512, 128, 64)] {
        let mut stft = DualWindowStft::new(analysis, synthesis, hop);
        for position in (0..analysis).chain(1024..1024 + hop) {
            stft.reset();
            let mut input = vec![0.0; 1024 + hop + analysis];
            input[position] = 1.0;
            let mut output = vec![0.0; input.len()];
            // Deliberately split at boundaries unrelated to the STFT hop.
            for (source, destination) in input.chunks(37).zip(output.chunks_mut(37)) {
                stft.process_block(source, destination, |_| {});
            }
            for (index, &sample) in output.iter().enumerate() {
                let expected = if index == position + stft.latency_samples() {
                    1.0
                } else {
                    0.0
                };
                assert!(
                    (sample - expected).abs() < 2e-5,
                    "N={analysis}, position={position}, index={index}: {sample}"
                );
            }
        }
    }
}

#[test]
fn finite_stream_finish_recovers_short_records_and_is_resettable() {
    for length in [1, 17, 129, 513] {
        let input: Vec<_> = (0..length).map(|n| (n as f32 * 0.173).cos()).collect();
        let mut stft = DualWindowStft::new(512, 128, 64);
        let mut output = vec![0.0; length + stft.latency_samples()];
        stft.process_block(&input, &mut output[..length], |_| {});
        stft.finish_into(&mut output[length..], |_| {});
        for (&actual, &expected) in output[stft.latency_samples()..].iter().zip(&input) {
            assert!((actual - expected).abs() < 2e-5);
        }
        let mut drain = vec![1.0; stft.latency_samples()];
        stft.finish_into(&mut drain, |_| {});
        assert!(drain.iter().all(|sample| sample.abs() < 1e-6));
    }
    let input = [0.7, -0.9, 0.8, -0.75];
    let mut finished = TruePeakDetector::new();
    let mut explicit = TruePeakDetector::new();
    for sample in input {
        finished.process_linear(sample);
        explicit.process_linear(sample);
    }
    finished.finish();
    for _ in 0..11 {
        explicit.process_linear(0.0);
    }
    assert_eq!(finished.peak_linear(), explicit.peak_linear());
    let peak = finished.peak_linear();
    finished.finish();
    assert_eq!(finished.peak_linear(), peak);
}

#[test]
fn integrated_loudness_retains_every_accepted_block_beyond_one_hour() {
    let signal = tone(8000, 0.1);
    let mut meter = EbuR128::new(1, 8000, Mode::I).unwrap();
    meter.reserve_gating_blocks(36020);
    for _ in 0..3602 {
        meter.add_frames_f32(&signal).unwrap();
    }
    assert_eq!(meter.gating_blocks_above_absolute_gate().len(), 36020 - 3);
}

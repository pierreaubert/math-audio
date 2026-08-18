#![allow(clippy::needless_range_loop)]
use super::super::analyze::analyze_recording;
use std::f32::consts::PI;

/// Write a mono f32 WAV file for testing
fn write_test_wav(path: &std::path::Path, samples: &[f32], sample_rate: u32) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for &s in samples {
        writer.write_sample(s).unwrap();
    }
    writer.finalize().unwrap();
}

/// Generate a log sweep signal (same as the recording system uses)
fn generate_test_sweep(
    start_freq: f32,
    end_freq: f32,
    duration_secs: f32,
    sample_rate: u32,
    amplitude: f32,
) -> Vec<f32> {
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    let mut signal = Vec::with_capacity(num_samples);
    let ln_ratio = (end_freq / start_freq).ln();
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let phase = 2.0 * PI * start_freq * duration_secs / ln_ratio
            * ((t / duration_secs * ln_ratio).exp() - 1.0);
        signal.push(amplitude * phase.sin());
    }
    signal
}

#[test]
fn test_load_wav_mono_int_pcm_respects_bits_per_sample() {
    // Integer PCM must be normalized by 1 << (bits_per_sample - 1):
    // hound sign-extends 16/24-bit samples into i32 without left-shifting,
    // so dividing by i32::MAX would make 16-bit files ~96 dB too quiet.
    let dir = std::env::temp_dir().join(format!("sotf_test_pcm_norm_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // 16-bit PCM at half full scale
    let path16 = dir.join("half16.wav");
    let spec16 = hound::WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path16, spec16).unwrap();
    let half_scale_16 = (0.5 * i16::MAX as f32) as i16;
    for _ in 0..100 {
        writer.write_sample(half_scale_16).unwrap();
    }
    writer.finalize().unwrap();

    let samples = super::super::load::load_wav_mono(&path16).unwrap();
    let peak = samples.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
    assert!(
        (peak - 0.5).abs() < 1e-3,
        "16-bit half-scale should load as ~0.5, got {peak}"
    );

    // 24-bit PCM at quarter full scale
    let path24 = dir.join("quarter24.wav");
    let spec24 = hound::WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 24,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path24, spec24).unwrap();
    let quarter_scale_24: i32 = 1 << 21; // 0.25 * 2^23
    for _ in 0..100 {
        writer.write_sample(quarter_scale_24).unwrap();
    }
    writer.finalize().unwrap();

    let samples = super::super::load::load_wav_mono(&path24).unwrap();
    let peak = samples.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
    assert!(
        (peak - 0.25).abs() < 1e-4,
        "24-bit quarter-scale should load as ~0.25, got {peak}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_analyze_recording_legacy_path_reports_zero_thd() {
    // The legacy (non-ESS) path has no sweep-timing reference for Farina
    // harmonic separation; THD is only produced by the canonical ESS path
    // (sweep_range = Some). Document that the legacy result reports zeros.
    let sample_rate = 48000;
    let reference = generate_test_sweep(20.0, 20000.0, 0.5, sample_rate, 0.5);
    let delay = 64;
    let mut recorded = vec![0.0_f32; reference.len() + delay];
    for (i, &s) in reference.iter().enumerate() {
        recorded[i + delay] = s * 0.5;
    }

    let dir = std::env::temp_dir().join(format!("sotf_test_legacy_thd_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wav_path = dir.join("legacy.wav");
    write_test_wav(&wav_path, &recorded, sample_rate);

    let result = analyze_recording(&wav_path, &reference, sample_rate, None).unwrap();
    std::fs::remove_dir_all(&dir).ok();

    assert!(
        result.thd_percent.iter().all(|&v| v == 0.0),
        "legacy path must report zero THD"
    );
    assert!(
        result.harmonic_distortion_db.is_empty(),
        "legacy path must report no harmonic curves"
    );
}

#[test]
fn test_analyze_recording_normal_channel() {
    // Simulate a normal speaker: reference sweep played back and recorded
    // with some attenuation and small delay
    let sample_rate = 48000;
    let duration = 1.0;
    let reference = generate_test_sweep(20.0, 20000.0, duration, sample_rate, 0.5);

    // Simulate recording: attenuate by ~-6dB (factor 0.5) and delay by 100 samples
    let delay = 100;
    let attenuation = 0.5;
    let mut recorded = vec![0.0_f32; reference.len() + delay];
    for (i, &s) in reference.iter().enumerate() {
        recorded[i + delay] = s * attenuation;
    }

    let dir = std::env::temp_dir().join(format!("sotf_test_normal_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wav_path = dir.join("test_normal.wav");
    write_test_wav(&wav_path, &recorded, sample_rate);

    let result = analyze_recording(&wav_path, &reference, sample_rate, None).unwrap();
    std::fs::remove_dir_all(&dir).ok();

    // Compute average SPL in the passband (200 Hz - 10 kHz)
    let mut sum = 0.0_f32;
    let mut count = 0;
    for (&freq, &db) in result.frequencies.iter().zip(result.spl_db.iter()) {
        if (200.0..=10000.0).contains(&freq) {
            sum += db;
            count += 1;
        }
    }
    let avg_db = sum / count as f32;

    // Expected: ~-6 dB (attenuation factor 0.5); measured -6.02 dB.
    assert!(
        (avg_db + 6.0).abs() <= 2.0,
        "Normal channel avg SPL should be -6 dB ± 2 dB, got {:.1} dB",
        avg_db
    );

    // No bin should exceed +6 dB (physically implausible for passive attenuation)
    let max_db = result
        .spl_db
        .iter()
        .zip(result.frequencies.iter())
        .filter(|&(_, &f)| (200.0..=10000.0).contains(&f))
        .map(|(&db, _)| db)
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_db < 6.0,
        "Normal channel should not have bins above +6 dB, got {:.1} dB",
        max_db
    );
}

#[test]
fn test_analyze_recording_sweep_range_uses_canonical_ess_path() {
    let sample_rate = 48_000;
    let reference = generate_test_sweep(100.0, 8_000.0, 0.08, sample_rate, 0.5);
    let delay = 41;
    let mut recorded = vec![0.0_f32; delay + reference.len() + 128];
    for (index, &sample) in reference.iter().enumerate() {
        recorded[delay + index] = sample * 0.5;
    }

    let dir = std::env::temp_dir().join(format!("sotf_test_canonical_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wav_path = dir.join("canonical.wav");
    write_test_wav(&wav_path, &recorded, sample_rate);

    let result =
        analyze_recording(&wav_path, &reference, sample_rate, Some((100.0, 8_000.0))).unwrap();
    std::fs::remove_dir_all(&dir).ok();

    assert_eq!(result.frequencies.len(), 2_000);
    assert_eq!(result.spl_db.len(), result.frequencies.len());
    assert_eq!(result.excess_group_delay_ms.len(), result.frequencies.len());
    assert!(
        result
            .excess_group_delay_ms
            .iter()
            .all(|value| value.is_finite())
    );
    assert!((result.estimated_lag_samples - delay as isize).abs() <= 1);
    assert!(
        result
            .spl_db
            .iter()
            .filter(|value| value.is_finite())
            .any(|value| *value < -3.0 && *value > -12.0)
    );
}

#[test]
fn test_analyze_recording_silent_channel() {
    // Simulate a disconnected speaker: reference sweep played but recording
    // is just low-level noise (no speaker output)
    let sample_rate = 48000;
    let duration = 1.0;
    let reference = generate_test_sweep(20.0, 20000.0, duration, sample_rate, 0.5);

    // Recording is pure noise at -60 dBFS (amplitude 0.001)
    let noise_amplitude = 0.001;
    let num_samples = reference.len();
    let mut recorded = Vec::with_capacity(num_samples);
    // Use deterministic "noise" (alternating small values)
    for i in 0..num_samples {
        let pseudo_noise =
            noise_amplitude * (((i as f32 * 0.1).sin() + (i as f32 * 0.37).cos()) * 0.5);
        recorded.push(pseudo_noise);
    }

    let dir = std::env::temp_dir().join(format!("sotf_test_silent_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wav_path = dir.join("test_silent.wav");
    write_test_wav(&wav_path, &recorded, sample_rate);

    let result = analyze_recording(&wav_path, &reference, sample_rate, None).unwrap();
    std::fs::remove_dir_all(&dir).ok();

    // For a disconnected channel, the transfer function should be very low
    // (noise / sweep ≈ noise floor). It must NOT show spurious high-dB peaks.
    let max_db = result
        .spl_db
        .iter()
        .zip(result.frequencies.iter())
        .filter(|&(_, &f)| (100.0..=10000.0).contains(&f))
        .map(|(&db, _)| db)
        .fold(f32::NEG_INFINITY, f32::max);

    assert!(
        max_db < 0.0,
        "Silent/disconnected channel should not have positive dB values, got max {:.1} dB",
        max_db
    );
}

#[test]
fn test_analyze_recording_lfe_narrow_sweep_same_point_count() {
    // Simulate a 5.1 scenario: LFE uses a narrow sweep (20-500 Hz) while
    // main channels use the full range (20-20000 Hz). Both must produce
    // the same number of output frequency points to avoid ndarray shape
    // mismatches when curves are combined in the optimizer.
    let sample_rate = 48000;
    let duration = 1.0;

    // Full-range reference (main channel)
    let ref_full = generate_test_sweep(20.0, 20000.0, duration, sample_rate, 0.5);
    // Narrow reference (LFE)
    let ref_lfe = generate_test_sweep(20.0, 500.0, duration, sample_rate, 0.5);

    // Simulate recordings: attenuated copies with delay
    let delay = 50;
    let atten = 0.3;

    let mut rec_full = vec![0.0_f32; ref_full.len() + delay];
    for (i, &s) in ref_full.iter().enumerate() {
        rec_full[i + delay] = s * atten;
    }

    let mut rec_lfe = vec![0.0_f32; ref_lfe.len() + delay];
    for (i, &s) in ref_lfe.iter().enumerate() {
        rec_lfe[i + delay] = s * atten;
    }

    let dir = std::env::temp_dir().join(format!("sotf_test_lfe_points_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let wav_full = dir.join("main.wav");
    let wav_lfe = dir.join("lfe.wav");
    write_test_wav(&wav_full, &rec_full, sample_rate);
    write_test_wav(&wav_lfe, &rec_lfe, sample_rate);

    let result_full = analyze_recording(&wav_full, &ref_full, sample_rate, None).unwrap();
    let result_lfe = analyze_recording(&wav_lfe, &ref_lfe, sample_rate, None).unwrap();
    std::fs::remove_dir_all(&dir).ok();

    // Both must produce the same number of frequency points
    assert_eq!(
        result_full.frequencies.len(),
        result_lfe.frequencies.len(),
        "Main ({}) and LFE ({}) must have the same number of frequency points",
        result_full.frequencies.len(),
        result_lfe.frequencies.len()
    );
    assert_eq!(
        result_full.spl_db.len(),
        result_lfe.spl_db.len(),
        "SPL arrays must match in length"
    );

    // LFE should have valid data below ~500 Hz and noise floor above
    let lfe_valid_count = result_lfe
        .spl_db
        .iter()
        .zip(result_lfe.frequencies.iter())
        .filter(|&(&db, &f)| f <= 500.0 && db > -100.0)
        .count();
    assert!(
        lfe_valid_count > 100,
        "LFE should have valid data below 500 Hz, got {} points",
        lfe_valid_count
    );

    let lfe_above_500_max = result_lfe
        .spl_db
        .iter()
        .zip(result_lfe.frequencies.iter())
        .filter(|&(_, &f)| f > 1000.0)
        .map(|(&db, _)| db)
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        lfe_above_500_max <= -100.0,
        "LFE above 1 kHz should be at noise floor, got {:.1} dB",
        lfe_above_500_max
    );
}

#[test]
fn test_analyze_recording_empty_reference_errors() {
    let dir = std::env::temp_dir().join(format!("sotf_test_empty_ref_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wav_path = dir.join("empty_ref.wav");
    write_test_wav(&wav_path, &[0.0f32; 100], 48000);
    let result = analyze_recording(&wav_path, &[], 48000, None);
    std::fs::remove_dir_all(&dir).ok();
    assert!(result.is_err(), "empty reference signal must error");
    assert!(
        result.unwrap_err().contains("empty"),
        "error should mention empty reference"
    );
}

#[test]
fn test_analyze_recording_empty_recorded_errors() {
    let dir = std::env::temp_dir().join(format!("sotf_test_empty_rec_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wav_path = dir.join("empty_rec.wav");
    write_test_wav(&wav_path, &[], 48000);
    let reference = generate_test_sweep(20.0, 20000.0, 0.1, 48000, 0.5);
    let result = analyze_recording(&wav_path, &reference, 48000, None);
    std::fs::remove_dir_all(&dir).ok();
    assert!(result.is_err(), "empty recorded signal must error");
    assert!(
        result.unwrap_err().contains("empty"),
        "error should mention empty recorded"
    );
}

#[test]
fn test_trim_impulse_empty() {
    let empty: &[f32] = &[];
    let trimmed = super::super::misc::trim_impulse_to_noise_floor(empty, 48000.0);
    assert_eq!(trimmed.len(), 0);
}

#[test]
fn test_trim_impulse_short_returns_unchanged() {
    let impulse: Vec<f32> = (0..100).map(|i| if i == 10 { 1.0 } else { 0.0 }).collect();
    let trimmed = super::super::misc::trim_impulse_to_noise_floor(&impulse, 48000.0);
    assert_eq!(trimmed, impulse.as_slice());
}

#[test]
fn test_trim_impulse_keeps_peak_and_headroom() {
    let sr = 48000.0;
    let mut impulse = vec![0.0f32; sr as usize * 2];
    impulse[1000] = 1.0;
    for i in 2000..impulse.len() {
        impulse[i] = 1e-5;
    }
    let trimmed = super::super::misc::trim_impulse_to_noise_floor(&impulse, sr);
    assert!(trimmed.len() < impulse.len());
    assert_eq!(trimmed[1000], 1.0);
}

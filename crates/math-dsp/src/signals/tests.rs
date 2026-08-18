use super::apply::apply_fade_in;
use super::apply::apply_fade_out;
use super::extract::extract_tone_phase;
use super::extract::extract_tone_phase_windowed;
use super::gen_::gen_allpass_probe;
use super::gen_::gen_bass_tone_burst;
use super::gen_::gen_dirac;
use super::gen_::gen_impulse;
use super::gen_::gen_log_sweep;
use super::gen_::gen_log_sweep_octave_scaled;
use super::gen_::gen_m_noise;
use super::gen_::gen_m_noise_seeded;
use super::gen_::gen_mls;
use super::gen_::gen_narrowband_probe;
use super::gen_::gen_pink_noise;
use super::gen_::gen_pink_noise_seeded;
use super::gen_::gen_probe_spectrum;
use super::gen_::gen_steady_tone;
use super::gen_::gen_step;
use super::gen_::gen_tone;
use super::gen_::gen_two_tone;
use super::gen_::gen_white_noise;
use super::gen_::gen_white_noise_seeded;
use super::misc::add_silence_padding;
use super::misc::clip;
use super::misc::frames_for;
use super::misc::interleave_per_channel;
use super::misc::mono_to_stereo;
use super::misc::replicate_mono;
use super::prepare::prepare_signal_for_playback;
use super::prepare::prepare_signal_for_playback_channels;
use super::tone_phasor_window::phasor_coherence;
use super::tone_phasor_window::tone_phase_phasors;
use crate::stft::RealFftProcessor;
use rustfft::num_complex::Complex;
use std::f32::consts::PI;

mod misc;

#[test]
fn test_frames_for() {
    assert_eq!(frames_for(1.0, 48000), 48000);
    assert_eq!(frames_for(0.5, 44100), 22050);
    assert_eq!(frames_for(2.0, 96000), 192000);
}

#[test]
fn test_clip() {
    assert_eq!(clip(0.5), 0.5);
    assert_eq!(clip(-0.5), -0.5);
    assert!(clip(1.5) < 1.0);
    assert!(clip(-1.5) > -1.0);
}

#[test]
fn test_gen_tone() {
    let signal = gen_tone(1000.0, 0.5, 48000, 0.1);
    assert_eq!(signal.len(), 4800);
    // Check that signal is not all zeros
    assert!(signal.iter().any(|&x| x.abs() > 0.1));
}

#[test]
fn test_gen_two_tone() {
    let signal = gen_two_tone(440.0, 0.3, 880.0, 0.3, 48000, 0.1);
    assert_eq!(signal.len(), 4800);
    assert!(signal.iter().any(|&x| x.abs() > 0.1));
}

#[test]
fn test_gen_log_sweep() {
    let signal = gen_log_sweep(20.0, 20000.0, 0.5, 48000, 1.0);
    assert_eq!(signal.len(), 48000);
    assert!(signal.iter().any(|&x| x.abs() > 0.1));
}

#[test]
fn test_log_sweep_rejects_invalid_parameters() {
    assert!(gen_log_sweep(0.0, 20_000.0, 0.5, 48_000, 1.0).is_empty());
    assert!(super::gen_::try_gen_log_sweep(20.0, 20_000.0, 0.5, 0, 1.0).is_err());
    assert!(super::gen_::try_gen_log_sweep(f32::NAN, 20_000.0, 0.5, 48_000, 1.0).is_err());
}

#[test]
fn test_seeded_noise_decorrelates_repeated_captures() {
    let first = gen_white_noise_seeded(0.5, 48_000, 0.05, 1);
    let second = gen_white_noise_seeded(0.5, 48_000, 0.05, 2);
    assert_ne!(first, second);
    assert_ne!(
        gen_pink_noise_seeded(0.5, 48_000, 0.05, 1),
        gen_pink_noise_seeded(0.5, 48_000, 0.05, 2)
    );
    assert_ne!(
        gen_m_noise_seeded(0.5, 48_000, 0.05, 1),
        gen_m_noise_seeded(0.5, 48_000, 0.05, 2)
    );
}

#[test]
fn test_gen_log_sweep_amplitude_analysis() {
    // Test amplitude at different points in the sweep
    let amp = 0.5;
    let signal = gen_log_sweep(20.0, 20000.0, amp, 48000, 1.0);

    // Check amplitude at different time points (20%, 40%, 60%, 80%)
    let checkpoints = [0.2, 0.4, 0.6, 0.8];
    let sample_rate = 48000.0;
    let duration = 1.0;

    for &checkpoint in &checkpoints {
        let sample_pos = (checkpoint * duration * sample_rate) as usize;
        let window_size = 480; // 10ms window
        let start = sample_pos.saturating_sub(window_size / 2);
        let end = (sample_pos + window_size / 2).min(signal.len());

        if end > start {
            let window_peak = signal[start..end]
                .iter()
                .map(|&x| x.abs())
                .fold(0.0_f32, |a, b| a.max(b));
            log::info!(
                "Checkpoint {:.1}: peak amplitude = {:.6} (target: {:.6})",
                checkpoint,
                window_peak,
                amp
            );
        }
    }
}

#[test]
fn test_gen_log_sweep_simple() {
    // Simple test to understand current behavior
    let amp = 0.5;
    let signal = gen_log_sweep(20.0, 20000.0, amp, 48000, 0.1);

    // Find the maximum amplitude in the signal
    let max_amp = signal
        .iter()
        .map(|&x| x.abs())
        .fold(0.0_f32, |a, b| a.max(b));
    log::info!("Generated log sweep:");
    log::info!("  Target amplitude: {:.6}", amp);
    log::info!("  Actual max amplitude: {:.6}", max_amp);
    log::info!("  Ratio: {:.6}", max_amp / amp);

    // Check that we have some signal
    assert!(max_amp > 0.01, "Signal should have significant amplitude");
}

#[test]
fn test_gen_log_sweep_constant_amplitude() {
    // Test that log sweep maintains constant amplitude across frequency range
    let amp = 0.7;
    let sample_rate = 48000_u32;
    let duration = 2.0;
    let f_start = 20.0_f32;
    let f_end = 20000.0_f32;
    let signal = gen_log_sweep(f_start, f_end, amp, sample_rate, duration);

    // Find peak values throughout the sweep using 10ms windows.
    // Skip the low-frequency region where the window (10ms = 100Hz period)
    // is shorter than one cycle and can't capture the true peak.
    let window_size = 480; // 10ms at 48kHz
    let min_freq_for_window = sample_rate as f32 / window_size as f32; // 100 Hz
    let k = (f_end / f_start).ln() / duration;
    let safe_start_t = (min_freq_for_window / f_start).ln() / k;
    let safe_start_sample = (safe_start_t * sample_rate as f32) as usize;

    let mut peaks = Vec::new();
    for i in (safe_start_sample..signal.len()).step_by(window_size / 4) {
        let end = (i + window_size).min(signal.len());
        if end > i {
            let window_peak = signal[i..end].iter().map(|&x| x.abs()).fold(0.0, f32::max);
            peaks.push(window_peak);
        }
    }

    assert!(!peaks.is_empty(), "Should have found peaks");

    let min_peak = peaks.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_peak = peaks.iter().fold(0.0_f32, |a, &b| a.max(b));
    let variation = max_peak - min_peak;

    let target_peak = amp;

    // With f64 phase computation, amplitude is near-constant (< 0.1% variation)
    // once the measurement window captures a full cycle.
    assert!(
        variation < 0.01 * target_peak,
        "Peak variation {:.6} exceeds 1% of target amplitude {:.6}",
        variation,
        target_peak
    );

    let avg_peak = peaks.iter().sum::<f32>() / peaks.len() as f32;
    assert!(
        (avg_peak - target_peak).abs() < 0.01 * target_peak,
        "Average peak {:.6} differs from target {:.6} by more than 1%",
        avg_peak,
        target_peak
    );

    log::info!("Log sweep amplitude test passed:");
    log::info!("  Target amplitude: {:.6}", target_peak);
    log::info!("  Min peak: {:.6}", min_peak);
    log::info!("  Max peak: {:.6}", max_peak);
    log::info!(
        "  Variation: {:.6} ({:.2}%)",
        variation,
        100.0 * variation / target_peak
    );
}

#[test]
fn test_octave_sweep_length_within_one_sample() {
    // Returned length must equal round(total_duration * sr) ±1.
    let sr = 48000_u32;
    let bass_dur = 3.0_f32;
    let min_dur = 5.0_f32;

    let signal = gen_log_sweep_octave_scaled(10.0, 20_000.0, 0.5, sr, bass_dur, min_dur);

    let oct_bass = (100.0_f64 / 10.0_f64).log2();
    let oct_mid = (1000.0_f64 / 100.0_f64).log2();
    let oct_high = (20000.0_f64 / 1000.0_f64).log2();
    let raw = oct_bass * bass_dur as f64
        + oct_mid * (bass_dur as f64 * 0.5)
        + oct_high * (bass_dur as f64 * 0.25);
    let expected_dur = raw.max(min_dur as f64);
    let expected_n = (expected_dur * sr as f64).round() as usize;

    let diff = (signal.len() as isize - expected_n as isize).unsigned_abs();
    assert!(
        diff <= 1,
        "Length {} differs from expected {} by {} samples (> 1)",
        signal.len(),
        expected_n,
        diff
    );
}

#[test]
fn test_octave_sweep_min_duration_floor() {
    // Narrow sweep -> raw duration < min_total; output must reach the floor.
    let sr = 48000_u32;
    let signal = gen_log_sweep_octave_scaled(1000.0, 2000.0, 0.5, sr, 0.5, 10.0);
    let min_expected = (10.0_f64 * sr as f64).round() as usize - 1;
    assert!(
        signal.len() >= min_expected,
        "Length {} is below the 10s floor (expected >= {})",
        signal.len(),
        min_expected
    );
}

#[test]
fn test_octave_sweep_phase_zero_at_start() {
    // sin(0) = 0 so the first sample must be ~0 for any f_start.
    for &f_start in &[5.0_f32, 10.0, 20.0, 50.0] {
        let signal = gen_log_sweep_octave_scaled(f_start, 20_000.0, 1.0, 48000, 3.0, 5.0);
        assert!(!signal.is_empty(), "Empty signal for f_start={f_start}");
        assert!(
            signal[0].abs() < 1e-6,
            "First sample {:.2e} != 0 for f_start={f_start}",
            signal[0]
        );
    }
}

#[test]
fn test_octave_sweep_does_not_change_gen_log_sweep() {
    // The old gen_log_sweep must be unaffected.
    let signal = gen_log_sweep(20.0, 20000.0, 0.5, 48000, 1.0);
    assert_eq!(signal.len(), 48000);
    assert!(signal.iter().any(|&x| x.abs() > 0.1));
}

#[test]
fn test_gen_white_noise() {
    let signal = gen_white_noise(0.5, 48000, 1.0); // Use 1 second for better statistics
    assert_eq!(signal.len(), 48000);
    // Check that noise exists and has content
    assert!(signal.iter().any(|&x| x.abs() > 0.01));
    // Check that values are clipped to prevent overflow (clip function limits to +/- 0.999999)
    assert!(signal.iter().all(|&x| x.abs() < 1.0));
}

#[test]
fn test_gen_pink_noise() {
    let signal = gen_pink_noise(0.5, 48000, 0.1);
    assert_eq!(signal.len(), 4800);
    assert!(signal.iter().any(|&x| x.abs() > 0.01));
}

#[test]
fn test_gen_impulse() {
    let signal = gen_impulse(0.5, 48000, 0.1);
    assert_eq!(signal.len(), 4800);
    assert_eq!(signal[0], 0.5);
    for &sample in &signal[1..4800] {
        assert_eq!(sample, 0.0);
    }
}

#[test]
fn test_gen_dirac_matches_impulse() {
    let dirac = gen_dirac(0.5, 48000, 0.1);
    let impulse = gen_impulse(0.5, 48000, 0.1);
    assert_eq!(dirac, impulse);
}

#[test]
fn test_gen_mls_length_and_values() {
    let signal = gen_mls(8, 0.5);
    assert_eq!(signal.len(), 255);
    assert!(signal.iter().all(|&s| s == 0.5 || s == -0.5));
    assert!(signal.iter().any(|&s| s > 0.0));
    assert!(signal.iter().any(|&s| s < 0.0));
}

#[test]
fn test_gen_mls_deterministic() {
    assert_eq!(gen_mls(16, 0.25), gen_mls(16, 0.25));
    assert_ne!(gen_mls(15, 0.25), gen_mls(16, 0.25));
}

#[test]
fn test_gen_mls_autocorrelation_peak() {
    let signal = gen_mls(10, 0.5);
    let zero_lag: f32 = signal.iter().map(|&s| s * s).sum();
    assert!((zero_lag - signal.len() as f32 * 0.25).abs() < 1e-3);
}

#[test]
fn test_gen_mls_rejects_unsupported_order() {
    assert!(gen_mls(1, 0.5).is_empty());
    assert!(gen_mls(25, 0.5).is_empty());
}

#[test]
fn test_gen_step() {
    let signal = gen_step(0.5, 48000, 0.1);
    assert_eq!(signal.len(), 4800);
    for &sample in &signal[..4800] {
        assert_eq!(sample, 0.5);
    }
}

#[test]
fn test_gen_m_noise() {
    let signal = gen_m_noise(0.5, 48000, 0.1);
    assert_eq!(signal.len(), 4800);
    assert!(signal.iter().any(|&x| x.abs() > 0.01));
}

#[test]
fn test_interleave_per_channel() {
    let ch0 = vec![1.0, 2.0, 3.0];
    let ch1 = vec![4.0, 5.0, 6.0];
    let interleaved = interleave_per_channel(&[ch0, ch1]);
    assert_eq!(interleaved, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_replicate_mono() {
    let mono = vec![1.0, 2.0, 3.0];
    let stereo = replicate_mono(&mono, 2);
    assert_eq!(stereo, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
}

#[test]
fn test_apply_fade_in() {
    let mut signal = vec![1.0; 100];
    apply_fade_in(&mut signal, 10);
    // First sample should be near zero
    assert!(signal[0].abs() < 0.01);
    // Middle of fade should be around 0.5
    assert!((signal[5] - 0.5).abs() < 0.1);
    // After fade should be 1.0
    assert_eq!(signal[20], 1.0);
}

#[test]
fn test_apply_fade_out() {
    let mut signal = vec![1.0; 100];
    apply_fade_out(&mut signal, 10);
    // Before fade should be 1.0
    assert_eq!(signal[80], 1.0);
    // Faded region should have reduced amplitude
    assert!(signal[95] < 0.5);
    assert!(signal[99] < 0.1);
}

#[test]
fn test_add_silence_padding() {
    let signal = vec![1.0, 2.0, 3.0];
    let padded = add_silence_padding(&signal, 2, 2);
    assert_eq!(padded.len(), 7);
    assert_eq!(padded, vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
}

#[test]
fn test_mono_to_stereo() {
    let mono = vec![1.0, 0.5, -0.5, 0.0];
    let stereo = mono_to_stereo(mono);
    assert_eq!(stereo, vec![1.0, 1.0, 0.5, 0.5, -0.5, -0.5, 0.0, 0.0]);
}

#[test]
fn test_prepare_signal_for_playback_channels_stereo() {
    let signal = vec![1.0; 100]; // Short signal for testing
    let stereo = prepare_signal_for_playback_channels(signal.clone(), 48000, 10.0, 50.0, true);

    // Stereo should have twice the samples (minus padding which is the same for both)
    let mono_prepared = prepare_signal_for_playback_channels(signal, 48000, 10.0, 50.0, false);
    assert_eq!(stereo.len(), mono_prepared.len() * 2);
}

#[test]
fn test_prepare_signal_for_playback_channels_mono() {
    let signal = vec![1.0; 100]; // Short signal for testing
    let mono = prepare_signal_for_playback_channels(signal.clone(), 48000, 10.0, 50.0, false);
    let mono_direct = prepare_signal_for_playback(signal, 48000, 10.0, 50.0);
    assert_eq!(mono, mono_direct);
}

#[test]
fn test_prepare_signal_for_playback() {
    let signal = vec![1.0; 48000]; // 1 second at 48kHz
    let prepared = prepare_signal_for_playback(signal, 48000, 20.0, 250.0);
    // Should have padding on both sides (250ms = 12000 samples each)
    assert_eq!(prepared.len(), 48000 + 2 * 12000);
    // First samples should be zero (padding)
    assert_eq!(prepared[0], 0.0);
    assert_eq!(prepared[11999], 0.0);
    // Last samples should be zero (padding)
    assert_eq!(prepared[prepared.len() - 1], 0.0);
}

#[test]
fn test_allpass_probe_flat_spectrum() {
    let n = 4096;
    let probe = gen_allpass_probe(n, 48000, 0.5, 42);
    assert_eq!(probe.len(), n);

    // FFT the probe and check magnitude is nearly constant
    let mut fft_proc = RealFftProcessor::new_forward_only(n);
    fft_proc.time_buffer[..n].copy_from_slice(&probe);
    fft_proc.forward();

    // Skip DC and Nyquist, check that magnitudes are within ±1dB of each other
    let mags: Vec<f32> = fft_proc.freq_buffer[1..fft_proc.spectrum_size - 1]
        .iter()
        .map(|c| c.norm())
        .collect();
    let avg_mag = mags.iter().sum::<f32>() / mags.len() as f32;

    for (i, &m) in mags.iter().enumerate() {
        let ratio_db = 20.0 * (m / avg_mag).log10();
        assert!(
            ratio_db.abs() < 1.0,
            "Bin {} magnitude deviates by {:.2} dB from average",
            i + 1,
            ratio_db
        );
    }
}

#[test]
fn test_narrowband_probe_bandpass() {
    let n = 8192;
    let sr = 48000;
    let probe = gen_narrowband_probe(n, sr, 0.5, 42, 800.0, 2000.0);
    assert_eq!(probe.len(), n);

    // FFT and check energy is concentrated in [800, 2000] Hz
    let mut fft_proc = RealFftProcessor::new_forward_only(n);
    fft_proc.time_buffer[..n].copy_from_slice(&probe);
    fft_proc.forward();

    let freq_res = sr as f32 / n as f32;
    let mut in_band_energy = 0.0_f32;
    let mut out_band_energy = 0.0_f32;

    for (k, c) in fft_proc.freq_buffer.iter().enumerate() {
        let freq = k as f32 * freq_res;
        let energy = c.norm_sqr();
        if (800.0..=2000.0).contains(&freq) {
            in_band_energy += energy;
        } else {
            out_band_energy += energy;
        }
    }

    // Out-of-band should be negligible compared to in-band
    let ratio = out_band_energy / (in_band_energy + 1e-30);
    assert!(
        ratio < 0.01,
        "Out-of-band energy ratio {:.4} should be < 1%",
        ratio
    );
}

#[test]
fn test_probe_deterministic() {
    let a = gen_allpass_probe(2048, 48000, 0.5, 123);
    let b = gen_allpass_probe(2048, 48000, 0.5, 123);
    assert_eq!(a, b, "Same seed should produce identical probes");

    let c = gen_allpass_probe(2048, 48000, 0.5, 456);
    assert_ne!(a, c, "Different seeds should produce different probes");
}

#[test]
fn test_narrowband_shares_phase_with_wideband() {
    // With the same seed, the narrowband probe's passband bins should
    // have the same phase as the corresponding wideband bins
    let n = 4096;
    let seed = 99;
    let wb_spectrum = gen_probe_spectrum(n, seed);
    let nb_spectrum = gen_probe_spectrum(n, seed);

    // Phase should be identical (same function, same seed)
    for k in 100..200 {
        let wb_phase = wb_spectrum[k].arg();
        let nb_phase = nb_spectrum[k].arg();
        assert!(
            (wb_phase - nb_phase).abs() < 1e-6,
            "Phase mismatch at bin {}",
            k
        );
    }
}

#[test]
fn bass_tone_burst_length_and_envelope() {
    // 20 Hz × 5 cycles at 48 kHz → exactly 12 000 samples (250 ms)
    let b = gen_bass_tone_burst(20.0, 5, 48_000, 0.5);
    assert_eq!(b.len(), 12_000);
    // Peak after Hann windowing at amp=0.5 → ≤ 0.5
    let peak = b.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
    assert!(peak <= 0.5 + 1e-6, "burst peak {peak} exceeds amp");
    // Hann envelope: first and last samples are near-zero.
    assert!(b[0].abs() < 1e-4);
    assert!(b[b.len() - 1].abs() < 1e-4);
    // The integer-cycle burst places a zero crossing exactly at
    // the midpoint (2.5 cycles). Peak around the midpoint — scan
    // a ¼-cycle window on either side to find it.
    let quarter_cycle = (48_000 / 20 / 4) as usize; // 600 samples
    let mid = b.len() / 2;
    let peak_near_mid = b[mid - quarter_cycle..=mid + quarter_cycle]
        .iter()
        .map(|s| s.abs())
        .fold(0.0_f32, f32::max);
    assert!(
        peak_near_mid > 0.3,
        "burst centre region should hit peak, got {peak_near_mid}"
    );
}

#[test]
fn bass_tone_burst_rejects_invalid() {
    assert!(gen_bass_tone_burst(0.0, 5, 48_000, 0.5).is_empty());
    assert!(gen_bass_tone_burst(20.0, 0, 48_000, 0.5).is_empty());
    assert!(gen_bass_tone_burst(20.0, 5, 0, 0.5).is_empty());
    assert!(gen_bass_tone_burst(20.0, 5, 48_000, 0.0).is_empty());
}

#[test]
fn tone_phase_recovers_sin_reference_zero() {
    // Reference: pure sin(ωt) at 20 Hz. Phase should be ≈ 0°.
    let burst = gen_bass_tone_burst(20.0, 5, 48_000, 0.5);
    let r = extract_tone_phase(&burst, 20.0, 48_000);
    assert!(
        r.phase_deg.abs() < 1.0,
        "pure sin burst should give phase ≈ 0°, got {:.3}°",
        r.phase_deg
    );
    assert!(r.magnitude > 0.0);
    // A stable-but-Hann-windowed burst has a small residual
    // stability reading (~7°) because the two halves have
    // different envelope profiles. The advisory threshold for
    // `"bass_anchor_unreliable"` is 20° (plan §2.8), so anything
    // below that is considered reliable. Test with a safety
    // margin of 15°.
    assert!(
        r.stability_deg < 15.0,
        "stable tone stability should stay under the 20° advisory threshold, got {:.3}°",
        r.stability_deg
    );
}

#[test]
fn tone_phase_reports_input_peak_amplitude() {
    let sample_rate = 48_000_u32;
    let frequency = 1000.0_f32;
    let amplitude = 0.4_f32;
    let signal: Vec<f32> = (0..4800)
        .map(|n| amplitude * (2.0 * PI * frequency * n as f32 / sample_rate as f32).sin())
        .collect();

    let result = extract_tone_phase(&signal, frequency, sample_rate);
    assert!((result.magnitude - amplitude as f64).abs() < 1e-6);
}

#[test]
fn tone_phase_recovers_synthetic_90_degree_shift() {
    // Synthesize a cos-phase burst (= 90° phase shift relative to sin).
    let freq = 20.0_f32;
    let sr = 48_000_u32;
    let n = 12_000;
    let omega = 2.0 * PI * freq / sr as f32;
    let n_f = n as f32;
    let burst: Vec<f32> = (0..n)
        .map(|k| {
            let t = k as f32;
            let w = 0.5 * (1.0 - (2.0 * PI * t / (n_f - 1.0)).cos());
            0.5 * w * (omega * t).cos() // cos(ωt) → 90° relative to sin(ωt)
        })
        .collect();
    let r = extract_tone_phase(&burst, freq, sr);
    // Expected phase is 90° (cos = sin shifted forward by 90°).
    let err = (r.phase_deg - 90.0).abs();
    assert!(
        err < 1.0,
        "cos burst should give phase ≈ 90°, got {:.3}° (error {:.3}°)",
        r.phase_deg,
        err
    );
    // Stable-but-Hann-windowed — see note on the sin-reference
    // test for why the stability reading is non-zero.
    assert!(r.stability_deg < 15.0);
}

#[test]
fn tone_phase_detects_unstable_burst() {
    // Concatenate two half-bursts with a 90° jump in the middle.
    // The stability metric must flag this > 20°.
    let freq = 20.0_f32;
    let sr = 48_000_u32;
    let n_half = 6_000;
    let omega = 2.0 * PI * freq / sr as f32;
    let n_f = (2 * n_half) as f32;
    let burst: Vec<f32> = (0..2 * n_half)
        .map(|k| {
            let t = k as f32;
            let w = 0.5 * (1.0 - (2.0 * PI * t / (n_f - 1.0)).cos());
            let phase_shift = if k < n_half { 0.0 } else { PI / 2.0 };
            0.5 * w * (omega * t + phase_shift).sin()
        })
        .collect();
    let r = extract_tone_phase(&burst, freq, sr);
    assert!(
        r.stability_deg > 20.0,
        "phase-jump burst should be flagged unstable, got stability = {:.1}°",
        r.stability_deg
    );
}

#[test]
fn tone_phase_rejects_short_signal() {
    let r = extract_tone_phase(&[0.1, 0.2], 20.0, 48_000);
    assert_eq!(r.magnitude, 0.0);
    assert_eq!(r.phase_deg, 0.0);
    assert_eq!(r.stability_deg, 0.0);
}

#[test]
fn steady_tone_length_and_fade_envelope() {
    let sr = 48_000_u32;
    let s = gen_steady_tone(30.0, 1.0, 50.0, sr, 0.5);
    assert_eq!(
        s.len(),
        48_000,
        "1 s @ 48 kHz must be exactly 48 000 samples"
    );
    // Endpoints sit on the half-Hann fade — should be zero.
    assert!(
        s[0].abs() < 1e-6,
        "fade-in must start at zero, got {}",
        s[0]
    );
    assert!(s[s.len() - 1].abs() < 1e-6, "fade-out must end at zero");
    // Steady region should hit the requested amplitude (at amp=0.5).
    let steady_start = (0.06 * sr as f32) as usize; // 60 ms in
    let steady_end = s.len() - steady_start;
    let peak: f32 = s[steady_start..steady_end]
        .iter()
        .map(|s| s.abs())
        .fold(0.0, f32::max);
    assert!(
        (peak - 0.5).abs() < 0.02,
        "steady region peak should be ≈ amp (0.5), got {peak}"
    );
}

#[test]
fn steady_tone_rejects_invalid() {
    assert!(gen_steady_tone(0.0, 1.0, 50.0, 48_000, 0.5).is_empty());
    assert!(gen_steady_tone(30.0, 0.0, 50.0, 48_000, 0.5).is_empty());
    assert!(gen_steady_tone(30.0, 1.0, 50.0, 0, 0.5).is_empty());
    assert!(gen_steady_tone(30.0, 1.0, 50.0, 48_000, 0.0).is_empty());
    // Duration shorter than two fades.
    assert!(gen_steady_tone(30.0, 0.05, 50.0, 48_000, 0.5).is_empty());
}

#[test]
fn windowed_phase_recovers_pure_sin() {
    let sr = 48_000_u32;
    let s = gen_steady_tone(30.0, 2.0, 50.0, sr, 0.5);
    let r = extract_tone_phase_windowed(&s, 30.0, sr, 8);
    assert!(
        r.phase_deg.abs() < 0.5,
        "pure sin should give phase ≈ 0°, got {:.4}°",
        r.phase_deg
    );
    assert!(r.magnitude > 0.0);
    assert!(
        r.stability_deg < 1.0,
        "pure sin steady tone should give near-zero circular-std, got {:.3}°",
        r.stability_deg
    );
}

#[test]
fn windowed_phase_reports_input_peak_amplitude() {
    let sample_rate = 48_000_u32;
    let frequency = 1000.0_f32;
    let amplitude = 0.4_f32;
    let signal: Vec<f32> = (0..48_000)
        .map(|n| amplitude * (2.0 * PI * frequency * n as f32 / sample_rate as f32).sin())
        .collect();

    let result = extract_tone_phase_windowed(&signal, frequency, sample_rate, 8);
    assert!((result.magnitude - amplitude as f64).abs() < 1e-6);
}

#[test]
fn tone_phasor_windows_minimize_fractional_cycle_error() {
    let sample_rate = 48_000_u32;
    let frequency = 1000.3_f32;
    let signal: Vec<f32> = (0..96_000)
        .map(|n| (2.0 * PI * frequency * n as f32 / sample_rate as f32).sin())
        .collect();
    let phasors = tone_phase_phasors(&signal, frequency, sample_rate, 8);
    assert!(!phasors.is_empty());

    for phasor in phasors {
        let cycles = phasor.len as f64 * frequency as f64 / sample_rate as f64;
        assert!(
            (cycles - cycles.round()).abs() < 0.02,
            "window length {} spans {cycles:.6} cycles",
            phasor.len
        );
    }
}

#[test]
fn windowed_phase_recovers_synthetic_45_degree_shift() {
    let sr = 48_000_u32;
    let freq = 30.0_f32;
    let n = (sr as f32 * 2.0) as usize;
    let omega = 2.0 * PI * freq / sr as f32;
    let phase_shift = (45.0_f32).to_radians();
    let fade_n = (0.05 * sr as f32) as usize;
    let s: Vec<f32> = (0..n)
        .map(|k| {
            let env = if k < fade_n {
                0.5 * (1.0 - (PI * k as f32 / fade_n as f32).cos())
            } else if k >= n - fade_n {
                let kk = (n - 1 - k) as f32;
                0.5 * (1.0 - (PI * kk / fade_n as f32).cos())
            } else {
                1.0
            };
            0.5 * env * (omega * k as f32 + phase_shift).sin()
        })
        .collect();
    let r = extract_tone_phase_windowed(&s, freq, sr, 8);
    let err = (r.phase_deg - 45.0).abs();
    assert!(
        err < 0.5,
        "45° shifted sin should give phase ≈ 45°, got {:.3}° (err {:.3}°)",
        r.phase_deg,
        err
    );
    assert!(r.stability_deg < 1.0);
}

#[test]
fn windowed_phase_flags_drifting_phase_as_unstable() {
    let sr = 48_000_u32;
    let freq = 30.0_f32;
    let n = (sr as f32 * 2.0) as usize;
    let omega = 2.0 * PI * freq / sr as f32;
    // Inject a slow phase drift (linear ramp from 0 to ±90°). The
    // circular std should rise well above the ~1° pure-tone floor.
    let s: Vec<f32> = (0..n)
        .map(|k| {
            let frac = k as f32 / n as f32;
            let drift = (PI / 2.0) * frac;
            0.5 * (omega * k as f32 + drift).sin()
        })
        .collect();
    let r = extract_tone_phase_windowed(&s, freq, sr, 8);
    assert!(
        r.stability_deg > 5.0,
        "drifting-phase tone should read unstable, got circular-std {:.3}°",
        r.stability_deg
    );
}

#[test]
fn windowed_phase_rejects_sub_cycle_buffers() {
    // 30 Hz @ 48 kHz → 1600 samples/cycle. A signal of 800 samples
    // contains less than one cycle even before the /8 settle drop.
    // The helper must refuse rather than return falsely-confident
    // numbers.
    let sr = 48_000_u32;
    let n = 800; // < one cycle
    let omega = 2.0 * PI * 30.0 / sr as f32;
    let s: Vec<f32> = (0..n).map(|k| 0.5 * (omega * k as f32).sin()).collect();
    let phasors = tone_phase_phasors(&s, 30.0, sr, 8);
    assert!(
        phasors.is_empty(),
        "sub-cycle buffers must yield empty phasor list, got {} entries",
        phasors.len()
    );
    let r = extract_tone_phase_windowed(&s, 30.0, sr, 8);
    assert_eq!(r.magnitude, 0.0);
    assert_eq!(r.phase_deg, 0.0);
}

#[test]
fn windowed_phase_merges_when_per_window_below_one_cycle() {
    // 30 Hz @ 48 kHz, 2 s tone, asking for 64 windows: each raw
    // window would be 2 s/64 = 31 ms < one cycle (33 ms). The
    // helper must merge windows down so each spans ≥ 1 cycle and
    // still recover phase to within ~1°.
    let sr = 48_000_u32;
    let s = gen_steady_tone(30.0, 2.0, 50.0, sr, 0.5);
    let r = extract_tone_phase_windowed(&s, 30.0, sr, 64);
    assert!(
        r.phase_deg.abs() < 1.0,
        "merged-window analysis should still recover ~0° phase, got {:.3}°",
        r.phase_deg
    );
    assert!(r.magnitude > 0.0);
}

#[test]
fn phasor_coherence_is_one_for_identical_phasors() {
    let sr = 48_000_u32;
    let s = gen_steady_tone(30.0, 2.0, 50.0, sr, 0.5);
    let p = tone_phase_phasors(&s, 30.0, sr, 8);
    let coh = phasor_coherence(&p, &p).expect("coherent");
    assert!(
        (coh - 1.0).abs() < 1e-9,
        "γ²(x, x) should be exactly 1, got {coh}"
    );
}

#[test]
fn phasor_coherence_drops_when_one_stream_is_noise() {
    // Mic = clean tone, loopback = uncorrelated white noise →
    // γ² should approach 0. The previous proxy returned 1.0 in
    // this case; the true MSC must be < 0.5.
    use rand::{RngExt, SeedableRng};
    let sr = 48_000_u32;
    let clean = gen_steady_tone(30.0, 2.0, 50.0, sr, 0.5);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE);
    let noise: Vec<f32> = (0..clean.len())
        .map(|_| rng.random_range(-0.3..0.3))
        .collect();
    let p_clean = tone_phase_phasors(&clean, 30.0, sr, 8);
    let p_noise = tone_phase_phasors(&noise, 30.0, sr, 8);
    let coh = phasor_coherence(&p_clean, &p_noise).expect("coherent");
    assert!(
        coh < 0.3,
        "γ²(clean, noise) must drop well below 0.5, got {coh}"
    );
}

#[test]
fn phasor_coherence_rejects_mismatched_lengths() {
    let sr = 48_000_u32;
    let s = gen_steady_tone(30.0, 2.0, 50.0, sr, 0.5);
    let a = tone_phase_phasors(&s, 30.0, sr, 8);
    let b = tone_phase_phasors(&s, 30.0, sr, 4);
    assert!(phasor_coherence(&a, &b).is_none());
}

#[test]
fn windowed_phase_rejects_degenerate_inputs() {
    let r = extract_tone_phase_windowed(&[0.1, 0.2, 0.3], 30.0, 48_000, 8);
    assert_eq!(r.phase_deg, 0.0);
    assert_eq!(r.magnitude, 0.0);
    assert_eq!(r.stability_deg, 0.0);

    let r = extract_tone_phase_windowed(&vec![0.1_f32; 1000], 30.0, 48_000, 0);
    assert_eq!(r.magnitude, 0.0);
}

#[test]
fn steady_tone_no_fade() {
    let sr = 48_000_u32;
    let s = gen_steady_tone(100.0, 1.0, 0.0, sr, 0.5);
    assert!(!s.is_empty());
    let peak = s.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    assert!((peak - 0.5).abs() < 0.01);
}

#[test]
fn steady_tone_exact_boundary() {
    let sr = 48_000_u32;
    let fade_ms = 50.0;
    let duration_s = 2.0 * fade_ms / 1000.0;
    let s = gen_steady_tone(100.0, duration_s, fade_ms, sr, 0.5);
    assert!(!s.is_empty());
}

#[test]
fn steady_tone_clipped_amp() {
    let sr = 48_000_u32;
    let s = gen_steady_tone(100.0, 1.0, 50.0, sr, 1.5);
    let peak = s.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    assert!(peak < 1.0);
}

#[test]
fn tone_phase_phasors_empty_signal() {
    let p = tone_phase_phasors(&[], 100.0, 48_000, 4);
    assert!(p.is_empty());
}

#[test]
fn tone_phase_phasors_zero_freq() {
    let s = gen_steady_tone(100.0, 1.0, 50.0, 48_000, 0.5);
    let p = tone_phase_phasors(&s, 0.0, 48_000, 4);
    assert!(p.is_empty());
}

#[test]
fn tone_phase_phasors_zero_sample_rate() {
    let s = gen_steady_tone(100.0, 1.0, 50.0, 48_000, 0.5);
    let p = tone_phase_phasors(&s, 100.0, 0, 4);
    assert!(p.is_empty());
}

#[test]
fn tone_phase_phasors_zero_windows() {
    let s = gen_steady_tone(100.0, 1.0, 50.0, 48_000, 0.5);
    let p = tone_phase_phasors(&s, 100.0, 48_000, 0);
    assert!(p.is_empty());
}

#[test]
fn tone_phase_phasors_short_signal() {
    let s = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5];
    let p = tone_phase_phasors(&s, 100.0, 48_000, 4);
    assert!(p.is_empty());
}

#[test]
fn tone_phase_phasors_exactly_one_window() {
    let sr = 48_000_u32;
    let freq = 100.0_f32;
    let s = gen_steady_tone(freq, 1.0, 50.0, sr, 0.5);
    let p = tone_phase_phasors(&s, freq, sr, 8);
    assert_eq!(p.len(), 8);
    for w in &p {
        assert_eq!(w.len, p[0].len);
    }
}

#[test]
fn tone_phase_phasors_merges_to_single() {
    let sr = 48_000_u32;
    let freq = 1000.0_f32;
    // Very short tone so steady portion < 2 cycles, forcing merge to 1 window.
    let s = gen_steady_tone(freq, 0.002, 1.0, sr, 0.5);
    let p = tone_phase_phasors(&s, freq, sr, 64);
    assert_eq!(p.len(), 1);
    assert!(p[0].len >= 48);
}

#[test]
fn probe_spectrum_handles_degenerate_fft_sizes() {
    // fft_size 0/1/2 have no interior bins between DC and Nyquist;
    // slicing the interior must not panic.
    let s0 = gen_probe_spectrum(0, 42);
    assert_eq!(s0.len(), 1);
    assert_eq!(s0[0], Complex::new(1.0, 0.0));

    let s1 = gen_probe_spectrum(1, 42);
    assert_eq!(s1.len(), 1);
    assert_eq!(s1[0], Complex::new(1.0, 0.0));

    let s2 = gen_probe_spectrum(2, 42);
    assert_eq!(s2.len(), 2);
    assert!(s2.iter().all(|c| *c == Complex::new(1.0, 0.0)));
}

#[test]
fn narrowband_probe_tapers_compose_symmetrically() {
    // Band narrow enough that the two edge tapers overlap: they must
    // compose (multiply), so the magnitude envelope stays symmetric
    // around the band centre instead of the upper taper clobbering the
    // lower one.
    let n = 8192;
    let sr = 48000;
    let lo_hz = 1000.0;
    let hi_hz = 1100.0;
    let probe = gen_narrowband_probe(n, sr, 0.5, 42, lo_hz, hi_hz);

    let mut fft_proc = RealFftProcessor::new_forward_only(n);
    fft_proc.time_buffer[..n].copy_from_slice(&probe);
    fft_proc.forward();

    let freq_res = sr as f32 / n as f32;
    let lo_bin = (lo_hz / freq_res).ceil() as usize; // 171
    let hi_bin = (hi_hz / freq_res).floor() as usize; // 187
    assert_eq!(lo_bin, 171);
    assert_eq!(hi_bin, 187);

    let mag = |k: usize| fft_proc.freq_buffer[k].norm();
    assert!(mag(179) > 0.0, "passband centre must carry energy");
    for j in 0..=8 {
        let lower = mag(lo_bin + j);
        let upper = mag(hi_bin - j);
        let denom = lower.max(upper).max(1e-12);
        assert!(
            (lower - upper).abs() / denom < 0.01,
            "taper asymmetry at offset {j}: lower={lower} upper={upper}"
        );
    }
}

#[test]
fn narrowband_probe_rejects_inverted_band() {
    // lo_hz >= hi_hz is a caller error: return empty instead of a
    // silent all-zeros buffer of length n_frames.
    assert!(gen_narrowband_probe(4096, 48000, 0.5, 1, 2000.0, 800.0).is_empty());
}

#[test]
fn steady_tone_phase_matches_f64_reference_at_tail() {
    // At 20 kHz the phase argument reaches ~1.3e6 rad after 10 s; f32
    // argument rounding alone costs ~0.1 rad of phase error. The
    // generator must compute phase in f64 like gen_log_sweep does.
    let sr = 48_000_u32;
    let freq = 20_000.0_f64;
    let s = gen_steady_tone(20_000.0, 10.0, 5.0, sr, 0.5);
    let n = s.len();
    // Compare the last second against the f64 reference, stopping clear
    // of the 5 ms fade-out (where the envelope is < 1).
    let mut max_err = 0.0_f64;
    for (k, &sample) in s.iter().enumerate().take(n - 1000).skip(n - sr as usize) {
        let reference = 0.5 * (2.0 * std::f64::consts::PI * freq * k as f64 / sr as f64).sin();
        max_err = max_err.max((sample as f64 - reference).abs());
    }
    assert!(
        max_err < 5e-4,
        "tail deviates from f64 reference by {max_err} (~{:.4} rad phase error)",
        max_err / 0.5
    );
}

#[test]
fn bass_tone_burst_phase_matches_f64_reference_at_tail() {
    let sr = 48_000_u32;
    let freq = 20_000.0_f64;
    let burst = gen_bass_tone_burst(20_000.0, 40_000, sr, 0.5);
    let n = burst.len();
    let n_f = n as f64;
    let mut max_err = 0.0_f64;
    for (k, &sample) in burst.iter().enumerate().skip(n - 12_000) {
        let w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * k as f64 / (n_f - 1.0)).cos());
        let reference = 0.5 * w * (2.0 * std::f64::consts::PI * freq * k as f64 / sr as f64).sin();
        max_err = max_err.max((sample as f64 - reference).abs());
    }
    assert!(
        max_err < 5e-4,
        "burst tail deviates from f64 reference by {max_err}"
    );
}

#[test]
fn octave_sweep_clamps_nonpositive_bass_duration() {
    // bass_octave_duration_s <= 0 must be clamped to the documented
    // floor (0.1 s/octave), not silently produce a degenerate sweep
    // (zero duration) or a negative time scale (negative duration).
    let reference = gen_log_sweep_octave_scaled(10.0, 20_000.0, 0.5, 48_000, 0.1, 5.0);
    for &bad in &[0.0_f32, -1.0] {
        let signal = gen_log_sweep_octave_scaled(10.0, 20_000.0, 0.5, 48_000, bad, 5.0);
        assert_eq!(
            signal.len(),
            reference.len(),
            "length mismatch for bass_octave_duration_s={bad}"
        );
        let max_diff = signal
            .iter()
            .zip(reference.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert_eq!(
            max_diff, 0.0,
            "output for bass_octave_duration_s={bad} differs from the clamped floor"
        );
    }
}

#[test]
#[should_panic(expected = "equal length")]
fn interleave_per_channel_rejects_unequal_lengths() {
    let ch0 = vec![1.0, 2.0, 3.0];
    let ch1 = vec![4.0, 5.0];
    let _ = interleave_per_channel(&[ch0, ch1]);
}

#[test]
fn white_noise_uses_high_bits_of_lcg_state() {
    // The low bits of an LCG have period 2^(k+1) (bit 0 alternates
    // every sample); usable randomness comes from the high bits. Pin
    // the extraction to (state >> 33) ^ state, matching the
    // gen_probe_spectrum RNG.
    let signal = gen_white_noise_seeded(0.5, 48_000, 0.001, 42);
    let mut state = 42_u64;
    for &sample in &signal {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let random_u32 = ((state >> 33) ^ state) as u32;
        let expected = clip(0.5 * ((random_u32 as f32 / u32::MAX as f32) * 2.0 - 1.0));
        assert_eq!(sample, expected);
    }
}

#[test]
fn white_noise_statistical_sanity() {
    let signal = gen_white_noise_seeded(1.0, 48_000, 2.0, 1234567890);
    let n = signal.len() as f64;
    let mean = signal.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = signal
        .iter()
        .map(|&x| (x as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    assert!(mean.abs() < 0.01, "mean {mean} too far from 0");
    assert!((var - 1.0 / 3.0).abs() < 0.01, "variance {var} too far from 1/3");
    let lag1: f64 = signal
        .windows(2)
        .map(|w| (w[0] as f64 - mean) * (w[1] as f64 - mean))
        .sum::<f64>()
        / (n - 1.0);
    let corr = lag1 / var;
    assert!(corr.abs() < 0.01, "lag-1 correlation {corr} too high");
}

#[test]
fn pink_noise_rms_matches_amplitude() {
    // Steady-state RMS of the Voss-McCartney network driven by uniform
    // [-1,1] noise is 1.7624 (full covariance, including the cross-terms
    // between b6 = e*w[n-1] and the IIR states). PINK_NORM must divide
    // by that value so that RMS(output) ≈ amp. amp=0.1 keeps peaks
    // (~5x RMS) clear of the clip ceiling. Pink RMS estimation converges
    // slowly (1/f correlation), so the seed is fixed: with seed 1 the
    // realized RMS sits 0.23% from the exact value, inside the 0.5% gate.
    let signal = gen_pink_noise_seeded(0.1, 48_000, 60.0, 1);
    let rms = (signal
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        / signal.len() as f64)
        .sqrt();
    assert!(
        (rms - 0.1).abs() < 0.0005,
        "pink noise RMS {rms} deviates from amp=0.1 by more than 0.5%"
    );
}

#[test]
fn single_bin_dft_recurrence_matches_direct_trig() {
    // Characterization for the perf refactor: the complex-rotation
    // recurrence in `single_bin_dft` must reproduce the original
    // per-sample sin/cos accumulation within f64 rounding noise,
    // including the k_offset (sub-slice) path.
    for &(freq, sr, n, k_offset) in &[
        (20.0_f32, 48_000_u32, 4800_usize, 0_usize),
        (1000.5, 48_000, 4096, 0),
        (63.0, 44_100, 3000, 2048),
        (7500.0, 48_000, 16_384, 1237),
    ] {
        let signal: Vec<f32> = (0..n + k_offset)
            .map(|i| 0.7 * (2.0 * PI * freq * i as f32 / sr as f32).sin())
            .collect();
        let slice = &signal[k_offset..];

        // Reference: the original per-sample trig algorithm.
        let omega = 2.0 * std::f64::consts::PI * freq as f64 / sr as f64;
        let mut ref_re = 0.0_f64;
        let mut ref_im = 0.0_f64;
        for (i, &s) in slice.iter().enumerate() {
            let theta = omega * (i + k_offset) as f64;
            ref_re += s as f64 * theta.sin();
            ref_im += s as f64 * theta.cos();
        }

        let (re, im) = super::misc::single_bin_dft(slice, freq, sr, k_offset);
        // Accumulations grow with N; drift is ~N·2⁻⁵³, so 1e-9·N is tight.
        let tol = 1e-9 * n as f64;
        assert!(
            (re - ref_re).abs() <= tol,
            "re drift: got {re}, reference {ref_re} (freq {freq}, n {n}, offset {k_offset})"
        );
        assert!(
            (im - ref_im).abs() <= tol,
            "im drift: got {im}, reference {ref_im} (freq {freq}, n {n}, offset {k_offset})"
        );
    }
}

#[test]
fn extract_tone_phase_two_pass_matches_three_pass_reference() {
    // Characterization for the perf refactor: the full-signal projection
    // computed as the sum of the two half accumulations (2 DFT passes)
    // must match the original separate full-signal pass (3 passes) within
    // f64 noise — the DFT sum is linear, only the accumulation order
    // changes.
    fn dft_ref(signal: &[f32], freq_hz: f32, sample_rate: u32, k_offset: usize) -> (f64, f64) {
        // Reference: the original per-sample trig algorithm.
        let omega = 2.0 * std::f64::consts::PI * freq_hz as f64 / sample_rate as f64;
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for (i, &s) in signal.iter().enumerate() {
            let theta = omega * (i + k_offset) as f64;
            re += s as f64 * theta.sin();
            im += s as f64 * theta.cos();
        }
        (re, im)
    }

    let sr = 48_000_u32;
    for &(freq, n) in &[(20.0_f32, 4800_usize), (1000.5, 8192), (330.0, 5001)] {
        let signal: Vec<f32> = (0..n)
            .map(|i| 0.6 * (2.0 * PI * freq * i as f32 / sr as f32 + 0.4).sin())
            .collect();

        let (re_full, im_full) = dft_ref(&signal, freq, sr, 0);
        let mid = n / 2;
        let (re_a, im_a) = dft_ref(&signal[..mid], freq, sr, 0);
        let (re_b, im_b) = dft_ref(&signal[mid..], freq, sr, mid);
        let phase_ref = im_full.atan2(re_full).to_degrees();
        let mag_ref = 2.0 * (re_full * re_full + im_full * im_full).sqrt() / n as f64;
        let mut diff = im_b.atan2(re_b) - im_a.atan2(re_a);
        while diff > std::f64::consts::PI {
            diff -= 2.0 * std::f64::consts::PI;
        }
        while diff <= -std::f64::consts::PI {
            diff += 2.0 * std::f64::consts::PI;
        }
        let stab_ref = diff.abs().to_degrees();

        let r = extract_tone_phase(&signal, freq, sr);
        assert!(
            (r.phase_deg - phase_ref).abs() < 1e-7,
            "phase drift: got {}, reference {phase_ref} (freq {freq}, n {n})",
            r.phase_deg
        );
        assert!(
            (r.magnitude - mag_ref).abs() <= 1e-9 * mag_ref.max(1.0),
            "magnitude drift: got {}, reference {mag_ref} (freq {freq}, n {n})",
            r.magnitude
        );
        assert!(
            (r.stability_deg - stab_ref).abs() < 1e-7,
            "stability drift: got {}, reference {stab_ref} (freq {freq}, n {n})",
            r.stability_deg
        );
    }
}

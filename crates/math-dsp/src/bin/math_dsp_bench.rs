use math_audio_dsp::analysis::{analyze_wav_buffer, SpectrumAnalyzer, WavAnalysisConfig};
use math_audio_dsp::audio_features::{analyze_audio_features, spectral, tempo};
use std::time::Instant;

const WELCH_ITERS: usize = 100;
const FEATURE_ITERS: usize = 50;

fn checksum_f32_slice(s: &[f32]) -> f64 {
    s.iter().map(|v| *v as f64).sum()
}

fn main() {
    // Welch bench inputs
    let signal_48k: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.01).sin()).collect();
    let welch_cfg = WavAnalysisConfig {
        single_fft: false,
        fft_size: Some(4096),
        overlap: 0.5,
        num_points: 500,
        ..Default::default()
    };

    // Audio features bench inputs
    let samples_22k: Vec<f32> = (0..88_200).map(|i| (i as f32 * 0.01).sin()).collect();

    // analyze_wav_buffer_welch
    let start = Instant::now();
    let mut analyze_checksum = 0.0_f64;
    for _ in 0..WELCH_ITERS {
        let out = analyze_wav_buffer(&signal_48k, 48_000, &welch_cfg).unwrap();
        analyze_checksum += checksum_f32_slice(&out.frequencies);
        analyze_checksum += checksum_f32_slice(&out.magnitude_db);
        analyze_checksum += checksum_f32_slice(&out.phase_deg);
    }
    let analyze_ms = start.elapsed().as_secs_f64() * 1000.0 / WELCH_ITERS as f64;

    // spectrum_analyzer_welch
    let mut analyzer = SpectrumAnalyzer::new(4096);
    let start = Instant::now();
    let mut analyzer_checksum = 0.0_f64;
    for _ in 0..WELCH_ITERS {
        let (freqs, mags, phases) = analyzer.welch(&signal_48k, 48_000, 0.5).unwrap();
        analyzer_checksum += checksum_f32_slice(&freqs);
        analyzer_checksum += checksum_f32_slice(&mags);
        analyzer_checksum += checksum_f32_slice(&phases);
    }
    let analyzer_ms = start.elapsed().as_secs_f64() * 1000.0 / WELCH_ITERS as f64;

    // analyze_audio_features
    let start = Instant::now();
    let mut features_checksum = 0.0_f64;
    for _ in 0..FEATURE_ITERS {
        let features = analyze_audio_features(&samples_22k, 22_050).unwrap();
        features_checksum += checksum_f32_slice(&features);
    }
    let features_ms = start.elapsed().as_secs_f64() * 1000.0 / FEATURE_ITERS as f64;

    // compute_tempo
    let start = Instant::now();
    let mut tempo_checksum = 0.0_f64;
    for _ in 0..FEATURE_ITERS {
        tempo_checksum += tempo::compute_tempo(&samples_22k, 22_050) as f64;
    }
    let tempo_ms = start.elapsed().as_secs_f64() * 1000.0 / FEATURE_ITERS as f64;

    // compute_spectral_features
    let start = Instant::now();
    let mut spectral_checksum = 0.0_f64;
    for _ in 0..FEATURE_ITERS {
        let spectral_features = spectral::compute_spectral_features(&samples_22k, 22_050);
        spectral_checksum += checksum_f32_slice(&spectral_features);
    }
    let spectral_ms = start.elapsed().as_secs_f64() * 1000.0 / FEATURE_ITERS as f64;

    let score = analyze_ms + analyzer_ms + features_ms + tempo_ms + spectral_ms;

    println!(
        "{{\"score\":{:.6},\"tasks\":{{\"analyze_wav_buffer_welch\":{:.6},\"spectrum_analyzer_welch\":{:.6},\"analyze_audio_features\":{:.6},\"compute_tempo\":{:.6},\"compute_spectral_features\":{:.6}}},\"checksums\":{{\"analyze_wav_buffer_welch\":{:.6},\"spectrum_analyzer_welch\":{:.6},\"analyze_audio_features\":{:.6},\"compute_tempo\":{:.6},\"compute_spectral_features\":{:.6}}}}}",
        score,
        analyze_ms,
        analyzer_ms,
        features_ms,
        tempo_ms,
        spectral_ms,
        analyze_checksum,
        analyzer_checksum,
        features_checksum,
        tempo_checksum,
        spectral_checksum
    );
}

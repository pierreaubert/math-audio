use criterion::{Criterion, criterion_group, criterion_main};
use math_audio_dsp::analysis::{SpectrumAnalyzer, WavAnalysisConfig, analyze_wav_buffer};
use std::hint::black_box;

fn bench_analyze_wav_buffer_welch(c: &mut Criterion) {
    let signal: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.01).sin()).collect();
    let cfg = WavAnalysisConfig {
        single_fft: false,
        fft_size: Some(4096),
        overlap: 0.5,
        num_points: 500,
        ..Default::default()
    };
    c.bench_function("analyze_wav_buffer_welch", |b| {
        b.iter(|| {
            analyze_wav_buffer(black_box(&signal), black_box(48_000), black_box(&cfg)).unwrap()
        })
    });
}

fn bench_spectrum_analyzer_welch(c: &mut Criterion) {
    let signal: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut analyzer = SpectrumAnalyzer::new(4096);
    c.bench_function("spectrum_analyzer_welch", |b| {
        b.iter(|| {
            analyzer
                .welch(black_box(&signal), black_box(48_000), black_box(0.5))
                .unwrap()
        })
    });
}

criterion_group!(
    benches,
    bench_analyze_wav_buffer_welch,
    bench_spectrum_analyzer_welch
);
criterion_main!(benches);

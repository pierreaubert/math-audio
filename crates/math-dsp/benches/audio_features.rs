use criterion::{Criterion, criterion_group, criterion_main};
use math_audio_dsp::audio_features::{analyze_audio_features, spectral, tempo};
use std::hint::black_box;

fn bench_analyze_audio_features(c: &mut Criterion) {
    // 22.05 kHz mono, ~4 seconds, meets MIN_SAMPLES
    let samples: Vec<f32> = (0..88_200).map(|i| (i as f32 * 0.01).sin()).collect();
    c.bench_function("analyze_audio_features", |b| {
        b.iter(|| analyze_audio_features(black_box(&samples), black_box(22_050)).unwrap())
    });
}

fn bench_tempo(c: &mut Criterion) {
    let samples: Vec<f32> = (0..88_200).map(|i| (i as f32 * 0.01).sin()).collect();
    c.bench_function("compute_tempo", |b| {
        b.iter(|| tempo::compute_tempo(black_box(&samples), black_box(22_050)))
    });
}

fn bench_spectral(c: &mut Criterion) {
    let samples: Vec<f32> = (0..88_200).map(|i| (i as f32 * 0.01).sin()).collect();
    c.bench_function("compute_spectral_features", |b| {
        b.iter(|| spectral::compute_spectral_features(black_box(&samples), black_box(22_050)))
    });
}

criterion_group!(
    benches,
    bench_analyze_audio_features,
    bench_tempo,
    bench_spectral
);
criterion_main!(benches);

use criterion::{criterion_group, criterion_main, Criterion};
use math_audio_iir_fir::{FirDesignConfig, FirPhase, WindowType, generate_fir_from_response, generate_kirkeby_correction};
use std::hint::black_box;

fn bench_generate_fir_from_response(c: &mut Criterion) {
    let freqs: Vec<f64> = (0..512).map(|i| 20.0 + i as f64 * 39.0).collect();
    let mags: Vec<f64> = freqs.iter().map(|f| (-f / 1000.0).exp()).collect();
    let config = FirDesignConfig {
        n_taps: 255,
        sample_rate: 48000.0,
        phase: FirPhase::Linear,
        min_freq: 20.0,
        max_freq: 20000.0,
        window: WindowType::Blackman,
        correct_excess_phase: false,
        phase_smoothing_octaves: 0.167,
        pre_ringing: None,
    };
    c.bench_function("generate_fir_from_response_512", |b| {
        b.iter(|| generate_fir_from_response(black_box(&freqs), black_box(&mags), black_box(&config)))
    });
}

fn bench_generate_kirkeby_correction(c: &mut Criterion) {
    let freqs: Vec<f64> = (0..512).map(|i| 20.0 + i as f64 * 39.0).collect();
    let meas: Vec<f64> = freqs.iter().map(|_| -5.0).collect();
    let target: Vec<f64> = freqs.iter().map(|_| 0.0).collect();
    let config = FirDesignConfig {
        n_taps: 255,
        sample_rate: 48000.0,
        phase: FirPhase::Linear,
        min_freq: 20.0,
        max_freq: 20000.0,
        window: WindowType::Blackman,
        correct_excess_phase: false,
        phase_smoothing_octaves: 0.167,
        pre_ringing: None,
    };
    c.bench_function("generate_kirkeby_correction_512", |b| {
        b.iter(|| {
            generate_kirkeby_correction(
                black_box(&freqs),
                black_box(&meas),
                None,
                black_box(&target),
                black_box(&config),
            )
        })
    });
}

criterion_group!(benches, bench_generate_fir_from_response, bench_generate_kirkeby_correction);
criterion_main!(benches);

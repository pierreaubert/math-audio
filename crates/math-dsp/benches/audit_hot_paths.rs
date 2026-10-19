//! Representative reusable analysis and small-block spectral workloads.

// Rust guideline compliant 2026-02-21
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use math_audio_dsp::analysis::SpectrumAnalyzer;
use math_audio_dsp::capture_resample::{CommonClockResampler, resample_to_common_clock};
use math_audio_dsp::capture_tdoa::ClockSkew;
use math_audio_dsp::stft::DualWindowStft;
use math_audio_dsp::tonal_transient::TonalTransientSeparator;
use std::hint::black_box;

fn benchmarks(c: &mut Criterion) {
    let signal: Vec<_> = (0..48000).map(|n| (n as f32 * 0.137).sin()).collect();
    let mut analyzer = SpectrumAnalyzer::new(4096);
    let (mut f, mut m, mut p) = (vec![0.0; 2048], vec![0.0; 2048], vec![0.0; 2048]);
    c.bench_function("welch_into_48k_4096", |b| {
        b.iter(|| {
            analyzer
                .welch_into(black_box(&signal), 48000, 0.5, &mut f, &mut m, &mut p)
                .unwrap()
        })
    });
    let skew = ClockSkew {
        offset_samples: 0.25,
        skew_ppm: 5.0,
        valid: true,
        low_confidence: false,
        inconsistent_spacing: false,
    };
    let resampler = CommonClockResampler::new(&skew).unwrap();
    let mut group = c.benchmark_group("capture_resampling");
    for size in [1, 256, 48000] {
        let mut output = vec![0.0; size];
        group.bench_with_input(BenchmarkId::new("prepared", size), &size, |b, _| {
            b.iter(|| {
                resampler
                    .process_into(black_box(&signal), 0.25, black_box(&mut output))
                    .unwrap()
            })
        });
        group.bench_with_input(BenchmarkId::new("with_setup", size), &size, |b, &size| {
            b.iter(|| black_box(resample_to_common_clock(black_box(&signal), &skew, size).unwrap()))
        });
    }
    group.finish();
    let mut group = c.benchmark_group("stft_identity");
    for size in [32, 64, 128, 256] {
        let mut stft = DualWindowStft::new(1024, 256, 128);
        let mut output = vec![0.0; size];
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                stft.process_block(black_box(&signal[..size]), black_box(&mut output), |_| {});
                black_box(&output);
            })
        });
    }
    group.finish();
    let magnitudes: Vec<_> = (0..1025).map(|n| (n as f32 * 1.31).sin() + 1.1).collect();
    let (mut tonal, mut transient) = (vec![0.0; 1025], vec![0.0; 1025]);
    let mut group = c.benchmark_group("tonal_separation");
    for kernel in [5, 15, 31] {
        let mut separator = TonalTransientSeparator::new(1025, kernel, kernel);
        group.bench_with_input(BenchmarkId::from_parameter(kernel), &kernel, |b, _| {
            b.iter(|| {
                separator.process(black_box(&magnitudes), &mut tonal, &mut transient);
                black_box(&tonal);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);

use criterion::{Criterion, criterion_group, criterion_main};
use math_rir::bands::{BandAnalysisContext, BandWidth, analyze_iso3382_bands_with_pool};
use math_rir::{analyze_iso3382, analyze_iso3382_octaves, analyze_iso3382_third_octaves};
use std::hint::black_box;

fn generate_rir(samples: usize) -> Vec<f32> {
    let mut rir = vec![0.0_f32; samples];
    // synthetic exponentially decaying impulse response
    for (i, v) in rir.iter_mut().enumerate() {
        *v = (-0.005 * i as f32).exp() * (i as f32 * 0.1).sin();
    }
    rir[0] = 1.0;
    rir
}

fn bench_iso3382_broadband(c: &mut Criterion) {
    let rir = generate_rir(48_000);
    c.bench_function("analyze_iso3382_broadband", |b| {
        b.iter(|| analyze_iso3382(black_box(&rir), black_box(48_000.0)))
    });
}

fn bench_iso3382_octaves(c: &mut Criterion) {
    let rir = generate_rir(48_000);
    c.bench_function("analyze_iso3382_octaves", |b| {
        b.iter(|| analyze_iso3382_octaves(black_box(&rir), black_box(48_000.0)))
    });
}

fn bench_iso3382_third_octaves(c: &mut Criterion) {
    let rir = generate_rir(48_000);
    c.bench_function("analyze_iso3382_third_octaves", |b| {
        b.iter(|| analyze_iso3382_third_octaves(black_box(&rir), black_box(48_000.0)))
    });
}

fn bench_workspace_matrix(c: &mut Criterion) {
    let bands = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0];
    let mut group = c.benchmark_group("iso3382_workspace_pool");
    for length in [4800, 48000, 480000] {
        let signal: Vec<_> = (0..length)
            .map(|n| {
                ((n as f64 * 0.13).sin()
                    * (-3.0 * std::f64::consts::LN_10 * n as f64 / 48000.0).exp())
                    as f32
            })
            .collect();
        for pool_size in [1, 2, 6] {
            let mut contexts: Vec<_> = (0..pool_size).map(|_| BandAnalysisContext::new()).collect();
            group.bench_with_input(
                criterion::BenchmarkId::new(format!("pool{pool_size}"), length),
                &length,
                |b, _| {
                    b.iter(|| {
                        black_box(analyze_iso3382_bands_with_pool(
                            black_box(&signal),
                            48000.0,
                            &bands,
                            BandWidth::Octave,
                            4,
                            &mut contexts,
                        ))
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_iso3382_broadband,
    bench_iso3382_octaves,
    bench_iso3382_third_octaves,
    bench_workspace_matrix
);
criterion_main!(benches);

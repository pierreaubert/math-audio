use criterion::{criterion_group, criterion_main, Criterion};
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

criterion_group!(
    benches,
    bench_iso3382_broadband,
    bench_iso3382_octaves,
    bench_iso3382_third_octaves
);
criterion_main!(benches);

use criterion::{Criterion, criterion_group, criterion_main};
use math_audio_test_functions::functions::{ackley, levy, rastrigin, rosenbrock};
use ndarray::Array1;
use std::hint::black_box;

fn bench_rosenbrock(c: &mut Criterion) {
    let x = Array1::from_vec(vec![0.5; 30]);
    c.bench_function("rosenbrock_30d", |b| b.iter(|| rosenbrock(black_box(&x))));
}

fn bench_rastrigin(c: &mut Criterion) {
    let x = Array1::from_vec(vec![0.5; 30]);
    c.bench_function("rastrigin_30d", |b| b.iter(|| rastrigin(black_box(&x))));
}

fn bench_ackley(c: &mut Criterion) {
    let x = Array1::from_vec(vec![0.5; 30]);
    c.bench_function("ackley_30d", |b| b.iter(|| ackley(black_box(&x))));
}

fn bench_levy(c: &mut Criterion) {
    let x = Array1::from_vec(vec![0.5; 30]);
    c.bench_function("levy_30d", |b| b.iter(|| levy(black_box(&x))));
}

criterion_group!(
    benches,
    bench_rosenbrock,
    bench_rastrigin,
    bench_ackley,
    bench_levy
);
criterion_main!(benches);

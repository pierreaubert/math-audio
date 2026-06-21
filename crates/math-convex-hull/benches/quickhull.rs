use criterion::{criterion_group, criterion_main, Criterion};
use math_convex_hull::{ConvexHull3D, Vertex};
use std::hint::black_box;

fn fibonacci_sphere(n: usize) -> Vec<Vertex> {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    (0..n)
        .map(|i| {
            let y = 1.0 - (i as f64 / (n - 1) as f64) * 2.0;
            let radius = (1.0 - y * y).sqrt();
            let theta = phi * i as f64;
            Vertex::new(radius * theta.cos(), y, radius * theta.sin())
        })
        .collect()
}

fn bench_quickhull_1k(c: &mut Criterion) {
    let vertices = fibonacci_sphere(1_000);
    c.bench_function("quickhull_1k", |b| {
        b.iter(|| ConvexHull3D::build(black_box(&vertices)).unwrap())
    });
}

fn bench_quickhull_10k(c: &mut Criterion) {
    let vertices = fibonacci_sphere(10_000);
    c.bench_function("quickhull_10k", |b| {
        b.iter(|| ConvexHull3D::build(black_box(&vertices)).unwrap())
    });
}

criterion_group!(benches, bench_quickhull_1k, bench_quickhull_10k);
criterion_main!(benches);

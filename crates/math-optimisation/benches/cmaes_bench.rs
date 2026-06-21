use criterion::{criterion_group, criterion_main, Criterion};
use math_audio_optimisation::{cma_es, CmaEsConfig};
use ndarray::Array1;
use std::hint::black_box;

fn bench_cmaes_sphere(c: &mut Criterion) {
    let dim = 20;
    let _config = CmaEsConfig {
        bounds: vec![(-5.0, 5.0); dim],
        maxeval: 10_000,
        seed: Some(42),
        ..Default::default()
    };
    let objective = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();

    c.bench_function("cmaes_sphere_20d", |b| {
        b.iter(|| {
            cma_es(
                black_box(&objective),
                CmaEsConfig {
                    bounds: vec![(-5.0, 5.0); dim],
                    maxeval: 10_000,
                    seed: Some(42),
                    ..Default::default()
                },
            )
            .unwrap()
        })
    });
}

criterion_group!(benches, bench_cmaes_sphere);
criterion_main!(benches);

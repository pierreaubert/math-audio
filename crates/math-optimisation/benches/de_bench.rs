use criterion::{Criterion, criterion_group, criterion_main};
use math_audio_optimisation::{DEConfigBuilder, differential_evolution};
use ndarray::Array1;
use std::hint::black_box;

fn bench_de_sphere(c: &mut Criterion) {
    let dim = 30;
    let bounds = vec![(-5.0, 5.0); dim];
    let _config = DEConfigBuilder::new()
        .maxiter(100)
        .seed(42)
        .build()
        .unwrap();
    let objective = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();

    c.bench_function("de_sphere_30d", |b| {
        b.iter(|| {
            differential_evolution(
                black_box(&objective),
                black_box(&bounds),
                // DEConfig is not Clone; move a fresh copy each iteration via builder.
                DEConfigBuilder::new()
                    .maxiter(100)
                    .seed(42)
                    .build()
                    .unwrap(),
            )
            .unwrap()
        })
    });
}

criterion_group!(benches, bench_de_sphere);
criterion_main!(benches);

use criterion::{Criterion, criterion_group, criterion_main};
use math_delaunay::Delaunay;
use std::hint::black_box;

fn random_points(n: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| {
            let x = (i * 9301 + 49297) % 233280;
            let y = (i * 49297 + 9301) % 233280;
            (x as f64 / 233280.0, y as f64 / 233280.0)
        })
        .collect()
}

fn bench_voronoi_cell_polygons(c: &mut Criterion) {
    let points = random_points(1_000);
    let delaunay = Delaunay::from_points(&points);
    let voronoi = delaunay.voronoi([0.0, 0.0, 1.0, 1.0]);
    c.bench_function("voronoi_cell_polygons_1k", |b| {
        b.iter(|| black_box(voronoi.cell_polygons()))
    });
}

criterion_group!(benches, bench_voronoi_cell_polygons);
criterion_main!(benches);

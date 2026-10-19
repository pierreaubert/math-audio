//! FIR precision, symmetry, tap-count and callback-size matrix.

// Rust guideline compliant 2026-02-21
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use math_audio_iir_fir::{FilterFloat, Fir};
use std::hint::black_box;

fn matrix<T: FilterFloat>(c: &mut Criterion, precision: &str) {
    let mut group = c.benchmark_group(format!("fir_{precision}"));
    for taps in [32, 128, 512] {
        for symmetric in [true, false] {
            let mut coefficients = vec![T::from_f64(1.0 / taps as f64).unwrap(); taps];
            if !symmetric {
                coefficients[0] *= T::from_f64(0.9).unwrap();
            }
            for size in [32, 64, 128, 256, 1024] {
                let input: Vec<T> = (0..size)
                    .map(|n| T::from_f64((n as f64 * 0.137).sin()).unwrap())
                    .collect();
                let mut output = input.clone();
                for block in [true, false] {
                    let mut filter =
                        Fir::new_custom(coefficients.clone(), T::from_f64(48000.0).unwrap());
                    filter.prepare(size);
                    let label = format!("taps{taps}_sym{symmetric}_block{block}");
                    group.bench_with_input(BenchmarkId::new(label, size), &size, |b, _| {
                        b.iter(|| {
                            output.copy_from_slice(black_box(&input));
                            if block {
                                filter.process_block(black_box(&mut output));
                            } else {
                                for sample in &mut output {
                                    *sample = filter.process(*sample);
                                }
                            }
                            black_box(&output);
                        })
                    });
                }
            }
        }
    }
    group.finish();
}

fn benchmarks(c: &mut Criterion) {
    matrix::<f32>(c, "f32");
    matrix::<f64>(c, "f64");
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);

use criterion::{Criterion, criterion_group, criterion_main};
use math_audio_iir_fir::{Biquad, BiquadFilterType, peq_spl, peq_spl_into};
use ndarray::Array1;
use std::hint::black_box;

fn build_peq(n: usize) -> Vec<(f64, Biquad<f64>)> {
    (0..n)
        .map(|i| {
            let freq = 100.0 + i as f64 * 200.0;
            (
                1.0,
                Biquad::new(BiquadFilterType::Peak, freq, 48000.0, 1.0, 3.0),
            )
        })
        .collect()
}

fn bench_peq_spl(c: &mut Criterion) {
    let freq = Array1::logspace(10.0, 2.0_f64.log10(), 4.0_f64.log10(), 512);
    let peq = build_peq(8);
    c.bench_function("peq_spl_8_biquads_512_freqs", |b| {
        b.iter(|| peq_spl(black_box(&freq), black_box(&peq)))
    });
}

fn bench_peq_spl_into(c: &mut Criterion) {
    let freq = Array1::logspace(10.0, 2.0_f64.log10(), 4.0_f64.log10(), 512);
    let peq = build_peq(10);
    let mut response = Array1::zeros(freq.len());
    let mut filter_scratch = Array1::zeros(freq.len());
    c.bench_function("peq_spl_into_10_filters_512_points", |b| {
        b.iter(|| {
            peq_spl_into(
                black_box(&freq),
                black_box(&peq),
                black_box(&mut response),
                black_box(&mut filter_scratch),
            )
        })
    });
}

fn bench_biquad_np_log_result(c: &mut Criterion) {
    let freq = Array1::logspace(10.0, 2.0_f64.log10(), 4.0_f64.log10(), 512);
    let biquad = Biquad::new(BiquadFilterType::Peak, 1000.0, 48000.0, 1.0, 6.0);
    c.bench_function("biquad_np_log_result_512", |b| {
        b.iter(|| biquad.np_log_result(black_box(&freq)))
    });
}

criterion_group!(
    benches,
    bench_peq_spl,
    bench_peq_spl_into,
    bench_biquad_np_log_result
);
criterion_main!(benches);

use criterion::{Criterion, criterion_group, criterion_main};
use math_audio_autodiff::{fft::Fft, iir::biquad::Biquad, module::DiffModule, tensor::DiffTensor};
use math_audio_iir_fir::BiquadFilterType;
use ndarray::Array3;
use num_complex::Complex;
use std::hint::black_box;

fn biquad_forward_benchmark(c: &mut Criterion) {
    const NFFT: usize = 8192;
    let fft = Fft::with_channels(NFFT, 1);
    let biquad = Biquad::new(NFFT, 48_000.0, 2, BiquadFilterType::Highpass, 1, 1, 30.0).unwrap();
    let input_time = Array3::<Complex<f64>>::zeros((1, NFFT, 1));
    let input = DiffTensor::from_array(input_time.into_dyn());
    let spectrum = fft.forward(&input).unwrap();

    c.bench_function("biquad forward", |b| {
        b.iter(|| black_box(biquad.forward(black_box(&spectrum)).unwrap()))
    });
}

fn biquad_backward_benchmark(c: &mut Criterion) {
    const NFFT: usize = 8192;
    let fft = Fft::with_channels(NFFT, 1);
    let mut biquad =
        Biquad::new(NFFT, 48_000.0, 2, BiquadFilterType::Highpass, 1, 1, 30.0).unwrap();
    let input_time = Array3::<Complex<f64>>::zeros((1, NFFT, 1));
    let input = DiffTensor::from_array(input_time.into_dyn());
    let spectrum = fft.forward(&input).unwrap();
    let output = biquad.forward(&spectrum).unwrap();
    let grad_output = DiffTensor::from_array(output.data.clone());

    c.bench_function("biquad backward", |b| {
        b.iter(|| {
            black_box(
                biquad
                    .backward(
                        black_box(&spectrum),
                        black_box(&output),
                        black_box(&grad_output),
                    )
                    .unwrap(),
            )
        })
    });
}

criterion_group!(benches, biquad_forward_benchmark, biquad_backward_benchmark);
criterion_main!(benches);

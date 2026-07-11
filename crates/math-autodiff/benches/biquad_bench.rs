use criterion::{Criterion, criterion_group, criterion_main};
use math_audio_autodiff::{
    fft::Fft, gain::Gain, iir::biquad::Biquad, module::DiffModule, recursion::Recursion,
    tensor::DiffTensor,
};
use math_audio_iir_fir::BiquadFilterType;
use ndarray::Array3;
use num_complex::Complex;
use std::hint::black_box;

fn fft_forward_benchmark(c: &mut Criterion) {
    const NFFT: usize = 8192;
    let fft = Fft::with_channels(NFFT, 1);
    let input = DiffTensor::from_array(
        Array3::from_shape_fn((1, NFFT, 1), |(_, sample, _)| {
            Complex::new((sample as f64 * 0.01).sin(), 0.0)
        })
        .into_dyn(),
    );

    c.bench_function("fft forward", |b| {
        b.iter(|| black_box(fft.forward(black_box(&input)).unwrap()))
    });
}

fn fft_backward_benchmark(c: &mut Criterion) {
    const NFFT: usize = 8192;
    let mut fft = Fft::with_channels(NFFT, 1);
    let input = DiffTensor::from_array(
        Array3::from_shape_fn((1, NFFT, 1), |(_, sample, _)| {
            Complex::new((sample as f64 * 0.01).sin(), 0.0)
        })
        .into_dyn(),
    );
    let output = fft.forward(&input).unwrap();
    let grad_output = DiffTensor::from_array(output.data.clone());

    c.bench_function("fft backward", |b| {
        b.iter(|| {
            black_box(
                fft.backward(
                    black_box(&input),
                    black_box(&output),
                    black_box(&grad_output),
                )
                .unwrap(),
            )
        })
    });
}

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
            biquad.zero_grad();
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

fn recursion_fixture(nfft: usize) -> (Recursion, DiffTensor<f64>) {
    let mut feedforward = Gain::new(nfft, 2, 2).unwrap();
    let mut feedback = Gain::new(nfft, 2, 2).unwrap();
    for channel in 0..2 {
        feedforward.param[[channel, channel]] = 1.0;
        feedback.param[[channel, channel]] = 0.1;
    }
    let recursion = Recursion::new(Box::new(feedforward), Box::new(feedback)).unwrap();
    let input = DiffTensor::from_array(Array3::from_elem(
        (1, nfft / 2 + 1, 2),
        Complex::new(1.0, 0.0),
    ));
    (recursion, input)
}

fn recursion_forward_benchmark(c: &mut Criterion) {
    const NFFT: usize = 1024;
    let (recursion, input) = recursion_fixture(NFFT);
    c.bench_function("recursion forward", |b| {
        b.iter(|| black_box(recursion.forward(black_box(&input)).unwrap()))
    });
}

fn recursion_backward_benchmark(c: &mut Criterion) {
    const NFFT: usize = 1024;
    let (mut recursion, input) = recursion_fixture(NFFT);
    let output = recursion.forward(&input).unwrap();
    let grad_output = DiffTensor::from_array(output.data.clone());
    c.bench_function("recursion backward", |b| {
        b.iter(|| {
            recursion.zero_grad();
            black_box(
                recursion
                    .backward(
                        black_box(&input),
                        black_box(&output),
                        black_box(&grad_output),
                    )
                    .unwrap(),
            )
        })
    });
}

criterion_group!(
    benches,
    fft_forward_benchmark,
    fft_backward_benchmark,
    biquad_forward_benchmark,
    biquad_backward_benchmark,
    recursion_forward_benchmark,
    recursion_backward_benchmark
);
criterion_main!(benches);

use criterion::{Criterion, criterion_group, criterion_main};
use math_audio_autodiff::{
    delay::Delay,
    fft::Fft,
    gain::Gain,
    iir::biquad::Biquad,
    loss::{magnitude_mse_loss, magnitude_mse_loss_backward, mse_loss, mse_loss_backward},
    module::DiffModule,
    recursion::Recursion,
    system::Series,
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

fn spectrum_fixture(nfft: usize, channels: usize) -> DiffTensor<f64> {
    let fft = Fft::with_channels(nfft, channels);
    let input_time = Array3::<Complex<f64>>::from_shape_fn((1, nfft, channels), |(_, s, _)| {
        Complex::new((s as f64 * 0.01).sin(), 0.0)
    });
    let input = DiffTensor::from_array(input_time.into_dyn());
    fft.forward(&input).unwrap()
}

fn gain_forward_benchmark(c: &mut Criterion) {
    const NFFT: usize = 8192;
    let gain = Gain::new(NFFT, 2, 2).unwrap();
    let spectrum = spectrum_fixture(NFFT, 2);
    c.bench_function("gain forward", |b| {
        b.iter(|| black_box(gain.forward(black_box(&spectrum)).unwrap()))
    });
}

fn gain_backward_benchmark(c: &mut Criterion) {
    const NFFT: usize = 8192;
    let mut gain = Gain::new(NFFT, 2, 2).unwrap();
    let spectrum = spectrum_fixture(NFFT, 2);
    let output = gain.forward(&spectrum).unwrap();
    let grad_output = DiffTensor::from_array(output.data.clone());
    c.bench_function("gain backward", |b| {
        b.iter(|| {
            gain.zero_grad();
            black_box(
                gain.backward(
                    black_box(&spectrum),
                    black_box(&output),
                    black_box(&grad_output),
                )
                .unwrap(),
            )
        })
    });
}

fn delay_forward_benchmark(c: &mut Criterion) {
    const NFFT: usize = 8192;
    let delay = Delay::new(NFFT, 2, 2, 0.0).unwrap();
    let spectrum = spectrum_fixture(NFFT, 2);
    c.bench_function("delay forward", |b| {
        b.iter(|| black_box(delay.forward(black_box(&spectrum)).unwrap()))
    });
}

fn into_api_benchmark(c: &mut Criterion) {
    const NFFT: usize = 8192;
    let gain = Gain::new(NFFT, 2, 2).unwrap();
    let delay = Delay::new(NFFT, 2, 2, 0.0).unwrap();
    let spectrum = spectrum_fixture(NFFT, 2);
    let mut out = gain.forward(&spectrum).unwrap();
    c.bench_function("gain forward_into (reused buffer)", |b| {
        b.iter(|| {
            gain.forward_into(black_box(&spectrum), black_box(&mut out))
                .unwrap();
            black_box(out.data[[0, 0, 0]])
        })
    });
    let mut dout = delay.forward(&spectrum).unwrap();
    c.bench_function("delay forward_into (reused buffer)", |b| {
        b.iter(|| {
            delay
                .forward_into(black_box(&spectrum), black_box(&mut dout))
                .unwrap();
            black_box(dout.data[[0, 0, 0]])
        })
    });
}

fn delay_backward_benchmark(c: &mut Criterion) {
    const NFFT: usize = 8192;
    let mut delay = Delay::new(NFFT, 2, 2, 0.0).unwrap();
    let spectrum = spectrum_fixture(NFFT, 2);
    let output = delay.forward(&spectrum).unwrap();
    let grad_output = DiffTensor::from_array(output.data.clone());
    c.bench_function("delay backward", |b| {
        b.iter(|| {
            delay.zero_grad();
            black_box(
                delay
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

fn series_fixture(nfft: usize) -> (Series, DiffTensor<f64>) {
    let gain = Gain::new(nfft, 2, 2).unwrap();
    let delay = Delay::new(nfft, 2, 2, 0.0).unwrap();
    let series = Series::new(vec![Box::new(gain), Box::new(delay)]).unwrap();
    (series, spectrum_fixture(nfft, 2))
}

fn series_forward_benchmark(c: &mut Criterion) {
    const NFFT: usize = 8192;
    let (series, spectrum) = series_fixture(NFFT);
    c.bench_function("series forward", |b| {
        b.iter(|| black_box(series.forward(black_box(&spectrum)).unwrap()))
    });
}

fn series_backward_benchmark(c: &mut Criterion) {
    const NFFT: usize = 8192;
    let (mut series, spectrum) = series_fixture(NFFT);
    let output = series.forward(&spectrum).unwrap();
    let grad_output = DiffTensor::from_array(output.data.clone());
    c.bench_function("series backward", |b| {
        b.iter(|| {
            series.zero_grad();
            black_box(
                series
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

fn loss_forward_benchmark(c: &mut Criterion) {
    const NFFT: usize = 8192;
    let pred = spectrum_fixture(NFFT, 2);
    let target = spectrum_fixture(NFFT, 2);
    c.bench_function("mse loss forward", |b| {
        b.iter(|| black_box(mse_loss(black_box(&pred), black_box(&target)).unwrap()))
    });
    c.bench_function("magnitude mse loss forward", |b| {
        b.iter(|| black_box(magnitude_mse_loss(black_box(&pred), black_box(&target)).unwrap()))
    });
}

fn loss_backward_benchmark(c: &mut Criterion) {
    const NFFT: usize = 8192;
    let pred = spectrum_fixture(NFFT, 2);
    let target = spectrum_fixture(NFFT, 2);
    c.bench_function("mse loss backward", |b| {
        b.iter(|| black_box(mse_loss_backward(black_box(&pred), black_box(&target)).unwrap()))
    });
    c.bench_function("magnitude mse loss backward", |b| {
        b.iter(|| {
            black_box(magnitude_mse_loss_backward(black_box(&pred), black_box(&target)).unwrap())
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
    recursion_backward_benchmark,
    gain_forward_benchmark,
    gain_backward_benchmark,
    into_api_benchmark,
    delay_forward_benchmark,
    delay_backward_benchmark,
    series_forward_benchmark,
    series_backward_benchmark,
    loss_forward_benchmark,
    loss_backward_benchmark
);
criterion_main!(benches);

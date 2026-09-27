//! Fast timing harness for the evo optimization loop (items: parallelism,
//! recursion). Not statistically rigorous like criterion; it runs fixed
//! repetition counts and prints one total-milliseconds score (lower is
//! better) as the last stdout line.

use math_audio_autodiff::{
    delay::Delay, fft::Fft, gain::Gain, iir::biquad::Biquad, module::DiffModule,
    recursion::Recursion, system::Series, tensor::DiffTensor,
};
use math_audio_iir_fir::BiquadFilterType;
use ndarray::Array3;
use num_complex::Complex;
use std::time::Instant;

const NFFT: usize = 8192;
const CHANNELS: usize = 2;
const REPS: usize = 200;

fn make_spectrum(nfft: usize, channels: usize) -> DiffTensor<f64> {
    let fft = Fft::with_channels(nfft, channels);
    let time = DiffTensor::from_array(
        Array3::from_shape_fn((1, nfft, channels), |(_, s, _)| {
            Complex::new((s as f64 * 0.01).sin(), 0.0)
        })
        .into_dyn(),
    );
    fft.forward(&time).unwrap()
}

fn time_it(label: &str, reps: usize, mut op: impl FnMut()) -> f64 {
    for _ in 0..5 {
        op();
    }
    let start = Instant::now();
    for _ in 0..reps {
        op();
    }
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    println!("{label}: {ms:.3} ms total for {reps} reps");
    ms
}

fn main() {
    let spectrum = make_spectrum(NFFT, CHANNELS);
    let mut total = 0.0;

    let gain = Gain::new(NFFT, CHANNELS, CHANNELS).unwrap();
    let gain_out = gain.forward(&spectrum).unwrap();
    let gain_grad = DiffTensor::from_array(gain_out.data.clone());
    total += time_it("gain fwd", REPS, || {
        std::hint::black_box(gain.forward(&spectrum).unwrap());
    });
    let mut gain_mut = gain;
    total += time_it("gain bwd", REPS, || {
        gain_mut.zero_grad();
        std::hint::black_box(gain_mut.backward(&spectrum, &gain_out, &gain_grad).unwrap());
    });

    let delay = Delay::new(NFFT, CHANNELS, CHANNELS, 0.0).unwrap();
    let delay_out = delay.forward(&spectrum).unwrap();
    let delay_grad = DiffTensor::from_array(delay_out.data.clone());
    total += time_it("delay fwd", REPS, || {
        std::hint::black_box(delay.forward(&spectrum).unwrap());
    });
    let mut delay_mut = delay;
    total += time_it("delay bwd", REPS, || {
        delay_mut.zero_grad();
        std::hint::black_box(
            delay_mut
                .backward(&spectrum, &delay_out, &delay_grad)
                .unwrap(),
        );
    });

    let biquad = Biquad::new(
        NFFT,
        48_000.0,
        2,
        BiquadFilterType::Highpass,
        1,
        1,
        30.0,
    )
    .unwrap();
    let spectrum_1ch = make_spectrum(NFFT, 1);
    let biquad_out = biquad.forward(&spectrum_1ch).unwrap();
    let biquad_grad = DiffTensor::from_array(biquad_out.data.clone());
    total += time_it("biquad fwd", REPS, || {
        std::hint::black_box(biquad.forward(&spectrum_1ch).unwrap());
    });
    let mut biquad_mut = biquad;
    total += time_it("biquad bwd", REPS, || {
        biquad_mut.zero_grad();
        std::hint::black_box(
            biquad_mut
                .backward(&spectrum_1ch, &biquad_out, &biquad_grad)
                .unwrap(),
        );
    });

    let series_gain = Gain::new(NFFT, CHANNELS, CHANNELS).unwrap();
    let series_delay = Delay::new(NFFT, CHANNELS, CHANNELS, 0.0).unwrap();
    let series = Series::new(vec![Box::new(series_gain), Box::new(series_delay)]).unwrap();
    let series_out = series.forward(&spectrum).unwrap();
    let series_grad = DiffTensor::from_array(series_out.data.clone());
    total += time_it("series fwd", REPS, || {
        std::hint::black_box(series.forward(&spectrum).unwrap());
    });
    let mut series_mut = series;
    total += time_it("series bwd", REPS, || {
        series_mut.zero_grad();
        std::hint::black_box(
            series_mut
                .backward(&spectrum, &series_out, &series_grad)
                .unwrap(),
        );
    });

    let rec_ff = Gain::new(NFFT, CHANNELS, CHANNELS).unwrap();
    let rec_fb = Gain::new(NFFT, CHANNELS, CHANNELS).unwrap();
    let recursion = Recursion::new(Box::new(rec_ff), Box::new(rec_fb)).unwrap();
    let rec_out = recursion.forward(&spectrum).unwrap();
    let rec_grad = DiffTensor::from_array(rec_out.data.clone());
    total += time_it("recursion fwd", REPS, || {
        std::hint::black_box(recursion.forward(&spectrum).unwrap());
    });
    let mut recursion_mut = recursion;
    total += time_it("recursion bwd", REPS, || {
        recursion_mut.zero_grad();
        std::hint::black_box(
            recursion_mut
                .backward(&spectrum, &rec_out, &rec_grad)
                .unwrap(),
        );
    });

    println!("{total:.6}");
}

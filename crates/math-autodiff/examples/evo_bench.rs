//! Fast timing harness for the evo optimization loop (items: parallelism,
//! recursion). Not statistically rigorous like criterion; it runs fixed
//! repetition counts (50 warmup + min-of-3 timed batches per task) and
//! prints one total-milliseconds score (lower is better) as the last
//! stdout line.

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
/// Warmup reps before timing (defeats cold-CPU frequency lottery).
const WARMUP: usize = 50;
/// Timed batches per task; the reported time is the min across batches,
/// which rejects transient contention outliers from concurrent builds
/// and bursty neighbor jobs on this shared machine.
const BATCHES: usize = 5;

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

fn time_it(tasks: &mut Vec<(String, f64)>, label: &str, reps: usize, mut op: impl FnMut()) {
    for _ in 0..WARMUP {
        op();
    }
    let mut best = f64::INFINITY;
    let mut worst = 0.0f64;
    for _ in 0..BATCHES {
        let start = Instant::now();
        for _ in 0..reps {
            op();
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        best = best.min(ms);
        worst = worst.max(ms);
    }
    println!("{label}: {best:.3} ms min-of-{BATCHES} for {reps} reps (worst {worst:.3})");
    tasks.push((label.to_string(), best));
}

fn main() {
    let spectrum = make_spectrum(NFFT, CHANNELS);
    let mut tasks: Vec<(String, f64)> = Vec::new();

    let gain = Gain::new(NFFT, CHANNELS, CHANNELS).unwrap();
    let gain_out = gain.forward(&spectrum).unwrap();
    let gain_grad = DiffTensor::from_array(gain_out.data.clone());
    time_it(&mut tasks, "gain fwd", REPS, || {
        std::hint::black_box(gain.forward(&spectrum).unwrap());
    });
    let mut gain_mut = gain;
    time_it(&mut tasks, "gain bwd", REPS, || {
        gain_mut.zero_grad();
        std::hint::black_box(gain_mut.backward(&spectrum, &gain_out, &gain_grad).unwrap());
    });

    let delay = Delay::new(NFFT, CHANNELS, CHANNELS, 0.0).unwrap();
    let delay_out = delay.forward(&spectrum).unwrap();
    let delay_grad = DiffTensor::from_array(delay_out.data.clone());
    time_it(&mut tasks, "delay fwd", REPS, || {
        std::hint::black_box(delay.forward(&spectrum).unwrap());
    });
    let mut delay_mut = delay;
    time_it(&mut tasks, "delay bwd", REPS, || {
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
    time_it(&mut tasks, "biquad fwd", REPS, || {
        std::hint::black_box(biquad.forward(&spectrum_1ch).unwrap());
    });
    let mut biquad_mut = biquad;
    time_it(&mut tasks, "biquad bwd", REPS, || {
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
    time_it(&mut tasks, "series fwd", REPS, || {
        std::hint::black_box(series.forward(&spectrum).unwrap());
    });
    let mut series_mut = series;
    time_it(&mut tasks, "series bwd", REPS, || {
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
    time_it(&mut tasks, "recursion fwd", REPS, || {
        std::hint::black_box(recursion.forward(&spectrum).unwrap());
    });
    let mut recursion_mut = recursion;
    time_it(&mut tasks, "recursion bwd", REPS, || {
        recursion_mut.zero_grad();
        std::hint::black_box(
            recursion_mut
                .backward(&spectrum, &rec_out, &rec_grad)
                .unwrap(),
        );
    });

    let total: f64 = tasks.iter().map(|(_, ms)| ms).sum();
    println!("{total:.6}");
    if let Ok(path) = std::env::var("EVO_RESULT_PATH") {
        let tasks_json: Vec<String> = tasks
            .iter()
            .map(|(name, ms)| format!("\"{name}\": {ms:.6}"))
            .collect();
        let json = format!(
            "{{\"score\": {total:.6}, \"tasks\": {{{}}}}}",
            tasks_json.join(", ")
        );
        std::fs::write(&path, &json).expect("write EVO_RESULT_PATH");
        println!("wrote {path}");
    }
}

//! Standalone throughput benchmark for IIR (biquad) and FIR filters.
//!
//! Prints a single JSON line with per-task throughput in Msamples/s and
//! deterministic checksums for the processed output.

use math_audio_iir_fir::{Biquad, BiquadFilterType, Fir, WindowType};
use std::time::Instant;

const SAMPLE_RATE: f64 = 48_000.0;

fn main() {
    // ---------- IIR biquad block ----------
    let mut biquad = Biquad::new(BiquadFilterType::Lowpass, 1_000.0, SAMPLE_RATE, 0.707, 0.0);
    let biquad_block_size: usize = 48_000;
    let biquad_iterations: usize = 100;
    let biquad_input: Vec<f64> = (0..biquad_block_size)
        .map(|i| 0.5 * (i as f64 * 0.01).sin())
        .collect();

    // Warmup
    for _ in 0..5 {
        let mut buf = biquad_input.clone();
        biquad.process_block(&mut buf);
        std::hint::black_box(&buf);
    }

    let start = Instant::now();
    for _ in 0..biquad_iterations {
        let mut buf = biquad_input.clone();
        biquad.process_block(&mut buf);
        std::hint::black_box(&buf);
    }
    let biquad_elapsed = start.elapsed().as_secs_f64();
    let biquad_samples = (biquad_block_size * biquad_iterations) as f64;
    let biquad_throughput = biquad_samples / biquad_elapsed / 1_000_000.0;

    let mut check_buf = biquad_input.clone();
    biquad.process_block(&mut check_buf);
    let biquad_checksum: f64 = check_buf.iter().sum();

    // ---------- FIR block ----------
    let mut fir = Fir::lowpass(101, 1_000.0, SAMPLE_RATE, WindowType::Hamming, 0.0);
    let fir_block_size: usize = 4_800;
    let fir_iterations: usize = 100;
    let fir_input: Vec<f64> = (0..fir_block_size)
        .map(|i| 0.5 * (i as f64 * 0.01).sin())
        .collect();

    for _ in 0..5 {
        let mut buf = fir_input.clone();
        fir.process_block(&mut buf);
        std::hint::black_box(&buf);
    }

    let start = Instant::now();
    for _ in 0..fir_iterations {
        let mut buf = fir_input.clone();
        fir.process_block(&mut buf);
        std::hint::black_box(&buf);
    }
    let fir_elapsed = start.elapsed().as_secs_f64();
    let fir_samples = (fir_block_size * fir_iterations) as f64;
    let fir_throughput = fir_samples / fir_elapsed / 1_000_000.0;

    let mut check_buf = fir_input.clone();
    fir.process_block(&mut check_buf);
    let fir_checksum: f64 = check_buf.iter().sum();

    println!(
        "{{\"iir_biquad_block_msamples_per_s\": {:.6}, \"fir_block_msamples_per_s\": {:.6}, \"iir_checksum\": {:.12}, \"fir_checksum\": {:.12}}}",
        biquad_throughput, fir_throughput, biquad_checksum, fir_checksum
    );
}

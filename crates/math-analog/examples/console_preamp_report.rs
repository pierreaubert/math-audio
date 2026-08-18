//! Print a deterministic level-matched Console/Preamp versus H2/H3 baseline
//! report.  The fixture is synthetic and is not a hardware or listening claim.

use math_audio_analog::analysis::{level_match_candidate, measure_harmonics};
use math_audio_analog::{AnalogProcessor, ConsolePreampModel, HarmonicModel, ProcessSpec};
use std::f32::consts::TAU;

const SAMPLE_RATE: f32 = 48_000.0;
const CHANNELS: usize = 2;
const FRAMES: usize = 240_000;

fn main() {
    let input: Vec<f32> = (0..FRAMES)
        .flat_map(|index| {
            let sample = 0.2
                * (TAU * 1_000.0 * index as f32 / SAMPLE_RATE).sin()
                * (0.8 + 0.2 * (TAU * 0.5 * index as f32 / SAMPLE_RATE).sin());
            [sample, sample]
        })
        .collect();

    let baseline = render_baseline(&input);
    let console = render_console(&input);
    let match_report =
        level_match_candidate(&baseline, &console, SAMPLE_RATE as u32, CHANNELS, 0.95).unwrap();
    let gain = 10.0_f32.powf(match_report.applied_gain_db / 20.0);
    let baseline_mono = every_other(&baseline);
    let console_mono: Vec<f32> = every_other(&console)
        .into_iter()
        .map(|sample| sample * gain)
        .collect();
    let baseline_harmonics = measure_harmonics(&baseline_mono, SAMPLE_RATE, 1_000.0, 5).unwrap();
    let console_harmonics = measure_harmonics(&console_mono, SAMPLE_RATE, 1_000.0, 5).unwrap();

    println!(
        "sample_rate_hz={SAMPLE_RATE} frames={FRAMES} channels={CHANNELS} fixture=synthetic_amplitude_modulated_1kHz"
    );
    println!(
        "level_match baseline_lufs={:.6} console_lufs={:.6} requested_gain_db={:.6} applied_gain_db={:.6} peak_limited={} baseline_peak={:.6} console_peak={:.6}",
        match_report.reference_lufs,
        match_report.candidate_lufs,
        match_report.requested_gain_db,
        match_report.applied_gain_db,
        match_report.peak_limited,
        match_report.reference_peak,
        match_report.candidate_peak,
    );
    println!(
        "harmonics_level_matched baseline_h1={:.9} console_h1={:.9} baseline_h2={:.9} console_h2={:.9} baseline_h3={:.9} console_h3={:.9} baseline_h4={:.9} console_h4={:.9} baseline_h5={:.9} console_h5={:.9}",
        baseline_harmonics.component(1).unwrap().amplitude,
        console_harmonics.component(1).unwrap().amplitude,
        baseline_harmonics.component(2).unwrap().amplitude,
        console_harmonics.component(2).unwrap().amplitude,
        baseline_harmonics.component(3).unwrap().amplitude,
        console_harmonics.component(3).unwrap().amplitude,
        baseline_harmonics.component(4).unwrap().amplitude,
        console_harmonics.component(4).unwrap().amplitude,
        baseline_harmonics.component(5).unwrap().amplitude,
        console_harmonics.component(5).unwrap().amplitude,
    );
}

fn render_baseline(input: &[f32]) -> Vec<f32> {
    let mut model = HarmonicModel::new();
    model.set_anti_aliasing(math_audio_analog::AntiAliasing::Off);
    model.set_drive_db(12.0).unwrap();
    model.set_h2_db(-18.0).unwrap();
    model.set_h3_db(-24.0).unwrap();
    model
        .prepare(ProcessSpec::new(SAMPLE_RATE, CHANNELS, FRAMES))
        .unwrap();
    let mut output = input.to_vec();
    model.process_interleaved(&mut output, FRAMES).unwrap();
    output
}

fn render_console(input: &[f32]) -> Vec<f32> {
    let mut model = ConsolePreampModel::new();
    model.set_anti_aliasing(math_audio_analog::AntiAliasing::Off);
    model.set_input_gain_db(12.0).unwrap();
    model
        .prepare(ProcessSpec::new(SAMPLE_RATE, CHANNELS, FRAMES))
        .unwrap();
    let mut output = input.to_vec();
    model.process_interleaved(&mut output, FRAMES).unwrap();
    output
}

fn every_other(interleaved: &[f32]) -> Vec<f32> {
    interleaved.iter().step_by(CHANNELS).copied().collect()
}

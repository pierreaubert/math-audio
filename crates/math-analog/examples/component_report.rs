//! Print deterministic component-model evidence rows (IDs 6-8).
//!
//! The rows characterize the Shockley clipper, Koren triode stage, and
//! Yeh tone-stack implementations against their frozen references in
//! `references/component-references.md`. They are implementation evidence,
//! not hardware fits or listening results.

use math_audio_analog::analysis::measure_harmonics;
use math_audio_analog::{
    AnalogProcessor, DiodeClipperModel, DiodeFlavor, ProcessSpec, ToneStackModel, ToneStackValues,
    TriodeStageModel,
};
use std::f32::consts::TAU;

const SAMPLE_RATE: f32 = 48_000.0;
const SETTLE: usize = 14_400;
const MEASURE: usize = 4_800;

fn coherent(target_hz: f32, record: usize) -> f32 {
    let bin = (target_hz * record as f32 / SAMPLE_RATE).round();
    bin * SAMPLE_RATE / record as f32
}

/// Render a settled sine through a prepared mono model and return the
/// fundamental amplitude plus the settled tail.
fn render<M: AnalogProcessor>(model: &mut M, frequency_hz: f32, amplitude: f32) -> (f32, Vec<f32>) {
    let frequency = coherent(frequency_hz, MEASURE);
    let mut samples: Vec<f32> = (0..SETTLE + MEASURE)
        .map(|index| amplitude * (TAU * frequency * index as f32 / SAMPLE_RATE).sin())
        .collect();
    model
        .process_interleaved(&mut samples, SETTLE + MEASURE)
        .expect("prepared render");
    let tail = samples[SETTLE..].to_vec();
    let report = measure_harmonics(&tail, SAMPLE_RATE, frequency, 4).expect("harmonic");
    (report.component(1).expect("H1").amplitude, tail)
}

fn clipper_rows() {
    for flavor in [DiodeFlavor::Silicon, DiodeFlavor::Germanium] {
        let mut model = DiodeClipperModel::new();
        model.set_flavor(flavor);
        model
            .prepare(ProcessSpec::new(SAMPLE_RATE, 1, SETTLE + MEASURE))
            .expect("valid spec");
        // Deep-clipped 100 Hz sine: settled peak output approximates the
        // Shockley threshold for the resistor current.
        let (_, tail) = render(&mut model, 100.0, 5.0);
        let peak = tail.iter().map(|sample| sample.abs()).fold(0.0, f32::max);
        let (solves, fallbacks) = model.solve_stats();
        println!(
            "clipper flavor={flavor:?} sine100hz_peak5v_out={peak:.6} solves={solves} fallbacks={fallbacks}"
        );
    }
    // Odd-symmetry spot check: H2 vs H3 under symmetric drive.
    let mut model = DiodeClipperModel::new();
    model.set_drive_db(12.0).expect("valid drive");
    model
        .prepare(ProcessSpec::new(SAMPLE_RATE, 1, SETTLE + MEASURE))
        .expect("valid spec");
    let frequency = coherent(1_000.0, MEASURE);
    let mut samples: Vec<f32> = (0..SETTLE + MEASURE)
        .map(|index| 0.5 * (TAU * frequency * index as f32 / SAMPLE_RATE).sin())
        .collect();
    model
        .process_interleaved(&mut samples, SETTLE + MEASURE)
        .expect("prepared render");
    let report =
        measure_harmonics(&samples[SETTLE..], SAMPLE_RATE, frequency, 4).expect("harmonic");
    println!(
        "clipper symmetry h1={:.6} h2={:.6} h3={:.6}",
        report.component(1).expect("H1").amplitude,
        report.component(2).expect("H2").amplitude,
        report.component(3).expect("H3").amplitude
    );
}

fn triode_rows() {
    let probe = TriodeStageModel::new();
    println!("triode gain_magnitude={:.4}", probe.gain_magnitude());
    // Small-signal unity check at -36 dBFS plus a THD drive sweep.
    for (drive_db, amplitude) in [
        (0.0, 0.016),
        (-12.0, 0.5),
        (0.0, 0.5),
        (12.0, 0.5),
        (24.0, 0.5),
    ] {
        let mut model = TriodeStageModel::new();
        model.set_drive_db(drive_db).expect("valid drive");
        model
            .prepare(ProcessSpec::new(SAMPLE_RATE, 1, SETTLE + MEASURE))
            .expect("valid spec");
        let frequency = coherent(1_000.0, MEASURE);
        let mut samples: Vec<f32> = (0..SETTLE + MEASURE)
            .map(|index| amplitude * (TAU * frequency * index as f32 / SAMPLE_RATE).sin())
            .collect();
        model
            .process_interleaved(&mut samples, SETTLE + MEASURE)
            .expect("prepared render");
        let report =
            measure_harmonics(&samples[SETTLE..], SAMPLE_RATE, frequency, 10).expect("harmonic");
        let fundamental = report.component(1).expect("H1").amplitude;
        let harmonics_rss: f32 = (2..=10)
            .map(|order| {
                let amplitude = report.component(order).expect("harmonic").amplitude;
                amplitude * amplitude
            })
            .sum::<f32>()
            .sqrt();
        let (solves, fallbacks) = model.solve_stats();
        println!(
            "triode drive_db={drive_db} in={amplitude} gain_db={:.3} thd={:.6} solves={solves} fallbacks={fallbacks}",
            20.0 * (fundamental / amplitude).log10(),
            harmonics_rss / fundamental
        );
    }
}

fn tonestack_rows() {
    for values in [ToneStackValues::Schematic59, ToneStackValues::Production] {
        // Noon reference + scoop depth.
        let mut model = ToneStackModel::new();
        model.set_values(values);
        model
            .prepare(ProcessSpec::new(SAMPLE_RATE, 1, SETTLE + MEASURE))
            .expect("valid spec");
        let (at_1k, _) = render(&mut model, 1_000.0, 0.5);
        let mut model = ToneStackModel::new();
        model.set_values(values);
        model
            .prepare(ProcessSpec::new(SAMPLE_RATE, 1, SETTLE + MEASURE))
            .expect("valid spec");
        let (at_100, _) = render(&mut model, 100.0, 0.5);
        let mut dip = f32::INFINITY;
        for frequency in [400.0, 500.0, 700.0, 1_000.0] {
            let mut model = ToneStackModel::new();
            model.set_values(values);
            model
                .prepare(ProcessSpec::new(SAMPLE_RATE, 1, SETTLE + MEASURE))
                .expect("valid spec");
            let (level, _) = render(&mut model, frequency, 0.5);
            dip = dip.min(level);
        }
        println!(
            "tonestack values={values:?} noon_1khz_db={:.3} scoop_db={:.2}",
            20.0 * (at_1k / 0.5).log10(),
            20.0 * (at_100 / dip).log10()
        );
    }
    // Knob sweeps (schematic set).
    let sweep = |knob: &str, frequency: f32| {
        let mut levels = Vec::new();
        for position in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let mut model = ToneStackModel::new();
            match knob {
                "treble" => model.set_treble(position).expect("valid knob"),
                "mid" => model.set_mid(position).expect("valid knob"),
                _ => model.set_bass(position).expect("valid knob"),
            }
            model
                .prepare(ProcessSpec::new(SAMPLE_RATE, 1, SETTLE + MEASURE))
                .expect("valid spec");
            let (level, _) = render(&mut model, frequency, 0.5);
            levels.push(20.0 * (level / 0.5).log10());
        }
        println!("tonestack sweep knob={knob} hz={frequency} db={levels:.2?}");
    };
    sweep("bass", 40.0);
    sweep("mid", 500.0);
    sweep("treble", 8_000.0);
}

fn main() {
    println!("sample_rate_hz={SAMPLE_RATE} settle={SETTLE} measure={MEASURE}");
    println!("fixture=settled coherent sines, rectangular one-sided amplitude");
    clipper_rows();
    triode_rows();
    tonestack_rows();
}

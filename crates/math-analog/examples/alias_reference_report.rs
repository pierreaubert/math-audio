//! Print the deterministic synthetic alias-reference report used by the
//! checked-in `reports/alias-reference.md` artifact.

use math_audio_analog::analysis::compare_alias_reference;
use math_audio_analog::{AnalogProcessor, AntiAliasing, HarmonicModel, ProcessSpec};
use std::f32::consts::TAU;

const BASE_RATE: f32 = 48_000.0;
const TONE_HZ: f32 = 10_000.0;
const BASE_FRAMES: usize = 4_800;

fn render(sample_rate: f32, frames: usize, anti_aliasing: AntiAliasing) -> Vec<f32> {
    let mut model = HarmonicModel::new();
    model.set_anti_aliasing(anti_aliasing);
    model.set_drive_db(24.0).expect("valid drive");
    model.set_h2_db(-120.0).expect("muted H2");
    model.set_h3_db(-120.0).expect("muted H3");
    model
        .prepare(ProcessSpec::new(sample_rate, 1, frames))
        .expect("valid process spec");

    let mut samples: Vec<f32> = (0..frames)
        .map(|index| 0.8 * (TAU * TONE_HZ * index as f32 / sample_rate).sin())
        .collect();
    model
        .process_interleaved(&mut samples, frames)
        .expect("prepared render");
    samples
}

fn main() {
    println!("sample_rate_hz={BASE_RATE}");
    println!("tone_hz={TONE_HZ}");
    println!("base_frames={BASE_FRAMES}");
    println!("input_amplitude=0.8");
    println!("drive_db=24");
    println!("h2_db=-120");
    println!("h3_db=-120");
    println!("fir=127-tap Blackman-windowed sinc, cutoff=0.45/factor");
    println!(
        "columns=anti_aliasing factor base_rms reference_rms error_rms error_peak error_level_db"
    );

    for anti_aliasing in [AntiAliasing::Off, AntiAliasing::Adaa1] {
        for factor in [2, 4] {
            let base = render(BASE_RATE, BASE_FRAMES, anti_aliasing);
            let high = render(
                BASE_RATE * factor as f32,
                BASE_FRAMES * factor,
                anti_aliasing,
            );
            let report = compare_alias_reference(&base, &high, BASE_RATE, factor)
                .expect("matching high-rate reference");
            println!(
                "{anti_aliasing:?} {factor} {:.9} {:.9} {:.9} {:.9} {:.9}",
                report.base_rms,
                report.reference_rms,
                report.error_rms,
                report.error_peak,
                report.error_level_db,
            );
        }
    }
}

//! Print deterministic harmonic, IMD, and transient reports for a synthetic
//! HarmonicModel fixture.

use math_audio_analog::analysis::{measure_harmonics, measure_transient, measure_two_tone_imd};
use math_audio_analog::{AnalogProcessor, HarmonicModel, ProcessSpec};
use std::f32::consts::TAU;

const SAMPLE_RATE: f32 = 48_000.0;
const RECORD_LENGTH: usize = 4_800;

fn configure(model: &mut HarmonicModel) {
    model.set_drive_db(12.0).expect("valid drive");
    model.set_h2_db(-18.0).expect("valid H2");
    model.set_h3_db(-24.0).expect("valid H3");
    model.set_output_gain_db(0.0).expect("valid output gain");
    model.set_amount(1.0).expect("valid amount");
    model.set_mix(1.0).expect("valid mix");
    model
        .prepare(ProcessSpec::new(SAMPLE_RATE, 1, RECORD_LENGTH))
        .expect("valid process spec");
}

fn render_tone(frequency_hz: f32) -> Vec<f32> {
    let mut model = HarmonicModel::new();
    configure(&mut model);
    let mut samples: Vec<f32> = (0..RECORD_LENGTH)
        .map(|index| 0.5 * (TAU * frequency_hz * index as f32 / SAMPLE_RATE).sin())
        .collect();
    model
        .process_interleaved(&mut samples, RECORD_LENGTH)
        .expect("prepared render");
    samples
}

fn render_two_tone() -> Vec<f32> {
    let mut model = HarmonicModel::new();
    configure(&mut model);
    let mut samples: Vec<f32> = (0..RECORD_LENGTH)
        .map(|index| {
            let phase_a = TAU * 1_000.0 * index as f32 / SAMPLE_RATE;
            let phase_b = TAU * 1_500.0 * index as f32 / SAMPLE_RATE;
            0.25 * phase_a.sin() + 0.125 * phase_b.sin()
        })
        .collect();
    model
        .process_interleaved(&mut samples, RECORD_LENGTH)
        .expect("prepared render");
    samples
}

fn main() {
    println!("sample_rate_hz={SAMPLE_RATE}");
    println!("record_length={RECORD_LENGTH}");
    println!("model=HarmonicModel, drive_db=12, h2_db=-18, h3_db=-24");
    println!("convention=rectangular one-sided amplitude, no zero-padding");

    let harmonic_samples = render_tone(1_000.0);
    let harmonic =
        measure_harmonics(&harmonic_samples, SAMPLE_RATE, 1_000.0, 5).expect("harmonic report");
    let distortion = harmonic
        .distortion(&harmonic_samples)
        .expect("distortion report");
    println!(
        "harmonic fundamental={:.9} h2={:.9} h3={:.9} h4={:.9} h5={:.9}",
        harmonic.component(1).expect("H1").amplitude,
        harmonic.component(2).expect("H2").amplitude,
        harmonic.component(3).expect("H3").amplitude,
        harmonic.component(4).expect("H4").amplitude,
        harmonic.component(5).expect("H5").amplitude,
    );
    println!(
        "distortion thd={:.9} thd_plus_n={:.9} alias_rms={:.9} alias_level_db={:.9}",
        distortion.thd, distortion.thd_plus_n, distortion.alias_rms, distortion.alias_level_db,
    );

    let imd_samples = render_two_tone();
    let imd =
        measure_two_tone_imd(&imd_samples, SAMPLE_RATE, 1_000.0, 1_500.0, 3).expect("IMD report");
    println!(
        "imd tone_a={:.9} tone_b={:.9} 2f1-f2={:.9} 2f2-f1={:.9}",
        imd.tone_a_amplitude,
        imd.tone_b_amplitude,
        imd.component(2, -1).expect("2f1-f2").amplitude,
        imd.component(1, -2).expect("2f2-f1").amplitude,
    );

    let transient =
        measure_transient(&harmonic_samples[..256], SAMPLE_RATE).expect("transient report");
    println!(
        "transient peak={:.9} peak_index={} rms={:.9} dc={:.9}",
        transient.peak_amplitude,
        transient.peak_index,
        transient.rms_amplitude,
        transient.dc_amplitude,
    );
}

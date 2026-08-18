//! Print a deterministic synthetic analysis row for every analog model family.
//!
//! The rows characterize the implementation contracts and are not hardware
//! fits, listening results, or evidence that one model is perceptually better.

use math_audio_analog::analysis::{measure_harmonics, measure_transient, measure_two_tone_imd};
use math_audio_analog::{AnalogModel, AnalogProcessor, AntiAliasing, ProcessSpec, StaticCurve};
use std::f32::consts::TAU;

const SAMPLE_RATE: f32 = 48_000.0;
const RECORD_LENGTH: usize = 4_800;

fn model_name(model_id: u32) -> &'static str {
    match model_id {
        AnalogModel::HARMONICS_ID => "Harmonics",
        AnalogModel::STATIC_ID => "Static",
        AnalogModel::HAMMERSTEIN_ID => "Hammerstein",
        AnalogModel::TAPE_ID => "Tape-style",
        AnalogModel::TRANSFORMER_ID => "Transformer-style",
        AnalogModel::CONSOLE_PREAMP_ID => "Console/Preamp-style",
        _ => "Unknown",
    }
}

fn configure(model: &mut AnalogModel) {
    match model {
        AnalogModel::Harmonics(model) => {
            model.set_anti_aliasing(AntiAliasing::Off);
            model.set_h2_db(-18.0).expect("valid H2");
            model.set_h3_db(-24.0).expect("valid H3");
            model.set_character(0.5).expect("valid character");
            model.set_drive_db(12.0).expect("valid drive");
            model.set_output_gain_db(0.0).expect("valid output gain");
            model.set_amount(1.0).expect("valid amount");
            model.set_mix(1.0).expect("valid mix");
        }
        AnalogModel::Static(model) => {
            model.set_curve(StaticCurve::TanhStyle);
            model.set_anti_aliasing(AntiAliasing::Off);
            model.set_character(0.5).expect("valid character");
            model.set_drive_db(12.0).expect("valid drive");
            model.set_output_gain_db(0.0).expect("valid output gain");
            model.set_amount(1.0).expect("valid amount");
            model.set_mix(1.0).expect("valid mix");
        }
        AnalogModel::Hammerstein(model) => {
            model.set_character(0.5).expect("valid character");
            model.set_drive_db(12.0).expect("valid drive");
            model.set_output_gain_db(0.0).expect("valid output gain");
            model.set_amount(1.0).expect("valid amount");
            model.set_mix(1.0).expect("valid mix");
        }
        AnalogModel::Tape(model) => {
            model.set_character(0.5).expect("valid character");
            model.set_drive_db(12.0).expect("valid drive");
            model.set_output_gain_db(0.0).expect("valid output gain");
            model.set_amount(1.0).expect("valid amount");
            model.set_mix(1.0).expect("valid mix");
        }
        AnalogModel::Transformer(model) => {
            model.set_character(0.5).expect("valid character");
            model.set_drive_db(12.0).expect("valid drive");
            model.set_output_gain_db(0.0).expect("valid output gain");
            model.set_amount(1.0).expect("valid amount");
            model.set_mix(1.0).expect("valid mix");
        }
        AnalogModel::ConsolePreamp(model) => {
            model.set_asymmetry(0.5).expect("valid asymmetry");
            model.set_input_gain_db(12.0).expect("valid input gain");
            model.set_output_gain_db(0.0).expect("valid output gain");
            model.set_amount(1.0).expect("valid amount");
            model.set_mix(1.0).expect("valid mix");
        }
    }
}

fn render_tone(model_id: u32, frequency_hz: f32) -> Vec<f32> {
    let mut model = AnalogModel::from_id(model_id).expect("stable model id");
    configure(&mut model);
    model
        .prepare(ProcessSpec::new(SAMPLE_RATE, 1, RECORD_LENGTH))
        .expect("valid process spec");
    let mut samples: Vec<f32> = (0..RECORD_LENGTH)
        .map(|index| 0.5 * (TAU * frequency_hz * index as f32 / SAMPLE_RATE).sin())
        .collect();
    model
        .process_interleaved(&mut samples, RECORD_LENGTH)
        .expect("prepared render");
    samples
}

fn render_two_tone(model_id: u32) -> Vec<f32> {
    let mut model = AnalogModel::from_id(model_id).expect("stable model id");
    configure(&mut model);
    model
        .prepare(ProcessSpec::new(SAMPLE_RATE, 1, RECORD_LENGTH))
        .expect("valid process spec");
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
    println!("sample_rate_hz={SAMPLE_RATE} record_length={RECORD_LENGTH}");
    println!("fixture=synthetic coherent sine/two-tone, rectangular one-sided amplitude");
    println!(
        "columns=model id finite h1 h2 h3 thd thd_plus_n imd_2f1_minus_f2 imd_2f2_minus_f1 transient_peak transient_rms dc"
    );

    for model_id in 0..=AnalogModel::CONSOLE_PREAMP_ID {
        let harmonic_samples = render_tone(model_id, 1_000.0);
        let harmonic =
            measure_harmonics(&harmonic_samples, SAMPLE_RATE, 1_000.0, 3).expect("harmonic");
        let distortion = harmonic.distortion(&harmonic_samples).expect("distortion");
        let imd_samples = render_two_tone(model_id);
        let imd =
            measure_two_tone_imd(&imd_samples, SAMPLE_RATE, 1_000.0, 1_500.0, 3).expect("IMD");
        let transient =
            measure_transient(&harmonic_samples[..256], SAMPLE_RATE).expect("transient");
        println!(
            "model={} id={} {} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9}",
            model_name(model_id),
            model_id,
            harmonic_samples.iter().all(|sample| sample.is_finite()),
            harmonic.component(1).expect("H1").amplitude,
            harmonic.component(2).expect("H2").amplitude,
            harmonic.component(3).expect("H3").amplitude,
            distortion.thd,
            distortion.thd_plus_n,
            imd.component(2, -1).expect("2f1-f2").amplitude,
            imd.component(1, -2).expect("2f2-f1").amplitude,
            transient.peak_amplitude,
            transient.rms_amplitude,
            transient.dc_amplitude,
        );
    }
}

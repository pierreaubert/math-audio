use math_audio_analog::analysis::measure_harmonics;
use math_audio_analog::{AnalogModel, AnalogProcessor, AntiAliasing, HarmonicModel, ProcessSpec};
use std::f32::consts::TAU;

const SAMPLE_RATES: [f32; 4] = [44_100.0, 48_000.0, 96_000.0, 192_000.0];
const LEVELS_DBFS: [f32; 6] = [-36.0, -24.0, -18.0, -12.0, -6.0, -1.0];

fn coherent_frequency(sample_rate: f32, record_length: usize, target_hz: f32) -> f32 {
    let bin = (target_hz * record_length as f32 / sample_rate).round();
    bin * sample_rate / record_length as f32
}

#[test]
fn harmonic_matrix_covers_rates_levels_frequencies_and_h10() {
    for sample_rate in SAMPLE_RATES {
        let record_length = sample_rate as usize / 10;
        let near_nyquist = sample_rate * 0.225;
        for target_hz in [50.0, 100.0, 1_000.0, 5_000.0, near_nyquist] {
            let frequency = coherent_frequency(sample_rate, record_length, target_hz);
            for level_dbfs in LEVELS_DBFS {
                let amplitude = 10.0_f32.powf(level_dbfs / 20.0);
                for drive_db in [0.0, 12.0, 24.0] {
                    let mut model = HarmonicModel::new();
                    model.set_drive_db(drive_db).unwrap();
                    model.set_h2_db(-18.0).unwrap();
                    model.set_h3_db(-24.0).unwrap();
                    model
                        .prepare(ProcessSpec::new(sample_rate, 1, record_length))
                        .unwrap();
                    let mut samples: Vec<f32> = (0..record_length)
                        .map(|index| {
                            amplitude * (TAU * frequency * index as f32 / sample_rate).sin()
                        })
                        .collect();
                    model
                        .process_interleaved(&mut samples, record_length)
                        .unwrap();
                    assert!(
                        samples.iter().all(|sample| sample.is_finite()),
                        "non-finite output at {sample_rate} Hz, {level_dbfs} dBFS, drive {drive_db} dB, {frequency} Hz"
                    );

                    let report = measure_harmonics(&samples, sample_rate, frequency, 10).unwrap();
                    assert!(report.component(1).unwrap().amplitude.is_finite());
                    for order in 2..=10 {
                        let component = report.component(order).unwrap();
                        if component.frequency_hz < sample_rate * 0.5 {
                            assert!(!component.aliases);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn direct_and_adaa_modes_produce_reproducible_alias_reports() {
    let sample_rate = 48_000.0;
    let frequency = 10_000.0;
    let record_length = 4_800;
    let render = |mode| {
        let mut model = HarmonicModel::new();
        model.set_anti_aliasing(mode);
        model.set_drive_db(24.0).unwrap();
        model.set_h2_db(-120.0).unwrap();
        model.set_h3_db(-120.0).unwrap();
        model
            .prepare(ProcessSpec::new(sample_rate, 1, record_length))
            .unwrap();
        let mut samples: Vec<f32> = (0..record_length)
            .map(|index| (TAU * frequency * index as f32 / sample_rate).sin() * 0.8)
            .collect();
        model
            .process_interleaved(&mut samples, record_length)
            .unwrap();
        let report = measure_harmonics(&samples, sample_rate, frequency, 3).unwrap();
        (report.distortion(&samples).unwrap(), report)
    };
    let (off, off_report) = render(AntiAliasing::Off);
    let (adaa, adaa_report) = render(AntiAliasing::Adaa1);
    assert!(off.alias_rms.is_finite());
    assert!(adaa.alias_rms.is_finite());
    assert!(off_report.component(3).unwrap().aliases);
    assert!(adaa_report.component(3).unwrap().aliases);
    assert_ne!(off.alias_rms.to_bits(), adaa.alias_rms.to_bits());
}

#[test]
fn every_model_family_exposes_an_in_crate_alias_reduction_path() {
    let sample_rate = 48_000.0;
    let frequency = 10_000.0;
    let record_length = 4_800;
    let render = |model_id: u32, mode: AntiAliasing| {
        let mut model = AnalogModel::from_id(model_id).unwrap();
        match &mut model {
            AnalogModel::Harmonics(model) => model.set_anti_aliasing(mode),
            AnalogModel::Static(model) => model.set_anti_aliasing(mode),
            AnalogModel::Hammerstein(model) => model.set_anti_aliasing(mode),
            AnalogModel::Tape(model) => model.set_anti_aliasing(mode),
            AnalogModel::Transformer(model) => model.set_anti_aliasing(mode),
            AnalogModel::ConsolePreamp(model) => model.set_anti_aliasing(mode),
            // Component models (IDs 6-8) expose no in-crate ADAA option:
            // implicit Newton solves have no closed-form antiderivative and
            // the tone stack is linear. Alias control is documented host
            // oversampling (see references/component-references.md); the
            // characterization test below records their finite behavior.
            AnalogModel::DiodeClipper(_)
            | AnalogModel::TriodeStage(_)
            | AnalogModel::ToneStack(_) => {}
        }
        match &mut model {
            AnalogModel::Harmonics(model) => {
                model.set_drive_db(24.0).unwrap();
                model.set_h2_db(0.0).unwrap();
                model.set_h3_db(-6.0).unwrap();
            }
            AnalogModel::Static(model) => model.set_drive_db(24.0).unwrap(),
            AnalogModel::Hammerstein(model) => model.set_drive_db(24.0).unwrap(),
            AnalogModel::Tape(model) => model.set_drive_db(24.0).unwrap(),
            AnalogModel::Transformer(model) => model.set_drive_db(24.0).unwrap(),
            AnalogModel::ConsolePreamp(model) => model.set_input_gain_db(24.0).unwrap(),
            AnalogModel::DiodeClipper(model) => model.set_drive_db(24.0).unwrap(),
            AnalogModel::TriodeStage(model) => model.set_drive_db(24.0).unwrap(),
            AnalogModel::ToneStack(model) => {
                model.set_treble(0.9).unwrap();
            }
        }
        model
            .prepare(ProcessSpec::new(sample_rate, 1, record_length))
            .unwrap();
        let mut samples: Vec<f32> = (0..record_length)
            .map(|index| (TAU * frequency * index as f32 / sample_rate).sin() * 0.8)
            .collect();
        model
            .process_interleaved(&mut samples, record_length)
            .unwrap();
        measure_harmonics(&samples, sample_rate, frequency, 5)
            .unwrap()
            .distortion(&samples)
            .unwrap()
            .alias_rms
    };

    // The 50% folded-energy guard stays scoped to families 0-5 (the models
    // it was pre-registered for in the Phase B exit criterion). Component
    // models carry no in-crate ADAA path by design; see the
    // characterization test below and references/component-references.md.
    for model_id in 0..=AnalogModel::CONSOLE_PREAMP_ID {
        let off = render(model_id, AntiAliasing::Off);
        let adaa = render(model_id, AntiAliasing::Adaa1);
        assert!(
            off.is_finite() && adaa.is_finite(),
            "model {model_id} was non-finite"
        );
        assert!(
            adaa < off * 0.5,
            "model {model_id} failed the provisional folded-energy guard: ADAA={adaa} Off={off}"
        );
    }
}

#[test]
fn component_models_record_finite_alias_characterization() {
    // IDs 6-8 have no in-crate ADAA path (implicit solves / linear): record
    // finite alias reports instead of asserting the folded-energy guard, and
    // prove the linear tone stack aliases at the measurement floor.
    let sample_rate = 48_000.0;
    let frequency = 10_000.0;
    let record_length = 4_800;
    let render = |model_id: u32| {
        let mut model = AnalogModel::from_id(model_id).unwrap();
        match &mut model {
            AnalogModel::DiodeClipper(model) => model.set_drive_db(24.0).unwrap(),
            AnalogModel::TriodeStage(model) => model.set_drive_db(24.0).unwrap(),
            AnalogModel::ToneStack(model) => {
                model.set_treble(0.9).unwrap();
            }
            _ => unreachable!("component fixture covers IDs 6-8 only"),
        }
        // Settle first: startup transients leak broadband energy into the
        // alias bins of every model, linear or not. The characterization
        // measures steady-state behavior on the second half.
        model
            .prepare(ProcessSpec::new(sample_rate, 1, 2 * record_length))
            .unwrap();
        let mut samples: Vec<f32> = (0..2 * record_length)
            .map(|index| (TAU * frequency * index as f32 / sample_rate).sin() * 0.8)
            .collect();
        model
            .process_interleaved(&mut samples, 2 * record_length)
            .unwrap();
        assert!(samples.iter().all(|sample| sample.is_finite()));
        let settled = &samples[record_length..];
        measure_harmonics(settled, sample_rate, frequency, 5)
            .unwrap()
            .distortion(settled)
            .unwrap()
            .alias_rms
    };

    let clipper = render(AnalogModel::DIODE_CLIPPER_ID);
    let triode = render(AnalogModel::TRIODE_STAGE_ID);
    let stack = render(AnalogModel::TONE_STACK_ID);
    assert!(clipper.is_finite() && triode.is_finite() && stack.is_finite());
    // The linear stack aliases at the f32 DFT floor (~2e-5 here); the
    // clipping components alias orders of magnitude higher.
    assert!(
        stack < 1e-4,
        "linear tone stack should alias at the floor, got {stack}"
    );
    assert!(
        stack < clipper * 0.01 && stack < triode * 0.01,
        "stack {stack} not well below clipper {clipper} / triode {triode}"
    );
    println!("component alias_rms: clipper={clipper} triode={triode} stack={stack}");
}

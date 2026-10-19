//! Shared driven-signal fixtures for callback contracts.

// Rust guideline compliant 2026-02-21
use math_audio_analog::AnalogModel;

pub fn drive_model(model: &mut AnalogModel, drive_db: f32) {
    match model {
        AnalogModel::Harmonics(model) => {
            model.set_drive_db(drive_db).unwrap();
            model.set_h2_db(-12.0).unwrap();
            model.set_h3_db(-18.0).unwrap();
        }
        AnalogModel::Static(model) => model.set_drive_db(drive_db).unwrap(),
        AnalogModel::Hammerstein(model) => model.set_drive_db(drive_db).unwrap(),
        AnalogModel::Tape(model) => model.set_drive_db(drive_db).unwrap(),
        AnalogModel::Transformer(model) => model.set_drive_db(drive_db).unwrap(),
        AnalogModel::ConsolePreamp(model) => model.set_input_gain_db(drive_db).unwrap(),
        AnalogModel::DiodeClipper(model) => model.set_drive_db(drive_db).unwrap(),
        AnalogModel::TriodeStage(model) => model.set_drive_db(drive_db).unwrap(),
        AnalogModel::ToneStack(model) => model
            .set_treble(if drive_db > 12.0 { 0.9 } else { 0.3 })
            .unwrap(),
    }
}

pub fn fill_input(block: &mut [f32], channels: usize, iteration: usize) {
    for (index, sample) in block.iter_mut().enumerate() {
        *sample = if iteration == 7 {
            f32::MIN_POSITIVE * 0.25
        } else {
            (((index / channels + iteration * 256) as f32) * 0.173).sin() * 0.85
        };
    }
}

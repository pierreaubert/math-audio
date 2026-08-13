use math_audio_analog::analysis::measure_harmonics;
use math_audio_analog::{AnalogProcessor, AntiAliasing, HarmonicModel, ProcessSpec};
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

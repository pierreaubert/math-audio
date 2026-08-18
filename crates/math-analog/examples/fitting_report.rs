//! Print a deterministic feature-gated fit-quality report.
//!
//! This is synthetic evidence for the fitting machinery.  It is not a
//! hardware fit, listening result, or claim that the recovered coefficients
//! describe a real device.

#[cfg(not(feature = "fitting"))]
fn main() {
    eprintln!(
        "fitting_report requires: cargo run -p math-analog --features fitting --example fitting_report"
    );
}

#[cfg(feature = "fitting")]
mod enabled {
    use math_audio_analog::analysis::{measure_harmonics, measure_transient, measure_two_tone_imd};
    use math_audio_analog::fitting::{FitCapture, FitDataset, FitOptions, fit_hammerstein};
    use math_audio_analog::{AnalogProcessor, HammersteinBranch, HammersteinModel, ProcessSpec};
    use std::f32::consts::TAU;

    const SAMPLE_RATE: f32 = 48_000.0;
    const FRAMES: usize = 2_048;

    pub fn run() {
        let source = HammersteinModel::with_branches(&[
            HammersteinBranch::new(1, 0.92, 0.0).unwrap(),
            HammersteinBranch::new(2, 0.08, 1_800.0).unwrap(),
            HammersteinBranch::new(3, 0.04, 6_000.0).unwrap(),
        ])
        .unwrap();

        let fit_sine = capture(&source, sine(1_000.0, 0.24));
        let fit_two_tone = capture(&source, two_tone(700.0, 2_700.0, 0.18, 0.11));
        let fit_transient = capture(&source, transient());
        let held_out_sine = capture(&source, sine(7_000.0, 0.31));
        let held_out_two_tone = capture(&source, two_tone(1_300.0, 5_900.0, 0.22, 0.09));
        let held_out_transient = capture(&source, held_out_transient());

        let dataset = FitDataset::from_captures(
            vec![
                fit_sine.clone(),
                fit_two_tone.clone(),
                fit_transient.clone(),
            ],
            vec![
                held_out_sine.clone(),
                held_out_two_tone.clone(),
                held_out_transient.clone(),
            ],
            SAMPLE_RATE,
        )
        .unwrap();
        let options = FitOptions {
            branch_orders: vec![1, 2, 3],
            de_max_iterations: 24,
            de_population_size: 8,
            lm_max_iterations: 20,
            target_description: "synthetic parallel-Hammerstein fixture".to_string(),
            capture_chain: "generated f32 in-memory loopback".to_string(),
            fit_date: "2026-08-18".to_string(),
            ..FitOptions::default()
        };
        let report = fit_hammerstein(&dataset, &options).unwrap();
        let fitted = report.coefficients.to_model().unwrap();

        println!("sample_rate_hz={SAMPLE_RATE} fit_captures=3 held_out_captures=3");
        println!(
            "quality fit_rms={:.9} held_out_rms={:.9} fit_spectral_rms={:.9} held_out_spectral_rms={:.9} de_objective={:.9} lm_objective={:.9}",
            report.quality.fit_rms,
            report.quality.held_out_rms,
            report.quality.fit_spectral_rms,
            report.quality.held_out_spectral_rms,
            report.quality.de_objective,
            report.quality.lm_objective,
        );
        for branch in report.coefficients.branches() {
            println!(
                "fitted_branch order={} gain={:.9} cutoff_hz={:.9}",
                branch.order, branch.gain, branch.cutoff_hz
            );
        }
        println!(
            "capture_hash={}",
            report.coefficients.provenance().capture_hash
        );

        harmonic_row("fit", "sine", &fit_sine, &fitted, 1_000.0);
        harmonic_row("held_out", "sine", &held_out_sine, &fitted, 7_000.0);
        imd_row("fit", &fit_two_tone, &fitted, 700.0, 2_700.0);
        imd_row("held_out", &held_out_two_tone, &fitted, 1_300.0, 5_900.0);
        transient_row("fit", &fit_transient, &fitted);
        transient_row("held_out", &held_out_transient, &fitted);
    }

    fn capture(model: &HammersteinModel, stimulus: Vec<f32>) -> FitCapture {
        let mut model = model.clone();
        model
            .prepare(ProcessSpec::new(SAMPLE_RATE, 1, stimulus.len()))
            .unwrap();
        let mut response = stimulus.clone();
        model
            .process_interleaved(&mut response, stimulus.len())
            .unwrap();
        FitCapture::new(stimulus, response).unwrap()
    }

    fn render(model: &HammersteinModel, capture: &FitCapture) -> Vec<f32> {
        let mut model = model.clone();
        model
            .prepare(ProcessSpec::new(SAMPLE_RATE, 1, capture.len()))
            .unwrap();
        let mut response = capture.stimulus().to_vec();
        model
            .process_interleaved(&mut response, capture.len())
            .unwrap();
        response
    }

    fn sine(frequency_hz: f32, amplitude: f32) -> Vec<f32> {
        (0..FRAMES)
            .map(|index| amplitude * (TAU * frequency_hz * index as f32 / SAMPLE_RATE).sin())
            .collect()
    }

    fn two_tone(
        frequency_a: f32,
        frequency_b: f32,
        amplitude_a: f32,
        amplitude_b: f32,
    ) -> Vec<f32> {
        (0..FRAMES)
            .map(|index| {
                let n = index as f32 / SAMPLE_RATE;
                amplitude_a * (TAU * frequency_a * n).sin()
                    + amplitude_b * (TAU * frequency_b * n).sin()
            })
            .collect()
    }

    fn transient() -> Vec<f32> {
        (0..FRAMES)
            .map(|index| {
                let decay = (-(index as f32) / 240.0).exp();
                if index == 0 {
                    0.9
                } else {
                    0.3 * decay * (index as f32 * 0.31).sin()
                }
            })
            .collect()
    }

    fn held_out_transient() -> Vec<f32> {
        (0..FRAMES)
            .map(|index| {
                let decay = (-(index as f32) / 410.0).exp();
                if index == 0 {
                    -0.75
                } else {
                    0.22 * decay * (index as f32 * 0.47).sin()
                }
            })
            .collect()
    }

    fn harmonic_row(
        split: &str,
        kind: &str,
        capture: &FitCapture,
        fitted: &HammersteinModel,
        fundamental_hz: f32,
    ) {
        let expected =
            measure_harmonics(capture.response(), SAMPLE_RATE, fundamental_hz, 5).unwrap();
        let actual =
            measure_harmonics(&render(fitted, capture), SAMPLE_RATE, fundamental_hz, 5).unwrap();
        println!(
            "{split}_{kind}_harmonics ref_h1={:.9} model_h1={:.9} ref_h2={:.9} model_h2={:.9} ref_h3={:.9} model_h3={:.9} ref_h4={:.9} model_h4={:.9} ref_h5={:.9} model_h5={:.9}",
            expected.component(1).unwrap().amplitude,
            actual.component(1).unwrap().amplitude,
            expected.component(2).unwrap().amplitude,
            actual.component(2).unwrap().amplitude,
            expected.component(3).unwrap().amplitude,
            actual.component(3).unwrap().amplitude,
            expected.component(4).unwrap().amplitude,
            actual.component(4).unwrap().amplitude,
            expected.component(5).unwrap().amplitude,
            actual.component(5).unwrap().amplitude,
        );
    }

    fn imd_row(
        split: &str,
        capture: &FitCapture,
        fitted: &HammersteinModel,
        frequency_a: f32,
        frequency_b: f32,
    ) {
        let expected =
            measure_two_tone_imd(capture.response(), SAMPLE_RATE, frequency_a, frequency_b, 3)
                .unwrap();
        let actual = measure_two_tone_imd(
            &render(fitted, capture),
            SAMPLE_RATE,
            frequency_a,
            frequency_b,
            3,
        )
        .unwrap();
        println!(
            "{split}_two_tone_imd ref_tone_a={:.9} model_tone_a={:.9} ref_tone_b={:.9} model_tone_b={:.9} ref_2f1_f2={:.9} model_2f1_f2={:.9} ref_2f2_f1={:.9} model_2f2_f1={:.9}",
            expected.tone_a_amplitude,
            actual.tone_a_amplitude,
            expected.tone_b_amplitude,
            actual.tone_b_amplitude,
            expected.component(2, -1).unwrap().amplitude,
            actual.component(2, -1).unwrap().amplitude,
            expected.component(1, -2).unwrap().amplitude,
            actual.component(1, -2).unwrap().amplitude,
        );
    }

    fn transient_row(split: &str, capture: &FitCapture, fitted: &HammersteinModel) {
        let expected = measure_transient(capture.response(), SAMPLE_RATE).unwrap();
        let actual = measure_transient(&render(fitted, capture), SAMPLE_RATE).unwrap();
        println!(
            "{split}_transient ref_peak={:.9} model_peak={:.9} ref_rms={:.9} model_rms={:.9} ref_dc={:.9} model_dc={:.9}",
            expected.peak_amplitude,
            actual.peak_amplitude,
            expected.rms_amplitude,
            actual.rms_amplitude,
            expected.dc_amplitude,
            actual.dc_amplitude,
        );
    }
}

#[cfg(feature = "fitting")]
fn main() {
    enabled::run();
}

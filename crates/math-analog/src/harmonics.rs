use math_audio_dsp::adaa::{Adaa1, adaa1_tanh};

use crate::chain::DcBlocker;
use crate::level::{calibrated_input_gain, db_to_gain};
use crate::process::{
    AnalogError, AnalogProcessor, ControlSmoother, ProcessSpec, checked_block_len, finite_output,
    sanitize_sample, validate_finite_range,
};

const CONTROL_SMOOTHING_MS: f32 = 10.0;
const MIN_DRIVE_DB: f32 = -60.0;
const MAX_DRIVE_DB: f32 = 36.0;
const MIN_HARMONIC_DB: f32 = -120.0;
const MAX_HARMONIC_DB: f32 = 12.0;
const MIN_OUTPUT_GAIN_DB: f32 = -60.0;
const MAX_OUTPUT_GAIN_DB: f32 = 24.0;

/// Antialiasing strategy for memoryless branches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntiAliasing {
    /// Evaluate the bounded curve directly.
    Off,
    /// Reuse math-dsp's first-order antiderivative processor.
    Adaa1,
}

/// A cheap, explainable second/third-harmonic coloration baseline.
///
/// For bounded input `u = tanh(drive * x)`, the model is:
///
/// ```text
/// v = u + h2 * (2*u² - 1) + h3 * (4*u³ - 3*u)
/// ```
///
/// `h2` and `h3` are linear branch strengths represented by dB controls
/// relative to the fundamental.  They are not independent spectral
/// oscillators: on programme material these branches also create DC and
/// intermodulation products.  The final DC blocker removes the generated
/// offset and is present in both antialiasing modes.
#[derive(Debug)]
pub struct HarmonicModel {
    drive_db: ControlSmoother,
    h2_db: ControlSmoother,
    h3_db: ControlSmoother,
    output_gain_db: ControlSmoother,
    character: ControlSmoother,
    amount: ControlSmoother,
    mix: ControlSmoother,
    anti_aliasing: AntiAliasing,
    spec: Option<ProcessSpec>,
    channels: Vec<HarmonicChannelState>,
}

impl Default for HarmonicModel {
    fn default() -> Self {
        Self::new()
    }
}

impl HarmonicModel {
    /// Create a baseline with unity drive, muted harmonic branches, and full
    /// wet amount.  Call [`AnalogProcessor::prepare`] before processing.
    pub fn new() -> Self {
        let sample_rate = 48_000.0;
        Self {
            drive_db: ControlSmoother::new(0.0, CONTROL_SMOOTHING_MS, sample_rate),
            h2_db: ControlSmoother::new(MIN_HARMONIC_DB, CONTROL_SMOOTHING_MS, sample_rate),
            h3_db: ControlSmoother::new(MIN_HARMONIC_DB, CONTROL_SMOOTHING_MS, sample_rate),
            output_gain_db: ControlSmoother::new(0.0, CONTROL_SMOOTHING_MS, sample_rate),
            character: ControlSmoother::new(0.5, CONTROL_SMOOTHING_MS, sample_rate),
            amount: ControlSmoother::new(1.0, CONTROL_SMOOTHING_MS, sample_rate),
            mix: ControlSmoother::new(1.0, CONTROL_SMOOTHING_MS, sample_rate),
            anti_aliasing: AntiAliasing::Adaa1,
            spec: None,
            channels: Vec::new(),
        }
    }

    pub fn drive_db(&self) -> f32 {
        self.drive_db.target()
    }

    pub fn h2_db(&self) -> f32 {
        self.h2_db.target()
    }

    pub fn h3_db(&self) -> f32 {
        self.h3_db.target()
    }

    pub fn output_gain_db(&self) -> f32 {
        self.output_gain_db.target()
    }

    /// Return the model-specific character macro.  `0.5` is neutral; moving
    /// away from it introduces a bounded input bias and therefore controlled
    /// asymmetry before the Chebyshev branches.
    pub fn character(&self) -> f32 {
        self.character.target()
    }

    pub fn amount(&self) -> f32 {
        self.amount.target()
    }

    pub fn mix(&self) -> f32 {
        self.mix.target()
    }

    pub fn anti_aliasing(&self) -> AntiAliasing {
        self.anti_aliasing
    }

    pub fn set_drive_db(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("drive_db", value, MIN_DRIVE_DB, MAX_DRIVE_DB)?;
        self.drive_db.set_target(value);
        Ok(())
    }

    pub fn set_h2_db(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("h2_db", value, MIN_HARMONIC_DB, MAX_HARMONIC_DB)?;
        self.h2_db.set_target(value);
        Ok(())
    }

    pub fn set_h3_db(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("h3_db", value, MIN_HARMONIC_DB, MAX_HARMONIC_DB)?;
        self.h3_db.set_target(value);
        Ok(())
    }

    pub fn set_output_gain_db(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range(
            "output_gain_db",
            value,
            MIN_OUTPUT_GAIN_DB,
            MAX_OUTPUT_GAIN_DB,
        )?;
        self.output_gain_db.set_target(value);
        Ok(())
    }

    pub fn set_character(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("character", value, 0.0, 1.0)?;
        self.character.set_target(value);
        Ok(())
    }

    pub fn set_amount(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("amount", value, 0.0, 1.0)?;
        if value == 0.0 {
            self.amount.set_immediate(value);
        } else {
            self.amount.set_target(value);
        }
        Ok(())
    }

    pub fn set_mix(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("mix", value, 0.0, 1.0)?;
        if value == 0.0 {
            self.mix.set_immediate(value);
        } else {
            self.mix.set_target(value);
        }
        Ok(())
    }

    /// Set the memoryless antialiasing mode.  The model has no added latency
    /// in either mode; host-owned oversampling remains an integration concern.
    /// Changing this structural mode clears the ADAA history so that a mode
    /// transition cannot reuse samples evaluated by the previous path.
    pub fn set_anti_aliasing(&mut self, mode: AntiAliasing) {
        if self.anti_aliasing != mode {
            for channel in &mut self.channels {
                channel.reset();
            }
        }
        self.anti_aliasing = mode;
    }

    fn process_frame(&mut self, frame: &mut [f32], channels: usize, anti_aliasing: AntiAliasing) {
        let drive = calibrated_input_gain(self.drive_db.advance());
        let h2 = db_to_gain(self.h2_db.advance());
        let h3 = db_to_gain(self.h3_db.advance());
        let output_gain = db_to_gain(self.output_gain_db.advance());
        let character_bias = (self.character.advance() - 0.5) * 0.8;
        let amount = self.amount.advance();
        let mix = self.mix.advance();
        let controls = HarmonicProcessControls {
            drive,
            h2,
            h3,
            output_gain,
            character_bias,
            anti_aliasing,
        };

        for (channel, sample) in frame.iter_mut().enumerate().take(channels) {
            let dry = sanitize_sample(*sample);
            let shaped = self.channels[channel].process(dry, controls);
            let effected = dry + amount * (shaped - dry);
            *sample = finite_output(dry + mix * (effected - dry));
        }
    }
}

impl AnalogProcessor for HarmonicModel {
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), AnalogError> {
        spec.validate()?;
        let mut channels = Vec::with_capacity(spec.channels);
        for _ in 0..spec.channels {
            channels.push(HarmonicChannelState::new(spec.sample_rate));
        }

        self.drive_db.reconfigure(spec.sample_rate);
        self.h2_db.reconfigure(spec.sample_rate);
        self.h3_db.reconfigure(spec.sample_rate);
        self.output_gain_db.reconfigure(spec.sample_rate);
        self.character.reconfigure(spec.sample_rate);
        self.amount.reconfigure(spec.sample_rate);
        self.mix.reconfigure(spec.sample_rate);
        self.channels = channels;
        self.spec = Some(spec);
        self.reset();
        Ok(())
    }

    fn reset(&mut self) {
        self.drive_db.reset();
        self.h2_db.reset();
        self.h3_db.reset();
        self.output_gain_db.reset();
        self.character.reset();
        self.amount.reset();
        self.mix.reset();
        for channel in &mut self.channels {
            channel.reset();
        }
    }

    fn process_interleaved(
        &mut self,
        samples: &mut [f32],
        frames: usize,
    ) -> Result<(), AnalogError> {
        let spec = self.spec.ok_or(AnalogError::NotPrepared)?;
        checked_block_len(spec, frames, samples.len())?;
        let anti_aliasing = self.anti_aliasing;
        for frame in samples.chunks_exact_mut(spec.channels).take(frames) {
            self.process_frame(frame, spec.channels, anti_aliasing);
        }
        Ok(())
    }

    fn latency_samples(&self) -> usize {
        0
    }
}

struct HarmonicChannelState {
    base: Adaa1,
    second: Adaa1,
    third: Adaa1,
    dc_blocker: DcBlocker,
}

#[derive(Clone, Copy)]
struct HarmonicProcessControls {
    drive: f32,
    h2: f32,
    h3: f32,
    output_gain: f32,
    character_bias: f32,
    anti_aliasing: AntiAliasing,
}

impl std::fmt::Debug for HarmonicChannelState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HarmonicChannelState")
            .finish_non_exhaustive()
    }
}

impl HarmonicChannelState {
    fn new(sample_rate: f32) -> Self {
        Self {
            base: adaa1_tanh(),
            second: Adaa1::new(second_harmonic, second_harmonic_ad1),
            third: Adaa1::new(third_harmonic, third_harmonic_ad1),
            dc_blocker: DcBlocker::new(sample_rate),
        }
    }

    #[inline]
    fn process(&mut self, input: f32, controls: HarmonicProcessControls) -> f32 {
        let driven = input * controls.drive + controls.character_bias;
        let (base, second, third) = match controls.anti_aliasing {
            AntiAliasing::Off => {
                let bounded = (driven as f64).tanh();
                (
                    bounded as f32,
                    second_harmonic(driven as f64) as f32,
                    third_harmonic(driven as f64) as f32,
                )
            }
            AntiAliasing::Adaa1 => (
                self.base.process(driven),
                self.second.process(driven),
                self.third.process(driven),
            ),
        };
        let shaped = base + controls.h2 * second + controls.h3 * third;
        let shaped = self.dc_blocker.process(shaped) * controls.output_gain;
        finite_output(shaped)
    }

    fn reset(&mut self) {
        self.base.reset();
        self.second.reset();
        self.third.reset();
        self.dc_blocker.reset();
    }
}

#[inline]
fn second_harmonic(x: f64) -> f64 {
    let bounded = x.tanh();
    2.0 * bounded * bounded - 1.0
}

#[inline]
fn second_harmonic_ad1(x: f64) -> f64 {
    x - 2.0 * x.tanh()
}

#[inline]
fn third_harmonic(x: f64) -> f64 {
    let bounded = x.tanh();
    4.0 * bounded * bounded * bounded - 3.0 * bounded
}

#[inline]
fn third_harmonic_ad1(x: f64) -> f64 {
    let bounded = x.tanh();
    let abs_x = x.abs();
    let tanh_ad1 = abs_x + (-2.0 * abs_x).exp().ln_1p() - std::f64::consts::LN_2;
    tanh_ad1 - 2.0 * bounded * bounded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prepared() -> HarmonicModel {
        let mut model = HarmonicModel::new();
        model
            .prepare(ProcessSpec {
                sample_rate: 48_000.0,
                channels: 2,
                max_block_frames: 128,
            })
            .expect("valid process spec");
        model
    }

    #[test]
    fn rejects_invalid_controls_without_mutation() {
        let mut model = HarmonicModel::new();
        assert!(model.set_h2_db(f32::NAN).is_err());
        assert_eq!(model.h2_db(), MIN_HARMONIC_DB);
        assert!(model.set_amount(2.0).is_err());
        assert_eq!(model.amount(), 1.0);
    }

    #[test]
    fn nonfinite_input_is_replaced_and_state_recovers() {
        let mut model = prepared();
        let mut samples = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.25];
        model.process_interleaved(&mut samples, 2).unwrap();
        assert!(samples.iter().all(|sample| sample.is_finite()));
        let mut silence = [0.0; 2];
        model.process_interleaved(&mut silence, 1).unwrap();
        assert!(silence.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn reset_matches_a_fresh_instance() {
        let mut model = HarmonicModel::new();
        model.set_h2_db(-12.0).unwrap();
        model.set_h3_db(-18.0).unwrap();
        model
            .prepare(ProcessSpec {
                sample_rate: 48_000.0,
                channels: 2,
                max_block_frames: 128,
            })
            .unwrap();
        let mut warmup = [0.7; 8];
        model.process_interleaved(&mut warmup, 4).unwrap();
        model.reset();

        let mut fresh = HarmonicModel::new();
        fresh.set_h2_db(-12.0).unwrap();
        fresh.set_h3_db(-18.0).unwrap();
        fresh
            .prepare(ProcessSpec {
                sample_rate: 48_000.0,
                channels: 2,
                max_block_frames: 128,
            })
            .unwrap();
        let mut expected = [0.2; 2];
        let mut actual = [0.2; 2];
        fresh.process_interleaved(&mut expected, 1).unwrap();
        model.process_interleaved(&mut actual, 1).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn dc_blocker_removes_constant_h2_offset() {
        let mut model = HarmonicModel::new();
        model
            .prepare(ProcessSpec {
                sample_rate: 48_000.0,
                channels: 1,
                max_block_frames: 128,
            })
            .unwrap();
        model.set_drive_db(-60.0).unwrap();
        model.set_h2_db(0.0).unwrap();
        let mut samples = vec![0.0; 48_000];
        for block in samples.chunks_exact_mut(128) {
            model.process_interleaved(block, 128).unwrap();
        }
        assert!(samples[47_000].abs() < 0.05);
    }

    #[test]
    fn adaa_and_direct_modes_are_finite() {
        let mut model = prepared();
        for mode in [AntiAliasing::Off, AntiAliasing::Adaa1] {
            model.set_anti_aliasing(mode);
            let mut samples = vec![0.0; 128 * 2];
            for (index, sample) in samples.iter_mut().enumerate() {
                *sample = ((index as f32) * 0.37).sin() * 8.0;
            }
            model.process_interleaved(&mut samples, 128).unwrap();
            assert!(samples.iter().all(|sample| sample.is_finite()));
        }
    }

    #[test]
    fn chebyshev_branches_have_the_declared_single_tone_identity() {
        for phase in [0.13_f64, 0.71, 1.37, 2.43, 4.91] {
            let cosine = phase.cos();
            let input = cosine.atanh();
            assert!((second_harmonic(input) - (2.0 * phase).cos()).abs() < 1e-12);
            assert!((third_harmonic(input) - (3.0 * phase).cos()).abs() < 1e-12);
        }
    }

    #[test]
    fn harmonic_antiderivatives_have_the_expected_derivatives() {
        let step = 1e-6;
        for x in [-3.0, -0.7, 0.2, 1.1, 4.0] {
            let second_derivative =
                (second_harmonic_ad1(x + step) - second_harmonic_ad1(x - step)) / (2.0 * step);
            let third_derivative =
                (third_harmonic_ad1(x + step) - third_harmonic_ad1(x - step)) / (2.0 * step);
            assert!((second_derivative - second_harmonic(x)).abs() < 1e-7);
            assert!((third_derivative - third_harmonic(x)).abs() < 1e-7);
        }
    }

    #[test]
    fn changing_antialiasing_mode_clears_history() {
        let mut transitioned = prepared();
        let mut warmup = [0.75_f32, -0.5];
        transitioned.process_interleaved(&mut warmup, 1).unwrap();
        transitioned.set_anti_aliasing(AntiAliasing::Off);
        transitioned.set_anti_aliasing(AntiAliasing::Adaa1);

        let mut fresh = prepared();
        let mut actual = [0.25_f32, -0.125];
        let mut expected = actual;
        transitioned.process_interleaved(&mut actual, 1).unwrap();
        fresh.process_interleaved(&mut expected, 1).unwrap();
        assert_eq!(actual, expected);
    }
}

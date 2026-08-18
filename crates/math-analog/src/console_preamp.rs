use math_audio_dsp::adaa::{Adaa2, adaa1_tanh, adaa2_tanh};
use math_audio_iir_fir::{Biquad, BiquadFilterType};

use crate::chain::DcBlocker;
use crate::effects::{DefectConfig, DefectState};
use crate::harmonics::AntiAliasing;
use crate::level::{calibrated_input_gain, db_to_gain};
use crate::process::{
    AnalogError, AnalogProcessor, ControlSmoother, ProcessSpec, checked_block_len, finite_output,
    flush_denormal, sanitize_sample, validate_finite_range,
};

const CONTROL_SMOOTHING_MS: f32 = 10.0;
const MAX_DEFECT_DELAY_SAMPLES: usize = 64;

/// A bounded, stylized Wiener–Hammerstein console/preamp model.
///
/// The structure is deliberately explicit: a coupling high-pass, an
/// asymmetric tanh core with envelope-dependent compression, and a bounded
/// output pole.  The default coefficients are synthetic and are not a claim
/// about a named console or preamp.  A fitted coefficient container can be
/// applied by an offline fitting workflow once a target capture exists.
#[derive(Debug)]
pub struct ConsolePreampModel {
    input_gain_db: ControlSmoother,
    output_gain_db: ControlSmoother,
    output_trim_db: ControlSmoother,
    compression: ControlSmoother,
    asymmetry: ControlSmoother,
    amount: ControlSmoother,
    mix: ControlSmoother,
    anti_aliasing: AntiAliasing,
    defects: DefectConfig,
    spec: Option<ProcessSpec>,
    channels: Vec<ConsoleChannelState>,
}

impl Default for ConsolePreampModel {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsolePreampModel {
    pub fn new() -> Self {
        let sample_rate = 48_000.0;
        Self {
            input_gain_db: ControlSmoother::new(0.0, CONTROL_SMOOTHING_MS, sample_rate),
            output_gain_db: ControlSmoother::new(0.0, CONTROL_SMOOTHING_MS, sample_rate),
            output_trim_db: ControlSmoother::new(-1.5, CONTROL_SMOOTHING_MS, sample_rate),
            compression: ControlSmoother::new(0.35, CONTROL_SMOOTHING_MS, sample_rate),
            asymmetry: ControlSmoother::new(0.5, CONTROL_SMOOTHING_MS, sample_rate),
            amount: ControlSmoother::new(1.0, CONTROL_SMOOTHING_MS, sample_rate),
            mix: ControlSmoother::new(1.0, CONTROL_SMOOTHING_MS, sample_rate),
            anti_aliasing: AntiAliasing::Adaa1,
            defects: DefectConfig::default(),
            spec: None,
            channels: Vec::new(),
        }
    }

    pub fn input_gain_db(&self) -> f32 {
        self.input_gain_db.target()
    }

    pub fn output_gain_db(&self) -> f32 {
        self.output_gain_db.target()
    }

    pub fn output_trim_db(&self) -> f32 {
        self.output_trim_db.target()
    }

    pub fn compression(&self) -> f32 {
        self.compression.target()
    }

    pub fn asymmetry(&self) -> f32 {
        self.asymmetry.target()
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

    pub fn defects(&self) -> DefectConfig {
        self.defects
    }

    pub fn set_anti_aliasing(&mut self, mode: AntiAliasing) {
        if self.anti_aliasing != mode {
            for channel in &mut self.channels {
                channel.reset();
            }
        }
        self.anti_aliasing = mode;
    }

    pub fn set_defects(&mut self, defects: DefectConfig) -> Result<(), AnalogError> {
        defects
            .validate(MAX_DEFECT_DELAY_SAMPLES)
            .map_err(|message| AnalogError::InvalidDefectConfig(message.to_string()))?;
        self.defects = defects;
        if let Some(spec) = self.spec {
            self.prepare(spec)?;
        } else {
            for channel in &mut self.channels {
                channel.reset();
            }
        }
        Ok(())
    }

    pub fn set_input_gain_db(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("input_gain_db", value, -60.0, 36.0)?;
        self.input_gain_db.set_target(value);
        Ok(())
    }

    pub fn set_output_gain_db(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("output_gain_db", value, -60.0, 24.0)?;
        self.output_gain_db.set_target(value);
        Ok(())
    }

    pub fn set_output_trim_db(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("output_trim_db", value, -24.0, 24.0)?;
        self.output_trim_db.set_target(value);
        Ok(())
    }

    pub fn set_compression(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("compression", value, 0.0, 1.0)?;
        self.compression.set_target(value);
        Ok(())
    }

    pub fn set_asymmetry(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("asymmetry", value, 0.0, 1.0)?;
        self.asymmetry.set_target(value);
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

    fn process_frame(&mut self, frame: &mut [f32], channels: usize) {
        let input_gain = calibrated_input_gain(self.input_gain_db.advance());
        let output_gain = db_to_gain(self.output_gain_db.advance());
        let output_trim = db_to_gain(self.output_trim_db.advance());
        let compression = self.compression.advance();
        let asymmetry = self.asymmetry.advance();
        let amount = self.amount.advance();
        let mix = self.mix.advance();
        let input_sum = frame
            .iter()
            .take(channels)
            .map(|sample| sanitize_sample(*sample))
            .sum::<f32>();
        for (channel, sample) in frame.iter_mut().enumerate().take(channels) {
            let dry = sanitize_sample(*sample);
            let crosstalk_input = if channels > 1 {
                (input_sum - dry) / (channels - 1) as f32
            } else {
                0.0
            };
            let shaped = self.channels[channel].process(
                dry,
                input_gain,
                output_gain * output_trim,
                compression,
                asymmetry,
                self.anti_aliasing,
                crosstalk_input,
            );
            let effected = dry + amount * (shaped - dry);
            *sample = finite_output(dry + mix * (effected - dry));
        }
    }
}

impl AnalogProcessor for ConsolePreampModel {
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), AnalogError> {
        spec.validate()?;
        let channels = (0..spec.channels)
            .map(|_| ConsoleChannelState::new(spec.sample_rate, self.defects))
            .collect();
        self.input_gain_db.reconfigure(spec.sample_rate);
        self.output_gain_db.reconfigure(spec.sample_rate);
        self.output_trim_db.reconfigure(spec.sample_rate);
        self.compression.reconfigure(spec.sample_rate);
        self.asymmetry.reconfigure(spec.sample_rate);
        self.amount.reconfigure(spec.sample_rate);
        self.mix.reconfigure(spec.sample_rate);
        self.channels = channels;
        self.spec = Some(spec);
        self.reset();
        Ok(())
    }

    fn reset(&mut self) {
        self.input_gain_db.reset();
        self.output_gain_db.reset();
        self.output_trim_db.reset();
        self.compression.reset();
        self.asymmetry.reset();
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
        for frame in samples.chunks_exact_mut(spec.channels).take(frames) {
            self.process_frame(frame, spec.channels);
        }
        Ok(())
    }

    fn latency_samples(&self) -> usize {
        0
    }
}

#[derive(Debug)]
struct ConsoleChannelState {
    pre_filter: Biquad<f32>,
    output_filter: Biquad<f32>,
    nonlinear: math_audio_dsp::adaa::Adaa1,
    nonlinear_second: Adaa2,
    envelope: f32,
    attack_coefficient: f32,
    release_coefficient: f32,
    defects: DefectState,
    dc_blocker: DcBlocker,
}

impl ConsoleChannelState {
    fn new(sample_rate: f32, defects: DefectConfig) -> Self {
        Self {
            pre_filter: Biquad::new(BiquadFilterType::Highpass, 35.0, sample_rate, 0.707, 0.0),
            output_filter: Biquad::new(
                BiquadFilterType::Lowpass,
                (sample_rate * 0.35).min(18_000.0),
                sample_rate,
                0.707,
                0.0,
            ),
            nonlinear: adaa1_tanh(),
            nonlinear_second: adaa2_tanh(),
            envelope: 0.0,
            attack_coefficient: time_coefficient(5.0, sample_rate),
            release_coefficient: time_coefficient(80.0, sample_rate),
            defects: DefectState::new(defects, sample_rate, MAX_DEFECT_DELAY_SAMPLES),
            dc_blocker: DcBlocker::new(sample_rate),
        }
    }

    #[inline]
    fn process(
        &mut self,
        input: f32,
        input_gain: f32,
        output_gain: f32,
        compression: f32,
        asymmetry: f32,
        anti_aliasing: AntiAliasing,
        crosstalk_input: f32,
    ) -> f32 {
        let filtered = self.pre_filter.process(input);
        let bias = (asymmetry - 0.5) * 0.32;
        let driven = filtered * input_gain + bias;
        let nonlinear = match anti_aliasing {
            AntiAliasing::Off => driven.tanh(),
            AntiAliasing::Adaa1 => self.nonlinear.process(driven),
            AntiAliasing::Adaa2 => self.nonlinear_second.process(driven),
        };
        let coefficient = if nonlinear.abs() > self.envelope {
            self.attack_coefficient
        } else {
            self.release_coefficient
        };
        self.envelope = coefficient * self.envelope + (1.0 - coefficient) * nonlinear.abs();
        self.envelope = flush_denormal(self.envelope);
        let compression_gain = 1.0 / (1.0 + compression * (self.envelope - 0.05).max(0.0));
        let asymmetric_gain = if nonlinear >= 0.0 {
            1.0 + 0.08 * asymmetry
        } else {
            1.0 - 0.08 * asymmetry
        };
        let output = self
            .output_filter
            .process(nonlinear * compression_gain * asymmetric_gain);
        let output = self.defects.process(output, crosstalk_input);
        finite_output(self.dc_blocker.process(output) * output_gain)
    }

    fn reset(&mut self) {
        self.pre_filter.reset();
        self.output_filter.reset();
        self.nonlinear.reset();
        self.nonlinear_second.reset();
        self.envelope = 0.0;
        self.defects.reset();
        self.dc_blocker.reset();
    }
}

#[inline]
fn time_coefficient(time_ms: f32, sample_rate: f32) -> f32 {
    (-1.0 / (time_ms * 0.001 * sample_rate)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AnalogModel;

    #[test]
    fn console_model_is_finite_and_resettable() {
        let mut model = ConsolePreampModel::new();
        model.prepare(ProcessSpec::new(48_000.0, 2, 128)).unwrap();
        let mut buffer = vec![16.0_f32; 256];
        model.process_interleaved(&mut buffer, 128).unwrap();
        assert!(buffer.iter().all(|sample| sample.is_finite()));
        model.reset();
        let mut actual = [0.25_f32, -0.25];
        model.process_interleaved(&mut actual, 1).unwrap();
        assert!(actual.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn console_model_exposes_a_distinct_append_only_id() {
        let model = AnalogModel::ConsolePreamp(ConsolePreampModel::new());
        assert_eq!(model.model_id(), AnalogModel::CONSOLE_PREAMP_ID);
        assert!(matches!(
            AnalogModel::from_id(AnalogModel::CONSOLE_PREAMP_ID),
            Ok(AnalogModel::ConsolePreamp(_))
        ));
    }
}

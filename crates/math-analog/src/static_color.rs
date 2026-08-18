use math_audio_dsp::adaa::{
    Adaa1, Adaa2, adaa1_hardclip, adaa1_softclip, adaa1_tanh, adaa2_hardclip, adaa2_softclip,
    adaa2_tanh,
};
use math_audio_dsp::simd::{hardclip_static_simd, softclip_static_simd};

use crate::chain::DcBlocker;
use crate::level::{calibrated_input_gain, db_to_gain};
use crate::process::{
    AnalogError, AnalogProcessor, ControlSmoother, ProcessSpec, checked_block_len, finite_output,
    sanitize_sample, validate_finite_range,
};

const CONTROL_SMOOTHING_MS: f32 = 10.0;

/// A named mathematical curve, not a claim about a physical circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaticCurve {
    /// Odd-symmetric `tanh(x)` saturation.
    TanhStyle,
    /// Odd-symmetric `x / (1 + |x|)` soft clipping.
    SoftClipStyle,
    /// Odd-symmetric clamp to `[-1, 1]`.
    HardClipStyle,
}

/// A bounded memoryless waveshaper with optional ADAA and level controls.
///
/// These curves have no hysteresis, frequency-dependent memory, or hardware
/// provenance.  The `Style` suffix is intentional: it prevents a convenient
/// mathematical curve from being mistaken for a tube, tape, or transformer
/// emulation.
#[derive(Debug)]
pub struct StaticColorModel {
    curve: StaticCurve,
    drive_db: ControlSmoother,
    output_gain_db: ControlSmoother,
    character: ControlSmoother,
    amount: ControlSmoother,
    mix: ControlSmoother,
    anti_aliasing: crate::harmonics::AntiAliasing,
    spec: Option<ProcessSpec>,
    channels: Vec<StaticChannelState>,
}

impl Default for StaticColorModel {
    fn default() -> Self {
        Self::new(StaticCurve::TanhStyle)
    }
}

impl StaticColorModel {
    pub fn new(curve: StaticCurve) -> Self {
        let sample_rate = 48_000.0;
        Self {
            curve,
            drive_db: ControlSmoother::new(0.0, CONTROL_SMOOTHING_MS, sample_rate),
            output_gain_db: ControlSmoother::new(0.0, CONTROL_SMOOTHING_MS, sample_rate),
            character: ControlSmoother::new(0.5, CONTROL_SMOOTHING_MS, sample_rate),
            amount: ControlSmoother::new(1.0, CONTROL_SMOOTHING_MS, sample_rate),
            mix: ControlSmoother::new(1.0, CONTROL_SMOOTHING_MS, sample_rate),
            anti_aliasing: crate::harmonics::AntiAliasing::Adaa1,
            spec: None,
            channels: Vec::new(),
        }
    }

    pub fn curve(&self) -> StaticCurve {
        self.curve
    }

    pub fn drive_db(&self) -> f32 {
        self.drive_db.target()
    }

    pub fn output_gain_db(&self) -> f32 {
        self.output_gain_db.target()
    }

    /// Return the model-specific character macro.  `0.5` is neutral; moving
    /// away from it introduces a bounded bias before the selected curve.
    pub fn character(&self) -> f32 {
        self.character.target()
    }

    pub fn amount(&self) -> f32 {
        self.amount.target()
    }

    pub fn mix(&self) -> f32 {
        self.mix.target()
    }

    pub fn set_curve(&mut self, curve: StaticCurve) {
        if self.curve != curve {
            for channel in &mut self.channels {
                channel.reset();
            }
        }
        self.curve = curve;
    }

    pub fn set_anti_aliasing(&mut self, mode: crate::harmonics::AntiAliasing) {
        if self.anti_aliasing != mode {
            for channel in &mut self.channels {
                channel.reset();
            }
        }
        self.anti_aliasing = mode;
    }

    pub fn set_drive_db(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("drive_db", value, -60.0, 36.0)?;
        self.drive_db.set_target(value);
        Ok(())
    }

    pub fn set_output_gain_db(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("output_gain_db", value, -60.0, 24.0)?;
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

    fn process_frame(
        &mut self,
        frame: &mut [f32],
        channels: usize,
        anti_aliasing: crate::harmonics::AntiAliasing,
    ) {
        let drive = calibrated_input_gain(self.drive_db.advance());
        let output_gain = db_to_gain(self.output_gain_db.advance());
        let character_bias = (self.character.advance() - 0.5) * 0.8;
        let amount = self.amount.advance();
        let mix = self.mix.advance();

        for (channel, sample) in frame.iter_mut().enumerate().take(channels) {
            let dry = sanitize_sample(*sample);
            let shaped = self.channels[channel].process(
                dry,
                drive,
                output_gain,
                self.curve,
                character_bias,
                anti_aliasing,
            );
            let effected = dry + amount * (shaped - dry);
            *sample = finite_output(dry + mix * (effected - dry));
        }
    }

    fn controls_are_settled(&self) -> bool {
        self.drive_db.is_settled()
            && self.output_gain_db.is_settled()
            && self.character.is_settled()
            && self.amount.is_settled()
            && self.mix.is_settled()
    }

    fn process_off_batch(&mut self, samples: &mut [f32], spec: ProcessSpec, frames: usize) {
        let drive = calibrated_input_gain(self.drive_db.advance());
        let output_gain = db_to_gain(self.output_gain_db.advance());
        let character_bias = (self.character.advance() - 0.5) * 0.8;
        let amount = self.amount.advance();
        let mix = self.mix.advance();
        for channel in 0..spec.channels {
            let state = &mut self.channels[channel];
            for frame in 0..frames {
                let index = frame * spec.channels + channel;
                state.simd_input[frame] = sanitize_sample(samples[index]) * drive + character_bias;
            }
            match self.curve {
                StaticCurve::TanhStyle => {
                    for frame in 0..frames {
                        state.simd_output[frame] = state.simd_input[frame].tanh();
                    }
                }
                StaticCurve::SoftClipStyle => {
                    softclip_static_simd(
                        &state.simd_input[..frames],
                        &mut state.simd_output[..frames],
                    );
                }
                StaticCurve::HardClipStyle => {
                    hardclip_static_simd(
                        &state.simd_input[..frames],
                        &mut state.simd_output[..frames],
                    );
                }
            }
            for frame in 0..frames {
                let index = frame * spec.channels + channel;
                let dry = sanitize_sample(samples[index]);
                let shaped =
                    finite_output(state.dc_blocker.process(state.simd_output[frame]) * output_gain);
                let effected = dry + amount * (shaped - dry);
                samples[index] = finite_output(dry + mix * (effected - dry));
            }
        }
    }
}

impl AnalogProcessor for StaticColorModel {
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), AnalogError> {
        spec.validate()?;
        let mut channels = Vec::with_capacity(spec.channels);
        for _ in 0..spec.channels {
            channels.push(StaticChannelState::new(
                spec.sample_rate,
                spec.max_block_frames,
            ));
        }
        self.drive_db.reconfigure(spec.sample_rate);
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
        if frames > 0
            && anti_aliasing == crate::harmonics::AntiAliasing::Off
            && self.controls_are_settled()
        {
            self.process_off_batch(samples, spec, frames);
            return Ok(());
        }
        for frame in samples.chunks_exact_mut(spec.channels).take(frames) {
            self.process_frame(frame, spec.channels, anti_aliasing);
        }
        Ok(())
    }

    fn latency_samples(&self) -> usize {
        0
    }
}

struct StaticChannelState {
    tanh: Adaa1,
    softclip: Adaa1,
    hardclip: Adaa1,
    tanh_second: Adaa2,
    softclip_second: Adaa2,
    hardclip_second: Adaa2,
    dc_blocker: DcBlocker,
    simd_input: Vec<f32>,
    simd_output: Vec<f32>,
}

impl std::fmt::Debug for StaticChannelState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StaticChannelState")
            .finish_non_exhaustive()
    }
}

impl StaticChannelState {
    fn new(sample_rate: f32, max_block_frames: usize) -> Self {
        Self {
            tanh: adaa1_tanh(),
            softclip: adaa1_softclip(),
            hardclip: adaa1_hardclip(),
            tanh_second: adaa2_tanh(),
            softclip_second: adaa2_softclip(),
            hardclip_second: adaa2_hardclip(),
            dc_blocker: DcBlocker::new(sample_rate),
            simd_input: vec![0.0; max_block_frames],
            simd_output: vec![0.0; max_block_frames],
        }
    }

    #[inline]
    fn process(
        &mut self,
        input: f32,
        drive: f32,
        output_gain: f32,
        curve: StaticCurve,
        character_bias: f32,
        anti_aliasing: crate::harmonics::AntiAliasing,
    ) -> f32 {
        let driven = input * drive + character_bias;
        let output = match (curve, anti_aliasing) {
            (StaticCurve::TanhStyle, crate::harmonics::AntiAliasing::Adaa1) => {
                self.tanh.process(driven)
            }
            (StaticCurve::SoftClipStyle, crate::harmonics::AntiAliasing::Adaa1) => {
                self.softclip.process(driven)
            }
            (StaticCurve::HardClipStyle, crate::harmonics::AntiAliasing::Adaa1) => {
                self.hardclip.process(driven)
            }
            (StaticCurve::TanhStyle, crate::harmonics::AntiAliasing::Adaa2) => {
                self.tanh_second.process(driven)
            }
            (StaticCurve::SoftClipStyle, crate::harmonics::AntiAliasing::Adaa2) => {
                self.softclip_second.process(driven)
            }
            (StaticCurve::HardClipStyle, crate::harmonics::AntiAliasing::Adaa2) => {
                self.hardclip_second.process(driven)
            }
            (StaticCurve::TanhStyle, crate::harmonics::AntiAliasing::Off) => {
                (driven as f64).tanh() as f32
            }
            (StaticCurve::SoftClipStyle, crate::harmonics::AntiAliasing::Off) => {
                let x = driven as f64;
                (x / (1.0 + x.abs())) as f32
            }
            (StaticCurve::HardClipStyle, crate::harmonics::AntiAliasing::Off) => {
                driven.clamp(-1.0, 1.0)
            }
        };
        finite_output(self.dc_blocker.process(output) * output_gain)
    }

    fn reset(&mut self) {
        self.tanh.reset();
        self.softclip.reset();
        self.hardclip.reset();
        self.tanh_second.reset();
        self.softclip_second.reset();
        self.hardclip_second.reset();
        self.dc_blocker.reset();
        self.simd_input.fill(0.0);
        self.simd_output.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prepared(curve: StaticCurve) -> StaticColorModel {
        let mut model = StaticColorModel::new(curve);
        model
            .prepare(ProcessSpec {
                sample_rate: 48_000.0,
                channels: 1,
                max_block_frames: 64,
            })
            .unwrap();
        model
    }

    #[test]
    fn all_curves_remain_finite_for_extreme_finite_input() {
        for curve in [
            StaticCurve::TanhStyle,
            StaticCurve::SoftClipStyle,
            StaticCurve::HardClipStyle,
        ] {
            let mut model = prepared(curve);
            let mut buffer = [f32::MAX; 4];
            model.process_interleaved(&mut buffer, 4).unwrap();
            assert!(buffer.iter().all(|sample| sample.is_finite()));
        }
    }

    #[test]
    fn amount_zero_is_aligned_dry_output() {
        let mut model = prepared(StaticCurve::TanhStyle);
        model.set_amount(0.0).unwrap();
        let mut buffer = [0.25, -0.5, 0.75, -1.0];
        let original = buffer;
        model.process_interleaved(&mut buffer, 4).unwrap();
        assert_eq!(buffer, original);
    }

    #[test]
    fn changing_curve_clears_history() {
        let mut transitioned = prepared(StaticCurve::TanhStyle);
        let mut warmup = [0.75_f32];
        transitioned.process_interleaved(&mut warmup, 1).unwrap();
        transitioned.set_curve(StaticCurve::SoftClipStyle);

        let mut fresh = prepared(StaticCurve::SoftClipStyle);
        let mut actual = [0.25_f32];
        let mut expected = actual;
        transitioned.process_interleaved(&mut actual, 1).unwrap();
        fresh.process_interleaved(&mut expected, 1).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn changing_antialiasing_mode_clears_history() {
        let mut transitioned = prepared(StaticCurve::TanhStyle);
        let mut warmup = [0.75_f32];
        transitioned.process_interleaved(&mut warmup, 1).unwrap();
        transitioned.set_anti_aliasing(crate::harmonics::AntiAliasing::Off);

        let mut fresh = prepared(StaticCurve::TanhStyle);
        fresh.set_anti_aliasing(crate::harmonics::AntiAliasing::Off);
        let mut actual = [0.25_f32];
        let mut expected = actual;
        transitioned.process_interleaved(&mut actual, 1).unwrap();
        fresh.process_interleaved(&mut expected, 1).unwrap();
        assert_eq!(actual, expected);
    }
}

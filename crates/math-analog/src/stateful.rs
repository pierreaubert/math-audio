use math_audio_dsp::adaa::{Adaa1, Adaa2, adaa1_tanh, adaa2_tanh};

use crate::chain::{DcBlocker, OnePoleLowpass};
use crate::effects::{
    DefectConfig, DefectState, HysteresisMode, HysteresisState, TapeEqCurve, TapeEqState,
};
use crate::harmonics::AntiAliasing;
use crate::level::{calibrated_input_gain, db_to_gain};
use crate::process::{
    AnalogError, AnalogProcessor, ControlSmoother, ProcessSpec, checked_block_len, finite_output,
    flush_denormal, sanitize_sample, validate_finite_range,
};

const CONTROL_SMOOTHING_MS: f32 = 10.0;
const MIN_DRIVE_DB: f32 = -60.0;
const MAX_DRIVE_DB: f32 = 36.0;
const MIN_OUTPUT_GAIN_DB: f32 = -60.0;
const MAX_OUTPUT_GAIN_DB: f32 = 24.0;
const MAX_DEFECT_DELAY_SAMPLES: usize = 64;

/// Transformer state equation choice. `Stylized` preserves the original
/// preset meaning; `Flux` enables the bounded voltage-integrated state used by
/// new fitted-capable models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformerMode {
    Stylized,
    Flux,
}

#[derive(Debug)]
struct StatefulControls {
    drive_db: ControlSmoother,
    output_gain_db: ControlSmoother,
    amount: ControlSmoother,
    mix: ControlSmoother,
    character: ControlSmoother,
}

#[derive(Debug, Clone, Copy)]
struct ControlFrame {
    drive: f32,
    output_gain: f32,
    amount: f32,
    mix: f32,
    character: f32,
}

impl StatefulControls {
    fn new(sample_rate: f32) -> Self {
        Self {
            drive_db: ControlSmoother::new(0.0, CONTROL_SMOOTHING_MS, sample_rate),
            output_gain_db: ControlSmoother::new(0.0, CONTROL_SMOOTHING_MS, sample_rate),
            amount: ControlSmoother::new(1.0, CONTROL_SMOOTHING_MS, sample_rate),
            mix: ControlSmoother::new(1.0, CONTROL_SMOOTHING_MS, sample_rate),
            character: ControlSmoother::new(0.5, CONTROL_SMOOTHING_MS, sample_rate),
        }
    }

    fn reconfigure(&mut self, sample_rate: f32) {
        self.drive_db.reconfigure(sample_rate);
        self.output_gain_db.reconfigure(sample_rate);
        self.amount.reconfigure(sample_rate);
        self.mix.reconfigure(sample_rate);
        self.character.reconfigure(sample_rate);
    }

    fn advance(&mut self) -> ControlFrame {
        ControlFrame {
            drive: calibrated_input_gain(self.drive_db.advance()),
            output_gain: db_to_gain(self.output_gain_db.advance()),
            amount: self.amount.advance(),
            mix: self.mix.advance(),
            character: self.character.advance(),
        }
    }

    fn reset(&mut self) {
        self.drive_db.reset();
        self.output_gain_db.reset();
        self.amount.reset();
        self.mix.reset();
        self.character.reset();
    }

    fn drive_db(&self) -> f32 {
        self.drive_db.target()
    }

    fn output_gain_db(&self) -> f32 {
        self.output_gain_db.target()
    }

    fn amount(&self) -> f32 {
        self.amount.target()
    }

    fn mix(&self) -> f32 {
        self.mix.target()
    }

    fn character(&self) -> f32 {
        self.character.target()
    }

    fn set_drive_db(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("drive_db", value, MIN_DRIVE_DB, MAX_DRIVE_DB)?;
        self.drive_db.set_target(value);
        Ok(())
    }

    fn set_output_gain_db(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range(
            "output_gain_db",
            value,
            MIN_OUTPUT_GAIN_DB,
            MAX_OUTPUT_GAIN_DB,
        )?;
        self.output_gain_db.set_target(value);
        Ok(())
    }

    fn set_amount(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("amount", value, 0.0, 1.0)?;
        if value == 0.0 {
            self.amount.set_immediate(value);
        } else {
            self.amount.set_target(value);
        }
        Ok(())
    }

    fn set_mix(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("mix", value, 0.0, 1.0)?;
        if value == 0.0 {
            self.mix.set_immediate(value);
        } else {
            self.mix.set_target(value);
        }
        Ok(())
    }

    fn set_character(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("character", value, 0.0, 1.0)?;
        self.character.set_target(value);
        Ok(())
    }
}

/// A bounded stylized tape-memory model, not a tape-machine emulation.
///
/// The documented target is a slow operating-point memory:
///
/// ```text
/// e[n] = a_e e[n-1] + (1-a_e)|u[n]|
/// m[n] = a_m m[n-1] + (1-a_m)u[n]
/// v[n] = tanh(u[n] + 0.25 * character * e[n] * m[n])
/// ```
///
/// `e` and `m` are independent per-channel states with 8 ms and 80 ms default
/// time constants.  Both are configurable for a prepared capture.  The
/// equations are intentionally stylized; optional hysteresis/EQ/defect modules
/// do not imply a measured-device claim.
#[derive(Debug)]
pub struct TapeModel {
    controls: StatefulControls,
    anti_aliasing: AntiAliasing,
    hysteresis_mode: HysteresisMode,
    hysteresis_amount: f32,
    head_bump_amount: f32,
    head_bump_hz: f32,
    hf_loss_amount: f32,
    envelope_time_ms: f32,
    memory_time_ms: f32,
    tape_eq: Option<TapeEqCurve>,
    defects: DefectConfig,
    spec: Option<ProcessSpec>,
    channels: Vec<TapeChannelState>,
}

impl Default for TapeModel {
    fn default() -> Self {
        Self::new()
    }
}

impl TapeModel {
    pub fn new() -> Self {
        Self {
            controls: StatefulControls::new(48_000.0),
            anti_aliasing: AntiAliasing::Adaa1,
            hysteresis_mode: HysteresisMode::Off,
            hysteresis_amount: 0.0,
            head_bump_amount: 0.0,
            head_bump_hz: 80.0,
            hf_loss_amount: 0.0,
            envelope_time_ms: 8.0,
            memory_time_ms: 80.0,
            tape_eq: None,
            defects: DefectConfig::default(),
            spec: None,
            channels: Vec::new(),
        }
    }

    pub fn drive_db(&self) -> f32 {
        self.controls.drive_db()
    }

    pub fn output_gain_db(&self) -> f32 {
        self.controls.output_gain_db()
    }

    pub fn amount(&self) -> f32 {
        self.controls.amount()
    }

    pub fn mix(&self) -> f32 {
        self.controls.mix()
    }

    pub fn character(&self) -> f32 {
        self.controls.character()
    }

    pub fn anti_aliasing(&self) -> AntiAliasing {
        self.anti_aliasing
    }

    pub fn envelope_time_ms(&self) -> f32 {
        self.envelope_time_ms
    }

    pub fn memory_time_ms(&self) -> f32 {
        self.memory_time_ms
    }

    pub fn hysteresis_mode(&self) -> HysteresisMode {
        self.hysteresis_mode
    }
    pub fn hysteresis_amount(&self) -> f32 {
        self.hysteresis_amount
    }
    pub fn head_bump_amount(&self) -> f32 {
        self.head_bump_amount
    }
    pub fn head_bump_hz(&self) -> f32 {
        self.head_bump_hz
    }
    pub fn hf_loss_amount(&self) -> f32 {
        self.hf_loss_amount
    }

    pub fn tape_eq_curve(&self) -> Option<&TapeEqCurve> {
        self.tape_eq.as_ref()
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

    pub fn set_envelope_time_ms(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("envelope_time_ms", value, 0.1, 2_000.0)?;
        self.envelope_time_ms = value;
        Ok(())
    }

    pub fn set_memory_time_ms(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("memory_time_ms", value, 0.1, 5_000.0)?;
        self.memory_time_ms = value;
        Ok(())
    }

    pub fn set_hysteresis(&mut self, mode: HysteresisMode, amount: f32) -> Result<(), AnalogError> {
        validate_finite_range("hysteresis_amount", amount, 0.0, 1.0)?;
        self.hysteresis_mode = mode;
        self.hysteresis_amount = amount;
        Ok(())
    }

    pub fn set_head_bump(&mut self, amount: f32, frequency_hz: f32) -> Result<(), AnalogError> {
        validate_finite_range("head_bump_amount", amount, 0.0, 1.0)?;
        validate_finite_range("head_bump_hz", frequency_hz, 5.0, 1_000.0)?;
        self.head_bump_amount = amount;
        self.head_bump_hz = frequency_hz;
        Ok(())
    }

    pub fn set_hf_loss_amount(&mut self, amount: f32) -> Result<(), AnalogError> {
        validate_finite_range("hf_loss_amount", amount, 0.0, 1.0)?;
        self.hf_loss_amount = amount;
        Ok(())
    }

    /// Install an optional record/replay EQ curve.  It is reduced to bounded
    /// shelves when the model is next prepared.
    pub fn set_tape_eq_curve(&mut self, curve: Option<TapeEqCurve>) -> Result<(), AnalogError> {
        self.tape_eq = curve;
        if let Some(spec) = self.spec {
            self.prepare(spec)?;
        } else {
            for channel in &mut self.channels {
                channel.reset();
            }
        }
        Ok(())
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

    pub fn set_drive_db(&mut self, value: f32) -> Result<(), AnalogError> {
        self.controls.set_drive_db(value)
    }

    pub fn set_output_gain_db(&mut self, value: f32) -> Result<(), AnalogError> {
        self.controls.set_output_gain_db(value)
    }

    pub fn set_amount(&mut self, value: f32) -> Result<(), AnalogError> {
        self.controls.set_amount(value)
    }

    pub fn set_mix(&mut self, value: f32) -> Result<(), AnalogError> {
        self.controls.set_mix(value)
    }

    pub fn set_character(&mut self, value: f32) -> Result<(), AnalogError> {
        self.controls.set_character(value)
    }

    fn process_frame(&mut self, frame: &mut [f32], channels: usize, anti_aliasing: AntiAliasing) {
        let controls = self.controls.advance();
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
            let shaped =
                self.channels[channel].process(dry, controls, anti_aliasing, crosstalk_input);
            let effected = dry + controls.amount * (shaped - dry);
            *sample = finite_output(dry + controls.mix * (effected - dry));
        }
    }
}

impl AnalogProcessor for TapeModel {
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), AnalogError> {
        spec.validate()?;
        let channels = (0..spec.channels)
            .map(|_| {
                TapeChannelState::new(
                    spec.sample_rate,
                    self.envelope_time_ms,
                    self.memory_time_ms,
                    self.hysteresis_mode,
                    self.hysteresis_amount,
                    self.head_bump_amount,
                    self.head_bump_hz,
                    self.hf_loss_amount,
                    self.tape_eq.as_ref(),
                    self.defects,
                )
            })
            .collect();
        self.controls.reconfigure(spec.sample_rate);
        self.channels = channels;
        self.spec = Some(spec);
        self.reset();
        Ok(())
    }

    fn reset(&mut self) {
        self.controls.reset();
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

#[derive(Debug)]
struct TapeChannelState {
    envelope: f32,
    memory: f32,
    envelope_coefficient: f32,
    memory_coefficient: f32,
    static_tanh: Adaa1,
    static_tanh_second: Adaa2,
    hysteresis: HysteresisState,
    head_bump: OnePoleLowpass,
    hf_loss: OnePoleLowpass,
    head_bump_amount: f32,
    hf_loss_amount: f32,
    eq: TapeEqState,
    defects: DefectState,
    dc_blocker: DcBlocker,
}

impl TapeChannelState {
    fn new(
        sample_rate: f32,
        envelope_time_ms: f32,
        memory_time_ms: f32,
        hysteresis_mode: HysteresisMode,
        hysteresis_amount: f32,
        head_bump_amount: f32,
        head_bump_hz: f32,
        hf_loss_amount: f32,
        tape_eq: Option<&TapeEqCurve>,
        defects: DefectConfig,
    ) -> Self {
        Self {
            envelope: 0.0,
            memory: 0.0,
            envelope_coefficient: time_coefficient(envelope_time_ms, sample_rate),
            memory_coefficient: time_coefficient(memory_time_ms, sample_rate),
            static_tanh: adaa1_tanh(),
            static_tanh_second: adaa2_tanh(),
            hysteresis: HysteresisState::new(hysteresis_mode, 0.12, hysteresis_amount),
            head_bump: OnePoleLowpass::new(head_bump_hz, sample_rate),
            hf_loss: OnePoleLowpass::new((sample_rate * 0.2).min(12_000.0), sample_rate),
            head_bump_amount,
            hf_loss_amount,
            eq: TapeEqState::new(tape_eq, sample_rate),
            defects: DefectState::new(defects, sample_rate, MAX_DEFECT_DELAY_SAMPLES),
            dc_blocker: DcBlocker::new(sample_rate),
        }
    }

    #[inline]
    fn process(
        &mut self,
        input: f32,
        controls: ControlFrame,
        anti_aliasing: AntiAliasing,
        crosstalk_input: f32,
    ) -> f32 {
        let driven = input * controls.drive;
        let mut u = match anti_aliasing {
            AntiAliasing::Off => driven.tanh(),
            AntiAliasing::Adaa1 => self.static_tanh.process(driven),
            AntiAliasing::Adaa2 => self.static_tanh_second.process(driven),
        };
        u = self.hysteresis.process(u);
        self.envelope =
            self.envelope_coefficient * self.envelope + (1.0 - self.envelope_coefficient) * u.abs();
        self.memory = self.memory_coefficient * self.memory + (1.0 - self.memory_coefficient) * u;
        self.envelope = flush_denormal(self.envelope);
        self.memory = flush_denormal(self.memory);
        let operating_point = 0.25 * controls.character * self.envelope * self.memory;
        let mut output = (u + operating_point).tanh();
        let bump = self.head_bump.process(output);
        output += self.head_bump_amount * self.envelope * 0.25 * bump;
        let high_frequency_component = output - self.hf_loss.process(output);
        let level_dependent_hf_loss = self.hf_loss_amount * self.envelope.clamp(0.0, 1.0);
        output -= level_dependent_hf_loss * high_frequency_component;
        output = self.eq.process(output);
        output = self.defects.process(output, crosstalk_input);
        finite_output(self.dc_blocker.process(output) * controls.output_gain)
    }

    fn reset(&mut self) {
        self.envelope = 0.0;
        self.memory = 0.0;
        self.static_tanh.reset();
        self.static_tanh_second.reset();
        self.hysteresis.reset();
        self.head_bump.reset();
        self.hf_loss.reset();
        self.eq.reset();
        self.defects.reset();
        self.dc_blocker.reset();
    }
}

/// A bounded stylized transformer-flux model, not a transformer emulation.
///
/// The stylized target uses a leaky flux state and a saturating magnetization
/// branch.  `Flux` mode replaces the old direct clamped state update with a
/// bounded integrated flux followed by a smooth finite fallback:
///
/// ```text
/// f[n] = clamp(a_f f[n-1] + (1-a_f)u[n], -4, 4)
/// q[n] = tanh((1 + 2*character)f[n])
/// v[n] = tanh(0.7u[n] + 0.3q[n] + 0.15*character*(u[n]-f[n]))
/// ```
///
/// This provides a deterministic low-frequency state target with bounded
/// response.  It is not fitted to a schematic or reference transformer.
#[derive(Debug)]
pub struct TransformerModel {
    controls: StatefulControls,
    anti_aliasing: AntiAliasing,
    mode: TransformerMode,
    defects: DefectConfig,
    spec: Option<ProcessSpec>,
    channels: Vec<TransformerChannelState>,
}

impl Default for TransformerModel {
    fn default() -> Self {
        Self::new()
    }
}

impl TransformerModel {
    pub fn new() -> Self {
        Self {
            controls: StatefulControls::new(48_000.0),
            anti_aliasing: AntiAliasing::Adaa1,
            mode: TransformerMode::Stylized,
            defects: DefectConfig::default(),
            spec: None,
            channels: Vec::new(),
        }
    }

    pub fn drive_db(&self) -> f32 {
        self.controls.drive_db()
    }

    pub fn output_gain_db(&self) -> f32 {
        self.controls.output_gain_db()
    }

    pub fn amount(&self) -> f32 {
        self.controls.amount()
    }

    pub fn mix(&self) -> f32 {
        self.controls.mix()
    }

    pub fn character(&self) -> f32 {
        self.controls.character()
    }

    pub fn anti_aliasing(&self) -> AntiAliasing {
        self.anti_aliasing
    }

    pub fn mode(&self) -> TransformerMode {
        self.mode
    }

    pub fn set_mode(&mut self, mode: TransformerMode) {
        if self.mode != mode {
            for channel in &mut self.channels {
                channel.reset();
            }
        }
        self.mode = mode;
    }

    pub fn set_anti_aliasing(&mut self, mode: AntiAliasing) {
        if self.anti_aliasing != mode {
            for channel in &mut self.channels {
                channel.reset();
            }
        }
        self.anti_aliasing = mode;
    }

    pub fn defects(&self) -> DefectConfig {
        self.defects
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

    pub fn set_drive_db(&mut self, value: f32) -> Result<(), AnalogError> {
        self.controls.set_drive_db(value)
    }

    pub fn set_output_gain_db(&mut self, value: f32) -> Result<(), AnalogError> {
        self.controls.set_output_gain_db(value)
    }

    pub fn set_amount(&mut self, value: f32) -> Result<(), AnalogError> {
        self.controls.set_amount(value)
    }

    pub fn set_mix(&mut self, value: f32) -> Result<(), AnalogError> {
        self.controls.set_mix(value)
    }

    pub fn set_character(&mut self, value: f32) -> Result<(), AnalogError> {
        self.controls.set_character(value)
    }

    fn process_frame(
        &mut self,
        frame: &mut [f32],
        channels: usize,
        anti_aliasing: AntiAliasing,
        mode: TransformerMode,
    ) {
        let controls = self.controls.advance();
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
            let shaped =
                self.channels[channel].process(dry, controls, anti_aliasing, mode, crosstalk_input);
            let effected = dry + controls.amount * (shaped - dry);
            *sample = finite_output(dry + controls.mix * (effected - dry));
        }
    }
}

impl AnalogProcessor for TransformerModel {
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), AnalogError> {
        spec.validate()?;
        let channels = (0..spec.channels)
            .map(|_| TransformerChannelState::new(spec.sample_rate, self.defects))
            .collect();
        self.controls.reconfigure(spec.sample_rate);
        self.channels = channels;
        self.spec = Some(spec);
        self.reset();
        Ok(())
    }

    fn reset(&mut self) {
        self.controls.reset();
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
        let mode = self.mode;
        for frame in samples.chunks_exact_mut(spec.channels).take(frames) {
            self.process_frame(frame, spec.channels, anti_aliasing, mode);
        }
        Ok(())
    }

    fn latency_samples(&self) -> usize {
        0
    }
}

#[derive(Debug)]
struct TransformerChannelState {
    flux: f32,
    flux_coefficient: f32,
    static_tanh: Adaa1,
    static_tanh_second: Adaa2,
    defects: DefectState,
    dc_blocker: DcBlocker,
}

impl TransformerChannelState {
    fn new(sample_rate: f32, defects: DefectConfig) -> Self {
        Self {
            flux: 0.0,
            flux_coefficient: time_coefficient(120.0, sample_rate),
            static_tanh: adaa1_tanh(),
            static_tanh_second: adaa2_tanh(),
            defects: DefectState::new(defects, sample_rate, MAX_DEFECT_DELAY_SAMPLES),
            dc_blocker: DcBlocker::new(sample_rate),
        }
    }

    #[inline]
    fn process(
        &mut self,
        input: f32,
        controls: ControlFrame,
        anti_aliasing: AntiAliasing,
        mode: TransformerMode,
        crosstalk_input: f32,
    ) -> f32 {
        let driven = input * controls.drive;
        let u = match anti_aliasing {
            AntiAliasing::Off => driven.tanh(),
            AntiAliasing::Adaa1 => self.static_tanh.process(driven),
            AntiAliasing::Adaa2 => self.static_tanh_second.process(driven),
        };
        self.flux = match mode {
            TransformerMode::Stylized => {
                self.flux_coefficient * self.flux + (1.0 - self.flux_coefficient) * u
            }
            TransformerMode::Flux => {
                // Voltage-integrated flux.  The smooth tanh saturation keeps
                // the state bounded without making the hard clamp part of the
                // nominal equation; non-finite input is a zero-state fallback.
                let integrated = self.flux + (1.0 - self.flux_coefficient) * u;
                if integrated.is_finite() {
                    4.0 * (integrated / 4.0).tanh()
                } else {
                    0.0
                }
            }
        };
        self.flux = flush_denormal(self.flux.clamp(-4.0, 4.0));
        let magnetization = ((1.0 + 2.0 * controls.character) * self.flux).tanh();
        let output =
            (0.7 * u + 0.3 * magnetization + 0.15 * controls.character * (u - self.flux)).tanh();
        let output = self.defects.process(output, crosstalk_input);
        finite_output(self.dc_blocker.process(output) * controls.output_gain)
    }

    fn reset(&mut self) {
        self.flux = 0.0;
        self.static_tanh.reset();
        self.static_tanh_second.reset();
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

    fn spec() -> ProcessSpec {
        ProcessSpec::new(48_000.0, 2, 128)
    }

    #[test]
    fn stylized_stateful_models_are_finite_and_resettable() {
        for mut model in [
            StatefulModel::Tape(TapeModel::new()),
            StatefulModel::Transformer(TransformerModel::new()),
        ] {
            model.prepare(spec()).unwrap();
            let mut warmup = vec![16.0_f32; 256];
            model.process_interleaved(&mut warmup, 128).unwrap();
            assert!(warmup.iter().all(|sample| sample.is_finite()));
            model.reset();
            let mut silence = [0.0_f32; 2];
            model.process_interleaved(&mut silence, 1).unwrap();
            assert!(silence.iter().all(|sample| sample.is_finite()));
        }
    }

    #[test]
    fn time_constants_are_sample_rate_aware() {
        for time_ms in [8.0, 80.0, 120.0] {
            let at_48k = time_coefficient(time_ms, 48_000.0).powf(48_000.0);
            let at_96k = time_coefficient(time_ms, 96_000.0).powf(96_000.0);
            assert!((at_48k - at_96k).abs() < 1e-6);
        }
    }

    #[test]
    fn time_constants_reach_one_tau_at_the_declared_millisecond_value() {
        for time_ms in [8.0, 80.0, 120.0] {
            let samples = (time_ms * 0.001 * 48_000.0_f32).round() as i32;
            let remaining = time_coefficient(time_ms, 48_000.0).powi(samples);
            let reached = 1.0 - remaining;
            let expected = 1.0 - (-(samples as f32) / (time_ms * 0.001 * 48_000.0_f32)).exp();
            assert!(
                (reached - expected).abs() < 1e-4,
                "time_ms={time_ms} samples={samples} reached={reached} expected={expected}"
            );
        }
    }

    #[test]
    fn stateful_history_is_bounded_and_reset_matches_fresh_state() {
        for mut model in [
            StatefulModel::Tape(TapeModel::new()),
            StatefulModel::Transformer(TransformerModel::new()),
        ] {
            model.prepare(spec()).unwrap();
            let mut saturated = vec![16.0_f32; 2 * 128];
            model.process_interleaved(&mut saturated, 128).unwrap();
            assert!(saturated.iter().all(|sample| sample.abs() <= 128.0));

            let mut after_history = [0.25_f32, -0.25];
            model.process_interleaved(&mut after_history, 1).unwrap();
            assert!(after_history.iter().all(|sample| sample.is_finite()));

            model.reset();
            let mut after_reset = [0.25_f32, -0.25];
            model.process_interleaved(&mut after_reset, 1).unwrap();
            let mut fresh = match model {
                StatefulModel::Tape(_) => StatefulModel::Tape(TapeModel::new()),
                StatefulModel::Transformer(_) => {
                    StatefulModel::Transformer(TransformerModel::new())
                }
            };
            fresh.prepare(spec()).unwrap();
            let mut expected = [0.25_f32, -0.25];
            fresh.process_interleaved(&mut expected, 1).unwrap();
            assert_eq!(after_reset, expected);
        }
    }

    #[test]
    fn tape_eq_and_optional_defects_are_prepared_and_default_off() {
        let curve = TapeEqCurve::new(&[
            crate::effects::TapeEqPoint {
                frequency_hz: 20.0,
                gain_db: 3.0,
            },
            crate::effects::TapeEqPoint {
                frequency_hz: 12_000.0,
                gain_db: -3.0,
            },
        ])
        .unwrap();
        let mut tape = TapeModel::new();
        assert!(tape.tape_eq_curve().is_none());
        tape.set_tape_eq_curve(Some(curve)).unwrap();
        tape.set_head_bump(0.5, 80.0).unwrap();
        tape.prepare(spec()).unwrap();
        let mut samples = vec![0.25_f32; 2 * 128];
        tape.process_interleaved(&mut samples, 128).unwrap();
        assert!(samples.iter().all(|sample| sample.is_finite()));

        let mut transformer = TransformerModel::new();
        transformer
            .set_defects(DefectConfig {
                crosstalk: 0.25,
                noise_floor: 0.001,
                ..DefectConfig::default()
            })
            .unwrap();
        transformer.prepare(spec()).unwrap();
        let mut stereo = vec![0.0_f32; 2 * 128];
        for frame in stereo.chunks_exact_mut(2) {
            frame[0] = 0.25;
            frame[1] = -0.25;
        }
        transformer.process_interleaved(&mut stereo, 128).unwrap();
        assert!(stereo.iter().all(|sample| sample.is_finite()));
        assert_ne!(transformer.defects().crosstalk, 0.0);
    }

    #[derive(Debug)]
    enum StatefulModel {
        Tape(TapeModel),
        Transformer(TransformerModel),
    }

    impl AnalogProcessor for StatefulModel {
        fn prepare(&mut self, spec: ProcessSpec) -> Result<(), AnalogError> {
            match self {
                Self::Tape(model) => model.prepare(spec),
                Self::Transformer(model) => model.prepare(spec),
            }
        }

        fn reset(&mut self) {
            match self {
                Self::Tape(model) => model.reset(),
                Self::Transformer(model) => model.reset(),
            }
        }

        fn process_interleaved(
            &mut self,
            samples: &mut [f32],
            frames: usize,
        ) -> Result<(), AnalogError> {
            match self {
                Self::Tape(model) => model.process_interleaved(samples, frames),
                Self::Transformer(model) => model.process_interleaved(samples, frames),
            }
        }

        fn latency_samples(&self) -> usize {
            0
        }
    }
}

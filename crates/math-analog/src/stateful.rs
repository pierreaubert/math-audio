use crate::chain::DcBlocker;
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
/// `e` and `m` are independent per-channel states with fixed 8 ms and 80 ms
/// time constants.  The equations are intentionally stylized; no hysteresis
/// loop area, tape EQ, bias, or measured-device claim is implied.
#[derive(Debug)]
pub struct TapeModel {
    controls: StatefulControls,
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

    fn process_frame(&mut self, frame: &mut [f32], channels: usize) {
        let controls = self.controls.advance();
        for (channel, sample) in frame.iter_mut().enumerate().take(channels) {
            let dry = sanitize_sample(*sample);
            let shaped = self.channels[channel].process(dry, controls);
            let effected = dry + controls.amount * (shaped - dry);
            *sample = finite_output(dry + controls.mix * (effected - dry));
        }
    }
}

impl AnalogProcessor for TapeModel {
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), AnalogError> {
        spec.validate()?;
        let channels = (0..spec.channels)
            .map(|_| TapeChannelState::new(spec.sample_rate))
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
        for frame in samples.chunks_exact_mut(spec.channels).take(frames) {
            self.process_frame(frame, spec.channels);
        }
        Ok(())
    }

    fn latency_samples(&self) -> usize {
        0
    }
}

#[derive(Debug, Clone, Copy)]
struct TapeChannelState {
    envelope: f32,
    memory: f32,
    envelope_coefficient: f32,
    memory_coefficient: f32,
    dc_blocker: DcBlocker,
}

impl TapeChannelState {
    fn new(sample_rate: f32) -> Self {
        Self {
            envelope: 0.0,
            memory: 0.0,
            envelope_coefficient: time_coefficient(8.0, sample_rate),
            memory_coefficient: time_coefficient(80.0, sample_rate),
            dc_blocker: DcBlocker::new(sample_rate),
        }
    }

    #[inline]
    fn process(&mut self, input: f32, controls: ControlFrame) -> f32 {
        let u = (input * controls.drive).tanh();
        self.envelope =
            self.envelope_coefficient * self.envelope + (1.0 - self.envelope_coefficient) * u.abs();
        self.memory = self.memory_coefficient * self.memory + (1.0 - self.memory_coefficient) * u;
        self.envelope = flush_denormal(self.envelope);
        self.memory = flush_denormal(self.memory);
        let operating_point = 0.25 * controls.character * self.envelope * self.memory;
        let output = (u + operating_point).tanh();
        finite_output(self.dc_blocker.process(output) * controls.output_gain)
    }

    fn reset(&mut self) {
        self.envelope = 0.0;
        self.memory = 0.0;
        self.dc_blocker.reset();
    }
}

/// A bounded stylized transformer-flux model, not a transformer emulation.
///
/// The target uses a leaky flux state and a saturating magnetization branch:
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

    fn process_frame(&mut self, frame: &mut [f32], channels: usize) {
        let controls = self.controls.advance();
        for (channel, sample) in frame.iter_mut().enumerate().take(channels) {
            let dry = sanitize_sample(*sample);
            let shaped = self.channels[channel].process(dry, controls);
            let effected = dry + controls.amount * (shaped - dry);
            *sample = finite_output(dry + controls.mix * (effected - dry));
        }
    }
}

impl AnalogProcessor for TransformerModel {
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), AnalogError> {
        spec.validate()?;
        let channels = (0..spec.channels)
            .map(|_| TransformerChannelState::new(spec.sample_rate))
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
        for frame in samples.chunks_exact_mut(spec.channels).take(frames) {
            self.process_frame(frame, spec.channels);
        }
        Ok(())
    }

    fn latency_samples(&self) -> usize {
        0
    }
}

#[derive(Debug, Clone, Copy)]
struct TransformerChannelState {
    flux: f32,
    flux_coefficient: f32,
    dc_blocker: DcBlocker,
}

impl TransformerChannelState {
    fn new(sample_rate: f32) -> Self {
        Self {
            flux: 0.0,
            flux_coefficient: time_coefficient(120.0, sample_rate),
            dc_blocker: DcBlocker::new(sample_rate),
        }
    }

    #[inline]
    fn process(&mut self, input: f32, controls: ControlFrame) -> f32 {
        let u = (input * controls.drive).tanh();
        self.flux = self.flux_coefficient * self.flux + (1.0 - self.flux_coefficient) * u;
        self.flux = flush_denormal(self.flux.clamp(-4.0, 4.0));
        let magnetization = ((1.0 + 2.0 * controls.character) * self.flux).tanh();
        let output =
            (0.7 * u + 0.3 * magnetization + 0.15 * controls.character * (u - self.flux)).tanh();
        finite_output(self.dc_blocker.process(output) * controls.output_gain)
    }

    fn reset(&mut self) {
        self.flux = 0.0;
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

//! Shockley diode shunt clipper component model (model ID 6).
//!
//! Circuit: input → series `R` → node `Vc` → (shunt `C` to ground in
//! parallel with an antiparallel Shockley diode pair to ground); the output
//! is `Vc`. The capacitor is integrated with the trapezoidal rule and the
//! resulting scalar nonlinear equation is solved per sample with the shared
//! bounded Newton utility. See `references/component-references.md` for the
//! frozen equations, defaults, and pre-registered tolerances.

use crate::chain::DcBlocker;
use crate::effects::{DefectConfig, DefectState, solve_bounded_nonlinear};
use crate::level::{calibrated_input_gain, db_to_gain};
use crate::process::{
    AnalogError, AnalogProcessor, ControlSmoother, ProcessSpec, checked_block_len, finite_output,
    flush_denormal, sanitize_sample, validate_finite_range,
};
use std::f32::consts::TAU;

const CONTROL_SMOOTHING_MS: f32 = 10.0;
const MAX_DEFECT_DELAY_SAMPLES: usize = 64;
/// Series resistance in ohms (documented model default).
const SERIES_RESISTANCE_OHMS: f32 = 10_000.0;
/// Thermal voltage at 300 K in volts.
const THERMAL_VOLTAGE: f32 = 0.02585;
/// Clamp for the diode exponent argument; keeps `exp` finite in f32.
const DIODE_EXPONENT_LIMIT: f32 = 80.0;
/// Newton budget per sample (the shared utility's cap).
///
/// Newton crawls on exponentials (~N*Vt per step from far away), so cold
/// starts and violent transients need dozens of iterations; typical audio
/// converges in under 8. Cost is bounded: 64 capped iterations worst case.
const SOLVER_MAX_ITERATIONS: usize = 64;
/// Newton residual tolerance in volts.
///
/// 100 uV with the diode conducting (slope ~175) pins the node within ~1 uV
/// (-120 dBFS); in the linear region (slope ~2.5) within ~40 uV (-88 dBFS).
/// The tolerance must clear the f32 correction-resolution floor
/// (slope * ulp ~ 1e-5 V at the clip point) with margin, or the shared
/// solver's stall detector trips on rounding noise just short of the root.
const SOLVER_TOLERANCE_V: f32 = 1e-4;
/// Solver rails in volts.
///
/// The diodes nail the node far inside these: with sanitized inputs
/// (<= 16 V peak) and <= +36 dB drive, the source stays under 1010 V, so
/// the silicon solution stays under 0.85 V and germanium under 0.4 V.
/// Tight rails keep escaped iterates where the diode exponential still has
/// slope instead of stranded on the exponent clamp with zero derivative.
const SOLVER_RAIL_V: f32 = 2.0;
/// Initial-guess clamp in volts.
///
/// The guess is the exact linear (diodes-off) node voltage, clamped into
/// the diode conduction zone so cold starts never begin a long exponential
/// crawl from the rails. Exact for small signals, within a few Newton
/// steps everywhere else.
const INITIAL_GUESS_LIMIT_V: f32 = 0.75;
/// Loose acceptance band in volts for non-converged iterates.
///
/// A stalled iterate within 1 mV of the root is inaudible (-60 dBFS) and
/// far closer than holding the previous sample across a fast transient, so
/// it is accepted (and still counted as a fallback). Only a truly lost
/// solve holds the previous bounded state.
const LOOSE_ACCEPT_V: f32 = 1e-3;
const MIN_CORNER_HZ: f32 = 20.0;
const MAX_CORNER_HZ: f32 = 20_000.0;
const DEFAULT_CORNER_HZ: f32 = 10_000.0;

/// Diode parameter flavor. Parameters are illustrative documented defaults
/// reproducing textbook-scale clipping thresholds, not datasheet values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DiodeFlavor {
    /// `Is = 2.0 nA, N = 1.8` (~0.50 V clipping at 100 uA).
    #[default]
    Silicon,
    /// `Is = 0.2 uA, N = 1.0` (~0.16 V clipping at 100 uA).
    Germanium,
}

impl DiodeFlavor {
    /// Saturation current in amperes and ideality factor.
    pub fn params(self) -> (f32, f32) {
        match self {
            Self::Silicon => (2.0e-9, 1.8),
            Self::Germanium => (0.2e-6, 1.0),
        }
    }
}

/// Antiparallel Shockley diode-pair current in amperes.
///
/// `2 * Is * sinh(V / (N * Vt))` with the exponent argument clamped so the
/// evaluation stays finite for any finite node voltage.
#[inline]
fn diode_pair_current(voltage: f32, saturation_current: f32, ideality: f32) -> f32 {
    let argument =
        (voltage / (ideality * THERMAL_VOLTAGE)).clamp(-DIODE_EXPONENT_LIMIT, DIODE_EXPONENT_LIMIT);
    2.0 * saturation_current * argument.sinh()
}

/// A bounded Shockley-diode shunt clipper with trapezoidal capacitor
/// integration and a per-sample Newton solve.
#[derive(Debug)]
pub struct DiodeClipperModel {
    drive_db: ControlSmoother,
    output_gain_db: ControlSmoother,
    amount: ControlSmoother,
    mix: ControlSmoother,
    corner_hz: f32,
    flavor: DiodeFlavor,
    defects: DefectConfig,
    spec: Option<ProcessSpec>,
    channels: Vec<ClipperChannelState>,
}

impl Default for DiodeClipperModel {
    fn default() -> Self {
        Self::new()
    }
}

impl DiodeClipperModel {
    pub fn new() -> Self {
        let sample_rate = 48_000.0;
        Self {
            drive_db: ControlSmoother::new(0.0, CONTROL_SMOOTHING_MS, sample_rate),
            output_gain_db: ControlSmoother::new(0.0, CONTROL_SMOOTHING_MS, sample_rate),
            amount: ControlSmoother::new(1.0, CONTROL_SMOOTHING_MS, sample_rate),
            mix: ControlSmoother::new(1.0, CONTROL_SMOOTHING_MS, sample_rate),
            corner_hz: DEFAULT_CORNER_HZ,
            flavor: DiodeFlavor::default(),
            defects: DefectConfig::default(),
            spec: None,
            channels: Vec::new(),
        }
    }

    pub fn drive_db(&self) -> f32 {
        self.drive_db.target()
    }

    pub fn output_gain_db(&self) -> f32 {
        self.output_gain_db.target()
    }

    pub fn amount(&self) -> f32 {
        self.amount.target()
    }

    pub fn mix(&self) -> f32 {
        self.mix.target()
    }

    pub fn corner_hz(&self) -> f32 {
        self.corner_hz
    }

    pub fn flavor(&self) -> DiodeFlavor {
        self.flavor
    }

    pub fn defects(&self) -> DefectConfig {
        self.defects
    }

    /// Total Newton solves and hold-previous fallbacks across channels.
    ///
    /// Counters are deterministic state: they never affect the output signal
    /// and are cleared by [`AnalogProcessor::reset`].
    pub fn solve_stats(&self) -> (u64, u64) {
        self.channels
            .iter()
            .fold((0_u64, 0_u64), |(solves, fallbacks), channel| {
                (
                    solves.saturating_add(channel.solves),
                    fallbacks.saturating_add(channel.fallbacks),
                )
            })
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

    /// Set the RC corner in Hz. Structural: applies immediately to the
    /// prepared companion models (no smoothing, deterministic).
    pub fn set_corner_hz(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("corner_hz", value, MIN_CORNER_HZ, MAX_CORNER_HZ)?;
        self.corner_hz = value;
        if let Some(spec) = self.spec {
            for channel in &mut self.channels {
                channel.reconfigure(spec.sample_rate, value);
            }
        }
        Ok(())
    }

    /// Set the diode flavor. Structural: applies immediately (no reset;
    /// existing capacitor state stays valid and bounded).
    pub fn set_flavor(&mut self, flavor: DiodeFlavor) {
        self.flavor = flavor;
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

    fn process_frame(&mut self, frame: &mut [f32], channels: usize) {
        let input_gain = calibrated_input_gain(self.drive_db.advance());
        let output_gain = db_to_gain(self.output_gain_db.advance());
        let amount = self.amount.advance();
        let mix = self.mix.advance();
        let (saturation_current, ideality) = self.flavor.params();
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
                output_gain,
                saturation_current,
                ideality,
                crosstalk_input,
            );
            let effected = dry + amount * (shaped - dry);
            *sample = finite_output(dry + mix * (effected - dry));
        }
    }
}

impl AnalogProcessor for DiodeClipperModel {
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), AnalogError> {
        spec.validate()?;
        let channels = (0..spec.channels)
            .map(|_| ClipperChannelState::new(spec.sample_rate, self.corner_hz, self.defects))
            .collect();
        self.drive_db.reconfigure(spec.sample_rate);
        self.output_gain_db.reconfigure(spec.sample_rate);
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
struct ClipperChannelState {
    capacitor_voltage: f32,
    capacitor_current: f32,
    trap_conductance: f32,
    defects: DefectState,
    dc_blocker: DcBlocker,
    solves: u64,
    fallbacks: u64,
}

impl ClipperChannelState {
    fn new(sample_rate: f32, corner_hz: f32, defects: DefectConfig) -> Self {
        let mut state = Self {
            capacitor_voltage: 0.0,
            capacitor_current: 0.0,
            trap_conductance: 0.0,
            defects: DefectState::new(defects, sample_rate, MAX_DEFECT_DELAY_SAMPLES),
            dc_blocker: DcBlocker::new(sample_rate),
            solves: 0,
            fallbacks: 0,
        };
        state.reconfigure(sample_rate, corner_hz);
        state
    }

    fn reconfigure(&mut self, sample_rate: f32, corner_hz: f32) {
        let capacitance = 1.0 / (TAU * SERIES_RESISTANCE_OHMS * corner_hz);
        self.trap_conductance = 2.0 * capacitance * sample_rate;
    }

    /// Solve one capacitor step for a source voltage in volts.
    ///
    /// The KCL residual is scaled by the series resistance so the solver
    /// tolerance is in volts. The initial guess is the exact linear
    /// (diodes-off) node voltage clamped into the conduction zone. A
    /// non-converged iterate inside [`LOOSE_ACCEPT_V`] is accepted (and
    /// counted); only a truly lost solve holds the previous state.
    fn solve_node(&mut self, source: f32, saturation_current: f32, ideality: f32) -> f32 {
        // Trapezoidal companion model: I = Gc * V - Ieq, with Ieq known
        // from the previous step.
        let equivalent_current =
            self.trap_conductance * self.capacitor_voltage + self.capacitor_current;
        let residual = |node: f32| {
            SERIES_RESISTANCE_OHMS
                * ((node - source) / SERIES_RESISTANCE_OHMS + self.trap_conductance * node
                    - equivalent_current
                    + diode_pair_current(node, saturation_current, ideality))
        };
        let linear_estimate = (source / SERIES_RESISTANCE_OHMS + equivalent_current)
            / (1.0 / SERIES_RESISTANCE_OHMS + self.trap_conductance);
        let solve = solve_bounded_nonlinear(
            residual,
            linear_estimate.clamp(-INITIAL_GUESS_LIMIT_V, INITIAL_GUESS_LIMIT_V),
            -SOLVER_RAIL_V,
            SOLVER_RAIL_V,
            SOLVER_MAX_ITERATIONS,
            SOLVER_TOLERANCE_V,
        );
        self.solves = self.solves.saturating_add(1);
        let solved = if solve.converged {
            solve.value
        } else {
            self.fallbacks = self.fallbacks.saturating_add(1);
            if solve.residual.is_finite() && solve.residual.abs() <= LOOSE_ACCEPT_V {
                solve.value
            } else {
                self.capacitor_voltage
            }
        };
        self.capacitor_voltage = flush_denormal(solved);
        self.capacitor_current =
            flush_denormal(self.trap_conductance * self.capacitor_voltage - equivalent_current);
        self.capacitor_voltage
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn process(
        &mut self,
        input: f32,
        input_gain: f32,
        output_gain: f32,
        saturation_current: f32,
        ideality: f32,
        crosstalk_input: f32,
    ) -> f32 {
        // Level convention: 0 dBFS = 1 V peak at the clipper input.
        let source = input * input_gain;
        let solved = self.solve_node(source, saturation_current, ideality);
        let output = self.defects.process(solved, crosstalk_input);
        finite_output(self.dc_blocker.process(output) * output_gain)
    }

    fn reset(&mut self) {
        self.capacitor_voltage = 0.0;
        self.capacitor_current = 0.0;
        self.defects.reset();
        self.dc_blocker.reset();
        self.solves = 0;
        self.fallbacks = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AnalogModel;

    /// Settle one channel's solver with an explicit flavor.
    ///
    /// Uses the real per-sample solver (bypassing only the output DC
    /// blocker, which strips DC by design).
    fn settled_dc_node_flavored(
        channel: &mut ClipperChannelState,
        source: f32,
        flavor: DiodeFlavor,
    ) -> f32 {
        let (saturation_current, ideality) = flavor.params();
        channel.reset();
        let mut node = 0.0;
        for _ in 0..4_800 {
            node = channel.solve_node(source, saturation_current, ideality);
        }
        node
    }

    /// Independent f64 static solution: R current + diode current = 0.
    fn static_solution(input: f64, saturation_current: f64, ideality: f64) -> f64 {
        let mut lo = -32.0_f64;
        let mut hi = 32.0_f64;
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            let residual = (mid - input) / 10_000.0
                + 2.0 * saturation_current * ((mid / (ideality * 0.02585)).sinh());
            // Residual rises monotonically in `mid` (passive circuit).
            if residual > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        0.5 * (lo + hi)
    }

    #[test]
    fn clipper_model_is_finite_and_resettable() {
        let mut model = DiodeClipperModel::new();
        model.prepare(ProcessSpec::new(48_000.0, 2, 128)).unwrap();
        let mut buffer = vec![16.0_f32; 256];
        model.process_interleaved(&mut buffer, 128).unwrap();
        assert!(buffer.iter().all(|sample| sample.is_finite()));
        model.reset();
        assert_eq!(model.solve_stats(), (0, 0));
        let mut actual = [0.25_f32, -0.25];
        model.process_interleaved(&mut actual, 1).unwrap();
        assert!(actual.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn clipper_model_exposes_a_distinct_append_only_id() {
        let model = AnalogModel::DiodeClipper(DiodeClipperModel::new());
        assert_eq!(model.model_id(), AnalogModel::DIODE_CLIPPER_ID);
        assert!(matches!(
            AnalogModel::from_id(AnalogModel::DIODE_CLIPPER_ID),
            Ok(AnalogModel::DiodeClipper(_))
        ));
    }

    #[test]
    fn dc_transfer_matches_the_static_solution_and_is_odd_symmetric() {
        for flavor in [DiodeFlavor::Silicon, DiodeFlavor::Germanium] {
            let (is_at, ideality) = flavor.params();
            let mut channel =
                ClipperChannelState::new(48_000.0, DEFAULT_CORNER_HZ, DefectConfig::default());
            for input in [-2.0_f32, -0.5, -0.1, 0.1, 0.5, 2.0] {
                let output = settled_dc_node_flavored(&mut channel, input, flavor);
                let expected =
                    static_solution(f64::from(input), f64::from(is_at), f64::from(ideality));
                assert!(
                    (f64::from(output) - expected).abs() < 1e-3,
                    "{flavor:?} DC {input}: got {output}, static {expected}"
                );
            }
            let positive = settled_dc_node_flavored(&mut channel, 1.5, flavor);
            let negative = settled_dc_node_flavored(&mut channel, -1.5, flavor);
            assert!(
                (positive + negative).abs() < 1e-4,
                "{flavor:?} odd symmetry: {positive} vs {negative}"
            );
        }
    }

    #[test]
    fn clipping_threshold_matches_the_shockley_prediction() {
        // Deep-clipped 5 V input: the resistor drops ~4.5 V, so the node sits
        // at the Shockley voltage for ~450 uA, computed independently here.
        for flavor in [DiodeFlavor::Silicon, DiodeFlavor::Germanium] {
            let (is_at, ideality) = flavor.params();
            let mut channel =
                ClipperChannelState::new(48_000.0, DEFAULT_CORNER_HZ, DefectConfig::default());
            let output = settled_dc_node_flavored(&mut channel, 5.0, flavor);
            let predicted = static_solution(5.0, f64::from(is_at), f64::from(ideality));
            let relative = ((f64::from(output) - predicted) / predicted).abs();
            assert!(
                relative < 0.05,
                "{flavor:?} threshold: got {output}, predicted {predicted}"
            );
        }
    }

    #[test]
    fn small_signal_gain_is_unity_and_output_is_passive() {
        use crate::analysis::measure_harmonics;
        use std::f32::consts::TAU;
        let mut model = DiodeClipperModel::new();
        model.prepare(ProcessSpec::new(48_000.0, 1, 4_800)).unwrap();
        let frequency = 1_000.0_f32;
        let mut samples: Vec<f32> = (0..4_800)
            .map(|index| 0.001 * (TAU * frequency * index as f32 / 48_000.0).sin())
            .collect();
        let peak_in = samples
            .iter()
            .map(|sample| sample.abs())
            .fold(0.0, f32::max);
        model.process_interleaved(&mut samples, 4_800).unwrap();
        // Skip the startup transient; steady-state gain of a passive clipper
        // cannot exceed unity beyond float noise.
        let peak_out = samples[1_000..]
            .iter()
            .map(|sample| sample.abs())
            .fold(0.0, f32::max);
        assert!(
            peak_out <= peak_in * 1.001,
            "passive clipper gained: {peak_out} > {peak_in}"
        );
        let report = measure_harmonics(&samples, 48_000.0, frequency, 3).unwrap();
        let fundamental = report.component(1).unwrap().amplitude;
        let gain_db = 20.0 * (fundamental / 0.001).log10();
        assert!(
            gain_db.abs() < 0.1,
            "small-signal gain should be unity, got {gain_db} dB"
        );
    }

    #[test]
    fn symmetric_clipping_produces_odd_harmonics_only() {
        use crate::analysis::measure_harmonics;
        use std::f32::consts::TAU;
        let mut model = DiodeClipperModel::new();
        model.set_drive_db(12.0).unwrap();
        model.prepare(ProcessSpec::new(48_000.0, 1, 4_800)).unwrap();
        let frequency = 1_000.0_f32;
        let mut samples: Vec<f32> = (0..4_800)
            .map(|index| 0.5 * (TAU * frequency * index as f32 / 48_000.0).sin())
            .collect();
        model.process_interleaved(&mut samples, 4_800).unwrap();
        let report = measure_harmonics(&samples, 48_000.0, frequency, 4).unwrap();
        let h2 = report.component(2).unwrap().amplitude;
        let h3 = report.component(3).unwrap().amplitude;
        assert!(h3 > 1e-4, "driven clipper should produce H3, got {h3}");
        assert!(
            20.0 * (h2 / h3).log10() < -10.0,
            "symmetric clipper leaked H2={h2} vs H3={h3}"
        );
    }

    #[test]
    fn solver_fallback_rate_stays_below_one_per_thousand() {
        use std::f32::consts::TAU;
        let mut model = DiodeClipperModel::new();
        model.set_drive_db(24.0).unwrap();
        model.prepare(ProcessSpec::new(48_000.0, 1, 4_800)).unwrap();
        let mut samples: Vec<f32> = (0..4_800)
            .map(|index| 0.8 * (TAU * 10_000.0 * index as f32 / 48_000.0).sin())
            .collect();
        model.process_interleaved(&mut samples, 4_800).unwrap();
        let (solves, fallbacks) = model.solve_stats();
        assert_eq!(solves, 4_800);
        assert!(
            f64::from(fallbacks as u32) / f64::from(solves as u32) < 1e-3,
            "fallback rate too high: {fallbacks}/{solves}"
        );
    }

    #[test]
    fn corner_and_flavor_setters_validate_and_apply() {
        let mut model = DiodeClipperModel::new();
        assert!(model.set_corner_hz(0.0).is_err());
        assert!(model.set_corner_hz(100_000.0).is_err());
        assert!(model.set_corner_hz(f32::NAN).is_err());
        model.set_corner_hz(5_000.0).unwrap();
        assert_eq!(model.corner_hz(), 5_000.0);
        model.prepare(ProcessSpec::new(48_000.0, 1, 64)).unwrap();
        model.set_corner_hz(15_000.0).unwrap();
        model.set_flavor(DiodeFlavor::Germanium);
        assert_eq!(model.flavor(), DiodeFlavor::Germanium);
        let mut block = vec![0.25_f32; 64];
        model.process_interleaved(&mut block, 64).unwrap();
        assert!(block.iter().all(|sample| sample.is_finite()));
    }
}

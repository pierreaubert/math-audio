//! Koren 12AX7A common-cathode triode stage component model (model ID 7).
//!
//! A fixed-bias common-cathode stage: the grid is driven by the AC-coupled
//! input, and the plate node is solved implicitly per sample (Newton via the
//! shared bounded utility) against the resistive plate load. Plate AC is
//! normalized by the measured small-signal gain so the stage is unity at low
//! levels and saturates with drive. See
//! `references/component-references.md` for the frozen equations, operating
//! point, and pre-registered tolerances.
//!
//! Documented limitations: no grid current (positive-grid drive stays
//! bounded but unmodeled) and fixed bias (no cathode RC). Cost note: every
//! sample runs a small Newton solve (~2-8 Koren evaluations in typical
//! audio, 64 capped iterations worst case); silence converges in one
//! evaluation.

use math_audio_iir_fir::{Biquad, BiquadFilterType};

use crate::chain::DcBlocker;
use crate::effects::{DefectConfig, DefectState, solve_bounded_nonlinear};
use crate::level::{calibrated_input_gain, db_to_gain};
use crate::process::{
    AnalogError, AnalogProcessor, ControlSmoother, ProcessSpec, checked_block_len, finite_output,
    flush_denormal, sanitize_sample, validate_finite_range,
};

const CONTROL_SMOOTHING_MS: f32 = 10.0;
const MAX_DEFECT_DELAY_SAMPLES: usize = 64;

// Koren 12AX7A parameters (Tube_params.html, Matlab-optimized).
const KOREN_MU: f32 = 101.24;
const KOREN_EX: f32 = 1.267;
const KOREN_KG1: f32 = 1002.9;
const KOREN_KP: f32 = 699.73;
const KOREN_KVB: f32 = 300.0;

// Documented operating point: 350 V / 150 kOhm published typical load line,
// 1 MOhm next-stage grid leak, fixed -1.5 V grid bias.
const SUPPLY_V: f32 = 350.0;
const PLATE_LOAD_OHMS: f32 = 150_000.0;
const NEXT_GRID_LEAK_OHMS: f32 = 1_000_000.0;
const GRID_BIAS_V: f32 = -1.5;
// Pinned quiescent point, verified against the Koren equation and the DC
// load line in tests: Vp = 204.174 V, Ip = 0.9722 mA.
const QUIESCENT_PLATE_V: f32 = 204.174;
const QUIESCENT_PLATE_A: f32 = 0.9722e-3;
// Level convention: 1 V peak grid swing per 0 dBFS before drive.
const GRID_VOLTS_PER_DBFS: f32 = 1.0;
const COUPLING_HP_HZ: f32 = 8.0;
/// Newton budget per sample (the shared utility's cap).
const SOLVER_MAX_ITERATIONS: usize = 64;
/// Newton residual tolerance in plate volts.
///
/// The residual slope ranges from ~1 (cutoff) to ~200 (saturation knee),
/// so 5 mV pins the plate within 5 mV worst case (73 uV FS, -83 dBFS) and
/// within ~25 uV typically (sub-uV FS). The tolerance must clear the
/// correction-resolution floor (slope * ulp(200 V) ~ 3e-3 in the knee);
/// tighter tolerances stall-fail on rounding noise with the root already
/// found to inaudible precision.
const SOLVER_TOLERANCE_V: f32 = 5e-3;
/// Loose acceptance band in plate volts (~1 mV FS after normalization).
const LOOSE_ACCEPT_V: f32 = 0.25;
/// Finite-difference step in grid volts for the prepare-time gain probe.
const GAIN_PROBE_V: f32 = 1e-3;

/// Koren 12AX7A plate current in amperes.
///
/// `E1 = (EP / KP) * ln(1 + exp(KP * (1 / MU + EG / sqrt(KVB + EP^2))))`,
/// `IP = (E1^EX / KG1) * (1 + sgn(E1))`. The softplus is evaluated in a
/// stable form so any finite grid/plate voltage stays finite (no exp
/// overflow); non-positive `E1` yields zero current.
fn koren_plate_current(grid_v: f32, plate_v: f32) -> f32 {
    if !grid_v.is_finite() || !plate_v.is_finite() || plate_v <= 0.0 {
        return 0.0;
    }
    let argument = KOREN_KP * (1.0 / KOREN_MU + grid_v / (KOREN_KVB + plate_v * plate_v).sqrt());
    // Stable softplus: ln(1 + e^x) = x for large x, ln1p(e^x) otherwise.
    // At x = 20 the neglected term is 2e-9; continuity is inaudible.
    let softplus = if argument > 20.0 {
        argument
    } else {
        argument.exp().ln_1p()
    };
    let e1 = (plate_v / KOREN_KP) * softplus;
    if e1 <= 0.0 {
        0.0
    } else {
        2.0 * e1.powf(KOREN_EX) / KOREN_KG1
    }
}

/// AC plate load: the plate resistor in parallel with the next grid leak.
fn ac_plate_load() -> f32 {
    PLATE_LOAD_OHMS * NEXT_GRID_LEAK_OHMS / (PLATE_LOAD_OHMS + NEXT_GRID_LEAK_OHMS)
}

/// A bounded Koren-12AX7A common-cathode stage with an implicit plate solve.
#[derive(Debug)]
pub struct TriodeStageModel {
    drive_db: ControlSmoother,
    output_gain_db: ControlSmoother,
    amount: ControlSmoother,
    mix: ControlSmoother,
    defects: DefectConfig,
    spec: Option<ProcessSpec>,
    /// Measured small-signal voltage gain magnitude (inverting stage).
    gain_magnitude: f32,
    channels: Vec<TriodeChannelState>,
}

impl Default for TriodeStageModel {
    fn default() -> Self {
        Self::new()
    }
}

impl TriodeStageModel {
    pub fn new() -> Self {
        let sample_rate = 48_000.0;
        Self {
            drive_db: ControlSmoother::new(0.0, CONTROL_SMOOTHING_MS, sample_rate),
            output_gain_db: ControlSmoother::new(0.0, CONTROL_SMOOTHING_MS, sample_rate),
            amount: ControlSmoother::new(1.0, CONTROL_SMOOTHING_MS, sample_rate),
            mix: ControlSmoother::new(1.0, CONTROL_SMOOTHING_MS, sample_rate),
            defects: DefectConfig::default(),
            spec: None,
            gain_magnitude: Self::measure_gain_magnitude(),
            channels: Vec::new(),
        }
    }

    /// Probe the small-signal gain with a symmetric finite difference of the
    /// solved stage at the quiescent point. Pure function of the documented
    /// constants: deterministic and identical for every channel.
    fn measure_gain_magnitude() -> f32 {
        let plate_load = ac_plate_load();
        let solve_plate = |grid_ac: f32| {
            let residual = |plate_ac: f32| {
                let plate = QUIESCENT_PLATE_V + plate_ac;
                let current = koren_plate_current(GRID_BIAS_V + grid_ac, plate) - QUIESCENT_PLATE_A;
                plate_ac + current * plate_load
            };
            solve_bounded_nonlinear(
                residual,
                0.0,
                -QUIESCENT_PLATE_V,
                SUPPLY_V - QUIESCENT_PLATE_V,
                SOLVER_MAX_ITERATIONS,
                SOLVER_TOLERANCE_V,
            )
            .value
        };
        let up = solve_plate(GAIN_PROBE_V);
        let down = solve_plate(-GAIN_PROBE_V);
        ((up - down) / (2.0 * GAIN_PROBE_V)).abs().max(1.0)
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

    pub fn defects(&self) -> DefectConfig {
        self.defects
    }

    /// Measured small-signal gain magnitude used for output normalization.
    pub fn gain_magnitude(&self) -> f32 {
        self.gain_magnitude
    }

    /// Total Newton solves and loose/hold fallbacks across channels.
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
        let gain_magnitude = self.gain_magnitude;
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
                gain_magnitude,
                crosstalk_input,
            );
            let effected = dry + amount * (shaped - dry);
            *sample = finite_output(dry + mix * (effected - dry));
        }
    }
}

impl AnalogProcessor for TriodeStageModel {
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), AnalogError> {
        spec.validate()?;
        let channels = (0..spec.channels)
            .map(|_| TriodeChannelState::new(spec.sample_rate, self.defects))
            .collect();
        self.drive_db.reconfigure(spec.sample_rate);
        self.output_gain_db.reconfigure(spec.sample_rate);
        self.amount.reconfigure(spec.sample_rate);
        self.mix.reconfigure(spec.sample_rate);
        self.gain_magnitude = Self::measure_gain_magnitude();
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
struct TriodeChannelState {
    input_filter: Biquad<f32>,
    output_filter: Biquad<f32>,
    plate_ac: f32,
    defects: DefectState,
    dc_blocker: DcBlocker,
    solves: u64,
    fallbacks: u64,
}

impl TriodeChannelState {
    fn new(sample_rate: f32, defects: DefectConfig) -> Self {
        Self {
            input_filter: Biquad::new(
                BiquadFilterType::Highpass,
                COUPLING_HP_HZ,
                sample_rate,
                0.707,
                0.0,
            ),
            output_filter: Biquad::new(
                BiquadFilterType::Highpass,
                COUPLING_HP_HZ,
                sample_rate,
                0.707,
                0.0,
            ),
            plate_ac: 0.0,
            defects: DefectState::new(defects, sample_rate, MAX_DEFECT_DELAY_SAMPLES),
            dc_blocker: DcBlocker::new(sample_rate),
            solves: 0,
            fallbacks: 0,
        }
    }

    /// Solve the plate node for a grid AC voltage. The plate load is passive
    /// so the plate stays inside `[0, SUPPLY]`.
    ///
    /// Two-phase deterministic solve: Newton from the previous plate value
    /// (the fast path — adjacent samples usually agree), then, only on
    /// failure, a coarse grid search for a fresh initial followed by a
    /// second Newton run. Newton overshoots across the 350 V span on
    /// cutoff/saturation transitions and can ping-pong between the rails;
    /// the grid restart breaks those limit cycles. A loose iterate is
    /// accepted and only a truly lost solve holds the previous state.
    fn solve_plate(&mut self, grid_ac: f32) -> f32 {
        let plate_load = ac_plate_load();
        let lower = -QUIESCENT_PLATE_V;
        let upper = SUPPLY_V - QUIESCENT_PLATE_V;
        let residual = |plate_ac: f32| {
            let plate = QUIESCENT_PLATE_V + plate_ac;
            let current = koren_plate_current(GRID_BIAS_V + grid_ac, plate) - QUIESCENT_PLATE_A;
            plate_ac + current * plate_load
        };
        let first = solve_bounded_nonlinear(
            residual,
            self.plate_ac,
            lower,
            upper,
            SOLVER_MAX_ITERATIONS,
            SOLVER_TOLERANCE_V,
        );
        self.solves = self.solves.saturating_add(1);
        if first.converged {
            self.plate_ac = flush_denormal(first.value);
            return self.plate_ac;
        }
        // Phase 2: scan a fixed 17-point grid for the smallest |residual|
        // and restart Newton there. Fixed grid, no allocation.
        let mut best_value = self.plate_ac;
        let mut best_residual = first.residual.abs();
        for step in 0..=16_u8 {
            let candidate = lower + (upper - lower) * f32::from(step) / 16.0;
            let magnitude = residual(candidate).abs();
            if magnitude < best_residual {
                best_residual = magnitude;
                best_value = candidate;
            }
        }
        let second = solve_bounded_nonlinear(
            residual,
            best_value,
            lower,
            upper,
            SOLVER_MAX_ITERATIONS,
            SOLVER_TOLERANCE_V,
        );
        self.solves = self.solves.saturating_add(1);
        let solved = if second.converged {
            second.value
        } else {
            self.fallbacks = self.fallbacks.saturating_add(1);
            if second.residual.is_finite() && second.residual.abs() <= LOOSE_ACCEPT_V {
                second.value
            } else {
                self.plate_ac
            }
        };
        self.plate_ac = flush_denormal(solved);
        self.plate_ac
    }

    #[inline]
    fn process(
        &mut self,
        input: f32,
        input_gain: f32,
        output_gain: f32,
        gain_magnitude: f32,
        crosstalk_input: f32,
    ) -> f32 {
        let coupled = self.input_filter.process(input);
        let grid_ac = coupled * input_gain * GRID_VOLTS_PER_DBFS;
        let plate_ac = self.solve_plate(grid_ac);
        let normalized = self.output_filter.process(plate_ac / gain_magnitude);
        let output = self.defects.process(normalized, crosstalk_input);
        finite_output(self.dc_blocker.process(output) * output_gain)
    }

    fn reset(&mut self) {
        self.input_filter.reset();
        self.output_filter.reset();
        self.plate_ac = 0.0;
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

    /// Independent f64 transcription of the published Koren equation.
    fn koren_reference(grid_v: f64, plate_v: f64) -> f64 {
        if plate_v <= 0.0 {
            return 0.0;
        }
        const MU: f64 = 101.24;
        const EX: f64 = 1.267;
        const KG1: f64 = 1002.9;
        const KP: f64 = 699.73;
        const KVB: f64 = 300.0;
        let argument = KP * (1.0 / MU + grid_v / (KVB + plate_v * plate_v).sqrt());
        // f64 handles the full exp range of this grid directly.
        let e1 = (plate_v / KP) * (1.0 + argument.exp()).ln();
        if e1 <= 0.0 {
            0.0
        } else {
            2.0 * e1.powf(EX) / KG1
        }
    }

    #[test]
    fn triode_model_is_finite_and_resettable() {
        let mut model = TriodeStageModel::new();
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
    fn triode_model_exposes_a_distinct_append_only_id() {
        let model = AnalogModel::TriodeStage(TriodeStageModel::new());
        assert_eq!(model.model_id(), AnalogModel::TRIODE_STAGE_ID);
        assert!(matches!(
            AnalogModel::from_id(AnalogModel::TRIODE_STAGE_ID),
            Ok(AnalogModel::TriodeStage(_))
        ));
    }

    #[test]
    fn koren_implementation_matches_the_published_equation() {
        // Cover cutoff, the quiescent neighborhood, saturation, and the
        // positive-grid edge (bounded, grid current unmodeled).
        for grid_tenths in -80..=10_i16 {
            for plate in [0.0, 1.0, 50.0, 150.0, 204.174, 300.0, 350.0, 400.0] {
                let grid = f32::from(grid_tenths) / 10.0;
                let actual = koren_plate_current(grid, plate);
                assert!(
                    actual.is_finite() && actual >= 0.0,
                    "grid {grid} plate {plate}: {actual}"
                );
                let expected = koren_reference(f64::from(grid), f64::from(plate));
                let tolerance = 1e-9 + expected.abs() * 1e-5;
                assert!(
                    (f64::from(actual) - expected).abs() <= tolerance,
                    "grid {grid} plate {plate}: got {actual}, published {expected}"
                );
            }
        }
        // Extreme drive stays finite (stable softplus, no exp overflow).
        assert!(koren_plate_current(1010.0, 350.0).is_finite());
        assert!(koren_plate_current(-1010.0, 350.0).is_finite());
        assert_eq!(koren_plate_current(0.0, 0.0), 0.0);
        assert_eq!(koren_plate_current(1.0, -5.0), 0.0);
    }

    #[test]
    fn pinned_quiescent_point_satisfies_koren_and_the_load_line() {
        let current = koren_reference(f64::from(GRID_BIAS_V), f64::from(QUIESCENT_PLATE_V));
        let expected = f64::from(QUIESCENT_PLATE_A);
        assert!(
            ((current - expected) / expected).abs() < 1e-3,
            "Q current {current} vs pinned {expected}"
        );
        let plate = f64::from(SUPPLY_V) - current * f64::from(PLATE_LOAD_OHMS);
        assert!(
            ((plate - f64::from(QUIESCENT_PLATE_V)) / f64::from(QUIESCENT_PLATE_V)).abs() < 1e-3,
            "load-line plate {plate} vs pinned {QUIESCENT_PLATE_V}"
        );
    }

    #[test]
    fn measured_gain_is_inverting_and_matches_transconductance() {
        // Cross-check the implicit solve against direct differentiation of
        // the published equation: gain = gm * Rac / (1 + gp * Rac), where
        // the plate resistance loads the AC plate load (a common-cathode
        // stage gains -gm * (Rac || rp), not -gm * Rac).
        let grid = f64::from(GRID_BIAS_V);
        let plate = f64::from(QUIESCENT_PLATE_V);
        let step = 1e-4;
        let gm = (koren_reference(grid + step, plate) - koren_reference(grid - step, plate))
            / (2.0 * step);
        let gp = (koren_reference(grid, plate + 1.0) - koren_reference(grid, plate - 1.0)) / 2.0;
        let rac = f64::from(ac_plate_load());
        let predicted = gm * rac / (1.0 + gp * rac);
        let gain = f64::from(TriodeStageModel::new().gain_magnitude());
        assert!(
            ((gain - predicted) / predicted).abs() < 0.01,
            "gain magnitude {gain} vs transconductance prediction {predicted}"
        );
        assert!(
            (gain - 68.2).abs() / 68.2 < 0.02,
            "gain magnitude {gain} vs design ~68.2 (see reference note correction)"
        );
        // Sign check through the solved plate: a positive grid step must pull
        // the plate down (common-cathode inversion).
        let mut channel = TriodeChannelState::new(48_000.0, DefectConfig::default());
        let up = channel.solve_plate(0.01);
        channel.reset();
        assert!(
            up < 0.0,
            "positive grid step should pull the plate down, got {up}"
        );
    }

    #[test]
    fn small_signal_gain_is_unity_and_output_inverts() {
        use crate::analysis::measure_harmonics;
        use std::f32::consts::TAU;
        let mut model = TriodeStageModel::new();
        model.prepare(ProcessSpec::new(48_000.0, 1, 4_800)).unwrap();
        let frequency = 1_000.0_f32;
        let input: Vec<f32> = (0..4_800)
            .map(|index| 0.016 * (TAU * frequency * index as f32 / 48_000.0).sin())
            .collect();
        let mut samples = input.clone();
        model.process_interleaved(&mut samples, 4_800).unwrap();
        let correlation: f32 = input.iter().zip(samples.iter()).map(|(a, b)| a * b).sum();
        assert!(
            correlation < 0.0,
            "inverting stage must anticorrelate, got {correlation}"
        );
        let report = measure_harmonics(&samples, 48_000.0, frequency, 3).unwrap();
        let fundamental = report.component(1).unwrap().amplitude;
        let gain_db = 20.0 * (fundamental / 0.016).log10();
        assert!(
            gain_db.abs() < 0.5,
            "small-signal gain should be unity, got {gain_db} dB"
        );
    }

    #[test]
    fn distortion_rises_monotonically_with_drive() {
        use crate::analysis::measure_harmonics;
        use std::f32::consts::TAU;
        let mut thds = Vec::new();
        for drive_db in [-12.0_f32, 0.0, 12.0, 24.0] {
            let mut model = TriodeStageModel::new();
            model.set_drive_db(drive_db).unwrap();
            model.prepare(ProcessSpec::new(48_000.0, 1, 4_800)).unwrap();
            let frequency = 1_000.0_f32;
            let mut samples: Vec<f32> = (0..4_800)
                .map(|index| 0.5 * (TAU * frequency * index as f32 / 48_000.0).sin())
                .collect();
            model.process_interleaved(&mut samples, 4_800).unwrap();
            let report = measure_harmonics(&samples, 48_000.0, frequency, 10).unwrap();
            let fundamental = report.component(1).unwrap().amplitude;
            let harmonics_rss: f32 = (2..=10)
                .map(|order| {
                    let amplitude = report.component(order).unwrap().amplitude;
                    amplitude * amplitude
                })
                .sum::<f32>()
                .sqrt();
            thds.push(harmonics_rss / fundamental);
        }
        assert!(
            thds.windows(2).all(|pair| pair[1] > pair[0]),
            "THD should rise with drive, got {thds:?}"
        );
        assert!(
            thds[0] < 0.05,
            "low-drive THD should be small, got {}",
            thds[0]
        );
    }

    #[test]
    fn positive_grid_drive_stays_finite_and_mostly_converged() {
        use std::f32::consts::TAU;
        // +24 dB on a 0.8 peak: grid swings +/-12.7 V around the -1.5 V bias,
        // deep into unmodeled positive-grid territory. The model must stay
        // finite and keep solving (documented limitation, not a crash).
        let mut model = TriodeStageModel::new();
        model.set_drive_db(24.0).unwrap();
        model.prepare(ProcessSpec::new(48_000.0, 1, 4_800)).unwrap();
        let mut samples: Vec<f32> = (0..4_800)
            .map(|index| 0.8 * (TAU * 1_000.0 * index as f32 / 48_000.0).sin())
            .collect();
        model.process_interleaved(&mut samples, 4_800).unwrap();
        assert!(samples.iter().all(|sample| sample.is_finite()));
        let (solves, fallbacks) = model.solve_stats();
        // Two-phase solve: retries add extra solves beyond one per sample.
        assert!(solves >= 4_800);
        assert!(
            f64::from(fallbacks as u32) / 4_800.0 < 0.05,
            "fallback rate too high under grid overdrive: {fallbacks}/{solves}"
        );
    }
}

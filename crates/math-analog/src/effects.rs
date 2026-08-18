//! Reusable bounded stateful primitives for analog-style models.
//!
//! Every type in this module has an explicit prepared state and bounded work.
//! Defect controls default to zero, so adding the module to a model never
//! silently adds noise, hum, modulation, or crosstalk to an existing preset.

use crate::chain::OnePoleLowpass;
use crate::process::flush_denormal;

/// One point in a measured record/replay EQ curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TapeEqPoint {
    pub frequency_hz: f32,
    pub gain_db: f32,
}

/// Bounded immutable EQ data for a measured tape response.  At most 32 points
/// are retained so lookup work remains bounded in a realtime model.
#[derive(Debug, Clone, PartialEq)]
pub struct TapeEqCurve {
    points: Vec<TapeEqPoint>,
}

impl TapeEqCurve {
    pub fn new(points: &[TapeEqPoint]) -> Result<Self, &'static str> {
        if points.is_empty() || points.len() > 32 {
            return Err("tape EQ curve requires 1..=32 points");
        }
        if points
            .windows(2)
            .any(|window| window[0].frequency_hz >= window[1].frequency_hz)
            || points.iter().any(|point| {
                !point.frequency_hz.is_finite()
                    || point.frequency_hz <= 0.0
                    || !point.gain_db.is_finite()
            })
        {
            return Err("tape EQ points must have increasing finite positive frequencies");
        }
        Ok(Self {
            points: points.to_vec(),
        })
    }

    pub fn points(&self) -> &[TapeEqPoint] {
        &self.points
    }

    pub fn gain_db_at(&self, frequency_hz: f32) -> f32 {
        if self.points.len() == 1 || frequency_hz <= self.points[0].frequency_hz {
            return self.points[0].gain_db;
        }
        for window in self.points.windows(2) {
            if frequency_hz <= window[1].frequency_hz {
                let ratio = ((frequency_hz / window[0].frequency_hz).ln()
                    / (window[1].frequency_hz / window[0].frequency_hz).ln())
                .clamp(0.0, 1.0);
                return window[0].gain_db + ratio * (window[1].gain_db - window[0].gain_db);
            }
        }
        self.points.last().map_or(0.0, |point| point.gain_db)
    }
}

/// Prepared two-shelf approximation of a bounded measured/replay curve.
/// Keeping the curve itself immutable and reducing it to two one-pole shelves
/// at prepare time makes the realtime path allocation-free and deterministic.
#[derive(Debug)]
pub(crate) struct TapeEqState {
    lowpass: OnePoleLowpass,
    high_band: OnePoleLowpass,
    low_gain: f32,
    high_gain: f32,
}

impl TapeEqState {
    pub(crate) fn new(curve: Option<&TapeEqCurve>, sample_rate: f32) -> Self {
        let (low_gain, high_gain) = curve.map_or((1.0, 1.0), |curve| {
            (
                db_to_gain(curve.gain_db_at(50.0)),
                db_to_gain(curve.gain_db_at((sample_rate * 0.2).min(18_000.0))),
            )
        });
        Self {
            lowpass: OnePoleLowpass::new(120.0, sample_rate),
            high_band: OnePoleLowpass::new((sample_rate * 0.2).min(18_000.0), sample_rate),
            low_gain,
            high_gain,
        }
    }

    #[inline]
    pub(crate) fn process(&mut self, input: f32) -> f32 {
        let low = self.lowpass.process(input);
        let high = input - self.high_band.process(input);
        finite_or_zero(input + (self.low_gain - 1.0) * low + (self.high_gain - 1.0) * high)
    }

    pub(crate) fn reset(&mut self) {
        self.lowpass.reset();
        self.high_band.reset();
    }
}

/// Explicit hysteresis choice for stateful magnetic-style models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HysteresisMode {
    Off,
    /// A bounded rate-independent loop approximation. It is intentionally
    /// named as an approximation until a measured hysteresis target is fitted.
    BoundedLoop,
    /// Normalized Jiles–Atherton differential update with a bounded
    /// four-iteration finite fallback. Parameters are normalized for the
    /// stylized model and are not a material-unit calibration.
    JilesAtherton,
}

#[derive(Debug, Clone, Copy)]
pub struct HysteresisState {
    state: f32,
    mode: HysteresisMode,
    coercivity: f32,
    loop_gain: f32,
    previous_input: f32,
}

impl HysteresisState {
    pub fn new(mode: HysteresisMode, coercivity: f32, loop_gain: f32) -> Self {
        Self {
            state: 0.0,
            mode,
            coercivity: coercivity.abs().max(1e-5),
            loop_gain: loop_gain.clamp(0.0, 1.0),
            previous_input: 0.0,
        }
    }

    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        if self.mode == HysteresisMode::Off {
            return input;
        }
        let mut state = self.state;
        match self.mode {
            HysteresisMode::BoundedLoop => {
                // Four fixed relaxation iterations are the declared bounded
                // solver; the final clamp is the finite click-safe fallback.
                for _ in 0..4 {
                    let target = ((input - state) / self.coercivity).tanh();
                    state += 0.25 * (target - state);
                }
            }
            HysteresisMode::JilesAtherton => {
                // Normalized Jiles–Atherton form: anhysteretic Langevin
                // magnetization plus irreversible/reversible differential
                // terms. The capped relaxation keeps extreme inputs safe.
                let delta_h = input - self.previous_input;
                let direction = if delta_h >= 0.0 { 1.0 } else { -1.0 };
                let alpha = 0.2 * self.loop_gain;
                let pinning = self.coercivity * (1.0 + self.loop_gain);
                let reversible_fraction = 0.5 * self.loop_gain;
                for _ in 0..4 {
                    let effective = input + alpha * state;
                    let anhysteretic = langevin(effective / self.coercivity);
                    let difference = anhysteretic - state;
                    let denominator = pinning * direction - alpha * difference;
                    let irreversible = if denominator.abs() > 1e-5 {
                        difference / denominator
                    } else {
                        0.0
                    };
                    let reversible = reversible_fraction * difference;
                    let proposal =
                        state + delta_h * ((1.0 - reversible_fraction) * irreversible + reversible);
                    state += 0.25 * (proposal - state);
                }
                self.previous_input = input;
            }
            HysteresisMode::Off => unreachable!("off mode returned before state update"),
        }
        self.state = if state.is_finite() {
            state.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        input + self.loop_gain * (self.state - input)
    }

    pub fn reset(&mut self) {
        self.state = 0.0;
        self.previous_input = 0.0;
    }
}

/// Envelope-driven rail compression used by preamp and component models.
#[derive(Debug, Clone, Copy)]
pub struct PowerSupplySag {
    rail: f32,
    coefficient: f32,
    depth: f32,
}

impl PowerSupplySag {
    pub fn new(sample_rate: f32, time_ms: f32, depth: f32) -> Self {
        Self {
            rail: 1.0,
            coefficient: time_coefficient(time_ms, sample_rate),
            depth: depth.clamp(0.0, 1.0),
        }
    }

    #[inline]
    pub fn process(&mut self, envelope: f32) -> f32 {
        let demand = envelope.abs().clamp(0.0, 1.0);
        let target = (1.0 - self.depth * demand).clamp(0.0, 1.0);
        self.rail = self.coefficient * self.rail + (1.0 - self.coefficient) * target;
        self.rail = flush_denormal(self.rail.clamp(0.0, 1.0));
        self.rail
    }

    pub fn reset(&mut self) {
        self.rail = 1.0;
    }

    pub fn rail(&self) -> f32 {
        self.rail
    }
}

/// Explicit V/µs slew-rate limiter.
#[derive(Debug, Clone, Copy)]
pub struct SlewRateLimiter {
    previous: f32,
    max_delta_per_sample: f32,
}

impl SlewRateLimiter {
    pub fn new(sample_rate: f32, volts_per_microsecond: f32) -> Self {
        Self {
            previous: 0.0,
            max_delta_per_sample: (volts_per_microsecond.max(0.0) * 1_000_000.0 / sample_rate)
                .max(f32::MIN_POSITIVE),
        }
    }

    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let delta =
            (input - self.previous).clamp(-self.max_delta_per_sample, self.max_delta_per_sample);
        self.previous = flush_denormal(self.previous + delta);
        self.previous
    }

    pub fn reset(&mut self) {
        self.previous = 0.0;
    }
}

/// Parameters for individually enabled defect modules.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DefectConfig {
    pub wow_depth_samples: f32,
    pub wow_rate_hz: f32,
    pub flutter_depth_samples: f32,
    pub flutter_rate_hz: f32,
    pub noise_floor: f32,
    pub hum_amplitude: f32,
    pub hum_frequency_hz: f32,
    pub crosstalk: f32,
}

impl Default for DefectConfig {
    fn default() -> Self {
        Self {
            wow_depth_samples: 0.0,
            wow_rate_hz: 0.4,
            flutter_depth_samples: 0.0,
            flutter_rate_hz: 7.0,
            noise_floor: 0.0,
            hum_amplitude: 0.0,
            hum_frequency_hz: 50.0,
            crosstalk: 0.0,
        }
    }
}

impl DefectConfig {
    pub fn validate(&self, max_delay_samples: usize) -> Result<(), &'static str> {
        let finite_non_negative = |value: f32| value.is_finite() && value >= 0.0;
        if !finite_non_negative(self.wow_depth_samples)
            || !finite_non_negative(self.flutter_depth_samples)
            || !finite_non_negative(self.noise_floor)
            || !finite_non_negative(self.hum_amplitude)
            || !finite_non_negative(self.crosstalk)
            || !self.wow_rate_hz.is_finite()
            || self.wow_rate_hz < 0.0
            || !self.flutter_rate_hz.is_finite()
            || self.flutter_rate_hz < 0.0
            || !self.hum_frequency_hz.is_finite()
            || self.hum_frequency_hz < 0.0
        {
            return Err("defect parameters must be finite and non-negative");
        }
        if self.wow_depth_samples.max(self.flutter_depth_samples)
            > max_delay_samples.saturating_sub(1) as f32
        {
            return Err("defect delay depth exceeds prepared delay capacity");
        }
        if self.noise_floor > 1.0 || self.hum_amplitude > 1.0 || self.crosstalk > 1.0 {
            return Err("defect amplitudes must be in [0, 1]");
        }
        Ok(())
    }
}

/// Prepared deterministic defect state.  The modulated delay is intentionally
/// short and capped; it is suitable for an optional coloration defect, not a
/// general resampler.
#[derive(Debug)]
pub struct DefectState {
    config: DefectConfig,
    sample_rate: f32,
    phase_wow: f32,
    phase_flutter: f32,
    phase_hum: f32,
    random_state: u32,
    delay: Vec<f32>,
    write_index: usize,
}

impl DefectState {
    pub fn new(config: DefectConfig, sample_rate: f32, max_delay_samples: usize) -> Self {
        Self {
            config,
            sample_rate,
            phase_wow: 0.0,
            phase_flutter: 0.0,
            phase_hum: 0.0,
            random_state: 0x1234_5678,
            delay: vec![0.0; max_delay_samples.max(2)],
            write_index: 0,
        }
    }

    pub fn config(&self) -> DefectConfig {
        self.config
    }

    #[inline]
    pub fn process(&mut self, input: f32, crosstalk_input: f32) -> f32 {
        let modulation = self.config.wow_depth_samples
            * (self.phase_wow * std::f32::consts::TAU).sin()
            + self.config.flutter_depth_samples
                * (self.phase_flutter * std::f32::consts::TAU).sin();
        self.delay[self.write_index] = input;
        let len = self.delay.len();
        let read = (self.write_index as f32 - modulation).rem_euclid(len as f32);
        let index_a = read.floor() as usize % len;
        let index_b = (index_a + 1) % len;
        let fraction = read.fract();
        let delayed = self.delay[index_a] * (1.0 - fraction) + self.delay[index_b] * fraction;
        self.write_index = (self.write_index + 1) % len;

        self.phase_wow = (self.phase_wow + self.config.wow_rate_hz / self.sample_rate).fract();
        self.phase_flutter =
            (self.phase_flutter + self.config.flutter_rate_hz / self.sample_rate).fract();
        self.phase_hum = (self.phase_hum + self.config.hum_frequency_hz / self.sample_rate).fract();
        let noise = self.next_noise() * self.config.noise_floor;
        let hum = (self.phase_hum * std::f32::consts::TAU).sin() * self.config.hum_amplitude;
        delayed + crosstalk_input * self.config.crosstalk.clamp(0.0, 1.0) + noise + hum
    }

    fn next_noise(&mut self) -> f32 {
        self.random_state = self
            .random_state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        (self.random_state as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    pub fn reset(&mut self) {
        self.phase_wow = 0.0;
        self.phase_flutter = 0.0;
        self.phase_hum = 0.0;
        self.random_state = 0x1234_5678;
        self.delay.fill(0.0);
        self.write_index = 0;
    }
}

/// Result of a bounded scalar nonlinear solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonlinearSolveResult {
    pub value: f32,
    pub residual: f32,
    pub iterations: usize,
    pub converged: bool,
}

/// Deterministic, capped Newton/fixed-point utility for component spikes.
pub fn solve_bounded_nonlinear<F>(
    function: F,
    initial: f32,
    lower: f32,
    upper: f32,
    max_iterations: usize,
    tolerance: f32,
) -> NonlinearSolveResult
where
    F: Fn(f32) -> f32,
{
    let lo = lower.min(upper);
    let hi = lower.max(upper);
    let mut value = initial.clamp(lo, hi);
    let tolerance = tolerance.max(1e-8);
    for iteration in 1..=max_iterations.min(64) {
        let residual = function(value);
        if !residual.is_finite() {
            return NonlinearSolveResult {
                value: value.clamp(lo, hi),
                residual: f32::INFINITY,
                iterations: iteration,
                converged: false,
            };
        }
        if residual.abs() <= tolerance {
            return NonlinearSolveResult {
                value,
                residual,
                iterations: iteration,
                converged: true,
            };
        }
        let step = (value.abs() + 1.0) * 1e-4;
        let plus = function((value + step).min(hi));
        let minus = function((value - step).max(lo));
        let derivative = (plus - minus) / (2.0 * step);
        let candidate = if derivative.is_finite() && derivative.abs() > 1e-8 {
            value - residual / derivative
        } else {
            value - residual.signum() * step
        };
        let next = candidate.clamp(lo, hi);
        if (next - value).abs() <= tolerance * 0.1 {
            return NonlinearSolveResult {
                value: next,
                residual: function(next),
                iterations: iteration,
                converged: false,
            };
        }
        value = next;
    }
    let residual = function(value);
    NonlinearSolveResult {
        value,
        residual,
        iterations: max_iterations.min(64),
        converged: residual.is_finite() && residual.abs() <= tolerance,
    }
}

#[inline]
fn time_coefficient(time_ms: f32, sample_rate: f32) -> f32 {
    (-1.0 / (time_ms.max(0.001) * 0.001 * sample_rate.max(1.0))).exp()
}

#[inline]
fn db_to_gain(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

#[inline]
fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

#[inline]
fn langevin(value: f32) -> f32 {
    if value.abs() < 1e-3 {
        value / 3.0
    } else {
        (1.0 / value.tanh() - 1.0 / value).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sag_and_slew_are_bounded() {
        let mut sag = PowerSupplySag::new(48_000.0, 20.0, 0.8);
        let mut slew = SlewRateLimiter::new(48_000.0, 0.001);
        for _ in 0..1_000 {
            assert!((0.0..=1.0).contains(&sag.process(1.0)));
            assert!(slew.process(100.0).is_finite());
        }
        assert!(slew.process(100.0) < 100.0);
    }

    #[test]
    fn defects_are_default_off_and_resettable() {
        let mut state = DefectState::new(DefectConfig::default(), 48_000.0, 32);
        let mut output = 0.0;
        for _ in 0..32 {
            output = state.process(0.25, 0.0);
        }
        assert!((output - 0.25).abs() < 1e-6);
        state.reset();
        assert_eq!(state.process(0.0, 0.0), 0.0);
    }

    #[test]
    fn enabled_defects_are_bounded_and_deterministic() {
        let config = DefectConfig {
            wow_depth_samples: 2.0,
            flutter_depth_samples: 1.0,
            noise_floor: 0.001,
            hum_amplitude: 0.002,
            crosstalk: 0.2,
            ..DefectConfig::default()
        };
        let mut state = DefectState::new(config, 48_000.0, 64);
        let mut changed = false;
        for _ in 0..256 {
            let output = state.process(0.25, -0.5);
            assert!(output.is_finite() && output.abs() < 2.0);
            changed |= (output - 0.25).abs() > 1e-6;
        }
        assert!(changed);
        state.reset();
        let first = state.process(0.25, -0.5);
        state.reset();
        assert_eq!(first, state.process(0.25, -0.5));
    }

    #[test]
    fn jiles_atherton_mode_has_bounded_directional_memory() {
        let mut ascending = HysteresisState::new(HysteresisMode::JilesAtherton, 0.2, 0.8);
        let mut descending = ascending;
        for value in [0.0, 0.2, 0.5, 0.8, 0.5] {
            ascending.process(value);
        }
        for value in [0.0, -0.2, -0.5, -0.8, 0.5] {
            descending.process(value);
        }
        let positive = ascending.process(0.5);
        let negative = descending.process(0.5);
        assert!(positive.is_finite() && negative.is_finite());
        assert!((positive - negative).abs() > 1e-4);
        ascending.reset();
        descending.reset();
        assert_eq!(ascending.process(0.25), descending.process(0.25));
    }

    #[test]
    fn defect_validation_bounds_each_optional_module() {
        assert!(DefectConfig::default().validate(32).is_ok());
        assert!(
            DefectConfig {
                wow_depth_samples: 32.0,
                ..DefectConfig::default()
            }
            .validate(32)
            .is_err()
        );
        assert!(
            DefectConfig {
                noise_floor: 1.1,
                ..DefectConfig::default()
            }
            .validate(32)
            .is_err()
        );
    }

    #[test]
    fn tape_eq_curve_is_applied_by_prepared_bounded_state() {
        let curve = TapeEqCurve::new(&[
            TapeEqPoint {
                frequency_hz: 20.0,
                gain_db: 6.0,
            },
            TapeEqPoint {
                frequency_hz: 12_000.0,
                gain_db: -6.0,
            },
        ])
        .unwrap();
        let mut state = TapeEqState::new(Some(&curve), 48_000.0);
        let mut output = 0.0;
        for _ in 0..128 {
            output = state.process(1.0);
        }
        assert!(output > 1.0);
        state.reset();
        assert_eq!(state.process(0.0), 0.0);
    }

    #[test]
    fn nonlinear_solver_converges_and_has_a_hard_iteration_cap() {
        let result = solve_bounded_nonlinear(|x| x * x - 4.0, 1.0, -4.0, 4.0, 20, 1e-5);
        assert!(result.converged);
        assert!((result.value.abs() - 2.0).abs() < 1e-3);
        let capped = solve_bounded_nonlinear(|_| 1.0, 0.0, -1.0, 1.0, 1_000, 1e-8);
        assert!(capped.iterations <= 64);
    }
}

use crate::chain::{DcBlocker, OnePoleLowpass};
use crate::level::{calibrated_input_gain, db_to_gain};
use crate::process::{
    AnalogError, AnalogProcessor, ControlSmoother, ProcessSpec, checked_block_len, finite_output,
    sanitize_sample, validate_finite_range,
};

const MAX_BRANCHES: usize = 5;
const CONTROL_SMOOTHING_MS: f32 = 10.0;

/// One bounded Chebyshev branch in a generic parallel-Hammerstein model.
///
/// `order` selects `T_order(tanh(drive * x))`, `gain` is linear, and
/// `cutoff_hz` describes a first-order output filter.  These coefficients are
/// deliberately supplied by the caller: until a measurement target and
/// provenance are attached, this is a generic model and not a branded device.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HammersteinBranch {
    order: usize,
    gain: f32,
    cutoff_hz: f32,
}

impl HammersteinBranch {
    pub fn new(order: usize, gain: f32, cutoff_hz: f32) -> Result<Self, AnalogError> {
        if !(1..=MAX_BRANCHES).contains(&order) {
            return Err(AnalogError::ParameterOutOfRange {
                parameter: "branch_order",
                value: order as f32,
                min: 1.0,
                max: MAX_BRANCHES as f32,
            });
        }
        validate_finite_range("branch_gain", gain, -128.0, 128.0)?;
        validate_finite_range("branch_cutoff_hz", cutoff_hz, 0.0, 1_000_000.0)?;
        Ok(Self {
            order,
            gain,
            cutoff_hz,
        })
    }

    pub fn order(&self) -> usize {
        self.order
    }

    pub fn gain(&self) -> f32 {
        self.gain
    }

    pub fn cutoff_hz(&self) -> f32 {
        self.cutoff_hz
    }
}

/// A low-order bounded parallel-Hammerstein processor.
///
/// The model is:
///
/// ```text
/// y[n] = sum_k H_k { gain_k * T_k(tanh(drive * x[n])) }
/// ```
///
/// It provides predictable frequency-dependent coloration with at most five
/// branches.  It does not claim hysteresis, component behaviour, or a fit to
/// a particular hardware unit.  The branch filters are allocated in
/// [`AnalogProcessor::prepare`], and processing is allocation-free.
#[derive(Debug, Clone)]
pub struct HammersteinModel {
    branches: Vec<HammersteinBranch>,
    drive_db: ControlSmoother,
    output_gain_db: ControlSmoother,
    character: ControlSmoother,
    amount: ControlSmoother,
    mix: ControlSmoother,
    spec: Option<ProcessSpec>,
    branch_states: Vec<Vec<OnePoleLowpass>>,
    dc_blockers: Vec<DcBlocker>,
}

impl Default for HammersteinModel {
    fn default() -> Self {
        Self::new()
    }
}

impl HammersteinModel {
    /// Create a unity, zero-latency linear branch.
    pub fn new() -> Self {
        Self::from_branches(vec![HammersteinBranch {
            order: 1,
            gain: 1.0,
            cutoff_hz: 0.0,
        }])
    }

    /// Create the crate's generic illustrative coloration preset.
    ///
    /// These coefficients are deliberately synthetic and have no measured
    /// device provenance. They make the serialized `Hammerstein` model ID a
    /// useful frequency-dependent example while [`Self::new`] remains the
    /// identity-branch constructor for callers that supply coefficients.
    pub fn generic_coloration() -> Result<Self, AnalogError> {
        Self::with_branches(&[
            HammersteinBranch {
                order: 1,
                gain: 0.92,
                cutoff_hz: 0.0,
            },
            HammersteinBranch {
                order: 2,
                gain: 0.08,
                cutoff_hz: 1_800.0,
            },
            HammersteinBranch {
                order: 3,
                gain: 0.04,
                cutoff_hz: 6_000.0,
            },
            HammersteinBranch {
                order: 4,
                gain: 0.02,
                cutoff_hz: 10_000.0,
            },
        ])
    }

    /// Build a model with at most five fixed branches.
    pub fn with_branches(branches: &[HammersteinBranch]) -> Result<Self, AnalogError> {
        if branches.is_empty() {
            return Err(AnalogError::ParameterOutOfRange {
                parameter: "branch_count",
                value: 0.0,
                min: 1.0,
                max: MAX_BRANCHES as f32,
            });
        }
        if branches.len() > MAX_BRANCHES {
            return Err(AnalogError::ParameterOutOfRange {
                parameter: "branch_count",
                value: branches.len() as f32,
                min: 1.0,
                max: MAX_BRANCHES as f32,
            });
        }
        for branch in branches {
            validate_finite_range("branch_gain", branch.gain, -128.0, 128.0)?;
            validate_finite_range("branch_cutoff_hz", branch.cutoff_hz, 0.0, 1_000_000.0)?;
            if !(1..=MAX_BRANCHES).contains(&branch.order) {
                return Err(AnalogError::ParameterOutOfRange {
                    parameter: "branch_order",
                    value: branch.order as f32,
                    min: 1.0,
                    max: MAX_BRANCHES as f32,
                });
            }
        }
        Ok(Self::from_branches(branches.to_vec()))
    }

    fn from_branches(branches: Vec<HammersteinBranch>) -> Self {
        let sample_rate = 48_000.0;
        Self {
            branches,
            drive_db: ControlSmoother::new(0.0, CONTROL_SMOOTHING_MS, sample_rate),
            output_gain_db: ControlSmoother::new(0.0, CONTROL_SMOOTHING_MS, sample_rate),
            character: ControlSmoother::new(0.5, CONTROL_SMOOTHING_MS, sample_rate),
            amount: ControlSmoother::new(1.0, CONTROL_SMOOTHING_MS, sample_rate),
            mix: ControlSmoother::new(1.0, CONTROL_SMOOTHING_MS, sample_rate),
            spec: None,
            branch_states: Vec::new(),
            dc_blockers: Vec::new(),
        }
    }

    pub fn branches(&self) -> &[HammersteinBranch] {
        &self.branches
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

    pub fn character(&self) -> f32 {
        self.character.target()
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

    fn process_frame(&mut self, frame: &mut [f32], channels: usize) {
        let drive = calibrated_input_gain(self.drive_db.advance());
        let output_gain = db_to_gain(self.output_gain_db.advance());
        let character_bias = (self.character.advance() - 0.5) * 0.8;
        let amount = self.amount.advance();
        let mix = self.mix.advance();

        for (channel, sample) in frame.iter_mut().enumerate().take(channels) {
            let dry = sanitize_sample(*sample);
            let driven = (dry * drive + character_bias).tanh();
            let mut basis = [0.0_f32; MAX_BRANCHES + 1];
            basis[0] = 1.0;
            basis[1] = driven;
            for order in 2..=MAX_BRANCHES {
                basis[order] = 2.0 * driven * basis[order - 1] - basis[order - 2];
            }

            let mut shaped = 0.0;
            for (branch_index, branch) in self.branches.iter().enumerate() {
                let branch_output = branch.gain * basis[branch.order];
                shaped += self.branch_states[branch_index][channel].process(branch_output);
            }
            shaped = self.dc_blockers[channel].process(shaped) * output_gain;
            let shaped = finite_output(shaped);
            let effected = dry + amount * (shaped - dry);
            *sample = finite_output(dry + mix * (effected - dry));
        }
    }
}

impl AnalogProcessor for HammersteinModel {
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), AnalogError> {
        spec.validate()?;
        let nyquist = spec.sample_rate * 0.5;
        let mut branch_states = Vec::with_capacity(self.branches.len());
        for branch in &self.branches {
            if branch.cutoff_hz > nyquist {
                return Err(AnalogError::ParameterOutOfRange {
                    parameter: "branch_cutoff_hz",
                    value: branch.cutoff_hz,
                    min: 0.0,
                    max: nyquist,
                });
            }
            let mut states = Vec::with_capacity(spec.channels);
            for _ in 0..spec.channels {
                states.push(OnePoleLowpass::new(branch.cutoff_hz, spec.sample_rate));
            }
            branch_states.push(states);
        }
        let mut dc_blockers = Vec::with_capacity(spec.channels);
        for _ in 0..spec.channels {
            dc_blockers.push(DcBlocker::new(spec.sample_rate));
        }

        self.drive_db.reconfigure(spec.sample_rate);
        self.output_gain_db.reconfigure(spec.sample_rate);
        self.character.reconfigure(spec.sample_rate);
        self.amount.reconfigure(spec.sample_rate);
        self.mix.reconfigure(spec.sample_rate);
        self.branch_states = branch_states;
        self.dc_blockers = dc_blockers;
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
        for states in &mut self.branch_states {
            for state in states {
                state.reset();
            }
        }
        for blocker in &mut self.dc_blockers {
            blocker.reset();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branch_limit_is_checked_at_construction() {
        let branches = [
            HammersteinBranch::new(1, 1.0, 0.0).unwrap(),
            HammersteinBranch::new(2, 0.1, 1000.0).unwrap(),
            HammersteinBranch::new(3, 0.1, 1000.0).unwrap(),
            HammersteinBranch::new(4, 0.1, 1000.0).unwrap(),
            HammersteinBranch::new(5, 0.1, 1000.0).unwrap(),
        ];
        assert!(HammersteinModel::with_branches(&branches).is_ok());
        let too_many = [
            branches[0],
            branches[1],
            branches[2],
            branches[3],
            branches[4],
            branches[0],
        ];
        assert!(HammersteinModel::with_branches(&too_many).is_err());
    }

    #[test]
    fn cutoff_is_checked_against_prepared_sample_rate() {
        let branch = HammersteinBranch::new(1, 1.0, 30_000.0).unwrap();
        let mut model = HammersteinModel::with_branches(&[branch]).unwrap();
        let result = model.prepare(ProcessSpec {
            sample_rate: 48_000.0,
            channels: 1,
            max_block_frames: 32,
        });
        assert!(matches!(
            result,
            Err(AnalogError::ParameterOutOfRange {
                parameter: "branch_cutoff_hz",
                ..
            })
        ));
    }

    #[test]
    fn nonlinear_branches_are_finite() {
        let branches = [
            HammersteinBranch::new(1, 1.0, 0.0).unwrap(),
            HammersteinBranch::new(2, 0.25, 1000.0).unwrap(),
            HammersteinBranch::new(3, 0.1, 5000.0).unwrap(),
        ];
        let mut model = HammersteinModel::with_branches(&branches).unwrap();
        model
            .prepare(ProcessSpec {
                sample_rate: 96_000.0,
                channels: 2,
                max_block_frames: 64,
            })
            .unwrap();
        let mut buffer = vec![f32::MAX; 128];
        model.process_interleaved(&mut buffer, 64).unwrap();
        assert!(buffer.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn branch_filter_has_declared_frequency_dependent_response() {
        let branch = HammersteinBranch::new(1, 1.0, 1_000.0).unwrap();
        let render = |frequency: f32| {
            let mut model = HammersteinModel::with_branches(&[branch]).unwrap();
            model.prepare(ProcessSpec::new(48_000.0, 1, 4_800)).unwrap();
            let mut samples: Vec<f32> = (0..4_800)
                .map(|index| {
                    (2.0 * std::f32::consts::PI * frequency * index as f32 / 48_000.0).sin() * 0.5
                })
                .collect();
            model.process_interleaved(&mut samples, 4_800).unwrap();
            (samples[2_800..]
                .iter()
                .map(|sample| sample * sample)
                .sum::<f32>()
                / 2_000.0)
                .sqrt()
        };
        let low_rms = render(100.0);
        let high_rms = render(10_000.0);
        assert!(low_rms > high_rms * 2.0);
    }

    #[test]
    fn generic_coloration_is_bounded_and_frequency_dependent() {
        let mut model = HammersteinModel::generic_coloration().unwrap();
        model.prepare(ProcessSpec::new(48_000.0, 1, 4_800)).unwrap();
        let mut low: Vec<f32> = (0..4_800)
            .map(|index| 0.5 * (2.0 * std::f32::consts::PI * 100.0 * index as f32 / 48_000.0).sin())
            .collect();
        let mut high: Vec<f32> = (0..4_800)
            .map(|index| {
                0.5 * (2.0 * std::f32::consts::PI * 10_000.0 * index as f32 / 48_000.0).sin()
            })
            .collect();
        model.process_interleaved(&mut low, 4_800).unwrap();
        model.reset();
        model.process_interleaved(&mut high, 4_800).unwrap();
        let rms = |samples: &[f32]| {
            (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32)
                .sqrt()
        };
        assert!(
            low.iter()
                .all(|sample| sample.is_finite() && sample.abs() <= 128.0)
        );
        assert!(
            high.iter()
                .all(|sample| sample.is_finite() && sample.abs() <= 128.0)
        );
        assert!((rms(&low) - rms(&high)).abs() > 1e-3);
    }
}

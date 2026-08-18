use math_audio_dsp::adaa::{Adaa1, Adaa2};
use math_audio_dsp::simd::chebyshev_basis_simd;
use math_audio_iir_fir::{Biquad, BiquadFilterType};

use crate::chain::{DcBlocker, OnePoleLowpass};
use crate::harmonics::AntiAliasing;
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

/// A bounded biquad section for a parallel-Hammerstein branch. Repeating an
/// order in a model creates a bounded SOS cascade.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HammersteinBiquadBranch {
    order: usize,
    gain: f32,
    filter_type: BiquadFilterType,
    frequency_hz: f32,
    q: f32,
    db_gain: f32,
}

/// Optional prepared Wiener-stage filter before the parallel nonlinear
/// branches.  It is disabled by default so existing generic models remain
/// bit-compatible and zero-latency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HammersteinPreFilter {
    filter_type: BiquadFilterType,
    frequency_hz: f32,
    q: f32,
    db_gain: f32,
}

impl HammersteinPreFilter {
    pub fn new(
        filter_type: BiquadFilterType,
        frequency_hz: f32,
        q: f32,
        db_gain: f32,
    ) -> Result<Self, AnalogError> {
        validate_finite_range(
            "pre_filter_frequency_hz",
            frequency_hz,
            f32::EPSILON,
            1_000_000.0,
        )?;
        validate_finite_range("pre_filter_q", q, 0.0, 100.0)?;
        validate_finite_range("pre_filter_db_gain", db_gain, -128.0, 128.0)?;
        Ok(Self {
            filter_type,
            frequency_hz,
            q,
            db_gain,
        })
    }

    pub fn filter_type(&self) -> BiquadFilterType {
        self.filter_type
    }

    pub fn frequency_hz(&self) -> f32 {
        self.frequency_hz
    }

    pub fn q(&self) -> f32 {
        self.q
    }

    pub fn db_gain(&self) -> f32 {
        self.db_gain
    }
}

impl HammersteinBiquadBranch {
    pub fn new(
        order: usize,
        gain: f32,
        filter_type: BiquadFilterType,
        frequency_hz: f32,
        q: f32,
        db_gain: f32,
    ) -> Result<Self, AnalogError> {
        if !(1..=MAX_BRANCHES).contains(&order) {
            return Err(AnalogError::ParameterOutOfRange {
                parameter: "branch_order",
                value: order as f32,
                min: 1.0,
                max: MAX_BRANCHES as f32,
            });
        }
        validate_finite_range("branch_gain", gain, -128.0, 128.0)?;
        validate_finite_range("branch_frequency_hz", frequency_hz, 0.0, 1_000_000.0)?;
        validate_finite_range("branch_q", q, 0.0, 100.0)?;
        validate_finite_range("branch_db_gain", db_gain, -128.0, 128.0)?;
        Ok(Self {
            order,
            gain,
            filter_type,
            frequency_hz,
            q,
            db_gain,
        })
    }

    pub fn order(&self) -> usize {
        self.order
    }
    pub fn gain(&self) -> f32 {
        self.gain
    }
    pub fn filter_type(&self) -> BiquadFilterType {
        self.filter_type
    }
    pub fn frequency_hz(&self) -> f32 {
        self.frequency_hz
    }
    pub fn q(&self) -> f32 {
        self.q
    }
    pub fn db_gain(&self) -> f32 {
        self.db_gain
    }
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
    filter_specs: Vec<HammersteinFilterSpec>,
    pre_filter: Option<HammersteinPreFilter>,
    anti_aliasing: AntiAliasing,
    branch_states: Vec<Vec<HammersteinBranchState>>,
    dc_blockers: Vec<DcBlocker>,
    pre_filters: Vec<Option<Biquad<f32>>>,
    simd_bounded: Vec<f32>,
    simd_basis: Vec<f32>,
    simd_shaped: Vec<f32>,
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

    /// Build a model with prepared biquad/SOS branch filters. Repeating an
    /// order creates a bounded cascade of sections for that nonlinear branch.
    pub fn with_biquad_branches(branches: &[HammersteinBiquadBranch]) -> Result<Self, AnalogError> {
        if branches.is_empty() || branches.len() > MAX_BRANCHES {
            return Err(AnalogError::ParameterOutOfRange {
                parameter: "branch_count",
                value: branches.len() as f32,
                min: 1.0,
                max: MAX_BRANCHES as f32,
            });
        }
        let simple = branches
            .iter()
            .map(|branch| HammersteinBranch::new(branch.order, branch.gain, 0.0))
            .collect::<Result<Vec<_>, _>>()?;
        let mut model = Self::from_branches(simple);
        model.filter_specs = branches
            .iter()
            .map(|branch| HammersteinFilterSpec::Biquad {
                filter_type: branch.filter_type,
                frequency_hz: branch.frequency_hz,
                q: branch.q,
                db_gain: branch.db_gain,
            })
            .collect();
        Ok(model)
    }

    fn from_branches(branches: Vec<HammersteinBranch>) -> Self {
        let sample_rate = 48_000.0;
        let filter_specs = branches
            .iter()
            .map(|branch| HammersteinFilterSpec::OnePole(branch.cutoff_hz))
            .collect();
        Self {
            branches,
            drive_db: ControlSmoother::new(0.0, CONTROL_SMOOTHING_MS, sample_rate),
            output_gain_db: ControlSmoother::new(0.0, CONTROL_SMOOTHING_MS, sample_rate),
            character: ControlSmoother::new(0.5, CONTROL_SMOOTHING_MS, sample_rate),
            amount: ControlSmoother::new(1.0, CONTROL_SMOOTHING_MS, sample_rate),
            mix: ControlSmoother::new(1.0, CONTROL_SMOOTHING_MS, sample_rate),
            spec: None,
            filter_specs,
            pre_filter: None,
            anti_aliasing: AntiAliasing::Adaa1,
            branch_states: Vec::new(),
            dc_blockers: Vec::new(),
            pre_filters: Vec::new(),
            simd_bounded: Vec::new(),
            simd_basis: Vec::new(),
            simd_shaped: Vec::new(),
        }
    }

    pub fn branches(&self) -> &[HammersteinBranch] {
        &self.branches
    }

    pub fn anti_aliasing(&self) -> AntiAliasing {
        self.anti_aliasing
    }

    pub fn pre_filter(&self) -> Option<HammersteinPreFilter> {
        self.pre_filter
    }

    pub fn set_pre_filter(&mut self, filter: Option<HammersteinPreFilter>) {
        self.pre_filter = filter;
        for pre_filter in self.pre_filters.iter_mut().flatten() {
            pre_filter.reset();
        }
    }

    /// Select the branch nonlinearity path.  Host-owned oversampling remains
    /// useful for the filtered/stateful parts; this option only changes the
    /// per-branch memoryless evaluation.
    pub fn set_anti_aliasing(&mut self, mode: AntiAliasing) {
        if self.anti_aliasing != mode {
            for states in &mut self.branch_states {
                for state in states {
                    state.reset();
                }
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
            for frame in 0..frames {
                let dry = sanitize_sample(samples[frame * spec.channels + channel]);
                self.simd_bounded[frame] = (dry * drive + character_bias).tanh();
            }
            self.simd_shaped[..frames].fill(0.0);
            for (branch_index, branch) in self.branches.iter().enumerate() {
                chebyshev_basis_simd(
                    &self.simd_bounded[..frames],
                    &mut self.simd_basis[..frames],
                    branch.order,
                );
                for frame in 0..frames {
                    let branch_output = self.simd_basis[frame] * branch.gain;
                    self.simd_shaped[frame] += self.branch_states[branch_index][channel]
                        .filter
                        .process(branch_output);
                }
            }
            for frame in 0..frames {
                let index = frame * spec.channels + channel;
                let dry = sanitize_sample(samples[index]);
                let shaped = finite_output(
                    self.dc_blockers[channel].process(self.simd_shaped[frame]) * output_gain,
                );
                let effected = dry + amount * (shaped - dry);
                samples[index] = finite_output(dry + mix * (effected - dry));
            }
        }
    }

    fn process_frame(&mut self, frame: &mut [f32], channels: usize, anti_aliasing: AntiAliasing) {
        let drive = calibrated_input_gain(self.drive_db.advance());
        let output_gain = db_to_gain(self.output_gain_db.advance());
        let character_bias = (self.character.advance() - 0.5) * 0.8;
        let amount = self.amount.advance();
        let mix = self.mix.advance();

        for (channel, sample) in frame.iter_mut().enumerate().take(channels) {
            let dry = sanitize_sample(*sample);
            let filtered = self.pre_filters[channel]
                .as_mut()
                .map_or(dry, |pre_filter| pre_filter.process(dry));
            let branch_input = filtered * drive + character_bias;
            let driven = branch_input.tanh();
            let mut basis = [0.0_f32; MAX_BRANCHES + 1];
            basis[0] = 1.0;
            basis[1] = driven;
            for order in 2..=MAX_BRANCHES {
                basis[order] = 2.0 * driven * basis[order - 1] - basis[order - 2];
            }

            let mut shaped = 0.0;
            for (branch_index, branch) in self.branches.iter().enumerate() {
                let branch_output = match anti_aliasing {
                    AntiAliasing::Off => branch.gain * basis[branch.order],
                    AntiAliasing::Adaa1 | AntiAliasing::Adaa2 => self.branch_states[branch_index]
                        [channel]
                        .process(branch_input, branch.order, branch.gain, anti_aliasing),
                };
                shaped += self.branch_states[branch_index][channel]
                    .filter
                    .process(branch_output);
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
        for (branch_index, branch) in self.branches.iter().enumerate() {
            let filter_spec = self
                .filter_specs
                .get(branch_index)
                .copied()
                .unwrap_or(HammersteinFilterSpec::OnePole(branch.cutoff_hz));
            if branch.cutoff_hz > nyquist {
                return Err(AnalogError::ParameterOutOfRange {
                    parameter: "branch_cutoff_hz",
                    value: branch.cutoff_hz,
                    min: 0.0,
                    max: nyquist,
                });
            }
            if let HammersteinFilterSpec::Biquad {
                frequency_hz,
                q,
                db_gain,
                ..
            } = filter_spec
            {
                validate_finite_range("branch_frequency_hz", frequency_hz, f32::EPSILON, nyquist)?;
                validate_finite_range("branch_q", q, 0.0, 100.0)?;
                validate_finite_range("branch_db_gain", db_gain, -128.0, 128.0)?;
            }
            let mut states = Vec::with_capacity(spec.channels);
            for _ in 0..spec.channels {
                states.push(HammersteinBranchState::new(
                    branch.order,
                    filter_spec,
                    spec.sample_rate,
                ));
            }
            branch_states.push(states);
        }
        let mut dc_blockers = Vec::with_capacity(spec.channels);
        for _ in 0..spec.channels {
            dc_blockers.push(DcBlocker::new(spec.sample_rate));
        }

        let mut pre_filters = Vec::with_capacity(spec.channels);
        for _ in 0..spec.channels {
            let filter = match self.pre_filter {
                Some(pre_filter) if pre_filter.frequency_hz > nyquist => {
                    return Err(AnalogError::ParameterOutOfRange {
                        parameter: "pre_filter_frequency_hz",
                        value: pre_filter.frequency_hz,
                        min: f32::EPSILON,
                        max: nyquist,
                    });
                }
                Some(pre_filter) => Some(Biquad::new(
                    pre_filter.filter_type,
                    pre_filter.frequency_hz,
                    spec.sample_rate,
                    pre_filter.q,
                    pre_filter.db_gain,
                )),
                None => None,
            };
            pre_filters.push(filter);
        }

        self.drive_db.reconfigure(spec.sample_rate);
        self.output_gain_db.reconfigure(spec.sample_rate);
        self.character.reconfigure(spec.sample_rate);
        self.amount.reconfigure(spec.sample_rate);
        self.mix.reconfigure(spec.sample_rate);
        self.branch_states = branch_states;
        self.dc_blockers = dc_blockers;
        self.pre_filters = pre_filters;
        self.simd_bounded = vec![0.0; spec.max_block_frames];
        self.simd_basis = vec![0.0; spec.max_block_frames];
        self.simd_shaped = vec![0.0; spec.max_block_frames];
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
        for filter in self.pre_filters.iter_mut().flatten() {
            filter.reset();
        }
        self.simd_bounded.fill(0.0);
        self.simd_basis.fill(0.0);
        self.simd_shaped.fill(0.0);
    }

    fn process_interleaved(
        &mut self,
        samples: &mut [f32],
        frames: usize,
    ) -> Result<(), AnalogError> {
        let spec = self.spec.ok_or(AnalogError::NotPrepared)?;
        checked_block_len(spec, frames, samples.len())?;
        let anti_aliasing = self.anti_aliasing;
        if frames > 0 && anti_aliasing == AntiAliasing::Off && self.controls_are_settled() {
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

#[derive(Debug, Clone, Copy)]
enum HammersteinFilterSpec {
    OnePole(f32),
    Biquad {
        filter_type: BiquadFilterType,
        frequency_hz: f32,
        q: f32,
        db_gain: f32,
    },
}

#[derive(Debug, Clone)]
enum HammersteinFilterState {
    OnePole(OnePoleLowpass),
    Biquad(Biquad<f32>),
}

impl HammersteinFilterState {
    fn new(spec: HammersteinFilterSpec, sample_rate: f32) -> Self {
        match spec {
            HammersteinFilterSpec::OnePole(cutoff_hz) => {
                Self::OnePole(OnePoleLowpass::new(cutoff_hz, sample_rate))
            }
            HammersteinFilterSpec::Biquad {
                filter_type,
                frequency_hz,
                q,
                db_gain,
            } => Self::Biquad(Biquad::new(
                filter_type,
                frequency_hz,
                sample_rate,
                q,
                db_gain,
            )),
        }
    }

    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        match self {
            Self::OnePole(filter) => filter.process(input),
            Self::Biquad(filter) => filter.process(input),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::OnePole(filter) => filter.reset(),
            Self::Biquad(filter) => filter.reset(),
        }
    }
}

#[derive(Debug, Clone)]
struct HammersteinBranchState {
    filter: HammersteinFilterState,
    adaa1: Adaa1,
    adaa2: Adaa2,
}

impl HammersteinBranchState {
    fn new(order: usize, filter_spec: HammersteinFilterSpec, sample_rate: f32) -> Self {
        Self {
            filter: HammersteinFilterState::new(filter_spec, sample_rate),
            adaa1: Adaa1::new(chebyshev_f(order), chebyshev_ad1(order)),
            adaa2: Adaa2::new(
                chebyshev_f(order),
                chebyshev_ad1(order),
                chebyshev_ad2(order),
            ),
        }
    }

    #[inline]
    fn process(&mut self, input: f32, order: usize, gain: f32, anti_aliasing: AntiAliasing) -> f32 {
        let value = match anti_aliasing {
            AntiAliasing::Off => chebyshev_f(order)(input as f64) as f32,
            AntiAliasing::Adaa1 => self.adaa1.process(input),
            AntiAliasing::Adaa2 => self.adaa2.process(input),
        };
        finite_output(value * gain)
    }

    fn reset(&mut self) {
        self.filter.reset();
        self.adaa1.reset();
        self.adaa2.reset();
    }
}

#[inline]
fn chebyshev_f(order: usize) -> fn(f64) -> f64 {
    match order {
        1 => chebyshev_1,
        2 => chebyshev_2,
        3 => chebyshev_3,
        4 => chebyshev_4,
        _ => chebyshev_5,
    }
}

#[inline]
fn chebyshev_ad1(order: usize) -> fn(f64) -> f64 {
    match order {
        1 => chebyshev_1_ad1,
        2 => chebyshev_2_ad1,
        3 => chebyshev_3_ad1,
        4 => chebyshev_4_ad1,
        _ => chebyshev_5_ad1,
    }
}

#[inline]
fn chebyshev_ad2(order: usize) -> fn(f64) -> f64 {
    match order {
        1 => chebyshev_1_ad2,
        2 => chebyshev_2_ad2,
        3 => chebyshev_3_ad2,
        4 => chebyshev_4_ad2,
        _ => chebyshev_5_ad2,
    }
}

#[inline]
fn chebyshev_1(x: f64) -> f64 {
    x.tanh()
}

#[inline]
fn chebyshev_2(x: f64) -> f64 {
    let z = x.tanh();
    2.0 * z * z - 1.0
}

#[inline]
fn chebyshev_3(x: f64) -> f64 {
    let z = x.tanh();
    4.0 * z * z * z - 3.0 * z
}

#[inline]
fn chebyshev_4(x: f64) -> f64 {
    let z = x.tanh();
    8.0 * z.powi(4) - 8.0 * z * z + 1.0
}

#[inline]
fn chebyshev_5(x: f64) -> f64 {
    let z = x.tanh();
    16.0 * z.powi(5) - 20.0 * z.powi(3) + 5.0 * z
}

#[inline]
fn log_cosh(x: f64) -> f64 {
    let magnitude = x.abs();
    magnitude + (-2.0 * magnitude).exp().ln_1p() - std::f64::consts::LN_2
}

#[inline]
fn tanh_ad2(x: f64) -> f64 {
    let sign = x.signum();
    let magnitude = x.abs();
    sign * (0.5 * (magnitude * magnitude + dilog_neg((-2.0 * magnitude).exp()))
        - std::f64::consts::LN_2 * magnitude
        + std::f64::consts::PI * std::f64::consts::PI / 24.0)
}

#[inline]
fn dilog_neg(z: f64) -> f64 {
    if z < 1e-15 {
        return 0.0;
    }
    if (1.0 - z).abs() < 1e-12 {
        return -std::f64::consts::PI.powi(2) / 12.0;
    }
    if z <= 1.0 {
        let mut result = 0.0;
        let mut power = 1.0;
        for k in 1..=200 {
            power *= z;
            let term = power / (k * k) as f64;
            result += if k % 2 == 0 { term } else { -term };
            if term.abs() < 1e-15 {
                break;
            }
        }
        result
    } else {
        let log_z = z.ln();
        -dilog_neg(1.0 / z) - std::f64::consts::PI.powi(2) / 6.0 - 0.5 * log_z * log_z
    }
}

#[inline]
fn chebyshev_1_ad1(x: f64) -> f64 {
    log_cosh(x)
}

#[inline]
fn chebyshev_2_ad1(x: f64) -> f64 {
    x - 2.0 * x.tanh()
}

#[inline]
fn chebyshev_3_ad1(x: f64) -> f64 {
    log_cosh(x) - 2.0 * x.tanh().powi(2)
}

#[inline]
fn chebyshev_4_ad1(x: f64) -> f64 {
    x - (8.0 / 3.0) * x.tanh().powi(3)
}

#[inline]
fn chebyshev_5_ad1(x: f64) -> f64 {
    let z = x.tanh();
    log_cosh(x) + 2.0 * z.powi(2) - 4.0 * z.powi(4)
}

#[inline]
fn chebyshev_1_ad2(x: f64) -> f64 {
    tanh_ad2(x)
}

#[inline]
fn chebyshev_2_ad2(x: f64) -> f64 {
    0.5 * x * x - 2.0 * log_cosh(x)
}

#[inline]
fn chebyshev_3_ad2(x: f64) -> f64 {
    tanh_ad2(x) - 2.0 * (x - x.tanh())
}

#[inline]
fn chebyshev_4_ad2(x: f64) -> f64 {
    let z = x.tanh();
    0.5 * x * x - (8.0 / 3.0) * (log_cosh(x) + 0.5 * (1.0 - z * z) - 0.5)
}

#[inline]
fn chebyshev_5_ad2(x: f64) -> f64 {
    let z = x.tanh();
    tanh_ad2(x) - 2.0 * x + 2.0 * z + (4.0 / 3.0) * z.powi(3)
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
    fn optional_wiener_pre_filter_is_prepared_and_disabled_by_default() {
        let branch = HammersteinBranch::new(1, 1.0, 0.0).unwrap();
        let mut plain = HammersteinModel::with_branches(&[branch]).unwrap();
        let mut filtered = plain.clone();
        assert!(plain.pre_filter().is_none());
        filtered.set_pre_filter(Some(
            HammersteinPreFilter::new(BiquadFilterType::Highpass, 1_000.0, 0.707, 0.0).unwrap(),
        ));
        plain.prepare(ProcessSpec::new(48_000.0, 1, 256)).unwrap();
        filtered
            .prepare(ProcessSpec::new(48_000.0, 1, 256))
            .unwrap();
        let mut plain_samples = vec![0.5_f32; 256];
        let mut filtered_samples = plain_samples.clone();
        plain.process_interleaved(&mut plain_samples, 256).unwrap();
        filtered
            .process_interleaved(&mut filtered_samples, 256)
            .unwrap();
        assert!(filtered_samples[255].abs() < plain_samples[255].abs());
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

    #[test]
    fn adaa_antiderivatives_match_each_chebyshev_branch_order() {
        let step = 1e-5;
        for order in 1..=5 {
            let function = chebyshev_f(order);
            let first = chebyshev_ad1(order);
            let second = chebyshev_ad2(order);
            for x in [-3.0, -0.7, 0.2, 1.1, 4.0] {
                let first_derivative = (first(x + step) - first(x - step)) / (2.0 * step);
                let second_derivative = (second(x + step) - second(x - step)) / (2.0 * step);
                assert!(
                    (first_derivative - function(x)).abs() < 2e-5,
                    "order={order} x={x} first derivative={first_derivative} expected={} ",
                    function(x)
                );
                assert!(
                    (second_derivative - first(x)).abs() < 2e-5,
                    "order={order} x={x} second derivative={second_derivative} expected={} ",
                    first(x)
                );
            }
        }
    }
}

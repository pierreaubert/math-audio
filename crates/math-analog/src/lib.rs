//! Host-independent analog-style coloration models.
//!
//! This crate intentionally describes mathematical signal models rather than
//! pretending that a generic curve is a physical tube, tape machine, or
//! transformer.  Hardware-specific models belong here only after their
//! coefficients and validation target are documented.
//!
//! The public processing contract is deliberately small: prepare a model for
//! a fixed interleaved layout, reset it deterministically, and process blocks
//! in place.  Preparation is the allocation boundary; steady-state processing
//! does not allocate or change topology.

mod chain;
mod curves;
mod hammerstein;
mod harmonics;
mod level;
mod process;
mod stateful;
mod static_color;

pub mod analysis;

pub use curves::{asymmetric_style, normalized_soft_clip, tape_style, tube_style};
pub use hammerstein::{HammersteinBranch, HammersteinModel};
pub use harmonics::{AntiAliasing, HarmonicModel};
pub use level::{
    DEFAULT_REFERENCE_LEVEL_DBFS, calibrated_input_gain, db_to_gain, dbfs_to_vu, gain_to_db,
    vu_to_dbfs,
};
pub use process::{AnalogError, AnalogProcessor, ProcessSpec};
pub use stateful::{TapeModel, TransformerModel};
pub use static_color::{StaticColorModel, StaticCurve};

/// The model family selected for one prepared processor.
///
/// Dispatch happens once per block.  Each variant owns its sample loop, so
/// there is no virtual dispatch in the realtime path.
#[derive(Debug)]
pub enum AnalogModel {
    /// Controlled second/third-harmonic baseline.
    Harmonics(HarmonicModel),
    /// A bounded memoryless curve, explicitly not a hardware emulation.
    Static(StaticColorModel),
    /// A bounded, user-coefficient parallel-Hammerstein model.
    Hammerstein(HammersteinModel),
    /// A bounded stylized stateful tape-memory target, not a tape emulation.
    Tape(TapeModel),
    /// A bounded stylized stateful transformer-flux target, not a transformer emulation.
    Transformer(TransformerModel),
}

impl AnalogModel {
    /// Stable append-only identifiers for serialized model selection.
    pub const HARMONICS_ID: u32 = 0;
    pub const STATIC_ID: u32 = 1;
    pub const HAMMERSTEIN_ID: u32 = 2;
    pub const TAPE_ID: u32 = 3;
    pub const TRANSFORMER_ID: u32 = 4;

    /// Decode a serialized model identifier without mutating an existing
    /// model.  Unknown identifiers are rejected rather than guessed.
    pub fn from_id(id: u32) -> Result<Self, AnalogError> {
        match id {
            Self::HARMONICS_ID => Ok(Self::default()),
            Self::STATIC_ID => Ok(Self::Static(StaticColorModel::default())),
            Self::HAMMERSTEIN_ID => Ok(Self::Hammerstein(HammersteinModel::generic_coloration()?)),
            Self::TAPE_ID => Ok(Self::Tape(TapeModel::default())),
            Self::TRANSFORMER_ID => Ok(Self::Transformer(TransformerModel::default())),
            unknown => Err(AnalogError::UnknownModelId(unknown)),
        }
    }

    pub fn model_id(&self) -> u32 {
        match self {
            Self::Harmonics(_) => Self::HARMONICS_ID,
            Self::Static(_) => Self::STATIC_ID,
            Self::Hammerstein(_) => Self::HAMMERSTEIN_ID,
            Self::Tape(_) => Self::TAPE_ID,
            Self::Transformer(_) => Self::TRANSFORMER_ID,
        }
    }
}

impl Default for AnalogModel {
    fn default() -> Self {
        Self::Harmonics(HarmonicModel::default())
    }
}

impl AnalogProcessor for AnalogModel {
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), AnalogError> {
        match self {
            Self::Harmonics(model) => model.prepare(spec),
            Self::Static(model) => model.prepare(spec),
            Self::Hammerstein(model) => model.prepare(spec),
            Self::Tape(model) => model.prepare(spec),
            Self::Transformer(model) => model.prepare(spec),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Harmonics(model) => model.reset(),
            Self::Static(model) => model.reset(),
            Self::Hammerstein(model) => model.reset(),
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
            Self::Harmonics(model) => model.process_interleaved(samples, frames),
            Self::Static(model) => model.process_interleaved(samples, frames),
            Self::Hammerstein(model) => model.process_interleaved(samples, frames),
            Self::Tape(model) => model.process_interleaved(samples, frames),
            Self::Transformer(model) => model.process_interleaved(samples, frames),
        }
    }

    fn latency_samples(&self) -> usize {
        match self {
            Self::Harmonics(model) => model.latency_samples(),
            Self::Static(model) => model.latency_samples(),
            Self::Hammerstein(model) => model.latency_samples(),
            Self::Tape(model) => model.latency_samples(),
            Self::Transformer(model) => model.latency_samples(),
        }
    }
}

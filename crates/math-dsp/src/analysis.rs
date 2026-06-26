//! FFT-based frequency analysis for recorded signals
//!
//! This module provides functions to analyze recorded audio signals and extract:
//! - Frequency spectrum (magnitude in dBFS)
//! - Phase spectrum (compensated for latency)
//! - Latency estimation via cross-correlation
//! - Microphone compensation for calibrated measurements
//! - Standalone WAV buffer analysis (wav2csv functionality)

use rustfft::FftPlanner;
use std::cell::RefCell;

thread_local! {
    static FFT_PLANNER: RefCell<FftPlanner<f32>> = RefCell::new(FftPlanner::new());
}

mod analyze;
pub mod analyzer;
mod apply;
mod compute;
mod estimate;
mod interpolate;
mod load;
mod microphone_compensation;
mod misc;
mod plan;
mod smooth;
#[cfg(test)]
mod tests;
mod types;
mod wav_analysis_config;
mod write;

pub use analyze::*;
pub use analyzer::*;
pub use compute::*;
pub use estimate::*;
pub use microphone_compensation::*;
pub use misc::*;
pub use plan::*;
pub use smooth::*;
pub use types::*;
pub use wav_analysis_config::*;
pub use write::*;

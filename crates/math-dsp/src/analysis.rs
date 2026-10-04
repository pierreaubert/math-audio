//! FFT-based frequency analysis for recorded signals
//!
//! This module provides functions to analyze recorded audio signals and extract:
//! - Frequency spectrum (magnitude in dBFS)
//! - Phase spectrum (compensated for latency)
//! - Latency estimation via cross-correlation
//! - Microphone compensation for calibrated measurements
//! - Standalone WAV buffer analysis (wav2csv functionality)

use realfft::RealFftPlanner;
use rustfft::FftPlanner;
use std::cell::RefCell;

thread_local! {
    static FFT_PLANNER: RefCell<FftPlanner<f32>> = RefCell::new(FftPlanner::new());
    static REAL_FFT_PLANNER: RefCell<RealFftPlanner<f32>> = RefCell::new(RealFftPlanner::new());
}

mod analyze;
pub mod analyzer;
mod apply;
mod capture_quality;
mod common_clock;
mod compute;
mod estimate;
mod interpolate;
mod load;
pub mod lsqr;
mod measurement;
mod microphone_compensation;
mod misc;
pub mod parallel_hammerstein;
mod plan;
pub mod polynomial_convolution;
pub mod right_preconditioner;
mod smooth;
#[cfg(test)]
mod tests;
mod types;
mod wav_analysis_config;
mod write;

pub use analyze::*;
pub use analyzer::*;
pub use capture_quality::*;
pub use common_clock::*;
pub use compute::*;
pub use estimate::*;
pub use measurement::*;
pub use microphone_compensation::*;
pub use misc::*;
#[doc(inline)]
pub use parallel_hammerstein::{
    CapturedTrainingOutput, DesignNumericalDiagnostics, DesignNumericalStatus,
    DesignResourceEstimate, FitRecordDiagnostics, FitSolverStop, FitUnavailable,
    FitUnavailableReason, HeldOutRecord, ParallelHammersteinCandidate,
    ParallelHammersteinFitOptions, ParallelHammersteinOutcome, PolynomialFitDiagnostics,
    PolynomialFitError, PreparedParallelHammersteinDesign, QualificationStatus, SolverDiagnostics,
    TrainingInputReference, fit_parallel_hammerstein, prepare_parallel_hammerstein_design,
};
pub(crate) use plan::deconvolve_sweep_f64_spectrum;
pub use plan::*;
pub use smooth::*;
pub use types::*;
pub use wav_analysis_config::*;
pub use write::*;

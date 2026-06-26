//! FIR filter design from frequency response
//!
//! This module provides functions to generate FIR filters that match a target
//! frequency response, with support for different phase types including
//! Kirkeby regularized inversion for room correction.

mod context;
mod fir_design_config;
mod fir_phase;
mod generate;
mod misc;
mod pre_ringing_config;
#[cfg(test)]
mod tests;
mod types;

#[allow(unused_imports)]
pub use context::FirDesignContext;
pub use fir_design_config::*;
pub use fir_phase::*;
pub use generate::*;
pub use misc::*;
pub use pre_ringing_config::*;
pub use types::*;

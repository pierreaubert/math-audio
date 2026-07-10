//! Frequency-domain differentiable audio DSP.
//!
//! This crate provides LTI audio modules whose parameters can be optimized
//! via analytical gradients of the complex frequency response.

#![warn(clippy::pedantic)]

pub mod fft;
pub mod gain;
pub mod gradient;
pub mod iir;
pub mod loss;
pub mod module;
pub mod optim;
pub mod system;
pub mod tensor;

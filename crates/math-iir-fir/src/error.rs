//! Error types for IIR filter operations.
//!
//! This module provides structured error handling for IIR filter creation
//! and manipulation, following the Microsoft Rust Guidelines pattern.

use thiserror::Error;

/// Errors that can occur during IIR filter operations.
#[derive(Debug, Error)]
pub enum IirError {
    /// Q factor is invalid (must be > 0).
    #[error("invalid Q factor: {q} (must be > 0)")]
    InvalidQ {
        /// The invalid Q value
        q: f64,
    },

    /// Frequency is invalid (must be > 0 and < Nyquist).
    #[error("invalid frequency: {freq} Hz (must be > 0 and < Nyquist frequency {nyquist} Hz)")]
    InvalidFrequency {
        /// The invalid frequency value
        freq: f64,
        /// The Nyquist frequency (sample_rate / 2)
        nyquist: f64,
    },

    /// Sample rate is invalid (must be > 0).
    #[error("invalid sample rate: {sample_rate} Hz (must be > 0)")]
    InvalidSampleRate {
        /// The invalid sample rate value
        sample_rate: f64,
    },

    /// Gain value is invalid (non-finite).
    #[error("invalid gain: {gain_db} dB (must be finite)")]
    InvalidGain {
        /// The invalid gain value
        gain_db: f64,
    },
}

/// A specialized `Result` type for IIR operations.
pub type Result<T> = std::result::Result<T, IirError>;

/// Errors that can occur during FIR filter construction.
///
/// Returned by the fallible `Fir::try_*` constructors. The infallible
/// `Fir::new_custom` / `Fir::lowpass` / `Fir::highpass` / `Fir::bandpass` /
/// `Fir::bandstop` constructors panic with one of these errors as the payload
/// (via `expect`) when validation fails, in all build profiles.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum FirError {
    /// Coefficient list is empty (`Fir::try_new_custom` requires ≥ 1 tap).
    #[error("FIR filter must have at least one tap (got empty coefficients)")]
    EmptyCoeffs,

    /// Tap count is zero (windowed-sinc constructors require `n_taps > 0`).
    #[error("invalid tap count: {n_taps} (must be > 0)")]
    InvalidTaps {
        /// The invalid tap count.
        n_taps: usize,
    },

    /// Sample rate is invalid (must be > 0).
    #[error("invalid sample rate: {sample_rate} Hz (must be > 0)")]
    InvalidSampleRate {
        /// The invalid sample rate value.
        sample_rate: f64,
    },

    /// Cutoff frequency is invalid (must be > 0 and < Nyquist).
    #[error("invalid frequency: {freq} Hz (must be > 0 and < Nyquist frequency {nyquist} Hz)")]
    InvalidFrequency {
        /// The invalid frequency value.
        freq: f64,
        /// The Nyquist frequency (sample_rate / 2).
        nyquist: f64,
    },

    /// Band edges are invalid (require `0 < freq_low < freq_high < Nyquist`).
    #[error(
        "invalid band edges: low {freq_low} Hz, high {freq_high} Hz \
         (require 0 < low < high < Nyquist {nyquist} Hz)"
    )]
    InvalidBand {
        /// The lower cutoff frequency.
        freq_low: f64,
        /// The upper cutoff frequency.
        freq_high: f64,
        /// The Nyquist frequency (sample_rate / 2).
        nyquist: f64,
    },
}

/// A specialized `Result` type for FIR operations.
pub type FirResult<T> = std::result::Result<T, FirError>;

impl IirError {
    /// Returns `true` if this is a frequency-related error.
    pub fn is_frequency_error(&self) -> bool {
        matches!(self, IirError::InvalidFrequency { .. })
    }

    /// Returns `true` if this is a Q-factor error.
    pub fn is_q_error(&self) -> bool {
        matches!(self, IirError::InvalidQ { .. })
    }

    /// Returns `true` if this is a sample rate error.
    pub fn is_sample_rate_error(&self) -> bool {
        matches!(self, IirError::InvalidSampleRate { .. })
    }

    /// Returns `true` if this is a gain error.
    pub fn is_gain_error(&self) -> bool {
        matches!(self, IirError::InvalidGain { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = IirError::InvalidQ { q: -1.0 };
        assert_eq!(err.to_string(), "invalid Q factor: -1 (must be > 0)");
    }

    #[test]
    fn test_frequency_error_display() {
        let err = IirError::InvalidFrequency {
            freq: 25000.0,
            nyquist: 24000.0,
        };
        assert!(err.to_string().contains("25000"));
        assert!(err.to_string().contains("24000"));
    }

    #[test]
    fn test_is_frequency_error() {
        let freq_err = IirError::InvalidFrequency {
            freq: 0.0,
            nyquist: 24000.0,
        };
        let q_err = IirError::InvalidQ { q: -1.0 };

        assert!(freq_err.is_frequency_error());
        assert!(!q_err.is_frequency_error());
    }

    #[test]
    fn test_fir_error_display() {
        assert_eq!(
            FirError::EmptyCoeffs.to_string(),
            "FIR filter must have at least one tap (got empty coefficients)"
        );
        assert_eq!(
            FirError::InvalidTaps { n_taps: 0 }.to_string(),
            "invalid tap count: 0 (must be > 0)"
        );
        let err = FirError::InvalidSampleRate { sample_rate: 0.0 };
        assert!(err.to_string().contains('0'));
        let err = FirError::InvalidFrequency {
            freq: 30000.0,
            nyquist: 24000.0,
        };
        assert!(err.to_string().contains("30000"));
        assert!(err.to_string().contains("24000"));
        let err = FirError::InvalidBand {
            freq_low: 2000.0,
            freq_high: 500.0,
            nyquist: 24000.0,
        };
        assert!(err.to_string().contains("2000"));
        assert!(err.to_string().contains("500"));
    }

    #[test]
    fn test_is_q_error() {
        let q_err = IirError::InvalidQ { q: 0.0 };
        let gain_err = IirError::InvalidGain {
            gain_db: f64::INFINITY,
        };

        assert!(q_err.is_q_error());
        assert!(!gain_err.is_q_error());
    }
}

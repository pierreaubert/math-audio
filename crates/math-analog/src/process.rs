use std::fmt;

use thiserror::Error;

/// Fixed processing configuration for an interleaved processor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProcessSpec {
    pub sample_rate: f32,
    pub channels: usize,
    pub max_block_frames: usize,
}

impl ProcessSpec {
    pub const fn new(sample_rate: f32, channels: usize, max_block_frames: usize) -> Self {
        Self {
            sample_rate,
            channels,
            max_block_frames,
        }
    }

    pub(crate) fn validate(self) -> Result<(), AnalogError> {
        if !self.sample_rate.is_finite() || self.sample_rate <= 0.0 {
            return Err(AnalogError::InvalidSampleRate(self.sample_rate));
        }
        if self.channels == 0 {
            return Err(AnalogError::InvalidChannels);
        }
        if self.max_block_frames == 0 {
            return Err(AnalogError::InvalidMaxBlockFrames);
        }
        Ok(())
    }
}

/// Errors returned before a processor mutates its state.
#[derive(Debug, Error, PartialEq)]
pub enum AnalogError {
    #[error("sample rate must be finite and greater than zero, got {0}")]
    InvalidSampleRate(f32),
    #[error("channel count must be greater than zero")]
    InvalidChannels,
    #[error("maximum block size must be greater than zero")]
    InvalidMaxBlockFrames,
    #[error("requested {requested} frames, but the prepared maximum is {maximum}")]
    BlockTooLarge { requested: usize, maximum: usize },
    #[error("interleaved buffer has {actual} samples; expected {expected}")]
    BufferLengthMismatch { expected: usize, actual: usize },
    #[error("processor has not been prepared")]
    NotPrepared,
    #[error("parameter {parameter} must be finite")]
    NonFiniteParameter { parameter: &'static str },
    #[error("parameter {parameter}={value} is outside [{min}, {max}]")]
    ParameterOutOfRange {
        parameter: &'static str,
        value: f32,
        min: f32,
        max: f32,
    },
    #[error("invalid defect configuration: {0}")]
    InvalidDefectConfig(String),
    #[error("unknown analog model id {0}")]
    UnknownModelId(u32),
}

/// Processing contract shared by all analog models.
pub trait AnalogProcessor {
    /// Allocate/rebuild channel and filter state for a fixed stream layout.
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), AnalogError>;

    /// Clear filters, nonlinear history, and control interpolation state.
    fn reset(&mut self);

    /// Process exactly `frames` of interleaved audio in place.
    fn process_interleaved(
        &mut self,
        samples: &mut [f32],
        frames: usize,
    ) -> Result<(), AnalogError>;

    /// Additional latency introduced by the prepared model.
    fn latency_samples(&self) -> usize;
}

/// A finite, sample-rate-aware one-pole control smoother.
#[derive(Clone, Copy)]
pub(crate) struct ControlSmoother {
    target: f32,
    current: f32,
    time_ms: f32,
    coeff: f32,
}

impl fmt::Debug for ControlSmoother {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ControlSmoother")
            .field("target", &self.target)
            .field("current", &self.current)
            .field("time_ms", &self.time_ms)
            .field("coeff", &self.coeff)
            .finish()
    }
}

impl ControlSmoother {
    pub(crate) fn new(value: f32, time_ms: f32, sample_rate: f32) -> Self {
        Self {
            target: value,
            current: value,
            time_ms,
            coeff: Self::coefficient(time_ms, sample_rate),
        }
    }

    fn coefficient(time_ms: f32, sample_rate: f32) -> f32 {
        if time_ms <= 0.0 || !sample_rate.is_finite() || sample_rate <= 0.0 {
            0.0
        } else {
            (-1.0 / (time_ms * 0.001 * sample_rate)).exp()
        }
    }

    pub(crate) fn reconfigure(&mut self, sample_rate: f32) {
        self.coeff = Self::coefficient(self.time_ms, sample_rate);
    }

    pub(crate) fn set_target(&mut self, value: f32) {
        self.target = value;
        if self.coeff == 0.0 {
            self.current = value;
        }
    }

    pub(crate) fn set_immediate(&mut self, value: f32) {
        self.target = value;
        self.current = value;
    }

    #[inline]
    pub(crate) fn advance(&mut self) -> f32 {
        if self.coeff == 0.0 || (self.current - self.target).abs() < 1e-6 {
            self.current = self.target;
        } else {
            self.current = self.target + self.coeff * (self.current - self.target);
        }
        self.current
    }

    pub(crate) fn reset(&mut self) {
        self.current = self.target;
    }

    pub(crate) fn target(&self) -> f32 {
        self.target
    }

    pub(crate) fn is_settled(&self) -> bool {
        self.coeff == 0.0 || (self.current - self.target).abs() < 1e-6
    }
}

pub(crate) fn validate_finite_range(
    parameter: &'static str,
    value: f32,
    min: f32,
    max: f32,
) -> Result<(), AnalogError> {
    if !value.is_finite() {
        return Err(AnalogError::NonFiniteParameter { parameter });
    }
    if !(min..=max).contains(&value) {
        return Err(AnalogError::ParameterOutOfRange {
            parameter,
            value,
            min,
            max,
        });
    }
    Ok(())
}

pub(crate) fn checked_block_len(
    spec: ProcessSpec,
    frames: usize,
    sample_count: usize,
) -> Result<(), AnalogError> {
    if frames > spec.max_block_frames {
        return Err(AnalogError::BlockTooLarge {
            requested: frames,
            maximum: spec.max_block_frames,
        });
    }
    let expected = frames
        .checked_mul(spec.channels)
        .ok_or(AnalogError::BufferLengthMismatch {
            expected: usize::MAX,
            actual: sample_count,
        })?;
    if expected != sample_count {
        return Err(AnalogError::BufferLengthMismatch {
            expected,
            actual: sample_count,
        });
    }
    Ok(())
}

/// Replace invalid callback input before it can enter persistent state.
#[inline]
pub(crate) fn sanitize_sample(sample: f32) -> f32 {
    if sample.is_finite() {
        sample.clamp(-16.0, 16.0)
    } else {
        0.0
    }
}

#[inline]
pub(crate) fn finite_output(sample: f32) -> f32 {
    if sample.is_finite() {
        flush_denormal(sample.clamp(-128.0, 128.0))
    } else {
        0.0
    }
}

/// Keep subnormal values out of persistent realtime state and output buffers.
#[inline]
pub(crate) fn flush_denormal(sample: f32) -> f32 {
    if sample.abs() < f32::MIN_POSITIVE {
        0.0
    } else {
        sample
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subnormal_values_are_flushed() {
        assert_eq!(flush_denormal(f32::MIN_POSITIVE * 0.5), 0.0);
        assert_eq!(flush_denormal(-f32::MIN_POSITIVE * 0.5), 0.0);
        assert_eq!(flush_denormal(f32::MIN_POSITIVE), f32::MIN_POSITIVE);
    }

    #[test]
    fn reconfigure_preserves_the_configured_time_constant() {
        let mut smoother = ControlSmoother::new(0.0, 80.0, 48_000.0);
        smoother.set_target(1.0);
        smoother.reconfigure(96_000.0);
        let one_tau_samples = (0.080_f32 * 96_000.0_f32).round() as usize;
        for _ in 0..one_tau_samples {
            smoother.advance();
        }
        assert!((smoother.current - (1.0 - (-1.0_f32).exp())).abs() < 0.01);
    }
}

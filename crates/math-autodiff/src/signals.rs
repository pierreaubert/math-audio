//! Time-domain signal generators for differentiable DSP experiments.

#![allow(
    clippy::cast_precision_loss,
    reason = "sample counts and channel counts fit exactly in f64 for practical values"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "signal indices are derived from positive finite floating-point intervals and clamped to valid ranges"
)]

use std::f64::consts::PI;

use ndarray::Array3;
use num_complex::Complex;
use rand::Rng;

use crate::tensor::DiffTensor;

/// Available signal types for `signal_gallery`.
#[derive(Debug, Clone, Copy)]
pub enum SignalType {
    /// Single unit impulse at sample 0.
    Impulse,
    /// Sine wave at `freq_hz`.
    Sine {
        /// Frequency in Hz.
        freq_hz: f64,
    },
    /// Linear sweep from `f0_hz` to `f1_hz`.
    Sweep {
        /// Start frequency in Hz.
        f0_hz: f64,
        /// End frequency in Hz.
        f1_hz: f64,
    },
    /// Uniform white noise in `[-1, 1)`.
    WhiteNoise,
    /// Exponential decay with time constant controlled by `rate`.
    ExpDecay {
        /// Decay rate in s⁻¹ (higher = faster decay).
        rate: f64,
    },
    /// Velvet noise with average `density` impulses per second.
    VelvetNoise {
        /// Impulse density in impulses per second.
        density: f64,
    },
}

/// Generate a real-valued signal of shape `(1, n_samples, n_channels)`.
///
/// The returned tensor contains complex samples with zero imaginary part so it
/// can be fed directly into `Fft`/`Shell`.
///
/// # Panics
///
/// Panics if `n_samples` or `n_channels` is zero.
#[must_use]
pub fn signal_gallery(
    signal_type: SignalType,
    n_samples: usize,
    n_channels: usize,
    fs: f64,
) -> DiffTensor<f64> {
    assert!(
        n_samples > 0,
        "signal_gallery: n_samples must be greater than 0"
    );
    assert!(
        n_channels > 0,
        "signal_gallery: n_channels must be greater than 0"
    );
    assert!(fs > 0.0, "signal_gallery: fs must be greater than 0");

    let mut data = Array3::<Complex<f64>>::zeros((1, n_samples, n_channels));

    match signal_type {
        SignalType::Impulse => {
            for ch in 0..n_channels {
                data[[0, 0, ch]] = Complex::new(1.0, 0.0);
            }
        }
        SignalType::Sine { freq_hz } => {
            let omega = 2.0 * PI * freq_hz / fs;
            for n in 0..n_samples {
                let sample = Complex::new((omega * n as f64).sin(), 0.0);
                for ch in 0..n_channels {
                    data[[0, n, ch]] = sample;
                }
            }
        }
        SignalType::Sweep { f0_hz, f1_hz } => {
            let duration = n_samples as f64 / fs;
            for n in 0..n_samples {
                let t = n as f64 / fs;
                // Linear chirp phase: phi(t) = 2*pi*(f0*t + (f1-f0)*t^2/(2*T))
                let phase = 2.0 * PI * (f0_hz * t + (f1_hz - f0_hz) * t * t / (2.0 * duration));
                let sample = Complex::new(phase.sin(), 0.0);
                for ch in 0..n_channels {
                    data[[0, n, ch]] = sample;
                }
            }
        }
        SignalType::WhiteNoise => {
            let mut rng = rand::rng();
            for n in 0..n_samples {
                let sample = Complex::new(rng.random::<f64>() * 2.0 - 1.0, 0.0);
                for ch in 0..n_channels {
                    data[[0, n, ch]] = sample;
                }
            }
        }
        SignalType::ExpDecay { rate } => {
            for n in 0..n_samples {
                let sample = Complex::new((-rate * n as f64 / fs).exp(), 0.0);
                for ch in 0..n_channels {
                    data[[0, n, ch]] = sample;
                }
            }
        }
        SignalType::VelvetNoise { density } => {
            assert!(
                density > 0.0,
                "signal_gallery: velvet noise density must be positive"
            );
            let td = fs / density; // average spacing in samples
            let num_impulses = (n_samples as f64 / td).floor() as usize;
            let mut rng = rand::rng();

            for ch in 0..n_channels {
                for i in 0..num_impulses.max(1) {
                    let grid = i as f64 * td;
                    let jitter = rng.random::<f64>() * (td - 1.0);
                    let mut idx = (grid + jitter).ceil() as usize;
                    if i == 0 {
                        idx = 0;
                    }
                    idx = idx.min(n_samples - 1);
                    let sign = if rng.random::<bool>() { 1.0 } else { -1.0 };
                    data[[0, idx, ch]] = Complex::new(sign, 0.0);
                }
            }
        }
    }

    DiffTensor::from_array(data.into_dyn())
}

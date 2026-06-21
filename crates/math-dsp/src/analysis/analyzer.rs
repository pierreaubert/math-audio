//! Reusable spectrum analyzer that avoids per-call allocations.

use rustfft::num_complex::Complex;

use super::compute::compute_welch_spectrum_with_buffers;
use super::plan::plan_fft_forward;
use super::types::SpectrumResult;
use crate::stft::generate_hann_window_symmetric;

/// Reusable analyzer for FFT-based spectral estimation.
///
/// Keeps the FFT plan, window buffer, and FFT scratch buffers across calls
/// so that repeated analysis of same-sized blocks does not allocate.
pub struct SpectrumAnalyzer {
    fft_size: usize,
    hann_window: Vec<f32>,
    windowed: Vec<f32>,
    buffer: Vec<Complex<f32>>,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

impl SpectrumAnalyzer {
    /// Create a reusable analyzer for the given FFT size.
    #[must_use]
    pub fn new(fft_size: usize) -> Self {
        Self {
            fft_size,
            hann_window: generate_hann_window_symmetric(fft_size),
            windowed: vec![0.0_f32; fft_size],
            buffer: vec![Complex::new(0.0, 0.0); fft_size],
            fft: plan_fft_forward(fft_size),
        }
    }

    /// Analyze `signal` using Welch's method and reuse all internal buffers.
    ///
    /// # Errors
    /// Returns an error if the signal is empty.
    pub fn welch(&mut self, signal: &[f32], sample_rate: u32, overlap: f32) -> SpectrumResult {
        compute_welch_spectrum_with_buffers(
            signal,
            sample_rate,
            self.fft_size,
            overlap,
            &self.hann_window,
            &self.fft,
            &mut self.windowed,
            &mut self.buffer,
        )
    }
}

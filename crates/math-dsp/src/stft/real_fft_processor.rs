use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use std::sync::Arc;

/// Thin wrapper around `realfft` encapsulating planner + buffers for
/// single-channel use. Provides forward (real→complex) and optional
/// inverse (complex→real) FFT.
pub struct RealFftProcessor {
    #[allow(dead_code)]
    pub fft_size: usize,
    pub spectrum_size: usize,
    pub(super) fft_forward: Arc<dyn RealToComplex<f32>>,
    pub(super) fft_inverse: Option<Arc<dyn ComplexToReal<f32>>>,
    /// Cached scratch for the forward FFT (avoids per-call allocation).
    pub(super) forward_scratch: Vec<Complex<f32>>,
    /// Cached scratch for the inverse FFT (empty for forward-only processors).
    pub(super) inverse_scratch: Vec<Complex<f32>>,
    pub time_buffer: Vec<f32>,
    pub freq_buffer: Vec<Complex<f32>>,
}

impl RealFftProcessor {
    /// Create a forward-only FFT processor (no inverse).
    pub fn new_forward_only(fft_size: usize) -> Self {
        let spectrum_size = fft_size / 2 + 1;
        let mut planner = RealFftPlanner::<f32>::new();
        let fft_forward = planner.plan_fft_forward(fft_size);
        let forward_scratch = vec![Complex::new(0.0, 0.0); fft_forward.get_scratch_len()];

        Self {
            fft_size,
            spectrum_size,
            fft_forward,
            fft_inverse: None,
            forward_scratch,
            inverse_scratch: Vec::new(),
            time_buffer: vec![0.0; fft_size],
            freq_buffer: vec![Complex::new(0.0, 0.0); spectrum_size],
        }
    }

    /// Create a bidirectional FFT processor (forward + inverse).
    #[allow(dead_code)]
    pub fn new_bidirectional(fft_size: usize) -> Self {
        let spectrum_size = fft_size / 2 + 1;
        let mut planner = RealFftPlanner::<f32>::new();
        let fft_forward = planner.plan_fft_forward(fft_size);
        let fft_inverse = planner.plan_fft_inverse(fft_size);
        let forward_scratch = vec![Complex::new(0.0, 0.0); fft_forward.get_scratch_len()];
        let inverse_scratch = vec![Complex::new(0.0, 0.0); fft_inverse.get_scratch_len()];

        Self {
            fft_size,
            spectrum_size,
            fft_forward,
            fft_inverse: Some(fft_inverse),
            forward_scratch,
            inverse_scratch,
            time_buffer: vec![0.0; fft_size],
            freq_buffer: vec![Complex::new(0.0, 0.0); spectrum_size],
        }
    }

    /// Perform forward FFT: time_buffer → freq_buffer.
    /// The caller should fill `time_buffer` before calling this.
    pub fn forward(&mut self) {
        self.fft_forward
            .process_with_scratch(
                &mut self.time_buffer,
                &mut self.freq_buffer,
                &mut self.forward_scratch,
            )
            .expect("FFT forward failed");
    }

    /// Perform inverse FFT: freq_buffer → time_buffer.
    /// Panics if this processor was created with `new_forward_only`.
    #[allow(dead_code)]
    pub fn inverse(&mut self) {
        self.fft_inverse
            .as_ref()
            .expect("Inverse FFT not available (forward-only processor)")
            .process_with_scratch(
                &mut self.freq_buffer,
                &mut self.time_buffer,
                &mut self.inverse_scratch,
            )
            .expect("FFT inverse failed");
    }
}

use super::generate::design_dual_windows;
use super::real_fft_processor::RealFftProcessor;
use super::ring_accumulator::RingAccumulator;
use rustfft::num_complex::Complex;

/// Dual-window STFT processor.
///
/// Uses a long analysis window for frequency resolution and a shorter
/// synthesis window for low-latency output. The steady-state output latency
/// is `analysis_size - 1` samples, including startup. Initial zero padding
/// supplies the earlier frames needed to reconstruct the first input sample.
/// Use [`Self::finish_into`] to recover the delayed tail of a finite stream.
pub struct DualWindowStft {
    pub(super) analysis_window: Vec<f32>,
    pub(super) synthesis_window: Vec<f32>,
    pub(super) analysis_size: usize,
    /// Input ring buffer sized to analysis window
    pub(super) input_ring: RingAccumulator,
    /// Overlap-add output accumulator
    pub(super) output_accum: Vec<f32>,
    pub(super) output_read_pos: usize,
    /// FFT processor (analysis size)
    pub(super) fft: RealFftProcessor,
}

impl DualWindowStft {
    /// Create a new dual-window STFT processor.
    ///
    /// # Arguments
    /// * `analysis_size` - Analysis window size (determines frequency resolution)
    /// * `synthesis_size` - Synthesis window size (determines output latency)
    /// * `hop_size` - Hop size in samples
    pub fn new(analysis_size: usize, synthesis_size: usize, hop_size: usize) -> Self {
        let (analysis_window, synthesis_window) =
            design_dual_windows(analysis_size, synthesis_size, hop_size);

        let fft = RealFftProcessor::new_bidirectional(analysis_size);

        let mut processor = Self {
            analysis_window,
            synthesis_window,
            analysis_size,
            input_ring: RingAccumulator::new(analysis_size, hop_size),
            output_accum: vec![0.0; analysis_size * 3],
            output_read_pos: 0,
            fft,
        };
        processor.prime();
        processor
    }

    fn prime(&mut self) {
        for _ in 0..self.analysis_size - self.input_ring.hop_size {
            self.input_ring.push(0.0);
        }
    }

    /// Push a single sample. Returns `true` when a hop boundary is reached.
    ///
    /// When `true`, the spectrum is available in `freq_buffer_mut()` for
    /// in-place modification. Call `synthesize_in_place()` after modifying.
    pub fn analyze(&mut self, sample: f32) -> bool {
        if !self.input_ring.push(sample) {
            return false;
        }

        // Read the analysis window worth of samples directly into the FFT
        // time buffer, then apply the analysis window in place.
        self.input_ring.read_window(&mut self.fft.time_buffer);
        for (t, &w) in self
            .fft
            .time_buffer
            .iter_mut()
            .zip(self.analysis_window.iter())
        {
            *t *= w;
        }

        // Forward FFT
        self.fft.forward();

        true
    }

    /// Access the frequency buffer for in-place modification after `analyze()` returns `true`.
    pub fn freq_buffer_mut(&mut self) -> &mut [Complex<f32>] {
        &mut self.fft.freq_buffer
    }

    /// Synthesize output from the current frequency buffer (after in-place modification).
    ///
    /// Call this after `analyze()` returns `true` and the spectrum has been modified
    /// via `freq_buffer_mut()`. The output samples accumulate in the internal buffer
    /// and can be read via `read_output()`.
    ///
    /// # Scaling convention
    /// `realfft`'s IFFT output is unnormalized: `IFFT(FFT(x))[n] = N * x[n]`.
    /// `design_dual_windows` normalizes the synthesis window so that the COLA
    /// overlap-add sum equals 1 assuming a hypothetical normalized IFFT (output = 1).
    /// The `1/N` factor corrects for `realfft`'s unnormalized output.
    /// The two normalizations serve distinct roles: synthesis window → COLA unity;
    /// `1/N` → IFFT scale convention. Neither makes the other redundant.
    pub fn synthesize_in_place(&mut self) {
        // Inverse FFT (operates on self.fft.freq_buffer directly)
        self.fft.inverse();

        // Apply synthesis window and overlap-add.
        // 1/analysis_size compensates for realfft's unnormalized IFFT (output × N).
        let scale = 1.0 / self.analysis_size as f32;
        for i in 0..self.analysis_size {
            let pos = (self.output_read_pos + i) % self.output_accum.len();
            self.output_accum[pos] += self.fft.time_buffer[i] * self.synthesis_window[i] * scale;
        }
    }

    /// Read one output sample. Returns 0.0 if no output is ready yet.
    pub fn read_output(&mut self) -> f32 {
        let sample = self.output_accum[self.output_read_pos];
        self.output_accum[self.output_read_pos] = 0.0;
        self.output_read_pos = (self.output_read_pos + 1) % self.output_accum.len();
        sample
    }

    /// Process a block: analyze, apply user function, synthesize.
    ///
    /// # Arguments
    /// * `input` - Input samples
    /// * `output` - Output buffer (same length as input)
    /// * `process_fn` - Function to modify the spectrum (called at each hop boundary)
    pub fn process_block<F>(&mut self, input: &[f32], output: &mut [f32], mut process_fn: F)
    where
        F: FnMut(&mut [Complex<f32>]),
    {
        assert_eq!(
            input.len(),
            output.len(),
            "STFT input/output length mismatch"
        );
        for (i, &sample) in input.iter().enumerate() {
            if self.analyze(sample) {
                process_fn(&mut self.fft.freq_buffer);
                self.synthesize_in_place();
            }
            output[i] = self.read_output();
        }
    }

    /// Drain the delayed finite-stream tail into `output`, then reset.
    ///
    /// Calls the spectral callback for zero-padded final frames. This operation
    /// does not allocate. For zero-preserving callbacks, repeated drains after
    /// reset produce zeros. Only STFT state is reset; callback state is caller-owned.
    ///
    /// # Panics
    /// Panics unless `output.len()` equals [`Self::latency_samples`].
    pub fn finish_into<F>(&mut self, output: &mut [f32], mut process_fn: F)
    where
        F: FnMut(&mut [Complex<f32>]),
    {
        assert_eq!(
            output.len(),
            self.latency_samples(),
            "STFT drain length mismatch"
        );
        for sample in output {
            if self.analyze(0.0) {
                process_fn(&mut self.fft.freq_buffer);
                self.synthesize_in_place();
            }
            *sample = self.read_output();
        }
        self.reset();
    }

    /// Return the input-to-output delay, including startup, in samples.
    pub fn latency_samples(&self) -> usize {
        self.analysis_size - 1
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        self.input_ring.reset();
        self.prime();
        self.output_accum.fill(0.0);
        self.output_read_pos = 0;
    }
}

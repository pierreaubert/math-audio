use super::heap_entry::HeapEntry;

/// Phase reconstruction processor using RTPGHI.
pub struct RtpghiProcessor {
    pub(super) fft_size: usize,
    pub(super) hop_size: usize,
    /// Gamma parameter for the Gaussian-like window
    /// (Hann: gamma = Cg * fft_size² with Cg = 0.25645, per LTFAT `pghi_findgamma`)
    pub(super) gamma: f64,
    /// Previous frame log-magnitudes
    pub(super) prev_log_mag: Vec<f64>,
    /// Previous frame phases (wrapped to (-pi, pi])
    pub(super) prev_phase: Vec<f64>,
    /// Whether we have a previous frame
    pub(super) has_prev: bool,
    /// Relative log-magnitude tolerance: bins more than `log_mag_tol` below the
    /// frame maximum log-magnitude are skipped (LTFAT `tol = 1e-6` convention;
    /// the value is a natural-log offset, so ln(1e-6) ≈ -13.8155)
    pub(super) log_mag_tol: f64,
    /// Effective absolute log-magnitude threshold of the previous frame
    pub(super) prev_tol: f64,

    // Pre-allocated scratch buffers for process_frame_into (zero-alloc hot path)
    pub(super) scratch_log_mag: Vec<f64>,
    pub(super) scratch_phases: Vec<f64>,
    pub(super) scratch_integrated: Vec<bool>,
    pub(super) scratch_d_phase_time: Vec<f64>,
    pub(super) scratch_d_phase_freq: Vec<f64>,
    pub(super) scratch_heap: Vec<HeapEntry>,
}

impl RtpghiProcessor {
    /// Create a new RTPGHI processor.
    ///
    /// # Arguments
    /// * `fft_size` - FFT size (must be power of 2)
    /// * `hop_size` - Hop size in samples
    pub fn new(fft_size: usize, hop_size: usize) -> Self {
        let spectrum_size = fft_size / 2 + 1;

        // Gamma of a Hann window approximated as Gaussian:
        // gamma = Cg * M^2 with Cg = 0.25645 (LTFAT `pghi_findgamma` table)
        let gamma = 0.25645 * (fft_size as f64) * (fft_size as f64);

        Self {
            fft_size,
            hop_size,
            gamma,
            prev_log_mag: vec![f64::NEG_INFINITY; spectrum_size],
            prev_phase: vec![0.0; spectrum_size],
            has_prev: false,
            // ln(1e-6): keep bins within 1e-6 (-120 dB) of the frame maximum
            log_mag_tol: -13.815_510_557_964_274,
            prev_tol: f64::NEG_INFINITY,
            scratch_log_mag: vec![0.0; spectrum_size],
            scratch_phases: vec![0.0; spectrum_size],
            scratch_integrated: vec![false; spectrum_size],
            scratch_d_phase_time: vec![0.0; spectrum_size],
            scratch_d_phase_freq: vec![0.0; spectrum_size],
            scratch_heap: Vec::with_capacity(spectrum_size),
        }
    }

    /// Process one STFT frame: given magnitudes, reconstruct phases.
    ///
    /// # Arguments
    /// * `magnitudes` - Magnitude spectrum (spectrum_size = fft_size/2 + 1)
    ///
    /// # Returns
    /// Reconstructed phase values for each bin.
    ///
    /// This is a thin allocation wrapper over [`Self::process_frame_into`].
    /// For real-time / allocation-free streaming, call `process_frame_into` directly.
    pub fn process_frame(&mut self, magnitudes: &[f32]) -> Vec<f32> {
        let n = self.fft_size / 2 + 1;
        let mut out = vec![0.0f32; n];
        self.process_frame_into(magnitudes, &mut out);
        out
    }

    /// Process one STFT frame without allocations: given magnitudes, write
    /// reconstructed phases into the provided output slice.
    ///
    /// # Arguments
    /// * `magnitudes` - Magnitude spectrum (spectrum_size = fft_size/2 + 1)
    /// * `phases_out` - Output slice for reconstructed phases (same length)
    ///
    /// # Panics
    /// If `magnitudes` or `phases_out` length does not equal `fft_size/2 + 1`.
    pub fn process_frame_into(&mut self, magnitudes: &[f32], phases_out: &mut [f32]) {
        let spectrum_size = self.fft_size / 2 + 1;
        assert_eq!(magnitudes.len(), spectrum_size);
        assert_eq!(phases_out.len(), spectrum_size);

        let log_mag = &mut self.scratch_log_mag;
        let phases = &mut self.scratch_phases;
        let integrated = &mut self.scratch_integrated;
        let d_phase_time = &mut self.scratch_d_phase_time;
        let d_phase_freq = &mut self.scratch_d_phase_freq;

        // Compute log-magnitudes
        for (i, &m) in magnitudes.iter().enumerate() {
            log_mag[i] = if m > 0.0 {
                (m as f64).ln()
            } else {
                f64::NEG_INFINITY
            };
        }

        // Effective threshold: relative to the frame maximum magnitude
        // (LTFAT `tol` convention; log_mag_tol = ln(1e-6) by default)
        let frame_max = log_mag.iter().fold(f64::NEG_INFINITY, |acc, &v| acc.max(v));
        let tol = frame_max + self.log_mag_tol;

        // Zero scratch
        for v in phases.iter_mut() {
            *v = 0.0;
        }
        for v in integrated.iter_mut() {
            *v = false;
        }

        if !self.has_prev {
            // First frame: use zero phase
            self.prev_log_mag.copy_from_slice(log_mag);
            self.prev_phase.copy_from_slice(phases);
            self.prev_tol = tol;
            self.has_prev = true;
            for (out, &p) in phases_out.iter_mut().zip(phases.iter()) {
                *out = p as f32;
            }
            return;
        }

        // Phase gradient estimation via the Cauchy-Riemann relations for a
        // Gaussian-window STFT (LTFAT `comp_pghiphasegrad`):
        //   tgrad = omega_k * a + (a*M/gamma) * d(log|F|)/d(bin)
        //   fgrad = -(gamma/(a*M)) * d(log|F|)/d(frame)
        let a = self.hop_size as f64;
        let two_pi = 2.0 * std::f64::consts::PI;
        let gamma = self.gamma;
        let m = self.fft_size as f64;
        let am_over_gamma = a * m / gamma;
        let gamma_over_am = gamma / (a * m);
        let prev_tol = self.prev_tol;

        // Time-direction phase gradient: expected bin advance plus the
        // frequency-direction log-magnitude slope
        for k in 0..spectrum_size {
            let omega_k = two_pi * k as f64 / m;
            let expected_advance = omega_k * a;
            let time_grad =
                if k > 0 && k + 1 < spectrum_size && log_mag[k - 1] > tol && log_mag[k + 1] > tol {
                    am_over_gamma * (log_mag[k + 1] - log_mag[k - 1]) / 2.0
                } else {
                    0.0
                };
            d_phase_time[k] = expected_advance + time_grad;
        }

        // Frequency-direction phase gradient: minus the scaled time-direction
        // log-magnitude slope
        for k in 0..spectrum_size {
            d_phase_freq[k] = if log_mag[k] > tol && self.prev_log_mag[k] > prev_tol {
                -gamma_over_am * (log_mag[k] - self.prev_log_mag[k])
            } else {
                0.0
            };
        }

        // Build sorted list by magnitude descending (reuse pre-allocated vec)
        self.scratch_heap.clear();
        for (k, &mag) in log_mag.iter().enumerate() {
            if mag > tol {
                self.scratch_heap.push(HeapEntry {
                    magnitude: mag,
                    bin: k,
                });
            }
        }
        // Sort descending by magnitude (highest first) while reusing scratch storage.
        // This preserves the priority integration order without allocating a BinaryHeap.
        self.scratch_heap.sort_unstable_by(|a, b| b.cmp(a));

        // Integrate phases starting from loudest bins
        for idx in 0..self.scratch_heap.len() {
            let k = self.scratch_heap[idx].bin;
            if integrated[k] {
                continue;
            }

            // Time-direction estimate, brought onto the principal branch.
            // Neighbor estimates are aligned to it before averaging so that
            // values straddling the +/-pi wrap boundary do not cancel out.
            let phase_from_time = (self.prev_phase[k] + d_phase_time[k] + std::f64::consts::PI)
                .rem_euclid(two_pi)
                - std::f64::consts::PI;

            let phase_from_freq_below = if k > 0 && integrated[k - 1] {
                let est = phases[k - 1] + d_phase_freq[k - 1];
                Some(
                    phase_from_time
                        + (est - phase_from_time + std::f64::consts::PI).rem_euclid(two_pi)
                        - std::f64::consts::PI,
                )
            } else {
                None
            };

            let phase_from_freq_above = if k + 1 < spectrum_size && integrated[k + 1] {
                let est = phases[k + 1] - d_phase_freq[k + 1];
                Some(
                    phase_from_time
                        + (est - phase_from_time + std::f64::consts::PI).rem_euclid(two_pi)
                        - std::f64::consts::PI,
                )
            } else {
                None
            };

            let phase = match (phase_from_freq_below, phase_from_freq_above) {
                (Some(below), Some(above)) => {
                    let avg = (below + above) / 2.0;
                    if self.prev_log_mag[k] > prev_tol {
                        (avg + phase_from_time) / 2.0
                    } else {
                        avg
                    }
                }
                (Some(below), None) => {
                    if self.prev_log_mag[k] > prev_tol {
                        (below + phase_from_time) / 2.0
                    } else {
                        below
                    }
                }
                (None, Some(above)) => {
                    if self.prev_log_mag[k] > prev_tol {
                        (above + phase_from_time) / 2.0
                    } else {
                        above
                    }
                }
                (None, None) => phase_from_time,
            };

            // Wrap to (-pi, pi]: keeps prev_phase bounded so the f64 -> f32
            // output conversion does not accumulate quantization error on
            // long-running streams.
            phases[k] = (phase + std::f64::consts::PI).rem_euclid(two_pi) - std::f64::consts::PI;
            integrated[k] = true;
        }

        // Bins below threshold get zero phase
        for k in 0..spectrum_size {
            if !integrated[k] {
                phases[k] = 0.0;
            }
        }

        // Store for next frame
        self.prev_log_mag.copy_from_slice(log_mag);
        self.prev_phase.copy_from_slice(phases);
        self.prev_tol = tol;

        // Write output
        for (out, &p) in phases_out.iter_mut().zip(phases.iter()) {
            *out = p as f32;
        }
    }

    /// Reset the processor state.
    pub fn reset(&mut self) {
        self.prev_log_mag.fill(f64::NEG_INFINITY);
        self.prev_phase.fill(0.0);
        self.prev_tol = f64::NEG_INFINITY;
        self.has_prev = false;
    }

    /// Get the latency in samples.
    pub fn latency_samples(&self) -> usize {
        self.fft_size
    }
}

/// Convenience: time-stretch a signal using RTPGHI for phase reconstruction.
///
/// # Arguments
/// * `magnitudes_frames` - Sequence of magnitude spectra (one per STFT frame)
/// * `stretch_factor` - Time stretch factor (2.0 = twice as long)
/// * `fft_size` - FFT size used for STFT
/// * `hop_size` - Original hop size
///
/// # Returns
/// Reconstructed phases for stretched output (interpolated magnitude frames)
pub fn stretch_with_rtpghi(
    magnitude_frames: &[Vec<f32>],
    stretch_factor: f64,
    fft_size: usize,
    hop_size: usize,
) -> Vec<Vec<f32>> {
    if magnitude_frames.is_empty() || stretch_factor <= 0.0 {
        return Vec::new();
    }

    let num_input_frames = magnitude_frames.len();
    let num_output_frames = (num_input_frames as f64 * stretch_factor).ceil() as usize;
    let spectrum_size = fft_size / 2 + 1;

    // Reconstruct phases frame by frame, reusing the interpolation and phase
    // buffers instead of allocating per output frame.
    let mut processor = RtpghiProcessor::new(fft_size, hop_size);
    let mut mag_buf = vec![0.0f32; spectrum_size];
    let mut phases_out = vec![0.0f32; spectrum_size];
    let mut phases = Vec::with_capacity(num_output_frames);

    for i in 0..num_output_frames {
        let src_pos = i as f64 / stretch_factor;
        let src_idx = src_pos.floor() as usize;
        let frac = (src_pos - src_idx as f64) as f32;

        // Interpolate magnitudes
        if src_idx + 1 < num_input_frames {
            for ((out, &a), &b) in mag_buf
                .iter_mut()
                .zip(&magnitude_frames[src_idx])
                .zip(&magnitude_frames[src_idx + 1])
            {
                *out = a * (1.0 - frac) + b * frac;
            }
        } else if src_idx < num_input_frames {
            mag_buf.copy_from_slice(&magnitude_frames[src_idx]);
        } else {
            mag_buf.copy_from_slice(magnitude_frames.last().unwrap());
        }

        processor.process_frame_into(&mag_buf, &mut phases_out);
        phases.push(phases_out.clone());
    }

    phases
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::stft::RealFftProcessor;

    /// Helper: compute STFT magnitudes of a signal
    fn compute_stft_magnitudes(signal: &[f32], fft_size: usize, hop_size: usize) -> Vec<Vec<f32>> {
        let spectrum_size = fft_size / 2 + 1;
        let window: Vec<f32> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / fft_size as f32).cos()))
            .collect();

        let mut frames = Vec::new();
        let mut fft = RealFftProcessor::new_forward_only(fft_size);

        let mut pos = 0;
        while pos + fft_size <= signal.len() {
            for i in 0..fft_size {
                fft.time_buffer[i] = signal[pos + i] * window[i];
            }
            fft.forward();

            let mags: Vec<f32> = fft.freq_buffer[..spectrum_size]
                .iter()
                .map(|c| (c.re * c.re + c.im * c.im).sqrt())
                .collect();
            frames.push(mags);
            pos += hop_size;
        }

        frames
    }

    #[test]
    fn test_identity_stretch() {
        let fft_size = 256;
        let hop_size = 64;
        let sample_rate = 48000.0;

        // Generate a pure tone
        let num_samples = 4096;
        let signal: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();

        let mags = compute_stft_magnitudes(&signal, fft_size, hop_size);
        assert!(!mags.is_empty());

        // Identity stretch (factor = 1.0)
        let phases = stretch_with_rtpghi(&mags, 1.0, fft_size, hop_size);
        assert_eq!(phases.len(), mags.len());

        // All phases should be finite
        for frame in &phases {
            for &p in frame {
                assert!(p.is_finite(), "Phase should be finite, got {p}");
            }
        }
    }

    #[test]
    fn test_2x_stretch_doubles_frames() {
        let fft_size = 256;
        let hop_size = 64;

        // Simple magnitude frames
        let spectrum_size = fft_size / 2 + 1;
        let frame: Vec<f32> = (0..spectrum_size)
            .map(|i| (i as f32).exp().recip())
            .collect();
        let mags = vec![frame; 10];

        let stretched = stretch_with_rtpghi(&mags, 2.0, fft_size, hop_size);
        assert_eq!(stretched.len(), 20);
    }

    #[test]
    fn test_no_nan_inf() {
        let fft_size = 512;
        let hop_size = 128;
        let spectrum_size = fft_size / 2 + 1;

        let mut processor = RtpghiProcessor::new(fft_size, hop_size);

        // Process several frames with varying magnitudes
        for frame_idx in 0..20 {
            let mags: Vec<f32> = (0..spectrum_size)
                .map(|k| {
                    let freq_factor = 1.0 - k as f32 / spectrum_size as f32;
                    let time_factor = 1.0 + 0.5 * (frame_idx as f32 * 0.3).sin();
                    freq_factor * time_factor
                })
                .collect();

            let phases = processor.process_frame(&mags);
            for (k, &p) in phases.iter().enumerate() {
                assert!(
                    p.is_finite(),
                    "Phase at bin {k}, frame {frame_idx} is not finite: {p}"
                );
            }
        }
    }

    #[test]
    fn test_reset() {
        let fft_size = 256;
        let hop_size = 64;
        let spectrum_size = fft_size / 2 + 1;

        let mut processor = RtpghiProcessor::new(fft_size, hop_size);
        let mags = vec![0.5; spectrum_size];

        // Process then reset
        let _ = processor.process_frame(&mags);
        assert!(processor.has_prev);

        processor.reset();
        assert!(!processor.has_prev);
    }

    #[test]
    fn test_empty_stretch() {
        let result = stretch_with_rtpghi(&[], 2.0, 256, 64);
        assert!(result.is_empty());
    }

    #[test]
    fn test_zero_magnitude_bins() {
        let fft_size = 256;
        let hop_size = 64;
        let spectrum_size = fft_size / 2 + 1;

        let mut processor = RtpghiProcessor::new(fft_size, hop_size);

        // All-zero magnitudes
        let mags = vec![0.0f32; spectrum_size];
        let _ = processor.process_frame(&mags);
        let phases = processor.process_frame(&mags);

        for &p in &phases {
            assert!(p.is_finite());
        }
    }

    /// Verify that `process_frame_into` produces the same results as `process_frame`.
    #[test]
    fn test_process_frame_into_matches_process_frame() {
        let fft_size = 512;
        let hop_size = 128;
        let spectrum_size = fft_size / 2 + 1;

        let mut proc_alloc = RtpghiProcessor::new(fft_size, hop_size);
        let mut proc_noalloc = RtpghiProcessor::new(fft_size, hop_size);

        for frame_idx in 0..15 {
            let mags: Vec<f32> = (0..spectrum_size)
                .map(|k| {
                    let freq_factor = 1.0 - k as f32 / spectrum_size as f32;
                    let time_factor = 1.0 + 0.5 * (frame_idx as f32 * 0.3).sin();
                    freq_factor * time_factor
                })
                .collect();

            let phases_alloc = proc_alloc.process_frame(&mags);
            let mut phases_noalloc = vec![0.0f32; spectrum_size];
            proc_noalloc.process_frame_into(&mags, &mut phases_noalloc);

            for (k, (&a, &b)) in phases_alloc.iter().zip(phases_noalloc.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-5,
                    "Mismatch at bin {k}, frame {frame_idx}: alloc={a}, noalloc={b}"
                );
            }
        }
    }

    /// Verify that `process_frame_into` produces finite phases and does not panic.
    #[test]
    fn test_process_frame_into_no_nan() {
        let fft_size = 256;
        let hop_size = 64;
        let spectrum_size = fft_size / 2 + 1;

        let mut processor = RtpghiProcessor::new(fft_size, hop_size);
        let mut phases = vec![0.0f32; spectrum_size];

        for frame_idx in 0..10 {
            let mags: Vec<f32> = (0..spectrum_size)
                .map(|k| {
                    let v = 0.5 + 0.5 * ((frame_idx * k) as f32 * 0.1).sin();
                    v.max(0.0)
                })
                .collect();

            processor.process_frame_into(&mags, &mut phases);
            for (k, &p) in phases.iter().enumerate() {
                assert!(
                    p.is_finite(),
                    "Phase at bin {k}, frame {frame_idx} is not finite: {p}"
                );
            }
        }
    }

    #[test]
    fn test_process_frame_into_first_frame_is_zero() {
        let fft_size = 256;
        let hop_size = 64;
        let spectrum_size = fft_size / 2 + 1;
        let mut processor = RtpghiProcessor::new(fft_size, hop_size);
        let mags = vec![1.0f32; spectrum_size];
        let mut phases = vec![0.0f32; spectrum_size];
        processor.process_frame_into(&mags, &mut phases);
        for &p in &phases {
            assert_eq!(p, 0.0);
        }
    }

    #[test]
    fn test_process_frame_into_reset_restarts() {
        let fft_size = 256;
        let hop_size = 64;
        let spectrum_size = fft_size / 2 + 1;
        let mut processor = RtpghiProcessor::new(fft_size, hop_size);
        let mags = vec![1.0f32; spectrum_size];
        let mut phases = vec![0.0f32; spectrum_size];

        processor.process_frame_into(&mags, &mut phases);
        processor.process_frame_into(&mags, &mut phases);
        let some_nonzero = phases.iter().any(|&p| p != 0.0);
        assert!(some_nonzero, "second frame should have non-zero phases");

        processor.reset();
        processor.process_frame_into(&mags, &mut phases);
        for &p in &phases {
            assert_eq!(p, 0.0, "after reset, first frame should be zero again");
        }
    }

    #[test]
    fn test_process_frame_into_all_below_threshold() {
        let fft_size = 256;
        let hop_size = 64;
        let spectrum_size = fft_size / 2 + 1;
        let mut processor = RtpghiProcessor::new(fft_size, hop_size);
        let prime = vec![1.0f32; spectrum_size];
        let _ = processor.process_frame(&prime);

        let mags = vec![0.0f32; spectrum_size];
        let mut phases = vec![0.0f32; spectrum_size];
        processor.process_frame_into(&mags, &mut phases);
        for &p in &phases {
            assert_eq!(p, 0.0);
        }
    }

    /// B2: gamma must use the LTFAT `pghi_findgamma` Hann constant
    /// Cg = 0.25645, i.e. gamma = 0.25645 * M^2.
    #[test]
    fn test_gamma_matches_hann_pghi_constant() {
        for &fft_size in &[256usize, 1024, 4096] {
            let processor = RtpghiProcessor::new(fft_size, fft_size / 4);
            let cg = processor.gamma / (fft_size as f64 * fft_size as f64);
            assert!(
                (cg - 0.25645).abs() < 1e-4,
                "gamma/M^2 for M={fft_size} is {cg}, expected Hann Cg = 0.25645"
            );
        }
    }

    /// B1: for a stationary off-bin sinusoid, the reconstructed phase in the
    /// peak bins must advance at the TRUE tone frequency, not at the bin
    /// center frequency.
    #[test]
    fn test_offbin_tone_phase_advances_at_true_frequency() {
        let fft_size = 1024;
        let hop_size = 256;
        let sample_rate = 48000.0f64;
        let tone_bin = 20.5f64; // exactly between bins 20 and 21
        let freq = tone_bin * sample_rate / fft_size as f64;

        let num_samples = 8192;
        let signal: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sample_rate).sin() as f32)
            .collect();

        let mags = compute_stft_magnitudes(&signal, fft_size, hop_size);
        assert!(mags.len() >= 10);

        let mut processor = RtpghiProcessor::new(fft_size, hop_size);
        let mut prev: Option<Vec<f32>> = None;
        // True per-frame phase advance (mod 2*pi): 2*pi*tone_bin*hop/M
        let true_advance = (2.0 * std::f64::consts::PI * tone_bin * hop_size as f64
            / fft_size as f64)
            .rem_euclid(2.0 * std::f64::consts::PI);
        let two_pi = 2.0 * std::f64::consts::PI;

        for frame in &mags {
            let phases = processor.process_frame(frame);
            if let Some(prev_phases) = &prev {
                for k in [20usize, 21] {
                    let diff = (phases[k] as f64 - prev_phases[k] as f64).rem_euclid(two_pi);
                    let err = (diff - true_advance)
                        .rem_euclid(two_pi)
                        .min((true_advance - diff).rem_euclid(two_pi));
                    assert!(
                        err < 0.1,
                        "bin {k}: phase advanced by {diff} rad/frame, expected {true_advance} (err {err} rad)"
                    );
                }
            }
            prev = Some(phases);
        }
    }

    /// B1 (end-to-end): stretch a stationary off-bin tone by a non-integer
    /// factor through `stretch_with_rtpghi` and verify that the reconstructed
    /// peak-bin phase keeps rotating at the TRUE tone rate (phasor coherence),
    /// not at the bin-center rate.
    #[test]
    fn test_stretched_offbin_tone_phase_coherence() {
        let fft_size = 1024;
        let hop_size = 256;
        let sample_rate = 48000.0f64;
        let tone_bin = 20.5f64; // exactly between bins 20 and 21
        let freq = tone_bin * sample_rate / fft_size as f64;

        let num_samples = 16384;
        let signal: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sample_rate).sin() as f32)
            .collect();

        let mags = compute_stft_magnitudes(&signal, fft_size, hop_size);
        let phases = stretch_with_rtpghi(&mags, 1.5, fft_size, hop_size);
        assert_eq!(phases.len(), (mags.len() as f64 * 1.5).ceil() as usize);

        // Coherence of the bin-20 phase series against the expected rotation
        // theta = 2*pi*tone_bin*hop/M per frame. With swapped/mis-scaled
        // gradients the phase rotates at the bin-center rate and the series
        // decorrelates (coherence ~ 0).
        let two_pi = 2.0 * std::f64::consts::PI;
        let theta = (two_pi * tone_bin * hop_size as f64 / fft_size as f64).rem_euclid(two_pi);
        let (mut re, mut im, mut w) = (0.0f64, 0.0f64, 0.0f64);
        for (n, ph) in phases.iter().enumerate().skip(2) {
            let z = ph[20] as f64 - theta * n as f64;
            re += z.cos();
            im += z.sin();
            w += 1.0;
        }
        let rho = (re * re + im * im).sqrt() / w;
        assert!(
            rho > 0.95,
            "peak-bin phasor coherence {rho:.3}, expected > 0.95 (phase rotation is garbage otherwise)"
        );
    }

    /// B3: bins more than 1e-6 (-120 dB) below the frame maximum must be
    /// skipped (LTFAT relative `tol` convention), not integrated.
    #[test]
    fn test_quiet_bins_below_relative_threshold_are_skipped() {
        let fft_size = 1024;
        let hop_size = 256;
        let spectrum_size = fft_size / 2 + 1;

        let mut processor = RtpghiProcessor::new(fft_size, hop_size);
        let prime = vec![1.0f32; spectrum_size];
        let _ = processor.process_frame(&prime);

        // One loud bin, everything else at 1e-9 of it (far below tol = 1e-6)
        let mut mags = vec![1.0e-9f32; spectrum_size];
        mags[101] = 1.0;
        let phases = processor.process_frame(&mags);

        // Loud bin: advance = 2*pi*101*256/1024 = 50.5*pi == pi/2 (mod 2*pi)
        assert!(
            (phases[101].abs() - std::f32::consts::FRAC_PI_2).abs() < 1e-3,
            "loud bin phase should be pi/2, got {}",
            phases[101]
        );
        // Quiet bins must be skipped: zero phase
        for (k, &p) in phases.iter().enumerate() {
            if k != 101 {
                assert_eq!(p, 0.0, "bin {k} below relative threshold must be skipped");
            }
        }
    }

    /// B4: integrated phases must stay wrapped to (-pi, pi] no matter how many
    /// frames have been processed.
    #[test]
    fn test_phases_stay_wrapped() {
        let fft_size = 1024;
        let hop_size = 256;
        let sample_rate = 48000.0f64;
        let freq = 20.5 * sample_rate / fft_size as f64;

        let num_samples = 200 * hop_size + fft_size;
        let signal: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sample_rate).sin() as f32)
            .collect();
        let mags = compute_stft_magnitudes(&signal, fft_size, hop_size);

        let mut processor = RtpghiProcessor::new(fft_size, hop_size);
        let bound = std::f32::consts::PI + 1e-4;
        for (n, frame) in mags.iter().enumerate() {
            let phases = processor.process_frame(frame);
            for (k, &p) in phases.iter().enumerate() {
                assert!(
                    p.abs() <= bound,
                    "frame {n} bin {k}: phase {p} escapes (-pi, pi]"
                );
            }
        }
    }

    /// B4: even with a huge previously accumulated phase (simulating a
    /// long-running stream), the output phase must remain accurate.
    #[test]
    fn test_phase_accuracy_after_large_accumulated_phase() {
        let fft_size = 1024;
        let hop_size = 256;
        let sample_rate = 48000.0f64;
        let tone_bin = 20.5f64;
        let freq = tone_bin * sample_rate / fft_size as f64;

        let num_samples = 4096;
        let signal: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sample_rate).sin() as f32)
            .collect();
        let mags = compute_stft_magnitudes(&signal, fft_size, hop_size);

        let mut processor = RtpghiProcessor::new(fft_size, hop_size);
        let _ = processor.process_frame(&mags[0]);

        // Simulate hours of accumulated unwrapped phase
        for p in processor.prev_phase.iter_mut() {
            *p = 1.0e9;
        }

        let phases = processor.process_frame(&mags[1]);

        let two_pi = 2.0 * std::f64::consts::PI;
        let true_advance = (2.0 * std::f64::consts::PI * tone_bin * hop_size as f64
            / fft_size as f64)
            .rem_euclid(two_pi);
        let expected = (1.0e9f64 + true_advance + std::f64::consts::PI).rem_euclid(two_pi)
            - std::f64::consts::PI;
        let got = phases[20] as f64;
        let err = (got - expected)
            .rem_euclid(two_pi)
            .min((expected - got).rem_euclid(two_pi));
        assert!(
            err < 1e-2,
            "peak bin phase {got} vs expected {expected} (err {err} rad) after large accumulation"
        );
    }
}

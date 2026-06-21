//! Reusable workspace for FIR design operations.
//!
//! [`FirDesignContext`] owns an FFT planner and scratch buffers so that
//! repeated calls to [`generate_fir_from_response`] or
//! [`generate_kirkeby_correction`] avoid reallocating planners and large
//! complex buffers on every invocation.

use num_complex::Complex64;
use rustfft::FftPlanner;
use rustfft::num_traits::Zero;

use super::super::fir::{WindowType, generate_window};
use super::fir_design_config::FirDesignConfig;
use super::fir_phase::{FirPhase, finalize_impulse_response};
use super::misc::interpolate_log_space;
use super::pre_ringing_config::suppress_pre_ringing;

/// Reusable workspace for FIR design.
///
/// The context caches an [`FftPlanner`] and internal buffers.  It is safe to
/// reuse across calls with different `n_taps` because buffers are resized on
/// demand.
pub struct FirDesignContext {
    planner: FftPlanner<f64>,
    complex_buf: Vec<Complex64>,
    real_buf: Vec<f64>,
    window_buf: Vec<f64>,
}

impl std::fmt::Debug for FirDesignContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FirDesignContext")
            .field("complex_buf_len", &self.complex_buf.len())
            .field("real_buf_len", &self.real_buf.len())
            .field("window_buf_len", &self.window_buf.len())
            .finish()
    }
}

impl Default for FirDesignContext {
    fn default() -> Self {
        Self::new()
    }
}

impl FirDesignContext {
    /// Create a new, empty context.
    #[must_use]
    pub fn new() -> Self {
        Self {
            planner: FftPlanner::new(),
            complex_buf: Vec::new(),
            real_buf: Vec::new(),
            window_buf: Vec::new(),
        }
    }

    /// Ensure internal buffers can hold the requested sizes.
    pub fn ensure_capacity(&mut self, complex_len: usize, real_len: usize, window_len: usize) {
        if self.complex_buf.len() < complex_len {
            self.complex_buf.resize(complex_len, Complex64::zero());
        }
        if self.real_buf.len() < real_len {
            self.real_buf.resize(real_len, 0.0);
        }
        if self.window_buf.len() < window_len {
            self.window_buf.resize(window_len, 0.0);
        }
    }

    /// Generate an FIR filter to match a target frequency response.
    ///
    /// This is the reusable equivalent of [`super::generate::generate_fir_from_response`].
    pub fn generate_fir_from_response(
        &mut self,
        freqs: &[f64],
        magnitude_db: &[f64],
        config: &FirDesignConfig,
    ) -> Vec<f64> {
        assert_eq!(
            freqs.len(),
            magnitude_db.len(),
            "freqs and magnitude_db must have same length"
        );
        assert!(
            !freqs.is_empty(),
            "freqs and magnitude_db must not be empty"
        );
        assert!(
            freqs.iter().all(|f| f.is_finite() && *f > 0.0),
            "freqs must contain finite positive values"
        );
        assert!(
            freqs.windows(2).all(|w| w[0] < w[1]),
            "freqs must be strictly increasing"
        );
        assert!(
            magnitude_db.iter().all(|db| db.is_finite()),
            "magnitude_db must contain finite values"
        );
        assert!(
            config.phase != FirPhase::Kirkeby,
            "Kirkeby correction requires measurement and target responses; use generate_kirkeby_correction"
        );

        let n_taps = config.n_taps;
        let sample_rate = config.sample_rate;
        let fft_size = (n_taps * 8).next_power_of_two().max(4096);
        let n_bins = fft_size / 2 + 1;

        self.ensure_capacity(fft_size, n_bins.max(n_taps), n_taps);

        let freq_step = sample_rate / fft_size as f64;
        for i in 0..n_bins {
            self.real_buf[i] = i as f64 * freq_step;
        }
        let linear_freqs = &self.real_buf[..n_bins];

        let interpolated_db = interpolate_log_space(freqs, magnitude_db, linear_freqs);

        for (i, &db) in interpolated_db.iter().enumerate().take(n_bins) {
            self.real_buf[i] = 10.0_f64.powf(db / 20.0);
        }
        let magnitude: Vec<f64> = self.real_buf[..n_bins].to_vec();

        let spectrum = match config.phase {
            FirPhase::Linear => magnitude.iter().map(|&m| Complex64::new(m, 0.0)).collect(),
            FirPhase::Minimum => self.generate_minimum_phase_spectrum(&magnitude, fft_size),
            FirPhase::Kirkeby => {
                unreachable!("Kirkeby correction must be generated with measurement data")
            }
        };

        let ir = self.spectrum_to_impulse_response(&spectrum, fft_size);
        finalize_impulse_response(
            ir,
            n_taps,
            config.phase,
            &config.window,
            config.pre_ringing.as_ref(),
            config.sample_rate,
        )
    }

    /// Generate Kirkeby regularized FIR correction filter.
    ///
    /// This is the reusable equivalent of
    /// [`super::generate::generate_kirkeby_correction`].
    pub fn generate_kirkeby_correction(
        &mut self,
        meas_freqs: &[f64],
        meas_magnitude_db: &[f64],
        meas_phase_deg: Option<&[f64]>,
        target_magnitude_db: &[f64],
        config: &FirDesignConfig,
    ) -> Vec<f64> {
        assert_eq!(
            meas_freqs.len(),
            meas_magnitude_db.len(),
            "meas_freqs and meas_magnitude_db must have same length"
        );
        assert_eq!(
            meas_freqs.len(),
            target_magnitude_db.len(),
            "meas_freqs and target_magnitude_db must have same length"
        );
        if let Some(phase) = meas_phase_deg {
            assert_eq!(
                meas_freqs.len(),
                phase.len(),
                "meas_phase_deg must match measurement length"
            );
        }

        let n_taps = config.n_taps;
        let sample_rate = config.sample_rate;
        let min_freq = config.min_freq;
        let max_freq = config.max_freq;

        let fft_len = (n_taps * 4).max(65536).next_power_of_two();
        let num_bins = fft_len / 2 + 1;
        let freq_step = sample_rate / fft_len as f64;

        self.ensure_capacity(fft_len, fft_len.max(num_bins), n_taps);

        for i in 0..num_bins {
            self.real_buf[i] = i as f64 * freq_step;
        }
        let linear_freqs: Vec<f64> = self.real_buf[..num_bins].to_vec();

        let meas_spl_interp = interpolate_log_space(meas_freqs, meas_magnitude_db, &linear_freqs);
        let target_spl_interp =
            interpolate_log_space(meas_freqs, target_magnitude_db, &linear_freqs);

        let excess_phase_correction: Option<Vec<f64>> = if config.correct_excess_phase {
            meas_phase_deg.map(|phase_deg| {
                let meas_phase_rad: Vec<f64> = phase_deg.iter().map(|&d| d.to_radians()).collect();

                let smoothed_phase_rad = if config.phase_smoothing_octaves > 0.0 {
                    super::super::phase_smooth::smooth_phase_via_group_delay(
                        meas_freqs,
                        &meas_phase_rad,
                        config.phase_smoothing_octaves,
                    )
                } else {
                    meas_phase_rad
                };

                let meas_phase_interp = super::super::phase_smooth::interpolate_phase_complex(
                    meas_freqs,
                    &smoothed_phase_rad,
                    &linear_freqs,
                );

                let min_phase = self.compute_minimum_phase_from_magnitude(&meas_spl_interp);

                meas_phase_interp
                    .iter()
                    .zip(min_phase.iter())
                    .map(|(&measured_rad, &min_rad)| {
                        let excess_rad = measured_rad - min_rad;
                        -excess_rad
                    })
                    .collect()
            })
        } else {
            None
        };

        let max_boost_db = 15.0;
        let max_cut_db = 20.0;
        let max_boost_linear = 10.0_f64.powf(max_boost_db / 20.0);
        let max_cut_linear = 10.0_f64.powf(max_cut_db / 20.0);
        let beta = 1.0 / (2.0 * max_boost_linear);

        for i in 0..num_bins {
            let f = self.real_buf[i];
            let rel_mag = 10.0_f64.powf((meas_spl_interp[i] - target_spl_interp[i]) / 20.0);

            let width = 10.0;
            let transition = if f < min_freq {
                ((f - (min_freq - width)) / width).clamp(0.0, 1.0)
            } else if f > max_freq {
                1.0 - ((f - max_freq) / width).clamp(0.0, 1.0)
            } else {
                1.0
            };

            let regularized_mag = rel_mag / (rel_mag * rel_mag + beta * beta);
            let limited_mag = regularized_mag.clamp(1.0 / max_cut_linear, max_boost_linear);
            let c_mag = 1.0 + transition * (limited_mag - 1.0);
            let c_phase = excess_phase_correction
                .as_ref()
                .map(|epc| epc[i] * transition)
                .unwrap_or(0.0);

            self.complex_buf[i] = Complex64::from_polar(c_mag, c_phase);
        }

        self.complex_buf[num_bins..fft_len].fill(Complex64::zero());
        self.complex_buf[fft_len / 2] = self.complex_buf[num_bins - 1];
        for i in 1..fft_len / 2 {
            self.complex_buf[fft_len - i] = self.complex_buf[i].conj();
        }

        let ifft = self.planner.plan_fft_inverse(fft_len);
        ifft.process(&mut self.complex_buf[..fft_len]);

        for i in 0..fft_len {
            self.real_buf[i] = self.complex_buf[i].re / fft_len as f64;
        }

        let shift = fft_len / 2;
        self.real_buf[..fft_len].rotate_right(shift);

        let start_idx = shift - n_taps / 2;
        let window = generate_window(n_taps, WindowType::Hann, 0.0);
        self.window_buf[..n_taps].copy_from_slice(&window[..n_taps]);
        let mut coeffs = vec![0.0; n_taps];
        for (i, coeff) in coeffs.iter_mut().enumerate() {
            let src_idx = start_idx + i;
            if src_idx < fft_len {
                *coeff = self.real_buf[src_idx] * self.window_buf[i];
            }
        }

        if let Some(pr_config) = &config.pre_ringing {
            suppress_pre_ringing(&mut coeffs, pr_config, sample_rate);
        }

        coeffs
    }

    /// Convert a magnitude spectrum to a minimum-phase complex spectrum.
    fn generate_minimum_phase_spectrum(
        &mut self,
        magnitude: &[f64],
        fft_size: usize,
    ) -> Vec<Complex64> {
        let n_bins = magnitude.len();
        self.ensure_capacity(fft_size, 0, 0);
        self.complex_buf[..fft_size].fill(Complex64::zero());

        for (i, &m) in magnitude.iter().enumerate().take(n_bins) {
            self.complex_buf[i] = Complex64::new(m.max(1e-9).ln(), 0.0);
        }

        for i in 1..n_bins {
            self.complex_buf[fft_size - i] = self.complex_buf[i].conj();
        }
        if fft_size.is_multiple_of(2) {
            let nyquist = self.complex_buf[n_bins - 1];
            self.complex_buf[n_bins - 1] = Complex64::new(nyquist.re, 0.0);
        }

        let ifft = self.planner.plan_fft_inverse(fft_size);
        ifft.process(&mut self.complex_buf[..fft_size]);
        for x in &mut self.complex_buf[..fft_size] {
            *x /= fft_size as f64;
        }

        // Keep DC, double positive frequencies, keep Nyquist.
        for i in 1..fft_size / 2 {
            self.complex_buf[i] *= 2.0;
        }
        for i in (fft_size / 2 + 1)..fft_size {
            self.complex_buf[i] = Complex64::zero();
        }

        let fft = self.planner.plan_fft_forward(fft_size);
        fft.process(&mut self.complex_buf[..fft_size]);

        self.complex_buf[..n_bins].iter().map(|c| c.exp()).collect()
    }

    /// Convert a complex spectrum to a real impulse response via IFFT.
    fn spectrum_to_impulse_response(&mut self, spectrum: &[Complex64], fft_size: usize) -> &[f64] {
        let n_bins = spectrum.len();
        self.ensure_capacity(fft_size, fft_size, 0);
        self.complex_buf[..fft_size].fill(Complex64::zero());

        self.complex_buf[0] = spectrum[0];
        for (i, s) in spectrum.iter().enumerate().take(n_bins).skip(1) {
            self.complex_buf[i] = *s;
            self.complex_buf[fft_size - i] = s.conj();
        }
        if fft_size.is_multiple_of(2) {
            self.complex_buf[n_bins - 1] = Complex64::new(spectrum[n_bins - 1].norm(), 0.0);
        }

        let ifft = self.planner.plan_fft_inverse(fft_size);
        ifft.process(&mut self.complex_buf[..fft_size]);

        for i in 0..fft_size {
            self.real_buf[i] = self.complex_buf[i].re / fft_size as f64;
        }

        &self.real_buf[..fft_size]
    }

    /// Compute minimum phase from magnitude response using Hilbert transform.
    fn compute_minimum_phase_from_magnitude(&mut self, magnitude_db: &[f64]) -> Vec<f64> {
        let n = magnitude_db.len();
        if n == 0 {
            return Vec::new();
        }

        for (i, &db) in magnitude_db.iter().enumerate() {
            self.real_buf[i] = db / 20.0 * 10.0_f64.ln();
        }
        let ln_mag: Vec<f64> = self.real_buf[..n].to_vec();

        let phase_rad = self.hilbert_transform(&ln_mag);
        phase_rad.iter().map(|&p| -p).collect()
    }

    /// Compute the Hilbert transform of a signal using FFT.
    fn hilbert_transform(&mut self, signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        if n == 0 {
            return Vec::new();
        }

        let n_fft = n.next_power_of_two().max(n * 2);
        self.ensure_capacity(n_fft, n, 0);
        self.complex_buf[..n_fft].fill(Complex64::zero());

        for (i, &s) in signal.iter().enumerate().take(n) {
            self.complex_buf[i] = Complex64::new(s, 0.0);
        }

        let fft = self.planner.plan_fft_forward(n_fft);
        fft.process(&mut self.complex_buf[..n_fft]);

        let half = n_fft / 2;
        for s in self.complex_buf.iter_mut().take(half).skip(1) {
            *s *= 2.0;
        }
        for s in self.complex_buf.iter_mut().skip(half + 1) {
            *s = Complex64::zero();
        }

        let ifft = self.planner.plan_fft_inverse(n_fft);
        ifft.process(&mut self.complex_buf[..n_fft]);

        (0..n)
            .map(|i| self.complex_buf[i].im / n_fft as f64)
            .collect()
    }
}

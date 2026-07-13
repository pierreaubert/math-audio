//! Reusable audio feature extractors.
//!
//! `FeatureExtractor` caches FFT plans, Hann windows, and per-frame buffers so
//! that repeated tempo/spectral analysis does not re-allocate them on every
//! call. `AudioFeatureExtractor` builds the full 23-element bliss-compatible
//! feature vector on top of it.

use super::chroma::ChromaFeatureExtractor;
use super::loudness;
use super::utils::{geometric_mean, normalize};
use super::zcr;
use super::{AnalysisError, FEATURES_COUNT, MIN_SAMPLES};
use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;
use std::f32::consts::PI;
use std::sync::{Arc, OnceLock};

pub const DEFAULT_WINDOW_SIZE: usize = 512;
pub const HOP_SIZE_TEMPO: usize = 256;
pub const HOP_SIZE_SPECTRAL: usize = 128;

const MAX_BPM: f32 = 206.0;
const MIN_BPM: f32 = 0.0;

/// Reusable extractor for tempo and spectral descriptors.
///
/// Holds a forward FFT plan, a Hann window, and all per-frame scratch buffers
/// to avoid re-creating them on every analysis call.
pub struct FeatureExtractor {
    window_size: usize,
    hann_window: Vec<f32>,
    fft: Arc<dyn realfft::RealToComplex<f32>>,
    time_buf: Vec<f32>,
    freq_buf: Vec<Complex<f32>>,
    spectrum_buf: Vec<f32>,
    prev_spectrum: Vec<f32>,
    onset_buf: Vec<f32>,
    centered_buf: Vec<f32>,
}

impl FeatureExtractor {
    /// Create a new extractor with the default window size (512 samples).
    pub fn new() -> Self {
        Self::with_cached_fft(DEFAULT_WINDOW_SIZE)
    }

    /// Create a new extractor reusing the globally cached forward real FFT plan
    /// for the default window size. Avoids re-planning on every call.
    fn with_cached_fft(window_size: usize) -> Self {
        let fft = default_real_fft();
        let hann_window = build_hann_window(window_size);
        let n_bins = window_size / 2 + 1;

        Self {
            window_size,
            hann_window,
            fft,
            time_buf: vec![0.0; window_size],
            freq_buf: vec![Complex::new(0.0, 0.0); n_bins],
            spectrum_buf: vec![0.0; n_bins],
            prev_spectrum: vec![0.0; n_bins],
            onset_buf: Vec::new(),
            centered_buf: Vec::new(),
        }
    }

    /// Create a new extractor with a custom window size.
    pub fn with_window_size(window_size: usize) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(window_size);
        let hann_window = build_hann_window(window_size);
        let n_bins = window_size / 2 + 1;

        Self {
            window_size,
            hann_window,
            fft,
            time_buf: vec![0.0; window_size],
            freq_buf: vec![Complex::new(0.0, 0.0); n_bins],
            spectrum_buf: vec![0.0; n_bins],
            prev_spectrum: vec![0.0; n_bins],
            onset_buf: Vec::new(),
            centered_buf: Vec::new(),
        }
    }

    /// Reset internal state (currently clears the previous spectrum used for
    /// onset detection).
    pub fn reset(&mut self) {
        self.prev_spectrum.fill(0.0);
    }

    /// Compute the magnitude spectrum for a single windowed frame.
    ///
    /// `samples` must have length `self.window_size`. The result is stored in
    /// `self.spectrum_buf` (length `window_size / 2 + 1`).
    fn compute_magnitude_spectrum(&mut self, samples: &[f32]) {
        debug_assert_eq!(samples.len(), self.window_size);

        for (i, (&s, &w)) in samples.iter().zip(self.hann_window.iter()).enumerate() {
            self.time_buf[i] = s * w;
        }

        self.fft
            .process(&mut self.time_buf, &mut self.freq_buf)
            .expect("real FFT forward failed");

        for i in 0..self.spectrum_buf.len() {
            self.spectrum_buf[i] = self.freq_buf[i].norm();
        }
    }

    /// Compute tempo (BPM) from audio samples.
    ///
    /// Returns a single normalized value in [-1, 1], or -1 for silence or
    /// signals that are too short.
    pub fn compute_tempo(&mut self, samples: &[f32], sample_rate: u32) -> f32 {
        self.onset_buf.clear();
        self.prev_spectrum.fill(0.0);

        let n_bins = self.window_size / 2 + 1;

        for chunk in samples.windows(self.window_size).step_by(HOP_SIZE_TEMPO) {
            self.compute_magnitude_spectrum(chunk);

            // Half-wave rectified spectral difference (spectral flux)
            let flux: f32 = self.spectrum_buf[..n_bins]
                .iter()
                .zip(self.prev_spectrum[..n_bins].iter())
                .map(|(&curr, &prev)| (curr - prev).max(0.0))
                .sum();

            self.onset_buf.push(flux);
            self.prev_spectrum.copy_from_slice(&self.spectrum_buf);
        }

        if self.onset_buf.len() < 4 {
            return -1.0; // Too short
        }

        let max_onset = self.onset_buf.iter().copied().fold(0.0f32, f32::max);
        if max_onset < 1e-6 {
            return -1.0; // No onsets detected (silence or near-silence)
        }

        let bpm = self.estimate_bpm_autocorrelation(sample_rate);

        if bpm <= 0.0 {
            return -1.0;
        }

        normalize(bpm, MIN_BPM, MAX_BPM)
    }

    /// Compute spectral centroid, rolloff, and flatness.
    ///
    /// Returns `[centroid_mean, centroid_std, rolloff_mean, rolloff_std,
    /// flatness_mean, flatness_std]` all normalized to [-1, 1].
    pub fn compute_spectral_features(&mut self, samples: &[f32], sample_rate: u32) -> [f32; 6] {
        let sr = sample_rate as f32;
        let half_sr = sr / 2.0;

        // Running mean / variance (Welford) so we do not need to store every
        // per-frame descriptor in a growable buffer.
        let mut centroid_mean = 0.0f32;
        let mut centroid_m2 = 0.0f32;
        let mut rolloff_mean = 0.0f32;
        let mut rolloff_m2 = 0.0f32;
        let mut flatness_mean = 0.0f32;
        let mut flatness_m2 = 0.0f32;
        let mut n = 0u32;

        for chunk in samples.windows(self.window_size).step_by(HOP_SIZE_SPECTRAL) {
            self.compute_magnitude_spectrum(chunk);

            // --- Centroid (first pass, fused with energy) ---
            let mut sum_mag = 0.0_f32;
            let mut weighted_sum = 0.0_f32;
            let mut energy = 0.0_f32;
            for (i, &m) in self.spectrum_buf.iter().enumerate() {
                sum_mag += m;
                weighted_sum += i as f32 * m;
                energy += m * m;
            }
            let centroid_bin = if sum_mag > 0.0 {
                weighted_sum / sum_mag
            } else {
                0.0
            };
            let centroid_freq = centroid_bin * sr / self.window_size as f32;

            // --- Rolloff ---
            let threshold = 0.95 * energy;
            let mut cumulative = 0.0_f32;
            let mut rolloff_bin = 0.0_f32;
            for (i, &m) in self.spectrum_buf.iter().enumerate() {
                cumulative += m * m;
                if cumulative >= threshold {
                    rolloff_bin = i as f32;
                    break;
                }
            }
            if rolloff_bin > self.window_size as f32 / 2.0 {
                rolloff_bin = self.window_size as f32 / 2.0;
            }
            let rolloff_freq = rolloff_bin * sr / self.window_size as f32;

            // --- Flatness ---
            let geo_len = (self.spectrum_buf.len() / 8) * 8;
            let geo = geometric_mean(&self.spectrum_buf[..geo_len]);
            let mean_mag = sum_mag / self.spectrum_buf.len() as f32;
            let flatness = if geo == 0.0 || mean_mag == 0.0 {
                0.0
            } else {
                geo / mean_mag
            };

            // Update online statistics for this frame.
            n += 1;
            let nf = n as f32;
            let delta_c = centroid_freq - centroid_mean;
            centroid_mean += delta_c / nf;
            let delta2_c = centroid_freq - centroid_mean;
            centroid_m2 += delta_c * delta2_c;

            let delta_r = rolloff_freq - rolloff_mean;
            rolloff_mean += delta_r / nf;
            let delta2_r = rolloff_freq - rolloff_mean;
            rolloff_m2 += delta_r * delta2_r;

            let delta_f = flatness - flatness_mean;
            flatness_mean += delta_f / nf;
            let delta2_f = flatness - flatness_mean;
            flatness_m2 += delta_f * delta2_f;
        }

        let count = n as f32;
        let centroid_std = if n > 0 { (centroid_m2 / count).sqrt() } else { 0.0 };
        let rolloff_std = if n > 0 { (rolloff_m2 / count).sqrt() } else { 0.0 };
        let flatness_std = if n > 0 { (flatness_m2 / count).sqrt() } else { 0.0 };

        [
            normalize(centroid_mean, 0.0, half_sr),
            normalize(centroid_std, 0.0, half_sr),
            normalize(rolloff_mean, 0.0, half_sr),
            normalize(rolloff_std, 0.0, half_sr),
            normalize(flatness_mean, 0.0, 1.0),
            normalize(flatness_std, 0.0, 1.0),
        ]
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Full 23-element audio feature extractor built on reusable
/// [`FeatureExtractor`] instances.
///
/// Keeps separate extractors for tempo, spectral, and chroma descriptors so
/// that they can run in parallel while still reusing FFT plans and scratch
/// buffers.
pub struct AudioFeatureExtractor {
    tempo_features: FeatureExtractor,
    spectral_features: FeatureExtractor,
    chroma_features: ChromaFeatureExtractor,
}

impl AudioFeatureExtractor {
    /// Create a new full audio feature extractor.
    pub fn new() -> Self {
        Self {
            tempo_features: FeatureExtractor::new(),
            spectral_features: FeatureExtractor::new(),
            chroma_features: ChromaFeatureExtractor::new(),
        }
    }

    /// Analyze audio samples and return a 23-element feature vector.
    ///
    /// The returned array is ordered identically to bliss v2 Analysis:
    /// `[tempo, zcr, centroid(2), rolloff(2), flatness(2), loudness(2), chroma(13)]`.
    pub fn analyze(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<[f32; FEATURES_COUNT], AnalysisError> {
        if samples.len() < MIN_SAMPLES {
            return Err(AnalysisError::TooShort);
        }

        std::thread::scope(|s| {
            let child_tempo = s.spawn(|| self.tempo_features.compute_tempo(samples, sample_rate));
            let child_spectral = s.spawn(|| {
                self.spectral_features
                    .compute_spectral_features(samples, sample_rate)
            });
            let child_chroma =
                s.spawn(|| self.chroma_features.compute(samples, sample_rate));

            // These are cheap; run them on the caller thread while the heavy
            // feature threads are in flight.
            let zcr_val = zcr::compute_zcr(samples);
            let loudness_vals = loudness::compute_loudness(samples);

            let tempo_val = child_tempo
                .join()
                .map_err(|e| AnalysisError::ThreadPanic(format!("{e:?}")))?;
            let spectral_vals = child_spectral
                .join()
                .map_err(|e| AnalysisError::ThreadPanic(format!("{e:?}")))?;
            let chroma_vals = child_chroma
                .join()
                .map_err(|e| AnalysisError::ThreadPanic(format!("{e:?}")))?
                .map_err(|e| AnalysisError::ChromaError(e.0))?;

            let mut result = [0.0f32; FEATURES_COUNT];
            result[0] = tempo_val;
            result[1] = zcr_val;
            result[2..8].copy_from_slice(&spectral_vals);
            result[8..10].copy_from_slice(&loudness_vals);
            result[10..23].copy_from_slice(&chroma_vals);

            Ok(result)
        })
    }
}

impl Default for AudioFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Return a cached forward real FFT plan for the default window size.
fn default_real_fft() -> Arc<dyn realfft::RealToComplex<f32>> {
    static PLAN: OnceLock<Arc<dyn realfft::RealToComplex<f32>>> = OnceLock::new();
    PLAN.get_or_init(|| {
        let mut planner = RealFftPlanner::<f32>::new();
        planner.plan_fft_forward(DEFAULT_WINDOW_SIZE)
    })
    .clone()
}

fn build_hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|n| 0.5 - 0.5 * f32::cos(2.0 * PI * n as f32 / size as f32))
        .collect()
}

impl FeatureExtractor {
    /// Estimate BPM using autocorrelation of the onset envelope.
    fn estimate_bpm_autocorrelation(&mut self, sample_rate: u32) -> f32 {
        let onset_envelope = &self.onset_buf;
        let n = onset_envelope.len();
        if n < 4 {
            return 0.0;
        }

        let frame_rate = sample_rate as f32 / HOP_SIZE_TEMPO as f32;

        let min_lag = (frame_rate * 60.0 / MAX_BPM).ceil() as usize;
        let max_lag = (frame_rate * 60.0 / 30.0).floor() as usize; // 30 BPM lower bound
        let max_lag = max_lag.min(n / 2);

        if min_lag >= max_lag || max_lag >= n {
            return 0.0;
        }

        // Remove mean, reusing centered_buf
        let mean_val = onset_envelope.iter().sum::<f32>() / n as f32;
        self.centered_buf.resize(n, 0.0);
        for (i, &x) in onset_envelope.iter().enumerate() {
            self.centered_buf[i] = x - mean_val;
        }

        let mut best_lag = min_lag;
        let mut best_score = f32::NEG_INFINITY;

        for lag in min_lag..=max_lag {
            let mut corr = 0.0f32;
            for i in 0..n - lag {
                corr += self.centered_buf[i] * self.centered_buf[i + lag];
            }
            corr /= (n - lag) as f32;

            // Apply tempo prior: prefer tempos near 120 BPM (Gaussian weighting)
            let bpm = frame_rate * 60.0 / lag as f32;
            let prior = (-0.5 * ((bpm - 120.0) / 40.0).powi(2)).exp();
            let score = corr * prior;

            if score > best_score {
                best_score = score;
                best_lag = lag;
            }
        }

        frame_rate * 60.0 / best_lag as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extractor_reuse_produces_same_results() {
        let sr = 22050u32;
        let signal: Vec<f32> = (0..sr as usize * 2)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();

        let mut extractor = FeatureExtractor::new();
        let tempo1 = extractor.compute_tempo(&signal, sr);
        let spectral1 = extractor.compute_spectral_features(&signal, sr);

        let tempo2 = extractor.compute_tempo(&signal, sr);
        let spectral2 = extractor.compute_spectral_features(&signal, sr);

        assert_eq!(tempo1, tempo2);
        assert_eq!(spectral1, spectral2);
    }

    #[test]
    fn test_audio_feature_extractor_count() {
        let sr = 22050u32;
        let signal: Vec<f32> = (0..sr as usize * 5)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();

        let features = AudioFeatureExtractor::new().analyze(&signal, sr).unwrap();
        assert_eq!(features.len(), FEATURES_COUNT);
    }
}

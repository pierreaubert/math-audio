//! Reusable audio feature extractors.
//!
//! `FeatureExtractor` caches FFT plans, Hann windows, and per-frame buffers so
//! that repeated tempo/spectral analysis does not re-allocate them on every
//! call. `AudioFeatureExtractor` builds the full 23-element bliss-compatible
//! feature vector on top of it.

use super::chroma;
use super::loudness;
use super::utils::{geometric_mean, mean, normalize, std_deviation};
use super::zcr;
use super::{AnalysisError, FEATURES_COUNT, MIN_SAMPLES};
use rustfft::Fft;
use rustfft::num_complex::Complex;
use std::f32::consts::PI;
use std::sync::Arc;

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
    fft: Arc<dyn Fft<f32>>,
    complex_buf: Vec<Complex<f32>>,
    spectrum_buf: Vec<f32>,
    prev_spectrum: Vec<f32>,
    onset_buf: Vec<f32>,
    centroid_values: Vec<f32>,
    rolloff_values: Vec<f32>,
    flatness_values: Vec<f32>,
}

impl FeatureExtractor {
    /// Create a new extractor with the default window size (512 samples).
    pub fn new() -> Self {
        Self::with_window_size(DEFAULT_WINDOW_SIZE)
    }

    /// Create a new extractor with a custom window size.
    pub fn with_window_size(window_size: usize) -> Self {
        let mut planner = rustfft::FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(window_size);
        let hann_window = build_hann_window(window_size);
        let n_bins = window_size / 2 + 1;

        Self {
            window_size,
            hann_window,
            fft,
            complex_buf: vec![Complex::new(0.0, 0.0); window_size],
            spectrum_buf: vec![0.0; n_bins],
            prev_spectrum: vec![0.0; n_bins],
            onset_buf: Vec::new(),
            centroid_values: Vec::new(),
            rolloff_values: Vec::new(),
            flatness_values: Vec::new(),
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
            self.complex_buf[i] = Complex::new(s * w, 0.0);
        }

        self.fft.process(&mut self.complex_buf);

        for i in 0..self.spectrum_buf.len() {
            self.spectrum_buf[i] = self.complex_buf[i].norm();
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

        let bpm = estimate_bpm_autocorrelation(&self.onset_buf, sample_rate);

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

        self.centroid_values.clear();
        self.rolloff_values.clear();
        self.flatness_values.clear();

        for chunk in samples.windows(self.window_size).step_by(HOP_SIZE_SPECTRAL) {
            self.compute_magnitude_spectrum(chunk);

            // --- Centroid ---
            let sum_mag: f32 = self.spectrum_buf.iter().sum();
            let centroid_bin = if sum_mag > 0.0 {
                self.spectrum_buf
                    .iter()
                    .enumerate()
                    .map(|(i, &m)| i as f32 * m)
                    .sum::<f32>()
                    / sum_mag
            } else {
                0.0
            };
            let centroid_freq = centroid_bin * sr / self.window_size as f32;
            self.centroid_values.push(centroid_freq);

            // --- Rolloff ---
            let total_energy: f32 = self.spectrum_buf.iter().map(|&m| m * m).sum();
            let threshold = 0.95 * total_energy;
            let mut cumulative = 0.0;
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
            self.rolloff_values.push(rolloff_freq);

            // --- Flatness ---
            let geo_len = (self.spectrum_buf.len() / 8) * 8;
            let geo = geometric_mean(&self.spectrum_buf[..geo_len]);
            if geo == 0.0 {
                self.flatness_values.push(0.0);
            } else {
                let flatness = geo / mean(&self.spectrum_buf);
                self.flatness_values.push(flatness);
            }
        }

        let centroid_mean = normalize(mean(&self.centroid_values), 0.0, half_sr);
        let centroid_std = normalize(std_deviation(&self.centroid_values), 0.0, half_sr);
        let rolloff_mean = normalize(mean(&self.rolloff_values), 0.0, half_sr);
        let rolloff_std = normalize(std_deviation(&self.rolloff_values), 0.0, half_sr);
        let flatness_mean = normalize(mean(&self.flatness_values), 0.0, 1.0);
        let flatness_std = normalize(std_deviation(&self.flatness_values), 0.0, 1.0);

        [
            centroid_mean,
            centroid_std,
            rolloff_mean,
            rolloff_std,
            flatness_mean,
            flatness_std,
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
/// Keeps separate extractors for tempo and spectral descriptors so that both
/// can run in parallel while still reusing FFT plans and scratch buffers.
pub struct AudioFeatureExtractor {
    tempo_features: FeatureExtractor,
    spectral_features: FeatureExtractor,
}

impl AudioFeatureExtractor {
    /// Create a new full audio feature extractor.
    pub fn new() -> Self {
        Self {
            tempo_features: FeatureExtractor::new(),
            spectral_features: FeatureExtractor::new(),
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
            let child_tempo =
                s.spawn(|| self.tempo_features.compute_tempo(samples, sample_rate));
            let child_spectral =
                s.spawn(|| self.spectral_features.compute_spectral_features(samples, sample_rate));
            let child_zcr = s.spawn(|| zcr::compute_zcr(samples));
            let child_loudness = s.spawn(|| loudness::compute_loudness(samples));
            let child_chroma = s.spawn(|| chroma::compute_chroma_features(samples, sample_rate));

            let tempo_val = child_tempo
                .join()
                .map_err(|e| AnalysisError::ThreadPanic(format!("{e:?}")))?;
            let spectral_vals = child_spectral
                .join()
                .map_err(|e| AnalysisError::ThreadPanic(format!("{e:?}")))?;
            let zcr_val = child_zcr
                .join()
                .map_err(|e| AnalysisError::ThreadPanic(format!("{e:?}")))?;
            let loudness_vals = child_loudness
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

fn build_hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|n| 0.5 - 0.5 * f32::cos(2.0 * PI * n as f32 / size as f32))
        .collect()
}

/// Estimate BPM using autocorrelation of the onset envelope.
fn estimate_bpm_autocorrelation(onset_envelope: &[f32], sample_rate: u32) -> f32 {
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

    // Remove mean
    let mean_val = onset_envelope.iter().sum::<f32>() / n as f32;
    let centered: Vec<f32> = onset_envelope.iter().map(|&x| x - mean_val).collect();

    let mut best_lag = min_lag;
    let mut best_score = f32::NEG_INFINITY;

    for lag in min_lag..=max_lag {
        let mut corr = 0.0f32;
        for i in 0..n - lag {
            corr += centered[i] * centered[i + lag];
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

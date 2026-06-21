//! Spectral descriptors: centroid, rolloff, flatness.
//!
//! Replaces aubio PVoc + SpecDesc with a pure Rust Hann-windowed FFT.

use super::extractor::FeatureExtractor;

/// Compute spectral centroid, rolloff, and flatness.
///
/// Returns `[centroid_mean, centroid_std, rolloff_mean, rolloff_std, flatness_mean, flatness_std]`
/// all normalized to [-1, 1].
pub fn compute_spectral_features(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    FeatureExtractor::new()
        .compute_spectral_features(samples, sample_rate)
        .to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_spectral_silence() {
        let silence = vec![0.0; 1024];
        let features = compute_spectral_features(&silence, 22050);
        assert_eq!(features.len(), 6);
        // All should be -1 for silence
        for &f in &features {
            assert!(f <= -0.99, "expected ~-1 for silence, got {f}");
        }
    }

    #[test]
    fn test_spectral_features_length() {
        let signal: Vec<f32> = (0..22050)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 22050.0).sin())
            .collect();
        let features = compute_spectral_features(&signal, 22050);
        assert_eq!(features.len(), 6);
        // All values should be in [-1, 1]
        for &f in &features {
            assert!((-1.0..=1.0).contains(&f), "feature out of range: {f}");
        }
    }
}

#[test]
fn test_spectral_too_short() {
    let short = vec![0.5f32; 100];
    let features = compute_spectral_features(&short, 22050);
    assert_eq!(features.len(), 6);
    for &f in &features {
        assert!(
            (f - -1.0).abs() < 1e-6,
            "expected -1 for no windows, got {f}"
        );
    }
}

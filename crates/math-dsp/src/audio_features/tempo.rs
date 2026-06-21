//! Tempo (BPM) estimation.
//!
//! Replaces aubio Tempo(SpecFlux) with a pure Rust implementation:
//! 1. Spectral flux onset detection
//! 2. Autocorrelation-based BPM estimation
//! 3. Median of per-segment BPM estimates

use super::extractor::FeatureExtractor;

/// Compute tempo (BPM) from audio samples.
///
/// Returns a single normalized value in [-1, 1].
pub fn compute_tempo(samples: &[f32], sample_rate: u32) -> f32 {
    FeatureExtractor::new().compute_tempo(samples, sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tempo_silence() {
        let silence = vec![0.0; 22050 * 10];
        let tempo = compute_tempo(&silence, 22050);
        assert_eq!(-1.0, tempo);
    }

    #[test]
    fn test_tempo_60bpm() {
        // Create a signal with a beat every second (60 BPM)
        let sr = 22050u32;
        let duration_secs = 30;
        let total_samples = sr as usize * duration_secs;
        let mut signal = vec![0.0f32; total_samples];

        // Place impulses every second
        for beat in 0..duration_secs {
            let pos = beat * sr as usize;
            for i in 0..100 {
                if pos + i < total_samples {
                    signal[pos + i] = 1.0;
                }
            }
        }

        let tempo = compute_tempo(&signal, sr);
        // Should detect roughly 60 BPM → normalized ~ -0.42
        // We allow wide tolerance since our impl differs from aubio
        assert!(tempo > -0.8 && tempo < 0.0, "tempo = {tempo}");
    }
}

#[test]
fn test_tempo_too_short() {
    let short = vec![0.0f32; 10];
    let tempo = compute_tempo(&short, 22050);
    assert_eq!(tempo, -1.0);
}

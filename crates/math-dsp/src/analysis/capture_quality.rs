//! Frequency-dependent capture quality using equally normalized Welch spectra.

// Rust guideline compliant 2026-02-21
use rustfft::{FftPlanner, num_complex::Complex};

/// Evidence for one frequency band in a recorded sweep.
#[derive(Debug, Clone)]
pub struct CaptureBandSnr {
    /// Lower integration boundary in Hz.
    pub low_hz: f64,
    /// Upper integration boundary in Hz.
    pub high_hz: f64,
    /// Signal-minus-noise to noise ratio, or unavailable evidence.
    pub snr_db: Option<f64>,
}

/// Estimate band SNR from a sweep window and a separate quiet window.
///
/// Both inputs use the same FFT size, periodic Hann window, 50% overlap and
/// one-sided PSD normalization by sample rate times window squared energy.
/// Complete frames are averaged, so unequal window durations do not bias power.
/// The FFT size is the largest power of two no larger than half the shorter
/// input, capped at 4096 samples. No zero padding is used. Bins are integrated
/// over half-open bands `[low_hz, high_hz)`.
///
/// This is the sweep-window *time-averaged* signal-to-noise ratio, not the
/// instantaneous SNR as the sweep crosses a frequency or deconvolved IR SNR.
/// Noise is assumed stationary between the two windows. Digital silence,
/// nonpositive signal-minus-noise, and bands containing no bins yield `None`.
/// Microphone magnitude calibration cancels in a narrowband ratio; this
/// estimator consumes uncalibrated samples and does not claim absolute SPL.
///
/// # Errors
/// Returns an error for invalid sample rates, non-finite inputs, fewer than
/// 512 samples in either window, or invalid band boundaries.
pub fn capture_band_snr(
    sweep: &[f32],
    quiet: &[f32],
    sample_rate_hz: u32,
    bands_hz: &[(f64, f64)],
) -> Result<Vec<CaptureBandSnr>, String> {
    if sample_rate_hz == 0
        || sweep.len().min(quiet.len()) < 512
        || sweep.iter().chain(quiet).any(|value| !value.is_finite())
    {
        return Err(
            "SNR requires finite windows of at least 512 samples and a positive rate".into(),
        );
    }
    let nyquist = f64::from(sample_rate_hz) / 2.0;
    if bands_hz.iter().any(|&(low, high)| {
        !low.is_finite() || !high.is_finite() || low < 0.0 || low >= high || high > nyquist
    }) {
        return Err("SNR bands must be finite, ordered and within Nyquist".into());
    }
    // Require multiple frames even for the shorter quiet window.
    let limit = (sweep.len().min(quiet.len()) / 2).min(4096);
    let size = 1usize << limit.ilog2();
    let signal_psd = welch(sweep, size, sample_rate_hz);
    let noise_psd = welch(quiet, size, sample_rate_hz);
    let bin_hz = f64::from(sample_rate_hz) / size as f64;
    Ok(bands_hz
        .iter()
        .map(|&(low_hz, high_hz)| {
            let start = (low_hz / bin_hz).ceil() as usize;
            let end = ((high_hz / bin_hz).ceil() as usize).min(signal_psd.len());
            let signal = signal_psd[start..end].iter().sum::<f64>() * bin_hz;
            let noise = noise_psd[start..end].iter().sum::<f64>() * bin_hz;
            let snr_db = if noise > 0.0 && signal > noise {
                let value = 10.0 * ((signal - noise) / noise).log10();
                value.is_finite().then_some(value)
            } else {
                None
            };
            CaptureBandSnr {
                low_hz,
                high_hz,
                snr_db,
            }
        })
        .collect())
}

fn welch(samples: &[f32], size: usize, sample_rate_hz: u32) -> Vec<f64> {
    let window: Vec<_> = (0..size)
        .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / size as f64).cos())
        .collect();
    let energy = window.iter().map(|x| x * x).sum::<f64>();
    let fft = FftPlanner::<f64>::new().plan_fft_forward(size);
    let mut buffer = vec![Complex::new(0.0, 0.0); size];
    let mut psd = vec![0.0; size / 2 + 1];
    let mut count = 0;
    for start in (0..=samples.len() - size).step_by(size / 2) {
        for ((bin, sample), weight) in buffer
            .iter_mut()
            .zip(&samples[start..start + size])
            .zip(&window)
        {
            *bin = Complex::new(f64::from(*sample) * weight, 0.0);
        }
        fft.process(&mut buffer);
        for (index, power) in psd.iter_mut().enumerate() {
            let sidedness = if index == 0 || index == size / 2 {
                1.0
            } else {
                2.0
            };
            *power += buffer[index].norm_sqr() * sidedness;
        }
        count += 1;
    }
    for power in &mut psd {
        *power /= f64::from(sample_rate_hz) * energy * f64::from(count);
    }
    psd
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(length: usize, amplitude: f32) -> Vec<f32> {
        (0..length)
            .map(|i| amplitude * (std::f32::consts::TAU * i as f32 / 32.0).sin())
            .collect()
    }

    #[test]
    fn unequal_duration_windows_have_correct_power_ratio() {
        let result = capture_band_snr(
            &tone(32768, 0.1),
            &tone(4096, 0.01),
            32768,
            &[(900.0, 1200.0)],
        )
        .unwrap();
        assert!((result[0].snr_db.unwrap() - 10.0 * 99.0_f64.log10()).abs() < 0.001);
    }

    #[test]
    fn integrated_psd_obeys_parseval_for_periodic_tone() {
        let psd = welch(&tone(8192, 0.2), 2048, 32768);
        let power = psd.iter().sum::<f64>() * 16.0;
        assert!((power - 0.02).abs() < 1e-7);
    }

    #[test]
    fn unavailable_evidence_never_becomes_infinite_snr() {
        let signal = tone(2048, 0.1);
        for noise in [vec![0.0; 2048], signal.clone()] {
            let result = capture_band_snr(&signal, &noise, 48000, &[(1000.0, 2000.0)]).unwrap();
            assert!(result[0].snr_db.is_none());
        }
        assert!(
            capture_band_snr(&signal, &signal, 48000, &[(0.0, 1.0)]).unwrap()[0]
                .snr_db
                .is_none()
        );
        assert!(capture_band_snr(&signal, &signal, 0, &[]).is_err());
        assert!(capture_band_snr(&signal, &signal, 48000, &[(2000.0, 1000.0)]).is_err());
        assert!(capture_band_snr(&[f32::NAN; 512], &signal, 48000, &[]).is_err());
    }
}

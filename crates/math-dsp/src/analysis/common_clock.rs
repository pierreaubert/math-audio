//! Deconvolution that preserves a shared stimulus time origin.
//!
//! Unlike arrival-aligned measurement analysis, this module never estimates or
//! removes a delay. Inputs must already use the same sample clock and origin.

// Rust guideline compliant 2026-02-21
use super::deconvolve_sweep_f64_spectrum;
use num_complex::Complex64;
use rustfft::FftPlanner;

/// Uncalibrated transfer response with the input time origin preserved.
#[derive(Debug, Clone)]
pub struct CommonClockResponse {
    /// Sample rate of both input signals, in Hz.
    pub sample_rate_hz: u32,
    /// Nonnegative-frequency bins, including DC and Nyquist.
    pub spectrum: Vec<Complex64>,
    /// Real impulse response; sample zero retains the shared input origin.
    pub impulse_response: Vec<f64>,
}

impl CommonClockResponse {
    /// Apply nonnegative real gains without adding a phase calibration.
    ///
    /// Supply one gain per nonnegative-frequency bin. Zero gains explicitly
    /// exclude unsupported frequencies, yielding a band-limited periodic IR.
    /// The reconstructed real IR retains the original time origin. This cannot
    /// correct a microphone's unknown phase response.
    ///
    /// # Errors
    /// Rejects malformed response dimensions, non-finite bins or gains,
    /// negative gains, or an overflow in calibrated output.
    pub fn with_magnitude_gains(mut self, gains: &[f64]) -> Result<Self, String> {
        let size = self.impulse_response.len();
        if !size.is_power_of_two()
            || self.spectrum.len() != size / 2 + 1
            || gains.len() != self.spectrum.len()
            || gains.iter().any(|gain| !gain.is_finite() || *gain < 0.0)
        {
            return Err("invalid common-clock calibration dimensions or gains".into());
        }
        for (bin, gain) in self.spectrum.iter_mut().zip(gains) {
            *bin *= gain;
            if !bin.re.is_finite() || !bin.im.is_finite() {
                return Err("non-finite calibrated common-clock response".into());
            }
        }
        // A real response has real DC and Nyquist values. Rebuild the negative
        // frequencies explicitly so the inverse transform remains Hermitian.
        self.spectrum[0].im = 0.0;
        self.spectrum[size / 2].im = 0.0;
        let mut full = vec![Complex64::new(0.0, 0.0); size];
        full[..self.spectrum.len()].copy_from_slice(&self.spectrum);
        for index in 1..size.div_ceil(2) {
            full[size - index] = full[index].conj();
        }
        FftPlanner::<f64>::new()
            .plan_fft_inverse(size)
            .process(&mut full);
        for (sample, bin) in self.impulse_response.iter_mut().zip(full) {
            *sample = bin.re / size as f64;
            if !sample.is_finite() {
                return Err("non-finite calibrated common-clock impulse".into());
            }
        }
        Ok(self)
    }
}

/// Deconvolve signals without removing their relative acoustic delay.
///
/// Both buffers must start at the same stimulus time. The recording must contain
/// the complete sweep and decay; timing chirps must be excluded from both inputs.
/// This function cannot verify those physical preconditions or clock quality.
/// It applies neither microphone calibration nor phase compensation.
///
/// The forward DFT is unnormalized and the inverse is divided by the FFT length.
/// The FFT length is the next power of two of the recording length. Thus the
/// returned IR is periodic over that length; callers must capture sufficient
/// decay to avoid truncation artifacts. The shared sweep deconvolver regularizes
/// the denominator at one millionth of the peak reference spectral power.
/// Frequencies without adequate excitation are not reliable phase measurements.
///
/// # Errors
/// Returns an error for a zero sample rate, empty, silent or non-finite inputs,
/// a recording shorter than the reference, length overflow, or non-finite output.
pub fn deconvolve_common_clock(
    recording: &[f64],
    reference: &[f64],
    sample_rate_hz: u32,
) -> Result<CommonClockResponse, String> {
    if sample_rate_hz == 0 {
        return Err("common-clock sample rate must be positive".into());
    }
    for samples in [recording, reference] {
        if samples.is_empty()
            || samples.iter().any(|value| !value.is_finite())
            || samples.iter().all(|value| *value == 0.0)
        {
            return Err("common-clock inputs must be nonempty, finite and nonsilent".into());
        }
    }
    let fft_size = recording
        .len()
        .checked_next_power_of_two()
        .ok_or("common-clock FFT length overflow")?;
    let mut full_spectrum = deconvolve_sweep_f64_spectrum(recording, reference, fft_size)?;
    if full_spectrum
        .iter()
        .any(|bin| !bin.re.is_finite() || !bin.im.is_finite())
    {
        return Err("common-clock deconvolution produced non-finite bins".into());
    }
    let spectrum = full_spectrum[..=fft_size / 2].to_vec();
    FftPlanner::<f64>::new()
        .plan_fft_inverse(fft_size)
        .process(&mut full_spectrum);
    let impulse_response: Vec<_> = full_spectrum
        .iter()
        .map(|value| value.re / fft_size as f64)
        .collect();
    if impulse_response.iter().any(|value| !value.is_finite()) {
        return Err("common-clock deconvolution produced a non-finite impulse".into());
    }
    Ok(CommonClockResponse {
        sample_rate_hz,
        spectrum,
        impulse_response,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retains_physical_delay_and_phase() {
        // An impulse has an independently known flat spectrum. No correlation
        // or deconvolution implementation is used to construct the oracle.
        let reference = [1.0];
        for delay in [0, 7, 53] {
            let mut recording = vec![0.0; 128];
            recording[delay] = 0.4;
            let result = deconvolve_common_clock(&recording, &reference, 48_000).unwrap();
            for (index, &sample) in result.impulse_response.iter().enumerate() {
                let expected = if index == delay { 0.4 / 1.000001 } else { 0.0 };
                assert!((sample - expected).abs() < 1e-12);
            }
            for (bin, actual) in result.spectrum.iter().enumerate() {
                let phase = -std::f64::consts::TAU * bin as f64 * delay as f64 / 128.0;
                let expected = Complex64::from_polar(0.4 / 1.000001, phase);
                assert!((*actual - expected).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn preserves_sweep_direct_and_reflected_arrivals() {
        // Direct time-domain convolution supplies two independently known taps.
        let reference: Vec<_> = (0..256)
            .map(|index| (0.0007 * (index * index) as f64).sin())
            .collect();
        let mut recording = vec![0.0; 512];
        for (index, value) in reference.iter().enumerate() {
            recording[index + 19] += value * 0.7;
            recording[index + 81] += value * -0.2;
        }
        let result = deconvolve_common_clock(&recording, &reference, 48_000).unwrap();
        assert!((result.impulse_response[19] - 0.7).abs() < 0.01);
        assert!((result.impulse_response[81] + 0.2).abs() < 0.01);
        assert!(
            result
                .impulse_response
                .iter()
                .enumerate()
                .all(|(i, x)| i == 19 || i == 81 || x.abs() < 0.01)
        );
    }

    #[test]
    fn rejects_invalid_evidence() {
        for (recording, reference, rate) in [
            (vec![], vec![1.0], 48_000),
            (vec![0.0], vec![1.0], 48_000),
            (vec![1.0], vec![0.0], 48_000),
            (vec![f64::NAN], vec![1.0], 48_000),
            (vec![1.0], vec![f64::INFINITY], 48_000),
            (vec![1.0], vec![1.0, 0.0], 48_000),
            (vec![1.0], vec![1.0], 0),
        ] {
            assert!(deconvolve_common_clock(&recording, &reference, rate).is_err());
        }
    }
}

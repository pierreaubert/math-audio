use super::FFT_PLANNER;
use super::REAL_FFT_PLANNER;
use super::misc::next_power_of_two;
use super::types::CrossCorrelationEnvelopeResult;
use num_complex::Complex64;
use realfft::RealToComplex;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use std::sync::Arc;

/// Get a cached forward FFT plan for the given size (f32).
///
/// Uses a thread-local planner so repeated calls with the same size
/// return the same plan without recomputing twiddle factors.
pub fn plan_fft_forward(size: usize) -> Arc<dyn rustfft::Fft<f32>> {
    FFT_PLANNER.with(|p| p.borrow_mut().plan_fft_forward(size))
}

/// Get a cached inverse FFT plan for the given size (f32).
pub fn plan_fft_inverse(size: usize) -> Arc<dyn rustfft::Fft<f32>> {
    FFT_PLANNER.with(|p| p.borrow_mut().plan_fft_inverse(size))
}

/// Get a cached real-to-complex FFT plan for the given size (f32).
///
/// The returned plan transforms `size` real samples into `size/2 + 1`
/// complex bins.  Uses a thread-local planner for plan reuse.
pub fn plan_real_fft_forward(size: usize) -> Arc<dyn RealToComplex<f32>> {
    REAL_FFT_PLANNER.with(|p| p.borrow_mut().plan_fft_forward(size))
}

/// Shared f64 ESS deconvolution spectrum used by both the measurement
/// pipeline and the low-level binaural-matrix adapter. Keeping the
/// regularization and length rules here prevents the scalar paths from
/// silently drifting apart again.
pub(crate) fn deconvolve_sweep_f64_spectrum(
    recording: &[f64],
    reference: &[f64],
    fft_size: usize,
) -> Result<Vec<Complex64>, String> {
    if recording.is_empty() || reference.is_empty() {
        return Err("deconvolve_sweep: recording and reference must be non-empty".to_string());
    }
    if recording.len() < reference.len() {
        return Err(format!(
            "deconvolve_sweep: recording len {} != reference len {} (recording must not be shorter)",
            recording.len(),
            reference.len()
        ));
    }
    if fft_size < recording.len().max(reference.len()).next_power_of_two() {
        return Err("deconvolve_sweep: fft_size is too small".to_string());
    }
    if recording
        .iter()
        .chain(reference)
        .any(|sample| !sample.is_finite())
    {
        return Err("deconvolve_sweep: inputs must contain only finite samples".to_string());
    }

    let mut y: Vec<Complex64> = recording
        .iter()
        .map(|&sample| Complex64::new(sample, 0.0))
        .collect();
    let mut x: Vec<Complex64> = reference
        .iter()
        .map(|&sample| Complex64::new(sample, 0.0))
        .collect();
    y.resize(fft_size, Complex64::new(0.0, 0.0));
    x.resize(fft_size, Complex64::new(0.0, 0.0));

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(fft_size);
    fft.process(&mut y);
    fft.process(&mut x);

    let peak_power = x
        .iter()
        .map(|value| value.norm_sqr())
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);
    let regularization = (peak_power * 1e-6).max(f64::MIN_POSITIVE);
    for (output, input) in y.iter_mut().zip(&x) {
        *output = *output * input.conj() / (input.norm_sqr() + regularization);
    }
    Ok(y)
}

/// Cross-correlate a probe with a recording and compute the analytic envelope.
///
/// Uses FFT-based cross-correlation followed by the Hilbert transform
/// (via `analytic_signal`) to extract a smooth envelope whose peak
/// indicates the arrival time with sub-sample precision.
///
/// This is the matched-filter approach recommended by Johnston (AES):
/// narrowband probes give excellent noise rejection, and the analytic
/// envelope provides a clean, unambiguous peak even in reverberant rooms.
///
/// # Arguments
/// * `probe` - The known probe signal that was played
/// * `recorded` - The recorded signal from the microphone
/// * `sample_rate` - Sample rate in Hz
pub fn cross_correlate_envelope(
    probe: &[f32],
    recorded: &[f32],
    sample_rate: u32,
) -> Result<CrossCorrelationEnvelopeResult, String> {
    if probe.is_empty() || recorded.is_empty() {
        return Err("Probe and recorded signals must be non-empty".to_string());
    }
    if sample_rate == 0 {
        return Err("Sample rate must be greater than zero".to_string());
    }
    if probe
        .iter()
        .chain(recorded)
        .any(|sample| !sample.is_finite())
    {
        return Err("Probe and recorded signals must contain only finite samples".to_string());
    }

    // Zero-pad to avoid circular correlation artifacts
    let fft_size = next_power_of_two(probe.len() + recorded.len());

    // Raw FFT (no normalization) — we handle normalization once after IFFT.
    // Using unnormalized FFT avoids the scale-dependent gain errors that
    // occur when compute_fft_padded's 1/N normalization interacts with IFFT.
    let fft_forward = plan_fft_forward(fft_size);

    let mut probe_buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); fft_size];
    for (dst, &src) in probe_buf.iter_mut().zip(probe.iter()) {
        dst.re = src;
    }
    fft_forward.process(&mut probe_buf);

    let mut rec_buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); fft_size];
    for (dst, &src) in rec_buf.iter_mut().zip(recorded.iter()) {
        dst.re = src;
    }
    fft_forward.process(&mut rec_buf);

    // Cross-correlation: conj(Probe) * Recorded
    let mut cross_fft: Vec<Complex<f32>> = probe_buf
        .iter()
        .zip(rec_buf.iter())
        .map(|(p, r)| p.conj() * r)
        .collect();

    // IFFT to get cross-correlation in time domain
    let ifft = plan_fft_inverse(fft_size);
    ifft.process(&mut cross_fft);

    // Single 1/N normalization (standard for round-trip FFT→IFFT)
    let norm = 1.0 / fft_size as f32;
    let xcorr: Vec<f32> = cross_fft.iter().map(|c| c.re * norm).collect();

    // Compute analytic envelope via Hilbert transform
    let analytic = crate::instantaneous_frequency::analytic_signal(&xcorr);
    let envelope: Vec<f32> = analytic.iter().map(|c| c.norm()).collect();

    // Positive lags for conj(probe) * recorded occupy 0..recorded.len().
    // The remaining non-zero correlation samples wrap at the end and represent
    // negative lags, so searching the entire FFT half can select an alias when
    // the recording is short relative to the padded transform.
    let search_len = recorded.len().min(envelope.len());
    let mut peak_sample = 0_usize;
    let mut peak_value = 0.0_f32;
    for (i, &val) in envelope.iter().enumerate().take(search_len) {
        if val > peak_value {
            peak_value = val;
            peak_sample = i;
        }
    }

    if peak_value <= f32::EPSILON {
        return Err("Unable to detect a probe arrival: correlation has no energy".to_string());
    }

    let overlap = probe.len().min(recorded.len().saturating_sub(peak_sample));
    let probe_energy: f32 = probe[..overlap].iter().map(|sample| sample * sample).sum();
    let recorded_energy: f32 = recorded[peak_sample..peak_sample + overlap]
        .iter()
        .map(|sample| sample * sample)
        .sum();
    let normalized_peak = if probe_energy > f32::EPSILON && recorded_energy > f32::EPSILON {
        // The peak was selected from the analytic envelope, not the real
        // correlation sample. Use the envelope magnitude here as well;
        // otherwise a quadrature-phase envelope peak can report a false zero
        // confidence even when the matched filter is strong.
        (peak_value / (probe_energy * recorded_energy).sqrt()).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let guard = (probe.len().min(recorded.len()) / 64).max(2);
    let sidelobe = envelope
        .iter()
        .take(search_len)
        .enumerate()
        .filter(|(index, _)| index.abs_diff(peak_sample) > guard)
        .map(|(_, value)| *value)
        .fold(0.0_f32, f32::max);
    let sidelobe_ratio = (sidelobe / peak_value).min(1.0);
    let peak_to_sidelobe_db = if sidelobe <= f32::EPSILON {
        f32::INFINITY
    } else {
        20.0 * (peak_value / sidelobe).log10()
    };
    let confidence = (normalized_peak * (1.0 - sidelobe_ratio)).clamp(0.0, 1.0);
    if normalized_peak < 0.15 || confidence < 0.05 {
        return Err(format!(
            "Unable to detect a probe arrival confidently: normalized peak {:.3}, confidence {:.3}, PSR {:.1} dB",
            normalized_peak, confidence, peak_to_sidelobe_db
        ));
    }

    // Parabolic interpolation for sub-sample precision
    let peak_refined = if peak_sample > 0 && peak_sample + 1 < search_len {
        let y_prev = envelope[peak_sample - 1] as f64;
        let y_peak = envelope[peak_sample] as f64;
        let y_next = envelope[peak_sample + 1] as f64;
        let denom = 2.0 * (2.0 * y_peak - y_prev - y_next);
        if denom.abs() > 1e-12 {
            peak_sample as f64 + (y_prev - y_next) / denom
        } else {
            peak_sample as f64
        }
    } else {
        peak_sample as f64
    };

    let arrival_ms = peak_refined / sample_rate as f64 * 1000.0;

    Ok(CrossCorrelationEnvelopeResult {
        envelope,
        peak_sample,
        peak_sample_refined: peak_refined,
        peak_value,
        arrival_ms,
        normalized_peak,
        peak_to_sidelobe_db,
        confidence,
    })
}

/// Deconvolve a single recorded log sweep by dividing the recording's
/// spectrum by the emitted sweep's spectrum, producing a complex
/// frequency response on the FFT grid `[0, Nyquist]`.
///
/// The inverse-filter approach is the standard log-sweep
/// deconvolution:
///
/// ```text
///    H(f) = Y(f) / X(f)
/// ```
///
/// where Y is the recording and X is the emitted sweep. A small
/// regularisation term ε is added to the denominator to keep out-
/// of-band bins from blowing up — 60 dB below the sweep's peak is a
/// safe default.
///
/// The returned spectrum has `recording.len().next_power_of_two() / 2 + 1`
/// complex bins, indexed so bin k corresponds to frequency
/// `k * sample_rate / fft_size`.
///
/// Callers that want multiple realisations pass each captured sweep
/// through this function in turn and feed the collected `Vec<Vec<_>>`
/// to [`compute_coherence_from_realizations`].
pub fn deconvolve_sweep(
    recording: &[f32],
    reference: &[f32],
    sample_rate: u32,
) -> Result<Vec<Complex<f32>>, String> {
    if sample_rate == 0 {
        return Err("deconvolve_sweep: zero sample_rate".to_string());
    }

    // Keep the complete capture, including its reverberant tail. Padding only
    // to this common FFT length prevents a truncated tail from wrapping into
    // the start of the deconvolved impulse response.
    let n = recording.len().max(reference.len());
    let fft_size = n.next_power_of_two();
    let recording_f64: Vec<f64> = recording.iter().map(|&sample| sample as f64).collect();
    let reference_f64: Vec<f64> = reference.iter().map(|&sample| sample as f64).collect();
    let spectrum = deconvolve_sweep_f64_spectrum(&recording_f64, &reference_f64, fft_size)?;
    Ok(spectrum[..fft_size / 2 + 1]
        .iter()
        .map(|value| Complex::new(value.re as f32, value.im as f32))
        .collect())
}

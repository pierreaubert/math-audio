//! Measurement-oriented DSP building blocks.
//!
//! This module keeps the recording path in one place: lag validation, tail-aware
//! ESS deconvolution, harmonic extraction, repeated-capture averaging, and a
//! small quality verdict for callers such as room-EQ wizards.

use super::compute::compute_coherence_from_realizations;
use super::compute::compute_thd_from_ir;
use super::estimate::estimate_lag_with_confidence;
use super::estimate::estimate_noise_floor_db_from_silence;
use super::plan::deconvolve_sweep;
use super::plan::plan_fft_forward;
use super::plan::plan_fft_inverse;
use super::types::AveragedEssResponse;
use super::types::AveragedResponse;
use super::types::ClippingInfo;
use super::types::ClockDriftEstimate;
use super::types::EssAnalysisResult;
use super::types::LagEstimate;
use super::types::MeasurementQualityConfig;
use super::types::MeasurementQualityReport;
use rustfft::num_complex::Complex;

fn active_signal_span(signal: &[f32]) -> Option<(usize, usize)> {
    let peak = signal
        .iter()
        .filter(|sample| sample.is_finite())
        .map(|sample| sample.abs())
        .fold(0.0_f32, f32::max);
    if peak <= f32::EPSILON {
        return None;
    }
    let threshold = peak * 1e-6;
    let start = signal.iter().position(|sample| sample.abs() > threshold)?;
    let end = signal.iter().rposition(|sample| sample.abs() > threshold)? + 1;
    Some((start, end))
}

/// Return the active (non-padding) duration of a reference signal.
pub fn effective_sweep_duration_seconds(reference: &[f32], sample_rate: u32) -> Option<f32> {
    if sample_rate == 0 {
        return None;
    }
    let (start, end) = active_signal_span(reference)?;
    Some((end - start) as f32 / sample_rate as f32)
}

/// Run the canonical tail-aware exponential-sine-sweep measurement path.
///
/// `recording` may be longer than `reference`; its reverberant tail is retained
/// in the FFT and therefore cannot circularly wrap into the beginning of the
/// returned IR. Silence padding in `reference` is excluded from the Farina
/// harmonic timing calculation.
pub fn analyze_ess_recording(
    recording: &[f32],
    reference: &[f32],
    sample_rate: u32,
    sweep_range: (f32, f32),
) -> Result<EssAnalysisResult, String> {
    if sample_rate == 0 {
        return Err("analyze_ess_recording: sample rate must be greater than zero".to_string());
    }
    if recording.is_empty() || reference.is_empty() {
        return Err("analyze_ess_recording: recording and reference must be non-empty".to_string());
    }
    if recording.len() < reference.len() {
        return Err(format!(
            "analyze_ess_recording: recording len {} is shorter than reference len {}",
            recording.len(),
            reference.len()
        ));
    }
    if recording
        .iter()
        .chain(reference)
        .any(|sample| !sample.is_finite())
    {
        return Err("analyze_ess_recording: inputs must contain only finite samples".to_string());
    }

    let lag = estimate_lag_with_confidence(reference, recording)?;
    let lag_offset = lag.lag_samples.max(0) as usize;
    if lag.lag_samples < 0 {
        return Err(
            "analyze_ess_recording: recording starts before the reference; cannot preserve its tail"
                .to_string(),
        );
    }
    if lag_offset >= recording.len() {
        return Err("analyze_ess_recording: estimated lag exceeds recording length".to_string());
    }

    let aligned_recording = &recording[lag_offset..];
    if aligned_recording.len() < reference.len() {
        return Err(
            "analyze_ess_recording: aligned recording is shorter than reference".to_string(),
        );
    }
    let frequency_response = deconvolve_sweep(aligned_recording, reference, sample_rate)?;
    let fft_size = (frequency_response.len().saturating_sub(1)) * 2;
    if fft_size == 0 {
        return Err("analyze_ess_recording: deconvolution returned no FFT bins".to_string());
    }

    let mut full_spectrum = vec![Complex::new(0.0_f32, 0.0_f32); fft_size];
    for (index, value) in frequency_response.iter().enumerate() {
        full_spectrum[index] = *value;
    }
    for index in 1..fft_size / 2 {
        full_spectrum[fft_size - index] = frequency_response[index].conj();
    }
    let ifft = plan_fft_inverse(fft_size);
    ifft.process(&mut full_spectrum);
    let scale = 1.0 / fft_size as f32;
    let impulse_response: Vec<f32> = full_spectrum.iter().map(|value| value.re * scale).collect();

    let (start_freq, end_freq) = sweep_range;
    let sweep_duration_seconds = effective_sweep_duration_seconds(reference, sample_rate)
        .ok_or("analyze_ess_recording: reference contains no active sweep")?;
    let frequencies: Vec<f32> = (0..frequency_response.len())
        .map(|index| index as f32 * sample_rate as f32 / fft_size as f32)
        .collect();
    let magnitude_db: Vec<f32> = frequency_response
        .iter()
        .map(|value| {
            let magnitude = value.norm();
            if magnitude > 1e-10 {
                20.0 * magnitude.log10()
            } else {
                -200.0
            }
        })
        .collect();
    let harmonic_impulse_responses = extract_log_sweep_harmonic_impulse_responses(
        &impulse_response,
        sample_rate as f32,
        start_freq,
        end_freq,
        sweep_duration_seconds,
    )?;
    let (thd_percent, harmonic_distortion_db) = compute_thd_from_ir(
        &impulse_response,
        sample_rate as f32,
        &frequencies,
        &magnitude_db,
        start_freq,
        end_freq,
        sweep_duration_seconds,
    );

    Ok(EssAnalysisResult {
        frequencies,
        frequency_response,
        impulse_response,
        harmonic_impulse_responses,
        thd_percent,
        harmonic_distortion_db,
        lag,
        sweep_duration_seconds,
    })
}

/// Alias using the common "log sweep" terminology.
pub fn analyze_log_sweep_recording(
    recording: &[f32],
    reference: &[f32],
    sample_rate: u32,
    sweep_range: (f32, f32),
) -> Result<EssAnalysisResult, String> {
    analyze_ess_recording(recording, reference, sample_rate, sweep_range)
}

/// Extract full-length, windowed Farina harmonic impulse responses (H2..H5).
pub fn extract_log_sweep_harmonic_impulse_responses(
    impulse: &[f32],
    sample_rate: f32,
    start_freq: f32,
    end_freq: f32,
    duration: f32,
) -> Result<Vec<Vec<f32>>, String> {
    if impulse.is_empty() {
        return Err("harmonic extraction: impulse response is empty".to_string());
    }
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(
            "harmonic extraction: sample rate must be finite and greater than zero".to_string(),
        );
    }
    if !start_freq.is_finite()
        || !end_freq.is_finite()
        || start_freq <= 0.0
        || end_freq <= start_freq
    {
        return Err(
            "harmonic extraction: sweep frequencies must satisfy 0 < start < end".to_string(),
        );
    }
    if !duration.is_finite() || duration <= 0.0 {
        return Err(
            "harmonic extraction: duration must be finite and greater than zero".to_string(),
        );
    }

    let peak_idx = impulse
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
        .map(|(index, _)| index)
        .unwrap_or(0);
    let sweep_log_ratio = (end_freq / start_freq).ln();
    let mut harmonic_irs = Vec::with_capacity(4);

    for harmonic_order in 2..=5 {
        let order = harmonic_order as f32;
        let dt = duration * order.ln() / sweep_log_ratio;
        let center = peak_idx as isize - (dt * sample_rate).round() as isize;
        let dt_next = duration * ((order + 1.0).ln() - order.ln()) / sweep_log_ratio;
        let minimum_window = (3.0 * sample_rate / (order * start_freq)).max(16.0);
        let window_len = ((dt_next * sample_rate * 0.8).max(minimum_window) as usize)
            .min((impulse.len() / 2).max(1));
        let window_len = window_len.max(1);
        let mut harmonic_ir = vec![0.0_f32; impulse.len()];
        for index in 0..window_len {
            let window = if window_len == 1 {
                1.0
            } else {
                0.5 * (1.0
                    - (2.0 * std::f32::consts::PI * index as f32 / (window_len - 1) as f32).cos())
            };
            let source = (center - window_len as isize / 2 + index as isize)
                .rem_euclid(impulse.len() as isize) as usize;
            harmonic_ir[source] = impulse[source] * window;
        }
        harmonic_irs.push(harmonic_ir);
    }
    Ok(harmonic_irs)
}

/// Average repeated deconvolved transfer responses without rejecting captures.
pub fn average_complex_responses(
    realizations: &[Vec<Complex<f32>>],
) -> Result<Vec<Complex<f32>>, String> {
    if realizations.is_empty() {
        return Err("average_complex_responses: no realizations".to_string());
    }
    let bins = realizations[0].len();
    for (index, realization) in realizations.iter().enumerate() {
        if realization.len() != bins {
            return Err(format!(
                "average_complex_responses: realization {index} has {} bins, expected {bins}",
                realization.len()
            ));
        }
        if realization
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(format!(
                "average_complex_responses: realization {index} contains non-finite data"
            ));
        }
    }
    let mut average = vec![Complex::new(0.0_f32, 0.0_f32); bins];
    for realization in realizations {
        for (output, input) in average.iter_mut().zip(realization) {
            *output += *input;
        }
    }
    let scale = 1.0 / realizations.len() as f32;
    for value in &mut average {
        *value *= scale;
    }
    Ok(average)
}

fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

/// Median/MAD-select repeated sweeps, then return their complex mean and
/// coherence. The selection is global per capture, so a single bad sweep does
/// not create a discontinuous per-frequency response.
pub fn average_deconvolved_sweeps(
    realizations: &[Vec<Complex<f32>>],
) -> Result<AveragedResponse, String> {
    let _ = average_complex_responses(realizations)?;
    let bins = realizations[0].len();
    if realizations.len() == 1 || bins == 0 {
        return Ok(AveragedResponse {
            response: realizations[0].clone(),
            coherence: Vec::new(),
            accepted_indices: (0..realizations.len()).collect(),
            rejected_indices: Vec::new(),
        });
    }

    let num_realizations = realizations.len();
    // Reusable scratch buffers: recordings are typically <= 8 channels, so a
    // fill-and-sort per bin on a hoisted buffer beats two fresh Vec per bin.
    let mut scratch_re = Vec::with_capacity(num_realizations);
    let mut scratch_im = Vec::with_capacity(num_realizations);
    let mut medians_re = Vec::with_capacity(bins);
    let mut medians_im = Vec::with_capacity(bins);
    for bin in 0..bins {
        scratch_re.clear();
        scratch_im.clear();
        scratch_re.extend(realizations.iter().map(|value| value[bin].re));
        scratch_im.extend(realizations.iter().map(|value| value[bin].im));
        medians_re.push(median(&mut scratch_re));
        medians_im.push(median(&mut scratch_im));
    }

    let mut distances = Vec::with_capacity(bins);
    let scores: Vec<f32> = realizations
        .iter()
        .map(|realization| {
            distances.clear();
            for bin in 0..bins {
                let center = Complex::new(medians_re[bin], medians_im[bin]);
                let scale = center.norm().max(1e-6);
                distances.push((realization[bin] - center).norm() / scale);
            }
            median(&mut distances)
        })
        .collect();
    // `scores` is needed below, so median over a reusable sorted copy.
    scratch_re.clear();
    scratch_re.extend_from_slice(&scores);
    let center_score = median(&mut scratch_re);
    let mut deviations: Vec<f32> = scores
        .iter()
        .map(|score| (score - center_score).abs())
        .collect();
    let mad = median(&mut deviations);
    let threshold = center_score + 3.0 * mad.max(1e-6);
    let accepted_indices: Vec<usize> = scores
        .iter()
        .enumerate()
        .filter_map(|(index, score)| (*score <= threshold).then_some(index))
        .collect();
    if accepted_indices.is_empty() {
        return Err("average_deconvolved_sweeps: all realizations were rejected".to_string());
    }
    let rejected_indices: Vec<usize> = (0..realizations.len())
        .filter(|index| !accepted_indices.contains(index))
        .collect();
    let accepted: Vec<Vec<Complex<f32>>> = accepted_indices
        .iter()
        .map(|index| realizations[*index].clone())
        .collect();
    let response = average_complex_responses(&accepted)?;
    let coherence = if accepted.len() >= 4 {
        compute_coherence_from_realizations(&accepted)?
    } else {
        Vec::new()
    };

    Ok(AveragedResponse {
        response,
        coherence,
        accepted_indices,
        rejected_indices,
    })
}

/// Align, tail-pad, deconvolve, and robustly average repeated ESS captures.
///
/// Captures may have different leading latency and different reverberant-tail
/// lengths. Each capture is first aligned independently; all aligned captures
/// are then padded to a common length before deconvolution so their FFT grids
/// match. Outlier selection is global per capture and is delegated to
/// [`average_deconvolved_sweeps`].
pub fn average_ess_recordings(
    recordings: &[Vec<f32>],
    reference: &[f32],
    sample_rate: u32,
) -> Result<AveragedEssResponse, String> {
    if recordings.is_empty() {
        return Err("average_ess_recordings: no recordings".to_string());
    }
    if reference.is_empty() {
        return Err("average_ess_recordings: reference must be non-empty".to_string());
    }
    if sample_rate == 0 {
        return Err("average_ess_recordings: sample rate must be greater than zero".to_string());
    }
    if reference.iter().any(|sample| !sample.is_finite()) {
        return Err(
            "average_ess_recordings: reference must contain only finite samples".to_string(),
        );
    }

    let mut lag_estimates = Vec::with_capacity(recordings.len());
    let mut aligned_recordings = Vec::with_capacity(recordings.len());
    let mut common_length = reference.len();
    for (index, recording) in recordings.iter().enumerate() {
        let lag = estimate_lag_with_confidence(reference, recording).map_err(|error| {
            format!("average_ess_recordings: capture {index} lag estimation failed: {error}")
        })?;
        if lag.lag_samples < 0 {
            return Err(format!(
                "average_ess_recordings: capture {index} starts before the reference"
            ));
        }
        let start = lag.lag_samples as usize;
        if start >= recording.len() {
            return Err(format!(
                "average_ess_recordings: capture {index} lag exceeds recording length"
            ));
        }
        let aligned = recording[start..].to_vec();
        if aligned.len() < reference.len() {
            return Err(format!(
                "average_ess_recordings: capture {index} is shorter than the reference after alignment"
            ));
        }
        common_length = common_length.max(aligned.len());
        lag_estimates.push(lag);
        aligned_recordings.push(aligned);
    }

    let mut realizations = Vec::with_capacity(aligned_recordings.len());
    for aligned in &aligned_recordings {
        let mut padded = aligned.clone();
        padded.resize(common_length, 0.0);
        realizations.push(deconvolve_sweep(&padded, reference, sample_rate)?);
    }
    let averaged = average_deconvolved_sweeps(&realizations)?;
    Ok(AveragedEssResponse {
        averaged,
        lag_estimates,
    })
}

/// H1 transfer estimator from repeated input/output spectra:
/// `H1 = <Y X*> / <X X*>` per frequency bin.
pub fn compute_h1_transfer_response(
    input_spectra: &[Vec<Complex<f32>>],
    output_spectra: &[Vec<Complex<f32>>],
) -> Result<Vec<Complex<f32>>, String> {
    if input_spectra.is_empty() || input_spectra.len() != output_spectra.len() {
        return Err(
            "compute_h1_transfer_response: input/output realization counts differ".to_string(),
        );
    }
    let bins = input_spectra[0].len();
    if output_spectra.iter().any(|spectrum| spectrum.len() != bins)
        || input_spectra.iter().any(|spectrum| spectrum.len() != bins)
    {
        return Err("compute_h1_transfer_response: realization bin counts differ".to_string());
    }
    if input_spectra.iter().chain(output_spectra).any(|spectrum| {
        spectrum
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
    }) {
        return Err(
            "compute_h1_transfer_response: spectra must contain only finite data".to_string(),
        );
    }
    let mut result = vec![Complex::new(0.0_f32, 0.0_f32); bins];
    for bin in 0..bins {
        let mut numerator = Complex::new(0.0_f32, 0.0_f32);
        let mut denominator = 0.0_f32;
        for (input, output) in input_spectra.iter().zip(output_spectra) {
            numerator += output[bin] * input[bin].conj();
            denominator += input[bin].norm_sqr();
        }
        if denominator > f32::EPSILON {
            result[bin] = numerator / denominator;
        }
    }
    Ok(result)
}

/// Detect overloads in a recorded floating-point buffer.
pub fn detect_clipping(recording: &[f32]) -> ClippingInfo {
    let non_finite_samples = recording
        .iter()
        .filter(|sample| !sample.is_finite())
        .count();
    let clipped_samples = recording
        .iter()
        .filter(|sample| !sample.is_finite() || sample.abs() >= 0.999)
        .count();
    ClippingInfo {
        clipped_samples,
        non_finite_samples,
        total_samples: recording.len(),
        fraction: if recording.is_empty() {
            0.0
        } else {
            clipped_samples as f32 / recording.len() as f32
        },
    }
}

fn median_snr(snr_db: &[f32]) -> Option<f32> {
    let mut finite: Vec<f32> = snr_db
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    (!finite.is_empty()).then(|| median(&mut finite))
}

/// Combine lag confidence, coherence, SNR, and clipping into one verdict.
pub fn assess_measurement_quality(
    recording: &[f32],
    lag: &LagEstimate,
    coherence: Option<&[f32]>,
    measured_spectrum_db: Option<&[f32]>,
    noise_floor_db: Option<&[f32]>,
    config: MeasurementQualityConfig,
) -> MeasurementQualityReport {
    let clipping = detect_clipping(recording);
    let mut issues = Vec::new();
    let mut missing_metrics = Vec::new();
    let config_valid = config.minimum_lag_confidence.is_finite()
        && (0.0..=1.0).contains(&config.minimum_lag_confidence)
        && config.minimum_mean_coherence.is_finite()
        && (0.0..=1.0).contains(&config.minimum_mean_coherence)
        && config.minimum_median_snr_db.is_finite()
        && config.minimum_median_snr_db > 0.0
        && config.maximum_clip_fraction.is_finite()
        && (0.0..=1.0).contains(&config.maximum_clip_fraction);
    if !config_valid {
        issues.push("measurement quality thresholds are invalid".to_string());
    }
    if recording.is_empty() {
        issues.push("recording is empty".to_string());
    }
    if clipping.non_finite_samples > 0 {
        issues.push(format!(
            "recording contains {} non-finite samples",
            clipping.non_finite_samples
        ));
    }

    let lag_fields_valid = lag.normalized_peak.is_finite()
        && lag.confidence.is_finite()
        && !lag.peak_to_sidelobe_db.is_nan()
        && (0.0..=1.0).contains(&lag.normalized_peak)
        && (0.0..=1.0).contains(&lag.confidence);
    if !lag_fields_valid {
        issues.push("lag diagnostics contain invalid values".to_string());
    }
    if lag_fields_valid && lag.confidence < config.minimum_lag_confidence {
        issues.push(format!(
            "lag confidence {:.3} is below {:.3}",
            lag.confidence, config.minimum_lag_confidence
        ));
    }

    let mean_coherence = match coherence {
        None => {
            missing_metrics.push("coherence".to_string());
            None
        }
        Some([]) => {
            issues.push("coherence data was supplied but empty".to_string());
            None
        }
        Some(values) => {
            let invalid = values
                .iter()
                .filter(|value| {
                    let value = **value;
                    !value.is_finite() || !(0.0..=1.0).contains(&value)
                })
                .count();
            if invalid > 0 {
                issues.push(format!("coherence contains {invalid} invalid bins"));
            }
            let finite: Vec<f32> = values
                .iter()
                .copied()
                .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
                .collect();
            let mean =
                (!finite.is_empty()).then(|| finite.iter().sum::<f32>() / finite.len() as f32);
            if mean.is_none() {
                issues.push("coherence has no usable bins".to_string());
            } else if let Some(value) = mean
                && value < config.minimum_mean_coherence
            {
                issues.push(format!(
                    "mean coherence {:.3} is below {:.3}",
                    value, config.minimum_mean_coherence
                ));
            }
            mean
        }
    };
    if config.require_coherence && coherence.is_none() {
        issues.push("coherence is required by the quality policy".to_string());
    }

    let snr_db = match (measured_spectrum_db, noise_floor_db) {
        (None, None) => {
            missing_metrics.push("measured_spectrum_db".to_string());
            missing_metrics.push("noise_floor_db".to_string());
            Vec::new()
        }
        (Some(_), None) => {
            missing_metrics.push("noise_floor_db".to_string());
            issues.push("measured spectrum was supplied without a noise floor".to_string());
            Vec::new()
        }
        (None, Some(_)) => {
            missing_metrics.push("measured_spectrum_db".to_string());
            issues.push("noise floor was supplied without a measured spectrum".to_string());
            Vec::new()
        }
        (Some(measured), Some(noise)) if measured.len() != noise.len() => {
            issues.push(format!(
                "measured spectrum length {} does not match noise floor length {}",
                measured.len(),
                noise.len()
            ));
            Vec::new()
        }
        (Some(measured), Some(noise)) => {
            if measured.is_empty() {
                issues.push("SNR spectra were supplied but empty".to_string());
                Vec::new()
            } else if measured
                .iter()
                .chain(noise.iter())
                .any(|value| !value.is_finite())
            {
                issues.push("SNR spectra contain non-finite values".to_string());
                Vec::new()
            } else {
                measured
                    .iter()
                    .zip(noise)
                    .map(|(signal, floor)| signal - floor)
                    .collect()
            }
        }
    };
    if config.require_snr && snr_db.is_empty() {
        issues.push("SNR data is required by the quality policy".to_string());
    }
    let median_snr_db = median_snr(&snr_db);
    if let Some(snr) = median_snr_db
        && snr < config.minimum_median_snr_db
    {
        issues.push(format!(
            "median SNR {:.1} dB is below {:.1} dB",
            snr, config.minimum_median_snr_db
        ));
    }
    if clipping.fraction > config.maximum_clip_fraction {
        issues.push(format!(
            "recording clipping fraction {:.4} exceeds {:.4}",
            clipping.fraction, config.maximum_clip_fraction
        ));
    }

    let mut components = vec![lag.confidence.clamp(0.0, 1.0)];
    if let Some(coherence) = mean_coherence {
        components.push(coherence.clamp(0.0, 1.0));
    }
    if let Some(snr) = median_snr_db {
        components.push((snr / config.minimum_median_snr_db.max(1.0)).clamp(0.0, 1.0));
    }
    components.push(
        (1.0 - clipping.fraction / config.maximum_clip_fraction.max(f32::EPSILON)).clamp(0.0, 1.0),
    );
    let score = if config_valid && lag_fields_valid && clipping.non_finite_samples == 0 {
        (components.iter().sum::<f32>() / components.len() as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };

    MeasurementQualityReport {
        trustworthy: issues.is_empty(),
        score,
        quality_data_complete: missing_metrics.is_empty(),
        missing_metrics,
        lag_confidence: lag.confidence,
        mean_coherence,
        snr_db,
        median_snr_db,
        clipping,
        issues,
    }
}

/// Convenience quality assessment for callers that have a pre-roll/silence
/// capture instead of an already computed noise-floor curve.
pub fn assess_measurement_quality_from_silence(
    recording: &[f32],
    lag: &LagEstimate,
    coherence: Option<&[f32]>,
    measured_spectrum_db: Option<&[f32]>,
    silence: &[f32],
    sample_rate: u32,
    config: MeasurementQualityConfig,
) -> MeasurementQualityReport {
    let noise_floor_db = estimate_noise_floor_db_from_silence(silence, sample_rate);
    assess_measurement_quality(
        recording,
        lag,
        coherence,
        measured_spectrum_db,
        Some(&noise_floor_db),
        config,
    )
}

/// Estimate relative playback/capture clock drift from lags at the beginning
/// and end of the reference sweep.
/// Fit a linear phase ramp and return its equivalent propagation delay.
///
/// The input phase may be wrapped in degrees. This fit over the supplied
/// frequency grid is useful as a stable latency diagnostic for deconvolved
/// ESS responses.
pub fn fit_linear_phase_delay_seconds(
    frequencies_hz: &[f32],
    phase_deg: &[f32],
) -> Result<f64, String> {
    if frequencies_hz.len() != phase_deg.len() || frequencies_hz.len() < 2 {
        return Err("fit_linear_phase_delay_seconds: need two matching samples".to_string());
    }
    if frequencies_hz
        .iter()
        .zip(phase_deg)
        .any(|(frequency, phase)| !frequency.is_finite() || *frequency < 0.0 || !phase.is_finite())
    {
        return Err("fit_linear_phase_delay_seconds: inputs must be finite".to_string());
    }
    let mut unwrapped = Vec::with_capacity(phase_deg.len());
    for (index, &phase) in phase_deg.iter().enumerate() {
        let radians = phase.to_radians();
        if index == 0 {
            unwrapped.push(radians);
        } else {
            let mut delta = radians - unwrapped[index - 1];
            while delta > std::f32::consts::PI {
                delta -= 2.0 * std::f32::consts::PI;
            }
            while delta < -std::f32::consts::PI {
                delta += 2.0 * std::f32::consts::PI;
            }
            unwrapped.push(unwrapped[index - 1] + delta);
        }
    }
    let count = frequencies_hz.len() as f64;
    let sum_x: f64 = frequencies_hz.iter().map(|value| *value as f64).sum();
    let sum_y: f64 = unwrapped.iter().map(|value| *value as f64).sum();
    let sum_xx: f64 = frequencies_hz
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum();
    let sum_xy: f64 = frequencies_hz
        .iter()
        .zip(&unwrapped)
        .map(|(frequency, phase)| *frequency as f64 * *phase as f64)
        .sum();
    let denominator = count * sum_xx - sum_x * sum_x;
    if denominator.abs() <= f64::EPSILON {
        return Err("fit_linear_phase_delay_seconds: frequency grid has no span".to_string());
    }
    let slope = (count * sum_xy - sum_x * sum_y) / denominator;
    Ok(-slope / (2.0 * std::f64::consts::PI))
}

pub fn estimate_clock_drift(
    reference: &[f32],
    recording: &[f32],
    sample_rate: u32,
) -> Result<ClockDriftEstimate, String> {
    if sample_rate == 0 || reference.len() < 32 || recording.is_empty() {
        return Err(
            "estimate_clock_drift: signals are too short or sample rate is zero".to_string(),
        );
    }
    let window = (reference.len() / 4).clamp(16, 8192);
    if reference.len() < window * 2 {
        return Err(
            "estimate_clock_drift: reference must contain two analysis windows".to_string(),
        );
    }
    let start = estimate_lag_with_confidence(&reference[..window], recording)?;
    let end_offset = reference.len() - window;
    let end = estimate_lag_with_confidence(&reference[end_offset..], recording)?;
    let elapsed_samples = end_offset as f64;
    let lag_change = (end.lag_samples - start.lag_samples) as f64 - elapsed_samples;
    let elapsed_seconds = elapsed_samples / sample_rate as f64;
    let ppm = lag_change / elapsed_seconds * 1e6;
    Ok(ClockDriftEstimate {
        ppm,
        start_lag_samples: start.lag_samples,
        end_lag_samples: end.lag_samples,
        confidence: start.confidence.min(end.confidence),
    })
}

/// Correct a recorded buffer for a measured relative sample-clock drift.
///
/// The leading lag is used as the fixed time anchor. A positive drift means
/// that the capture clock accumulated samples relative to playback; the source
/// coordinate therefore advances slightly faster than the corrected output
/// coordinate. Linear interpolation keeps the operation deterministic and
/// avoids introducing a second FFT or an unbounded filter state.
pub fn correct_clock_drift(
    recording: &[f32],
    estimate: &ClockDriftEstimate,
) -> Result<Vec<f32>, String> {
    if recording.is_empty() {
        return Err("correct_clock_drift: recording must be non-empty".to_string());
    }
    if recording.iter().any(|sample| !sample.is_finite()) {
        return Err("correct_clock_drift: recording must contain only finite samples".to_string());
    }
    if !estimate.ppm.is_finite()
        || estimate.ppm <= -1_000_000.0
        || !estimate.confidence.is_finite()
        || !(0.0..=1.0).contains(&estimate.confidence)
    {
        return Err("correct_clock_drift: estimate contains invalid values".to_string());
    }

    let scale = 1.0 + estimate.ppm / 1_000_000.0;
    let anchor = estimate.start_lag_samples.max(0) as f64;
    let mut corrected = vec![0.0_f32; recording.len()];
    for (output_index, sample) in corrected.iter_mut().enumerate() {
        let output_position = output_index as f64;
        let source_position = if output_position < anchor {
            output_position
        } else {
            anchor + (output_position - anchor) * scale
        };
        if !(0.0..=(recording.len() - 1) as f64).contains(&source_position) {
            continue;
        }
        let left = source_position.floor() as usize;
        let right = (left + 1).min(recording.len() - 1);
        let fraction = (source_position - left as f64) as f32;
        *sample = recording[left] + fraction * (recording[right] - recording[left]);
    }
    Ok(corrected)
}

/// Deconvolve one period of a bipolar MLS recording into a circular IR.
///
/// The FFT implementation is O(N log N), uses the MLS spectrum directly, and
/// retains the periodic MLS semantics. Callers should provide a capture at
/// least one full MLS period; extra samples are ignored rather than wrapped
/// into the returned period.
pub fn deconvolve_mls_to_ir(recording: &[f32], mls: &[f32]) -> Result<Vec<f32>, String> {
    if mls.is_empty() || recording.len() < mls.len() {
        return Err("deconvolve_mls_to_ir: recording must contain a full MLS period".to_string());
    }
    if recording
        .iter()
        .chain(mls)
        .any(|sample| !sample.is_finite())
    {
        return Err("deconvolve_mls_to_ir: inputs must contain only finite samples".to_string());
    }
    let n = mls.len();
    // MLS periods are 2^m-1 samples, not powers of two. RustFFT supports the
    // odd period directly; zero-padding would change the circular-convolution
    // algebra and smear the recovered impulse response.
    let fft_size = n;
    let mut input: Vec<Complex<f32>> = mls
        .iter()
        .map(|sample| Complex::new(*sample, 0.0))
        .collect();
    let mut output: Vec<Complex<f32>> = recording[..n]
        .iter()
        .map(|sample| Complex::new(*sample, 0.0))
        .collect();
    input.resize(fft_size, Complex::new(0.0, 0.0));
    output.resize(fft_size, Complex::new(0.0, 0.0));
    let fft = plan_fft_forward(fft_size);
    fft.process(&mut input);
    fft.process(&mut output);
    let peak_power = input
        .iter()
        .map(|value| value.norm_sqr())
        .fold(0.0_f32, f32::max)
        .max(f32::MIN_POSITIVE);
    let regularization = peak_power * 1e-6;
    for (output_bin, input_bin) in output.iter_mut().zip(&input) {
        *output_bin = *output_bin * input_bin.conj() / (input_bin.norm_sqr() + regularization);
    }
    let ifft = plan_fft_inverse(fft_size);
    ifft.process(&mut output);
    let scale = 1.0 / fft_size as f32;
    Ok(output[..n].iter().map(|value| value.re * scale).collect())
}

/// Alias with the shorter name used by MLS measurement callers.
pub fn deconvolve_mls(recording: &[f32], mls: &[f32]) -> Result<Vec<f32>, String> {
    deconvolve_mls_to_ir(recording, mls)
}

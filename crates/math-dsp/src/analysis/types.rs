use std::path::Path;

/// Spectrum result: (frequencies, magnitudes_db, phases_deg)
pub(super) type SpectrumResult = Result<(Vec<f32>, Vec<f32>, Vec<f32>), String>;

/// Result of standalone WAV buffer analysis
#[derive(Debug, Clone)]
pub struct WavAnalysisOutput {
    /// Frequency points in Hz (log-spaced)
    pub frequencies: Vec<f32>,
    /// Peak-amplitude magnitude in dBFS
    pub magnitude_db: Vec<f32>,
    /// Phase in degrees
    pub phase_deg: Vec<f32>,
}

/// Result of FFT analysis
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Frequency bins in Hz
    pub frequencies: Vec<f32>,
    /// Magnitude in dBFS
    pub spl_db: Vec<f32>,
    /// Phase in degrees (compensated for latency)
    pub phase_deg: Vec<f32>,
    /// Estimated latency in samples
    pub estimated_lag_samples: isize,
    /// Impulse response (time domain)
    pub impulse_response: Vec<f32>,
    /// Time vector for impulse response in ms
    pub impulse_time_ms: Vec<f32>,
    /// Excess group delay in ms
    pub excess_group_delay_ms: Vec<f32>,
    /// Total Harmonic Distortion + Noise (%)
    pub thd_percent: Vec<f32>,
    /// Harmonic distortion curves (2nd, 3rd, etc) in dB
    pub harmonic_distortion_db: Vec<Vec<f32>>,
    /// RT60 decay time in ms
    pub rt60_ms: Vec<f32>,
    /// Clarity C50 in dB
    pub clarity_c50_db: Vec<f32>,
    /// Clarity C80 in dB
    pub clarity_c80_db: Vec<f32>,
    /// Spectrogram (time × frequency peak-amplitude magnitude in dBFS)
    pub spectrogram_db: Vec<Vec<f32>>,
}

/// Read analysis results from CSV file
///
/// Parses CSV with columns: frequency_hz, spl_db, phase_deg, thd_percent, rt60_ms, c50_db, c80_db, group_delay_ms
/// Also supports legacy format with just: frequency_hz, spl_db, phase_deg
pub fn read_analysis_csv(csv_path: &Path) -> Result<AnalysisResult, String> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(csv_path).map_err(|e| format!("Failed to open CSV: {}", e))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Read header
    let header = lines
        .next()
        .ok_or("Empty CSV file")?
        .map_err(|e| format!("Failed to read header: {}", e))?;

    let columns: Vec<&str> = header.split(',').map(|s| s.trim()).collect();
    let has_extended_format = columns.len() >= 8;

    let mut frequencies = Vec::new();
    let mut spl_db = Vec::new();
    let mut phase_deg = Vec::new();
    let mut thd_percent = Vec::new();
    let mut rt60_ms = Vec::new();
    let mut clarity_c50_db = Vec::new();
    let mut clarity_c80_db = Vec::new();
    let mut excess_group_delay_ms = Vec::new();

    for line in lines {
        let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

        if parts.len() < 3 {
            continue;
        }

        let freq: f32 = parts[0].parse().unwrap_or(0.0);
        let spl: f32 = parts[1].parse().unwrap_or(0.0);
        let phase: f32 = parts[2].parse().unwrap_or(0.0);

        frequencies.push(freq);
        spl_db.push(spl);
        phase_deg.push(phase);

        if has_extended_format && parts.len() >= 8 {
            thd_percent.push(parts[3].parse().unwrap_or(0.0));
            rt60_ms.push(parts[4].parse().unwrap_or(0.0));
            clarity_c50_db.push(parts[5].parse().unwrap_or(0.0));
            clarity_c80_db.push(parts[6].parse().unwrap_or(0.0));
            excess_group_delay_ms.push(parts[7].parse().unwrap_or(0.0));
        }
    }

    // If legacy format, fill with zeros
    let n = frequencies.len();
    if thd_percent.is_empty() {
        thd_percent = vec![0.0; n];
        rt60_ms = vec![0.0; n];
        clarity_c50_db = vec![0.0; n];
        clarity_c80_db = vec![0.0; n];
        excess_group_delay_ms = vec![0.0; n];
    }

    Ok(AnalysisResult {
        frequencies,
        spl_db,
        phase_deg,
        estimated_lag_samples: 0,
        impulse_response: Vec::new(),
        impulse_time_ms: Vec::new(),
        thd_percent,
        harmonic_distortion_db: Vec::new(),
        rt60_ms,
        clarity_c50_db,
        clarity_c80_db,
        excess_group_delay_ms,
        spectrogram_db: Vec::new(),
    })
}

/// Window function type for FFT
#[derive(Debug, Clone, Copy)]
pub(super) enum WindowType {
    Tukey(f32), // alpha parameter (0.0-1.0)
}

/// Result of cross-correlation with analytic envelope detection.
///
/// The envelope peak corresponds to the probe's arrival time, detected
/// via Hilbert transform of the cross-correlation.
#[derive(Debug, Clone)]
pub struct CrossCorrelationEnvelopeResult {
    /// Analytic envelope of the cross-correlation
    pub envelope: Vec<f32>,
    /// Sample index of the peak (integer arrival time)
    pub peak_sample: usize,
    /// Sub-sample refined peak position via parabolic interpolation
    pub peak_sample_refined: f64,
    /// Peak envelope value (proportional to channel gain)
    pub peak_value: f32,
    /// Arrival time in milliseconds (sub-sample precision)
    pub arrival_ms: f64,
    /// Peak correlation normalised by the energy of the overlapping signals.
    pub normalized_peak: f32,
    /// Peak-to-sidelobe ratio in dB. `INFINITY` means no measurable sidelobe.
    pub peak_to_sidelobe_db: f32,
    /// Combined bounded confidence score in the range `[0, 1]`.
    pub confidence: f32,
}

/// A lag estimate together with enough diagnostics to reject an arbitrary
/// peak from a noise-only or disconnected recording.
#[derive(Debug, Clone, Copy)]
pub struct LagEstimate {
    /// Positive values mean that the recording starts later than the reference.
    pub lag_samples: isize,
    /// Normalised cross-correlation at the selected lag.
    pub normalized_peak: f32,
    /// Peak-to-sidelobe ratio in dB.
    pub peak_to_sidelobe_db: f32,
    /// Combined bounded confidence score in the range `[0, 1]`.
    pub confidence: f32,
}

// Keep old in-crate assertions such as `assert_eq!(estimate_lag(...), 0)`
// readable while the estimator now carries diagnostics.
impl PartialEq<isize> for LagEstimate {
    fn eq(&self, other: &isize) -> bool {
        self.lag_samples == *other
    }
}

/// Result of the canonical exponential-sine-sweep measurement path.
#[derive(Debug, Clone)]
pub struct EssAnalysisResult {
    /// FFT-grid frequencies for `frequency_response`.
    pub frequencies: Vec<f32>,
    /// Complex positive-frequency transfer response.
    pub frequency_response: Vec<rustfft::num_complex::Complex<f32>>,
    /// Linear, non-circular deconvolved impulse response.
    pub impulse_response: Vec<f32>,
    /// One full-length, windowed impulse response per harmonic (H2..H5).
    pub harmonic_impulse_responses: Vec<Vec<f32>>,
    /// THD percentage evaluated at `frequencies`.
    pub thd_percent: Vec<f32>,
    /// Harmonic levels in dB, one curve per harmonic (H2..H5).
    pub harmonic_distortion_db: Vec<Vec<f32>>,
    /// Estimated playback-to-recording lag.
    pub lag: LagEstimate,
    /// Sweep duration used for Farina harmonic offsets, excluding silence padding.
    pub sweep_duration_seconds: f32,
}

/// Result of robustly averaging repeated complex transfer responses.
#[derive(Debug, Clone)]
pub struct AveragedResponse {
    /// Complex mean of the accepted responses.
    pub response: Vec<rustfft::num_complex::Complex<f32>>,
    /// Magnitude-squared coherence of the accepted responses, when at least
    /// four captures remain; otherwise an empty vector.
    pub coherence: Vec<f32>,
    /// Original indices retained for the average.
    pub accepted_indices: Vec<usize>,
    /// Original indices rejected as global median/MAD outliers.
    pub rejected_indices: Vec<usize>,
}

/// Result of aligning and averaging complete repeated ESS captures.
#[derive(Debug, Clone)]
pub struct AveragedEssResponse {
    /// Averaged transfer response and capture-selection diagnostics.
    pub averaged: AveragedResponse,
    /// Lag diagnostics in the same order as the input captures.
    pub lag_estimates: Vec<LagEstimate>,
}

/// Default thresholds used by [`assess_measurement_quality`].
#[derive(Debug, Clone, Copy)]
pub struct MeasurementQualityConfig {
    /// Minimum lag confidence required for a trustworthy result.
    pub minimum_lag_confidence: f32,
    /// Minimum mean coherence required when coherence is available.
    pub minimum_mean_coherence: f32,
    /// Minimum median SNR in dB required when a noise floor is available.
    pub minimum_median_snr_db: f32,
    /// Maximum allowed fraction of clipped recording samples.
    pub maximum_clip_fraction: f32,
    /// Require coherence data before a result can be trustworthy.
    pub require_coherence: bool,
    /// Require measured and noise-floor spectra before a result can be trustworthy.
    pub require_snr: bool,
}

impl Default for MeasurementQualityConfig {
    fn default() -> Self {
        Self {
            minimum_lag_confidence: 0.2,
            minimum_mean_coherence: 0.5,
            minimum_median_snr_db: 10.0,
            maximum_clip_fraction: 0.001,
            require_coherence: false,
            require_snr: false,
        }
    }
}

/// Recording clipping/overload diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct ClippingInfo {
    pub clipped_samples: usize,
    pub non_finite_samples: usize,
    pub total_samples: usize,
    pub fraction: f32,
}

/// Combined measurement-quality verdict and the metrics supporting it.
#[derive(Debug, Clone)]
pub struct MeasurementQualityReport {
    pub trustworthy: bool,
    pub score: f32,
    /// True when all optional quality inputs were supplied and usable.
    /// `trustworthy` may still be true for a partial report when the caller
    /// intentionally omitted optional inputs; this field makes that choice
    /// explicit to downstream quality gates.
    pub quality_data_complete: bool,
    /// Names of optional quality inputs that were not supplied.
    pub missing_metrics: Vec<String>,
    pub lag_confidence: f32,
    pub mean_coherence: Option<f32>,
    pub snr_db: Vec<f32>,
    pub median_snr_db: Option<f32>,
    pub clipping: ClippingInfo,
    pub issues: Vec<String>,
}

/// Estimated relative sample-clock drift between playback and capture.
#[derive(Debug, Clone, Copy)]
pub struct ClockDriftEstimate {
    pub ppm: f64,
    pub start_lag_samples: isize,
    pub end_lag_samples: isize,
    pub confidence: f32,
}

/// Frequency responses computed from different time windows of an impulse response.
///
/// Direct sound, early reflections, and late reverb each have different
/// perceptual roles (Toole, Johnston) and should be corrected differently.
#[derive(Debug, Clone)]
pub struct WindowedFrequencyResponse {
    /// Direct sound frequency response (frequencies in Hz, SPL in dB)
    pub direct_sound_freq: Vec<f32>,
    pub direct_sound_spl: Vec<f32>,
    /// Early reflections frequency response
    pub early_reflections_freq: Vec<f32>,
    pub early_reflections_spl: Vec<f32>,
    /// Late/reverberant field frequency response
    pub late_reverb_freq: Vec<f32>,
    pub late_reverb_spl: Vec<f32>,
    /// Time boundaries used (in ms)
    pub direct_end_ms: f64,
    pub early_end_ms: f64,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum Rt60FitMethod {
    T30,
    T20,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Rt60Fit {
    pub(super) rt60_seconds: f32,
    pub(super) method: Rt60FitMethod,
    pub(super) r_squared: f32,
    pub(super) fit_start_seconds: f32,
    pub(super) fit_end_seconds: f32,
}

pub(super) fn fit_rt60_decay(
    decay_db: &[f32],
    sample_rate: f32,
    start_db: f32,
    end_db: f32,
    method: Rt60FitMethod,
) -> Option<Rt60Fit> {
    const MIN_FIT_POINTS: usize = 32;
    const MIN_FIT_DURATION_SECONDS: f32 = 0.015;
    const MIN_R_SQUARED: f32 = 0.97;

    let start = decay_db.iter().position(|value| *value <= start_db)?;
    let end = decay_db.iter().position(|value| *value <= end_db)?;
    if end <= start || end - start + 1 < MIN_FIT_POINTS {
        return None;
    }

    let fit_duration = (end - start) as f32 / sample_rate;
    if fit_duration < MIN_FIT_DURATION_SECONDS {
        return None;
    }

    let n = (end - start + 1) as f32;
    let mut sum_x = 0.0_f32;
    let mut sum_y = 0.0_f32;
    let mut sum_xx = 0.0_f32;
    let mut sum_xy = 0.0_f32;

    for (offset, y) in decay_db[start..=end].iter().enumerate() {
        let x = offset as f32 / sample_rate;
        sum_x += x;
        sum_y += *y;
        sum_xx += x * x;
        sum_xy += x * *y;
    }

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() <= f32::EPSILON {
        return None;
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    if !slope.is_finite() || slope >= 0.0 {
        return None;
    }

    let mean_y = sum_y / n;
    let mut ss_total = 0.0_f32;
    let mut ss_residual = 0.0_f32;
    for (offset, y) in decay_db[start..=end].iter().enumerate() {
        let x = offset as f32 / sample_rate;
        let fitted = intercept + slope * x;
        ss_total += (*y - mean_y).powi(2);
        ss_residual += (*y - fitted).powi(2);
    }

    if ss_total <= f32::EPSILON {
        return None;
    }

    let r_squared = 1.0 - ss_residual / ss_total;
    let rt60_seconds = -60.0 / slope;
    if !rt60_seconds.is_finite() || rt60_seconds <= 0.0 || r_squared < MIN_R_SQUARED {
        return None;
    }

    Some(Rt60Fit {
        rt60_seconds,
        method,
        r_squared,
        fit_start_seconds: start as f32 / sample_rate,
        fit_end_seconds: end as f32 / sample_rate,
    })
}

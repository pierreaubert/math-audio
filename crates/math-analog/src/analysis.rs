use std::f32::consts::TAU;

use math_audio_dsp::ebur128::{EbuR128, Mode};
use thiserror::Error;

const MAX_HARMONIC_ORDER: usize = 128;

/// Errors returned by the offline deterministic spectral helpers.
#[derive(Debug, Error, PartialEq)]
pub enum SpectralError {
    #[error("the spectral record must not be empty")]
    EmptyRecord,
    #[error("sample rate must be finite and greater than zero, got {0}")]
    InvalidSampleRate(f32),
    #[error("fundamental frequency must be finite and greater than zero, got {0}")]
    InvalidFundamental(f32),
    #[error("maximum harmonic order must be greater than zero")]
    InvalidMaximumOrder,
    #[error("maximum harmonic order {requested} exceeds the report limit {maximum}")]
    MaximumOrderTooLarge { requested: usize, maximum: usize },
    #[error("fundamental frequency {fundamental_hz} Hz is at or above Nyquist {nyquist_hz} Hz")]
    FundamentalAtOrAboveNyquist {
        fundamental_hz: f32,
        nyquist_hz: f32,
    },
    #[error("the two test tones must be distinct")]
    EqualTestTones,
    #[error("test tone {tone_hz} Hz is at or above Nyquist {nyquist_hz} Hz")]
    TestToneAtOrAboveNyquist { tone_hz: f32, nyquist_hz: f32 },
    #[error("spectral record contains a non-finite sample at index {0}")]
    NonFiniteSample(usize),
    #[error("spectral report length {report_length} does not match record length {record_length}")]
    ReportLengthMismatch {
        report_length: usize,
        record_length: usize,
    },
    #[error("oversample factor must be 2 or 4, got {0}")]
    InvalidOversampleFactor(usize),
    #[error(
        "oversampled reference has {samples} samples, which is not divisible by factor {factor}"
    )]
    ReferenceLengthNotMultiple { samples: usize, factor: usize },
    #[error("channel count must be greater than zero")]
    InvalidChannelCount,
    #[error(
        "interleaved record has {samples} samples, which is not divisible by {channels} channels"
    )]
    InterleavedRecordLengthMismatch { samples: usize, channels: usize },
    #[error("integrated loudness is unavailable for this record")]
    LoudnessUnavailable,
    #[error("EBU R128 measurement failed: {0}")]
    LoudnessMeasurement(String),
    #[error("peak ceiling must be finite and in (0, 1], got {0}")]
    InvalidPeakCeiling(f32),
    #[error("reference peak {peak} exceeds peak ceiling {ceiling}")]
    ReferencePeakExceedsCeiling { peak: f32, ceiling: f32 },
}

/// The normalization metadata attached to a deterministic harmonic report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralConvention {
    pub sample_rate: f32,
    pub record_length: usize,
    pub bin_spacing_hz: f32,
    pub one_sided: bool,
    pub fft_normalization: FftNormalization,
    pub window: WindowKind,
    pub coherent_gain: f32,
}

/// FFT normalization used by [`measure_harmonics`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftNormalization {
    OneSidedAmplitude,
}

/// Window used by [`measure_harmonics`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowKind {
    Rectangular,
}

/// One measured harmonic component.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HarmonicComponent {
    pub order: usize,
    pub frequency_hz: f32,
    pub folded_frequency_hz: f32,
    pub bin: usize,
    pub amplitude: f32,
    pub level_db: f32,
    pub aliases: bool,
}

/// Deterministic one-sided report for a coherent or non-coherent record.
#[derive(Debug, Clone, PartialEq)]
pub struct HarmonicReport {
    pub convention: SpectralConvention,
    pub fundamental_hz: f32,
    pub dc_amplitude: f32,
    pub nyquist_amplitude: f32,
    pub components: Vec<HarmonicComponent>,
}

/// Explicit distortion metrics derived from a harmonic report.
///
/// `thd` includes only measured, in-band harmonic components from H2 onward.
/// `thd_plus_n` is the residual RMS after removing the measured fundamental,
/// so it also includes DC, unmeasured harmonics, aliases, and noise. The two
/// definitions are intentionally not interchangeable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistortionReport {
    pub fundamental_amplitude: f32,
    pub harmonic_rms: f32,
    pub residual_rms: f32,
    pub thd: f32,
    pub thd_plus_n: f32,
    pub alias_rms: f32,
    pub alias_level_db: f32,
}

impl HarmonicReport {
    pub fn component(&self, order: usize) -> Option<HarmonicComponent> {
        self.components
            .iter()
            .copied()
            .find(|component| component.order == order)
    }

    /// Calculate THD, THD+N, and folded alias energy for the reported record.
    pub fn distortion(&self, samples: &[f32]) -> Result<DistortionReport, SpectralError> {
        if samples.len() != self.convention.record_length {
            return Err(SpectralError::ReportLengthMismatch {
                report_length: self.convention.record_length,
                record_length: samples.len(),
            });
        }
        if let Some(index) = samples.iter().position(|sample| !sample.is_finite()) {
            return Err(SpectralError::NonFiniteSample(index));
        }
        let fundamental_amplitude = self
            .component(1)
            .map_or(0.0, |component| component.amplitude);
        let harmonic_rms = self
            .components
            .iter()
            .filter(|component| component.order >= 2 && !component.aliases)
            .map(|component| component.amplitude * component.amplitude)
            .sum::<f32>()
            .sqrt()
            / std::f32::consts::SQRT_2;
        let alias_power = self
            .components
            .iter()
            .filter(|component| component.aliases)
            .map(|component| component.amplitude * component.amplitude)
            .sum::<f32>();
        let alias_rms = if alias_power > 0.0 {
            alias_power.sqrt() / std::f32::consts::SQRT_2
        } else {
            0.0
        };
        let total_rms = (samples.iter().map(|sample| sample * sample).sum::<f32>()
            / samples.len() as f32)
            .sqrt();
        let fundamental_rms = fundamental_amplitude / std::f32::consts::SQRT_2;
        let residual_rms = (total_rms * total_rms - fundamental_rms * fundamental_rms)
            .max(0.0)
            .sqrt();
        let ratio = |numerator: f32| {
            if fundamental_amplitude > 0.0 {
                numerator / fundamental_amplitude
            } else {
                f32::INFINITY
            }
        };
        let alias_level_db = if alias_rms > 0.0 {
            20.0 * alias_rms.log10()
        } else {
            f32::NEG_INFINITY
        };
        Ok(DistortionReport {
            fundamental_amplitude,
            harmonic_rms,
            residual_rms,
            thd: ratio(harmonic_rms * std::f32::consts::SQRT_2),
            thd_plus_n: ratio(residual_rms * std::f32::consts::SQRT_2),
            alias_rms,
            alias_level_db,
        })
    }
}

/// One two-tone intermodulation product identified by its tone coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntermodulationComponent {
    pub coefficient_a: i32,
    pub coefficient_b: i32,
    pub order: usize,
    pub frequency_hz: f32,
    pub folded_frequency_hz: f32,
    pub bin: usize,
    pub amplitude: f32,
    pub level_db: f32,
    pub aliases: bool,
}

/// Deterministic two-tone intermodulation report.
#[derive(Debug, Clone, PartialEq)]
pub struct IntermodulationReport {
    pub convention: SpectralConvention,
    pub tone_a_hz: f32,
    pub tone_b_hz: f32,
    pub tone_a_amplitude: f32,
    pub tone_b_amplitude: f32,
    pub components: Vec<IntermodulationComponent>,
}

/// Offline loudness and peak information for one listening-comparison render.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevelMatchReport {
    /// Integrated BS.1770/EBU R128 loudness of the reference render.
    pub reference_lufs: f32,
    /// Integrated BS.1770/EBU R128 loudness of the candidate render.
    pub candidate_lufs: f32,
    /// Gain requested to match the candidate loudness to the reference.
    pub requested_gain_db: f32,
    /// Gain after enforcing the caller's sample-peak ceiling.
    pub applied_gain_db: f32,
    /// Maximum absolute sample in the reference render.
    pub reference_peak: f32,
    /// Maximum absolute sample in the unscaled candidate render.
    pub candidate_peak: f32,
    /// Maximum absolute sample allowed after applying the reported gain.
    pub peak_ceiling: f32,
    /// Whether the peak ceiling prevented exact loudness matching.
    pub peak_limited: bool,
}

/// Compute an offline, level-matched gain for listening comparisons.
///
/// Loudness uses integrated ITU-R BS.1770/EBU R128 measurement and therefore
/// requires a sufficiently long programme (at least one complete gated
/// loudness window).  The returned gain is additionally limited so the
/// candidate's sample peak does not exceed `peak_ceiling`.  This helper is
/// intentionally offline: it allocates the EBU meter and does not belong in a
/// realtime processing callback.  It produces a matching instruction, not a
/// preference or discrimination result.
pub fn level_match_candidate(
    reference_samples: &[f32],
    candidate_samples: &[f32],
    sample_rate: u32,
    channels: usize,
    peak_ceiling: f32,
) -> Result<LevelMatchReport, SpectralError> {
    if reference_samples.is_empty() || candidate_samples.is_empty() {
        return Err(SpectralError::EmptyRecord);
    }
    if reference_samples.len() != candidate_samples.len() {
        return Err(SpectralError::ReportLengthMismatch {
            report_length: reference_samples.len(),
            record_length: candidate_samples.len(),
        });
    }
    if channels == 0 {
        return Err(SpectralError::InvalidChannelCount);
    }
    if !reference_samples.len().is_multiple_of(channels) {
        return Err(SpectralError::InterleavedRecordLengthMismatch {
            samples: reference_samples.len(),
            channels,
        });
    }
    if !(peak_ceiling.is_finite() && peak_ceiling > 0.0 && peak_ceiling <= 1.0) {
        return Err(SpectralError::InvalidPeakCeiling(peak_ceiling));
    }
    if let Some(index) = reference_samples
        .iter()
        .chain(candidate_samples)
        .position(|sample| !sample.is_finite())
    {
        return Err(SpectralError::NonFiniteSample(index));
    }

    let (reference_lufs, reference_peak) =
        measure_loudness_and_peak(reference_samples, sample_rate, channels)?;
    let (candidate_lufs, candidate_peak) =
        measure_loudness_and_peak(candidate_samples, sample_rate, channels)?;
    if reference_peak > peak_ceiling {
        return Err(SpectralError::ReferencePeakExceedsCeiling {
            peak: reference_peak,
            ceiling: peak_ceiling,
        });
    }

    let requested_gain_db = reference_lufs - candidate_lufs;
    let peak_gain_db = if candidate_peak > 0.0 {
        20.0 * (peak_ceiling / candidate_peak).log10()
    } else {
        f32::INFINITY
    };
    let applied_gain_db = requested_gain_db.min(peak_gain_db);
    Ok(LevelMatchReport {
        reference_lufs,
        candidate_lufs,
        requested_gain_db,
        applied_gain_db,
        reference_peak,
        candidate_peak,
        peak_ceiling,
        peak_limited: applied_gain_db < requested_gain_db,
    })
}

fn measure_loudness_and_peak(
    samples: &[f32],
    sample_rate: u32,
    channels: usize,
) -> Result<(f32, f32), SpectralError> {
    let channel_count = u32::try_from(channels).map_err(|_| SpectralError::InvalidChannelCount)?;
    let mut meter = EbuR128::new(channel_count, sample_rate, Mode::I)
        .map_err(SpectralError::LoudnessMeasurement)?;
    meter
        .add_frames_f32(samples)
        .map_err(SpectralError::LoudnessMeasurement)?;
    let loudness = meter
        .loudness_global()
        .map_err(SpectralError::LoudnessMeasurement)?;
    if !loudness.is_finite() {
        return Err(SpectralError::LoudnessUnavailable);
    }
    let peak = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0_f32, f32::max);
    Ok((loudness as f32, peak))
}

impl IntermodulationReport {
    pub fn component(
        &self,
        coefficient_a: i32,
        coefficient_b: i32,
    ) -> Option<&IntermodulationComponent> {
        self.components.iter().find(|component| {
            component.coefficient_a == coefficient_a && component.coefficient_b == coefficient_b
        })
    }
}

/// Measure two-tone fundamentals and low-order intermodulation products.
///
/// Products are represented as `coefficient_a * tone_a + coefficient_b *
/// tone_b`, with the sign-canonical duplicate omitted.  The report uses the
/// same rectangular, one-sided DFT convention as [`measure_harmonics`].
pub fn measure_two_tone_imd(
    samples: &[f32],
    sample_rate: f32,
    tone_a_hz: f32,
    tone_b_hz: f32,
    max_order: usize,
) -> Result<IntermodulationReport, SpectralError> {
    if samples.is_empty() {
        return Err(SpectralError::EmptyRecord);
    }
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(SpectralError::InvalidSampleRate(sample_rate));
    }
    if !tone_a_hz.is_finite() || tone_a_hz <= 0.0 {
        return Err(SpectralError::InvalidFundamental(tone_a_hz));
    }
    if !tone_b_hz.is_finite() || tone_b_hz <= 0.0 {
        return Err(SpectralError::InvalidFundamental(tone_b_hz));
    }
    if tone_a_hz == tone_b_hz {
        return Err(SpectralError::EqualTestTones);
    }
    let nyquist_hz = sample_rate * 0.5;
    for tone_hz in [tone_a_hz, tone_b_hz] {
        if tone_hz >= nyquist_hz {
            return Err(SpectralError::TestToneAtOrAboveNyquist {
                tone_hz,
                nyquist_hz,
            });
        }
    }
    if max_order == 0 {
        return Err(SpectralError::InvalidMaximumOrder);
    }
    if max_order > MAX_HARMONIC_ORDER {
        return Err(SpectralError::MaximumOrderTooLarge {
            requested: max_order,
            maximum: MAX_HARMONIC_ORDER,
        });
    }
    if let Some(index) = samples.iter().position(|sample| !sample.is_finite()) {
        return Err(SpectralError::NonFiniteSample(index));
    }

    let record_length = samples.len();
    let convention = SpectralConvention {
        sample_rate,
        record_length,
        bin_spacing_hz: sample_rate / record_length as f32,
        one_sided: true,
        fft_normalization: FftNormalization::OneSidedAmplitude,
        window: WindowKind::Rectangular,
        coherent_gain: 1.0,
    };
    let nyquist_bin = record_length / 2;
    let tone_a_amplitude = dft_amplitude(
        samples,
        ((tone_a_hz / sample_rate) * record_length as f32).round() as usize,
    );
    let tone_b_amplitude = dft_amplitude(
        samples,
        ((tone_b_hz / sample_rate) * record_length as f32).round() as usize,
    );
    let mut components = Vec::with_capacity(max_order * max_order);
    let max_order = max_order as i32;
    for coefficient_a in -max_order..=max_order {
        for coefficient_b in -max_order..=max_order {
            let order =
                coefficient_a.unsigned_abs() as usize + coefficient_b.unsigned_abs() as usize;
            if !(2..=max_order as usize).contains(&order)
                || !(coefficient_a > 0 || (coefficient_a == 0 && coefficient_b > 0))
            {
                continue;
            }
            let frequency_hz =
                (coefficient_a as f32 * tone_a_hz + coefficient_b as f32 * tone_b_hz).abs();
            if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
                continue;
            }
            let folded_frequency_hz = fold_frequency(frequency_hz, sample_rate);
            let bin = (((folded_frequency_hz / sample_rate) * record_length as f32).round()
                as usize)
                .min(nyquist_bin);
            let amplitude = dft_amplitude(samples, bin);
            let level_db = if amplitude > 0.0 {
                20.0 * amplitude.log10()
            } else {
                f32::NEG_INFINITY
            };
            components.push(IntermodulationComponent {
                coefficient_a,
                coefficient_b,
                order,
                frequency_hz,
                folded_frequency_hz,
                bin,
                amplitude,
                level_db,
                aliases: frequency_hz >= nyquist_hz,
            });
        }
    }

    Ok(IntermodulationReport {
        convention,
        tone_a_hz,
        tone_b_hz,
        tone_a_amplitude,
        tone_b_amplitude,
        components,
    })
}

/// Summary statistics for a finite transient or burst record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransientReport {
    pub sample_rate: f32,
    pub record_length: usize,
    pub peak_amplitude: f32,
    pub peak_index: usize,
    pub rms_amplitude: f32,
    pub dc_amplitude: f32,
}

/// Metadata and error statistics for an offline high-rate reference
/// comparison.
///
/// The reference is low-pass filtered with a 127-tap Blackman-windowed sinc
/// FIR at the high rate and then decimated by `oversample_factor`.  The
/// resulting error is an aliasing proxy, not a hardware-model accuracy score:
/// it also includes the declared reconstruction filter's transition band and
/// any difference in state evolution between the two sample rates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AliasReferenceReport {
    pub sample_rate: f32,
    pub high_rate: f32,
    pub oversample_factor: usize,
    pub record_length: usize,
    pub fir_taps: usize,
    pub base_rms: f32,
    pub reference_rms: f32,
    pub error_rms: f32,
    pub error_peak: f32,
    pub error_level_db: f32,
}

const REFERENCE_FIR_TAPS: usize = 127;

/// Downsample an offline high-rate record with the declared reference filter.
///
/// This helper is intentionally separate from the realtime model API.  It
/// allocates its output and uses zero extension at both record boundaries;
/// callers comparing rendered audio should include a guard interval when
/// measuring steady-state error.
pub fn downsample_reference(
    high_rate_samples: &[f32],
    oversample_factor: usize,
) -> Result<Vec<f32>, SpectralError> {
    validate_oversample_factor(oversample_factor)?;
    if high_rate_samples.is_empty() {
        return Err(SpectralError::EmptyRecord);
    }
    if let Some(index) = high_rate_samples
        .iter()
        .position(|sample| !sample.is_finite())
    {
        return Err(SpectralError::NonFiniteSample(index));
    }
    if !high_rate_samples.len().is_multiple_of(oversample_factor) {
        return Err(SpectralError::ReferenceLengthNotMultiple {
            samples: high_rate_samples.len(),
            factor: oversample_factor,
        });
    }

    let coefficients = reference_fir(oversample_factor);
    let half = coefficients.len() / 2;
    let output_length = high_rate_samples.len() / oversample_factor;
    let mut output = vec![0.0_f32; output_length];
    for (output_index, sample) in output.iter_mut().enumerate() {
        let center = output_index * oversample_factor;
        let mut value = 0.0_f64;
        for (tap, &coefficient) in coefficients.iter().enumerate() {
            let high_index = center as isize + tap as isize - half as isize;
            if let Ok(high_index) = usize::try_from(high_index)
                && let Some(&input) = high_rate_samples.get(high_index)
            {
                value += f64::from(input) * coefficient;
            }
        }
        *sample = value as f32;
    }
    Ok(output)
}

/// Compare a base-rate render with a high-rate render after offline
/// reconstruction and decimation.
pub fn compare_alias_reference(
    base_rate_samples: &[f32],
    high_rate_samples: &[f32],
    sample_rate: f32,
    oversample_factor: usize,
) -> Result<AliasReferenceReport, SpectralError> {
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(SpectralError::InvalidSampleRate(sample_rate));
    }
    if base_rate_samples.is_empty() {
        return Err(SpectralError::EmptyRecord);
    }
    if let Some(index) = base_rate_samples
        .iter()
        .position(|sample| !sample.is_finite())
    {
        return Err(SpectralError::NonFiniteSample(index));
    }
    let reference = downsample_reference(high_rate_samples, oversample_factor)?;
    if reference.len() != base_rate_samples.len() {
        return Err(SpectralError::ReportLengthMismatch {
            report_length: base_rate_samples.len(),
            record_length: reference.len(),
        });
    }

    let mut base_energy = 0.0_f64;
    let mut reference_energy = 0.0_f64;
    let mut error_energy = 0.0_f64;
    let mut error_peak = 0.0_f32;
    for (&base, &reference) in base_rate_samples.iter().zip(&reference) {
        base_energy += f64::from(base) * f64::from(base);
        reference_energy += f64::from(reference) * f64::from(reference);
        let error = (base - reference).abs();
        error_peak = error_peak.max(error);
        error_energy += f64::from(error) * f64::from(error);
    }
    let length = base_rate_samples.len() as f64;
    let base_rms = (base_energy / length).sqrt() as f32;
    let reference_rms = (reference_energy / length).sqrt() as f32;
    let error_rms = (error_energy / length).sqrt() as f32;
    let error_level_db = if error_rms > 0.0 {
        20.0 * error_rms.log10()
    } else {
        f32::NEG_INFINITY
    };
    Ok(AliasReferenceReport {
        sample_rate,
        high_rate: sample_rate * oversample_factor as f32,
        oversample_factor,
        record_length: base_rate_samples.len(),
        fir_taps: REFERENCE_FIR_TAPS,
        base_rms,
        reference_rms,
        error_rms,
        error_peak,
        error_level_db,
    })
}

fn validate_oversample_factor(oversample_factor: usize) -> Result<(), SpectralError> {
    if matches!(oversample_factor, 2 | 4) {
        Ok(())
    } else {
        Err(SpectralError::InvalidOversampleFactor(oversample_factor))
    }
}

fn reference_fir(oversample_factor: usize) -> Vec<f64> {
    let cutoff = 0.45 / oversample_factor as f64;
    let half = REFERENCE_FIR_TAPS / 2;
    let mut coefficients = Vec::with_capacity(REFERENCE_FIR_TAPS);
    for tap in 0..REFERENCE_FIR_TAPS {
        let offset = tap as isize - half as isize;
        let sinc = if offset == 0 {
            1.0
        } else {
            let phase = std::f64::consts::PI * offset as f64;
            (2.0 * std::f64::consts::PI * cutoff * offset as f64).sin() / phase
        };
        let window_phase = std::f64::consts::TAU * tap as f64 / (REFERENCE_FIR_TAPS - 1) as f64;
        let blackman = 0.42 - 0.5 * window_phase.cos() + 0.08 * (2.0 * window_phase).cos();
        coefficients.push(2.0 * cutoff * sinc * blackman);
    }
    let normalization = coefficients.iter().sum::<f64>();
    for coefficient in &mut coefficients {
        *coefficient /= normalization;
    }
    coefficients
}

/// Measure peak, RMS, and DC behavior of a transient record.
pub fn measure_transient(
    samples: &[f32],
    sample_rate: f32,
) -> Result<TransientReport, SpectralError> {
    if samples.is_empty() {
        return Err(SpectralError::EmptyRecord);
    }
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(SpectralError::InvalidSampleRate(sample_rate));
    }
    if let Some(index) = samples.iter().position(|sample| !sample.is_finite()) {
        return Err(SpectralError::NonFiniteSample(index));
    }
    let (peak_index, peak_amplitude) = samples
        .iter()
        .enumerate()
        .map(|(index, sample)| (index, sample.abs()))
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .expect("non-empty samples checked above");
    let sum = samples.iter().map(|sample| *sample as f64).sum::<f64>();
    let energy = samples
        .iter()
        .map(|sample| (*sample as f64) * (*sample as f64))
        .sum::<f64>();
    let record_length = samples.len();
    Ok(TransientReport {
        sample_rate,
        record_length,
        peak_amplitude,
        peak_index,
        rms_amplitude: (energy / record_length as f64).sqrt() as f32,
        dc_amplitude: (sum / record_length as f64) as f32,
    })
}

/// Measure DC, Nyquist, and harmonic bins without zero-padding or a hidden
/// window.  The rectangular window and one-sided amplitude normalization are
/// recorded in the returned report so the values are reproducible.
///
/// Harmonics above Nyquist are reported at their folded bin and marked with
/// `aliases = true`; they are not treated as desired in-band content.
pub fn measure_harmonics(
    samples: &[f32],
    sample_rate: f32,
    fundamental_hz: f32,
    max_order: usize,
) -> Result<HarmonicReport, SpectralError> {
    if samples.is_empty() {
        return Err(SpectralError::EmptyRecord);
    }
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(SpectralError::InvalidSampleRate(sample_rate));
    }
    if !fundamental_hz.is_finite() || fundamental_hz <= 0.0 {
        return Err(SpectralError::InvalidFundamental(fundamental_hz));
    }
    let nyquist_hz = sample_rate * 0.5;
    if fundamental_hz >= nyquist_hz {
        return Err(SpectralError::FundamentalAtOrAboveNyquist {
            fundamental_hz,
            nyquist_hz,
        });
    }
    if max_order == 0 {
        return Err(SpectralError::InvalidMaximumOrder);
    }
    if max_order > MAX_HARMONIC_ORDER {
        return Err(SpectralError::MaximumOrderTooLarge {
            requested: max_order,
            maximum: MAX_HARMONIC_ORDER,
        });
    }
    if let Some(index) = samples.iter().position(|sample| !sample.is_finite()) {
        return Err(SpectralError::NonFiniteSample(index));
    }

    let record_length = samples.len();
    let convention = SpectralConvention {
        sample_rate,
        record_length,
        bin_spacing_hz: sample_rate / record_length as f32,
        one_sided: true,
        fft_normalization: FftNormalization::OneSidedAmplitude,
        window: WindowKind::Rectangular,
        coherent_gain: 1.0,
    };
    let dc_amplitude = dft_amplitude(samples, 0);
    let nyquist_bin = record_length / 2;
    let nyquist_amplitude = if record_length.is_multiple_of(2) {
        dft_amplitude(samples, nyquist_bin)
    } else {
        0.0
    };

    let mut components = Vec::with_capacity(max_order);
    for order in 1..=max_order {
        let frequency_hz = fundamental_hz * order as f32;
        let folded_frequency_hz = fold_frequency(frequency_hz, sample_rate);
        let bin = ((folded_frequency_hz / sample_rate) * record_length as f32).round() as usize;
        let bin = bin.min(nyquist_bin);
        let amplitude = dft_amplitude(samples, bin);
        let level_db = if amplitude > 0.0 {
            20.0 * amplitude.log10()
        } else {
            f32::NEG_INFINITY
        };
        components.push(HarmonicComponent {
            order,
            frequency_hz,
            folded_frequency_hz,
            bin,
            amplitude,
            level_db,
            aliases: frequency_hz >= nyquist_hz,
        });
    }

    Ok(HarmonicReport {
        convention,
        fundamental_hz,
        dc_amplitude,
        nyquist_amplitude,
        components,
    })
}

#[inline]
fn fold_frequency(frequency_hz: f32, sample_rate: f32) -> f32 {
    let wrapped = frequency_hz.rem_euclid(sample_rate);
    if wrapped <= sample_rate * 0.5 {
        wrapped
    } else {
        sample_rate - wrapped
    }
}

fn dft_amplitude(samples: &[f32], bin: usize) -> f32 {
    let n = samples.len();
    let mut real = 0.0_f64;
    let mut imaginary = 0.0_f64;
    for (index, &sample) in samples.iter().enumerate() {
        let phase = TAU as f64 * bin as f64 * index as f64 / n as f64;
        real += sample as f64 * phase.cos();
        imaginary -= sample as f64 * phase.sin();
    }
    let scale = if bin == 0 || (n.is_multiple_of(2) && bin == n / 2) {
        1.0 / n as f64
    } else {
        2.0 / n as f64
    };
    (scale * real.hypot(imaginary)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coherent_record_reports_one_sided_amplitudes() {
        let sample_rate = 48_000.0;
        let fundamental = 1_000.0;
        let samples: Vec<f32> = (0..4_800)
            .map(|index| {
                let phase = TAU * fundamental * index as f32 / sample_rate;
                0.5 + phase.cos() + 0.25 * (2.0 * phase).cos() + 0.125 * (3.0 * phase).cos()
            })
            .collect();
        let report = measure_harmonics(&samples, sample_rate, fundamental, 3).unwrap();
        assert!((report.dc_amplitude - 0.5).abs() < 1e-6);
        assert!((report.component(1).unwrap().amplitude - 1.0).abs() < 1e-6);
        assert!((report.component(2).unwrap().amplitude - 0.25).abs() < 1e-6);
        assert!((report.component(3).unwrap().amplitude - 0.125).abs() < 1e-6);
        assert_eq!(report.convention.bin_spacing_hz, 10.0);
        assert_eq!(report.convention.coherent_gain, 1.0);
    }

    #[test]
    fn out_of_band_components_are_marked_as_aliases() {
        let samples = vec![0.0_f32; 480];
        let report = measure_harmonics(&samples, 48_000.0, 10_000.0, 4).unwrap();
        assert!(!report.component(2).unwrap().aliases);
        assert!(report.component(3).unwrap().aliases);
        assert!((report.component(3).unwrap().folded_frequency_hz - 18_000.0).abs() < 1e-3);
        assert!(report.component(5).is_none());
    }

    #[test]
    fn invalid_records_fail_before_work() {
        assert_eq!(
            measure_harmonics(&[], 48_000.0, 1_000.0, 3),
            Err(SpectralError::EmptyRecord)
        );
        assert_eq!(
            measure_harmonics(&[0.0, f32::NAN], 48_000.0, 1_000.0, 3),
            Err(SpectralError::NonFiniteSample(1))
        );
        assert!(matches!(
            measure_harmonics(&[0.0], 48_000.0, 1_000.0, usize::MAX),
            Err(SpectralError::MaximumOrderTooLarge { .. })
        ));
    }

    #[test]
    fn two_tone_report_identifies_third_order_product() {
        let sample_rate = 48_000.0;
        let tone_a = 1_000.0;
        let tone_b = 1_500.0;
        let samples: Vec<f32> = (0..4_800)
            .map(|index| {
                let phase = TAU * tone_a * index as f32 / sample_rate;
                phase.cos() + 0.5 * (1.5 * phase).cos() + 0.2 * (0.5 * phase).cos()
            })
            .collect();
        let report = measure_two_tone_imd(&samples, sample_rate, tone_a, tone_b, 3).unwrap();
        assert!((report.tone_a_amplitude - 1.0).abs() < 1e-4);
        assert!((report.tone_b_amplitude - 0.5).abs() < 1e-4);
        let product = report.component(2, -1).unwrap();
        assert_eq!(product.order, 3);
        assert!((product.frequency_hz - 500.0).abs() < 1e-6);
        assert!((product.amplitude - 0.2).abs() < 1e-4);
    }

    #[test]
    fn transient_report_is_deterministic_and_validates_input() {
        let report = measure_transient(&[0.0, -0.5, 1.0, 0.5], 48_000.0).unwrap();
        assert_eq!(report.peak_index, 2);
        assert_eq!(report.peak_amplitude, 1.0);
        assert!((report.rms_amplitude - 0.61237246).abs() < 1e-6);
        assert_eq!(report.dc_amplitude, 0.25);
        assert_eq!(
            measure_transient(&[0.0, f32::NAN], 48_000.0),
            Err(SpectralError::NonFiniteSample(1))
        );
    }

    #[test]
    fn reference_downsampler_has_declared_length_and_finite_steady_state() {
        for factor in [2, 4] {
            let high_rate = vec![0.25_f32; 4_096 * factor];
            let downsampled = downsample_reference(&high_rate, factor).unwrap();
            assert_eq!(downsampled.len(), 4_096);
            assert!(downsampled.iter().all(|sample| sample.is_finite()));
            assert!(
                downsampled[128..3_968]
                    .iter()
                    .all(|sample| (*sample - 0.25).abs() < 1e-5)
            );
        }
        assert_eq!(
            downsample_reference(&[0.0; 8], 3),
            Err(SpectralError::InvalidOversampleFactor(3))
        );
        assert_eq!(
            downsample_reference(&[0.0; 7], 2),
            Err(SpectralError::ReferenceLengthNotMultiple {
                samples: 7,
                factor: 2,
            })
        );
    }

    #[test]
    fn high_rate_reference_comparison_is_reproducible() {
        use crate::{AnalogProcessor, AntiAliasing, HarmonicModel, ProcessSpec};

        let render = |sample_rate: f32, frames: usize| {
            let mut model = HarmonicModel::new();
            model.set_anti_aliasing(AntiAliasing::Off);
            model.set_drive_db(24.0).unwrap();
            model.set_h2_db(-120.0).unwrap();
            model.set_h3_db(-120.0).unwrap();
            model
                .prepare(ProcessSpec::new(sample_rate, 1, frames))
                .unwrap();
            let mut samples: Vec<f32> = (0..frames)
                .map(|index| 0.8 * (TAU * 10_000.0 * index as f32 / sample_rate).sin())
                .collect();
            model.process_interleaved(&mut samples, frames).unwrap();
            samples
        };

        let base = render(48_000.0, 4_800);
        let high_rate = render(96_000.0, 9_600);
        let report = compare_alias_reference(&base, &high_rate, 48_000.0, 2).unwrap();
        let repeat = compare_alias_reference(&base, &high_rate, 48_000.0, 2).unwrap();
        assert_eq!(report, repeat);
        assert_eq!(report.high_rate, 96_000.0);
        assert_eq!(report.fir_taps, REFERENCE_FIR_TAPS);
        assert!(report.error_rms.is_finite());
        assert!(report.error_rms > 0.0);
    }

    #[test]
    fn thd_and_thd_plus_n_have_distinct_declared_definitions() {
        let sample_rate = 48_000.0;
        let fundamental = 1_000.0;
        let samples: Vec<f32> = (0..4_800)
            .map(|index| {
                let phase = TAU * fundamental * index as f32 / sample_rate;
                0.25 + phase.cos() + 0.1 * (2.0 * phase).cos()
            })
            .collect();
        let report = measure_harmonics(&samples, sample_rate, fundamental, 3).unwrap();
        let distortion = report.distortion(&samples).unwrap();
        assert!((distortion.thd - 0.1).abs() < 1e-4);
        assert!(distortion.thd_plus_n > distortion.thd);
        assert_eq!(distortion.alias_rms, 0.0);
    }

    #[test]
    fn level_match_uses_integrated_loudness_and_peak_limit() {
        let sample_rate = 48_000_u32;
        let frames = sample_rate as usize * 5;
        let reference: Vec<f32> = (0..frames)
            .flat_map(|index| {
                let phase = TAU * 1_000.0 * index as f32 / sample_rate as f32;
                let sample = 0.2 * phase.sin();
                [sample, sample]
            })
            .collect();
        let candidate: Vec<f32> = reference.iter().map(|sample| sample * 2.0).collect();

        let report = level_match_candidate(&reference, &candidate, sample_rate, 2, 0.25)
            .expect("five-second stereo programme has an integrated loudness");
        assert!((report.requested_gain_db + 6.0206).abs() < 0.02);
        assert!((report.applied_gain_db - report.requested_gain_db).abs() < 1e-5);
        assert!(!report.peak_limited);
        let applied_gain = 10.0_f32.powf(report.applied_gain_db / 20.0);
        assert!((report.candidate_peak * applied_gain - 0.2).abs() < 1e-4);

        let mut transient_candidate = reference
            .iter()
            .map(|sample| sample * 0.25)
            .collect::<Vec<_>>();
        transient_candidate[frames] = 0.9;
        let limited = level_match_candidate(&reference, &transient_candidate, sample_rate, 2, 0.25)
            .expect("finite candidate should be measurable");
        assert!(limited.peak_limited);
        assert!(limited.applied_gain_db < limited.requested_gain_db);
        let limited_gain = 10.0_f32.powf(limited.applied_gain_db / 20.0);
        assert!(
            limited.candidate_peak * limited_gain <= 0.25 + 1e-5,
            "peak ceiling was not enforced: {limited:?}"
        );
    }

    #[test]
    fn level_match_rejects_invalid_records_before_allocating_a_meter() {
        assert_eq!(
            level_match_candidate(&[], &[0.0], 48_000, 1, 0.9),
            Err(SpectralError::EmptyRecord)
        );
        assert_eq!(
            level_match_candidate(&[0.0, 0.0], &[0.0, 0.0], 48_000, 0, 0.9),
            Err(SpectralError::InvalidChannelCount)
        );
        assert_eq!(
            level_match_candidate(&[0.0, 0.0], &[0.0, 0.0], 48_000, 2, 1.1),
            Err(SpectralError::InvalidPeakCeiling(1.1))
        );
    }
}

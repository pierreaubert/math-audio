//! Generates IEC 60268-16 direct-method STI test signals.
//!
//! Provides the two standardized modulated-noise signals used to measure
//! Speech Transmission Index by playing a known signal through a
//! transmission channel and analyzing the recording:
//!
//! - STIPA: one composite signal with two modulation frequencies per
//!   octave band, for rapid (~15-25 s) field assessment.
//! - Full STI: 98 concatenated single-modulation segments (14 modulation
//!   frequencies x 7 octave bands) for exhaustive laboratory measurement.
//!
//! Both carriers start from seeded pink noise filtered into the seven
//! STI octave bands, apply the standard speech-spectrum band levels, and
//! normalize active-signal RMS to 0.07. Use [`gen_stipa_signal`] and
//! [`gen_full_sti_signal`] for byte-stable default seeds, or the `_seeded`
//! variants to decorrelate repeated captures.
//!
//! Reference implementation:
//! [rajmic/Speech-Transmission-Index-STI](https://github.com/rajmic/Speech-Transmission-Index-STI)
//! (`generateStipaSignal.m`, `generateFullSTISignal.m`). Intentional
//! deviations from that reference:
//!
//! - Octave filtering uses a zero-phase Butterworth bandpass (6th order
//!   per edge plus forward/reverse `filtfilt`, the same recipe as
//!   `math_rir::bands`) instead of MATLAB's order-20 `octaveFilter`.
//! - The STIPA time axis is `n / sample_rate` for every sample; the
//!   reference uses endpoint-inclusive `linspace`, which detunes
//!   modulation frequencies by one part in `N`.
//! - The STIPA envelope radicand is clamped at zero before `sqrt`. The
//!   reference formula can dip to -0.05 for adversarial (duration, rate)
//!   pairs, where MATLAB would emit complex samples.
//! - Full STI reuses one filtered carrier set across all 98 segments,
//!   exactly like the reference; segments sharing a band share a carrier.
//!
//! The IEC band centers, modulation frequencies, and band levels mirror
//! `math_rir::sti` (`STI_OCTAVE_CENTERS_HZ`,
//! `STI_MODULATION_FREQUENCIES_HZ`); that crate cannot be reused here
//! because `math-dsp` and `math-rir` are sibling crates under
//! `math-iir-fir` with no dependency edge between them.

// Rust guideline compliant 2026-02-21

use super::gen_::gen_pink_noise_seeded;
use super::misc::clip;
use super::misc::frames_for;
use math_audio_iir_fir::filtfilt;
use math_audio_iir_fir::peq_butterworth_highpass;
use math_audio_iir_fir::peq_butterworth_lowpass;

/// Nominal IEC STI octave-band centers in ascending order (Hz).
///
/// Mirrors `math_rir::sti::STI_OCTAVE_CENTERS_HZ`; kept in sync by hand
/// because neither sibling crate may depend on the other.
pub const STI_OCTAVE_CENTERS_HZ: [f64; 7] = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0];

/// IEC 60268-16 revision-5 speech-spectrum band levels (dB), ascending bands.
///
/// Converted to linear gains with `10^(level / 20)` before weighting each
/// modulated carrier. Values come from the reference implementation.
pub const STI_BAND_LEVELS_DB: [f64; 7] = [-2.5, 0.5, 0.0, -6.0, -12.0, -18.0, -24.0];

/// Nominal IEC full-STI modulation frequencies (Hz).
///
/// Mirrors `math_rir::sti::STI_MODULATION_FREQUENCIES_HZ`.
pub const STI_FULL_MODULATION_FREQUENCIES_HZ: [f64; 14] = [
    0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5,
];

/// STIPA modulation-frequency pairs per octave band, ascending bands (Hz).
///
/// Each entry holds the low then high modulation frequency applied to one
/// band carrier. Values come from the reference implementation.
pub const STIPA_MODULATION_FREQUENCIES_HZ: [[f64; 2]; 7] = [
    [1.6, 8.0],
    [1.0, 5.0],
    [0.63, 3.15],
    [2.0, 10.0],
    [1.25, 6.25],
    [0.8, 4.0],
    [2.5, 12.5],
];

/// Target RMS of the active (non-silent) STI signal.
///
/// Empirically derived from the character of the STIPA test signal in the
/// reference implementation; the same target normalizes full STI. Peaks
/// stay far below full scale, so PCM conversion never clips.
pub const STI_TARGET_RMS: f32 = 0.07;

/// Minimum sample rate supporting all seven STI octave bands (Hz).
///
/// The 8 kHz octave reaches `8000 * sqrt(2)` ~= 11314 Hz, so rates below
/// 22050 Hz cannot carry it. Matches the reference implementation's
/// minimum; callers should reject lower rates before generating.
pub const STI_MIN_SAMPLE_RATE_HZ: u32 = 22_050;

/// Full-STI segment count: 14 modulation frequencies x 7 octave bands.
pub const STI_FULL_SEGMENT_COUNT: usize = 98;

/// Butterworth order per bandpass edge before forward/reverse filtering.
///
/// `filtfilt` doubles the effective slope, matching the 6th-order recipe
/// `math_rir::sti` uses for analysis. Higher orders narrow the 8 kHz
/// band's clamped top edge without audible benefit.
pub const STI_FILTER_ORDER: usize = 6;

/// Deterministic pink-noise seed behind [`gen_stipa_signal`].
///
/// Encodes IEC 60268-16 plus a STIPA tag; fixed so regenerated fixtures
/// are byte-stable, and distinct from [`FULL_STI_NOISE_SEED`] so the two
/// signals never share a carrier.
pub const STIPA_NOISE_SEED: u64 = 602681601;

/// Deterministic pink-noise seed behind [`gen_full_sti_signal`].
///
/// Encodes IEC 60268-16 plus the 98-segment count; fixed so regenerated
/// fixtures are byte-stable, and distinct from [`STIPA_NOISE_SEED`] so
/// the two signals never share a carrier.
pub const FULL_STI_NOISE_SEED: u64 = 602681698;

/// STIPA amplitude-modulation depth from the reference implementation.
///
/// The envelope is `sqrt(0.5 * (1 + depth * (sin1 - sin2)))`; 0.55 keeps
/// most of the radicand non-negative while preserving the standard
/// modulation strength.
const STIPA_MODULATION_DEPTH: f64 = 0.55;

/// Base-2 octave band edges around a center frequency (Hz).
///
/// `f_low = fc / sqrt(2)`, `f_high = fc * sqrt(2)`; the same convention
/// `math_rir::bands` documents for ISO 3382 work.
fn octave_band_edges(center_hz: f64) -> (f64, f64) {
    const SQRT_2: f64 = std::f64::consts::SQRT_2;
    (center_hz / SQRT_2, center_hz * SQRT_2)
}

/// Filter pink noise into the seven STI octave-band carriers.
///
/// Each band runs through a zero-phase Butterworth bandpass built from a
/// highpass plus lowpass leg (`filtfilt` removes group delay so carrier
/// energy stays time-aligned with the modulation envelopes). Edges clamp
/// to `[1 Hz, 0.99 * Nyquist]`; at the minimum 22050 Hz rate the 8 kHz
/// top edge narrows slightly instead of aliasing.
fn octave_band_carriers(pink_noise: &[f32], sample_rate: u32) -> Vec<Vec<f32>> {
    let sample_rate_f64 = f64::from(sample_rate);
    let nyquist = sample_rate_f64 * 0.5;
    let input: Vec<f64> = pink_noise.iter().map(|&v| f64::from(v)).collect();
    STI_OCTAVE_CENTERS_HZ
        .iter()
        .map(|&center| {
            let (low, high) = octave_band_edges(center);
            let low = low.max(1.0);
            let high = high.min(nyquist * 0.99);
            if high <= low {
                return pink_noise.to_vec();
            }
            let mut sections = filtfilt::peq_to_coefficients(&peq_butterworth_highpass(
                STI_FILTER_ORDER,
                low,
                sample_rate_f64,
            ));
            sections.extend(filtfilt::peq_to_coefficients(&peq_butterworth_lowpass(
                STI_FILTER_ORDER,
                high,
                sample_rate_f64,
            )));
            filtfilt::filtfilt(&input, &sections)
                .iter()
                .map(|&v| v as f32)
                .collect()
        })
        .collect()
}

/// Linear speech-spectrum gain for one octave band.
fn band_level_gain(band: usize) -> f64 {
    10.0_f64.powf(STI_BAND_LEVELS_DB[band] / 20.0)
}

/// Root-mean-square level of active (non-silent) samples.
fn active_rms(signal: &[f32], active_frames: usize) -> f64 {
    if active_frames == 0 {
        return 0.0;
    }
    let sum_squares: f64 = signal.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    (sum_squares / active_frames as f64).sqrt()
}

/// Generate a STIPA direct-method test signal.
///
/// Sums the seven octave-band pink-noise carriers after per-band dual
/// modulation and speech-spectrum weighting, then normalizes active RMS
/// to [`STI_TARGET_RMS`]. `sample_rate` is in Hz and `duration` in
/// seconds; rates below [`STI_MIN_SAMPLE_RATE_HZ`] narrow the 8 kHz band
/// and should be rejected by the caller. Output is mono; use
/// [`super::misc::replicate_mono`] for multichannel layouts.
///
/// Returns an empty vector when `sample_rate` is zero or `duration` is
/// not a positive finite value.
///
/// # Examples
/// ```
/// let signal = math_audio_dsp::signals::gen_stipa_signal(48_000, 0.1);
/// assert_eq!(signal.len(), 4_800);
/// ```
pub fn gen_stipa_signal(sample_rate: u32, duration: f32) -> Vec<f32> {
    gen_stipa_signal_seeded(sample_rate, duration, STIPA_NOISE_SEED)
}

/// Generate a STIPA test signal with a caller-controlled seed.
///
/// Behaves like [`gen_stipa_signal`] with deterministic pink noise drawn
/// from `seed`, so repeated captures can decorrelate their carriers.
///
/// # Examples
/// ```
/// use math_audio_dsp::signals::gen_stipa_signal_seeded;
/// assert_ne!(
///     gen_stipa_signal_seeded(48_000, 0.05, 1),
///     gen_stipa_signal_seeded(48_000, 0.05, 2)
/// );
/// ```
pub fn gen_stipa_signal_seeded(sample_rate: u32, duration: f32, seed: u64) -> Vec<f32> {
    if sample_rate == 0 || !duration.is_finite() || duration <= 0.0 {
        return Vec::new();
    }
    let frames = frames_for(duration, sample_rate);
    if frames == 0 {
        return Vec::new();
    }
    let pink = gen_pink_noise_seeded(1.0, sample_rate, duration, seed);
    debug_assert_eq!(pink.len(), frames);
    let carriers = octave_band_carriers(&pink, sample_rate);
    let sample_rate_f64 = f64::from(sample_rate);
    let mut signal = vec![0.0_f32; frames];
    for (band, carrier) in carriers.iter().enumerate() {
        let gain = band_level_gain(band);
        let [low_fm, high_fm] = STIPA_MODULATION_FREQUENCIES_HZ[band];
        let omega_low = std::f64::consts::TAU * low_fm / sample_rate_f64;
        let omega_high = std::f64::consts::TAU * high_fm / sample_rate_f64;
        for (n, sample) in signal.iter_mut().enumerate() {
            // Omega is radians per sample: the phase variable is the
            // sample index. Seconds here would divide by the sample rate
            // a second time and freeze the envelope (see §10).
            let index = n as f64;
            let beating = (omega_low * index).sin() - (omega_high * index).sin();
            let envelope = (0.5 * (1.0 + STIPA_MODULATION_DEPTH * beating).max(0.0)).sqrt();
            *sample += (f64::from(carrier[n]) * envelope * gain) as f32;
        }
    }
    let rms = active_rms(&signal, frames);
    if rms.is_finite() && rms > 0.0 {
        let scale = f64::from(STI_TARGET_RMS) / rms;
        for sample in signal.iter_mut() {
            *sample = clip((f64::from(*sample) * scale) as f32);
        }
    }
    signal
}

/// Generate a full-STI direct-method test signal.
///
/// Concatenates all 98 single-modulation segments in modulation-major
/// order (outer loop over [`STI_FULL_MODULATION_FREQUENCIES_HZ`], inner
/// loop over [`STI_OCTAVE_CENTERS_HZ`]) with `silence_gap` seconds of
/// digital silence between segments. `sample_rate` is in Hz and
/// `segment_duration` in seconds; rates below
/// [`STI_MIN_SAMPLE_RATE_HZ`] narrow the 8 kHz band and should be
/// rejected by the caller. Active-segment RMS normalizes to
/// [`STI_TARGET_RMS`] while gaps stay exactly zero. Output is mono; use
/// [`super::misc::replicate_mono`] for multichannel layouts.
///
/// Returns an empty vector when `sample_rate` is zero, `segment_duration`
/// is not a positive finite value, or `silence_gap` is negative or not
/// finite.
///
/// # Examples
/// ```
/// use math_audio_dsp::signals::STI_FULL_SEGMENT_COUNT;
/// use math_audio_dsp::signals::gen_full_sti_signal;
/// let signal = gen_full_sti_signal(48_000, 0.05, 0.0);
/// assert_eq!(signal.len(), STI_FULL_SEGMENT_COUNT * 2_400);
/// ```
pub fn gen_full_sti_signal(sample_rate: u32, segment_duration: f32, silence_gap: f32) -> Vec<f32> {
    gen_full_sti_signal_seeded(
        sample_rate,
        segment_duration,
        silence_gap,
        FULL_STI_NOISE_SEED,
    )
}

/// Generate a full-STI test signal with a caller-controlled seed.
///
/// Behaves like [`gen_full_sti_signal`] with deterministic pink noise
/// drawn from `seed`, so repeated captures can decorrelate carriers.
///
/// # Examples
/// ```
/// use math_audio_dsp::signals::gen_full_sti_signal_seeded;
/// assert_ne!(
///     gen_full_sti_signal_seeded(48_000, 0.05, 0.0, 1),
///     gen_full_sti_signal_seeded(48_000, 0.05, 0.0, 2)
/// );
/// ```
pub fn gen_full_sti_signal_seeded(
    sample_rate: u32,
    segment_duration: f32,
    silence_gap: f32,
    seed: u64,
) -> Vec<f32> {
    if sample_rate == 0
        || !segment_duration.is_finite()
        || segment_duration <= 0.0
        || !silence_gap.is_finite()
        || silence_gap < 0.0
    {
        return Vec::new();
    }
    let segment_frames = frames_for(segment_duration, sample_rate);
    let gap_frames = frames_for(silence_gap, sample_rate);
    if segment_frames == 0 {
        return Vec::new();
    }
    let pink = gen_pink_noise_seeded(1.0, sample_rate, segment_duration, seed);
    debug_assert_eq!(pink.len(), segment_frames);
    let carriers = octave_band_carriers(&pink, sample_rate);
    let sample_rate_f64 = f64::from(sample_rate);
    let total_frames =
        STI_FULL_SEGMENT_COUNT * segment_frames + (STI_FULL_SEGMENT_COUNT - 1) * gap_frames;
    let mut signal = Vec::with_capacity(total_frames);
    let mut sum_squares = 0.0_f64;
    for &modulation_hz in &STI_FULL_MODULATION_FREQUENCIES_HZ {
        let omega = std::f64::consts::TAU * modulation_hz / sample_rate_f64;
        for (band, carrier) in carriers.iter().enumerate() {
            let gain = band_level_gain(band);
            for (n, &carrier_sample) in carrier.iter().enumerate() {
                // Radians per sample: phase from the sample index (see §10).
                let index = n as f64;
                let envelope = (0.5 * (1.0 + (omega * index).cos())).sqrt();
                let sample = f64::from(carrier_sample) * envelope * gain;
                sum_squares += sample * sample;
                signal.push(sample as f32);
            }
            signal.extend(std::iter::repeat_n(0.0, gap_frames));
        }
    }
    signal.truncate(total_frames);
    let active_frames = STI_FULL_SEGMENT_COUNT * segment_frames;
    let rms = (sum_squares / active_frames as f64).sqrt();
    if rms.is_finite() && rms > 0.0 {
        let scale = f64::from(STI_TARGET_RMS) / rms;
        for sample in signal.iter_mut() {
            *sample = clip((f64::from(*sample) * scale) as f32);
        }
    }
    signal
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHORT_RATE: u32 = 22_050;
    const SHORT_DURATION: f32 = 2.0;

    fn rms_of(signal: &[f32]) -> f64 {
        active_rms(signal, signal.len())
    }

    /// Zero-phase octave bandpass of `signal` around `center_hz`.
    fn bandpass_octave(signal: &[f32], center_hz: f64, sample_rate: u32) -> Vec<f64> {
        let (low, high) = octave_band_edges(center_hz);
        let mut sections = filtfilt::peq_to_coefficients(&peq_butterworth_highpass(
            STI_FILTER_ORDER,
            low.max(1.0),
            f64::from(sample_rate),
        ));
        sections.extend(filtfilt::peq_to_coefficients(&peq_butterworth_lowpass(
            STI_FILTER_ORDER,
            high.min(f64::from(sample_rate) * 0.5 * 0.99),
            f64::from(sample_rate),
        )));
        let input: Vec<f64> = signal.iter().map(|&v| f64::from(v)).collect();
        filtfilt::filtfilt(&input, &sections)
    }

    /// Least-squares modulation depth of a squared band signal at `fm_hz`.
    ///
    /// Fits `c + a·cos(2π·fm·t) + b·sin(2π·fm·t)` over the samples and
    /// returns `hypot(a, b) / c`, the same least-squares demodulation the
    /// direct-method analyzer applies per modulation frequency.
    fn fitted_depth(squared: &[f64], fm_hz: f64, sample_rate: u32) -> f64 {
        let sample_rate_f64 = f64::from(sample_rate);
        // Radians per sample; the phase variable is the sample index.
        let omega = std::f64::consts::TAU * fm_hz / sample_rate_f64;
        let mut normal = [[0.0_f64; 4]; 3];
        for (n, &value) in squared.iter().enumerate() {
            let phase = omega * (n as f64);
            let basis = [1.0, phase.cos(), phase.sin()];
            for row in 0..3 {
                for col in 0..3 {
                    normal[row][col] += basis[row] * basis[col];
                }
                normal[row][3] += basis[row] * value;
            }
        }
        for pivot in 0..3 {
            let mut best = pivot;
            for (row, equation) in normal.iter().enumerate().skip(pivot + 1) {
                if equation[pivot].abs() > normal[best][pivot].abs() {
                    best = row;
                }
            }
            normal.swap(pivot, best);
            let divisor = normal[pivot][pivot];
            assert!(
                divisor.abs() > 0.0,
                "singular modulation basis at {fm_hz} Hz"
            );
            let pivot_row = normal[pivot];
            let (_, below) = normal.split_at_mut(pivot + 1);
            for equation in below.iter_mut() {
                let factor = equation[pivot] / divisor;
                for (entry, &pivot_entry) in equation.iter_mut().zip(pivot_row.iter()).skip(pivot) {
                    *entry -= factor * pivot_entry;
                }
            }
        }
        let mut beta = [0.0_f64; 3];
        for row in (0..3).rev() {
            let mut value = normal[row][3];
            for col in row + 1..3 {
                value -= normal[row][col] * beta[col];
            }
            beta[row] = value / normal[row][row];
        }
        beta[1].hypot(beta[2]) / beta[0]
    }

    #[test]
    fn stipa_hits_target_rms_without_clipping() {
        let signal = gen_stipa_signal(48_000, SHORT_DURATION);
        assert_eq!(signal.len(), frames_for(SHORT_DURATION, 48_000));
        assert!((rms_of(&signal) - f64::from(STI_TARGET_RMS)).abs() < 1e-4);
        let peak = signal.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        assert!(peak < 1.0, "STIPA peak {peak} approaches full scale");
        assert!(peak > STI_TARGET_RMS);
    }

    #[test]
    fn stipa_is_deterministic_per_seed() {
        assert_eq!(
            gen_stipa_signal_seeded(48_000, 0.2, 7),
            gen_stipa_signal_seeded(48_000, 0.2, 7)
        );
        assert_ne!(
            gen_stipa_signal_seeded(48_000, 0.2, 7),
            gen_stipa_signal_seeded(48_000, 0.2, 8)
        );
    }

    #[test]
    fn stipa_rejects_degenerate_input() {
        assert!(gen_stipa_signal(0, 1.0).is_empty());
        for duration in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                gen_stipa_signal(48_000, duration).is_empty(),
                "duration {duration} should yield no signal"
            );
        }
    }

    #[test]
    fn stipa_bands_follow_speech_weighting() {
        // Pink carriers hold roughly equal energy per octave, so each
        // output band should sit near its speech-level gain squared.
        // Wiring check only: it reuses the octave filter to prove band
        // levels and assembly, not the filter itself.
        let signal = gen_stipa_signal(SHORT_RATE, SHORT_DURATION);
        let trim = SHORT_RATE as usize / 4;
        let body = &signal[trim..signal.len() - trim];
        let total: f64 = body.iter().map(|&v| f64::from(v).powi(2)).sum();
        for (band, &center) in STI_OCTAVE_CENTERS_HZ.iter().enumerate() {
            let band_energy: f64 = bandpass_octave(body, center, SHORT_RATE)
                .iter()
                .map(|v| v.powi(2))
                .sum();
            let expected_share = band_level_gain(band).powi(2)
                / (0..7).map(band_level_gain).map(|g| g.powi(2)).sum::<f64>();
            let actual_share = band_energy / total;
            let error_db = 10.0 * (actual_share / expected_share).log10();
            assert!(
                error_db.abs() < 2.0,
                "band {center} Hz share {actual_share:.4} vs {expected_share:.4}"
            );
        }
    }

    #[test]
    fn full_sti_layout_matches_segment_grid() {
        let segment = 0.1_f32;
        let gap = 0.02_f32;
        let signal = gen_full_sti_signal(SHORT_RATE, segment, gap);
        let segment_frames = frames_for(segment, SHORT_RATE);
        let gap_frames = frames_for(gap, SHORT_RATE);
        assert_eq!(
            signal.len(),
            STI_FULL_SEGMENT_COUNT * segment_frames + (STI_FULL_SEGMENT_COUNT - 1) * gap_frames
        );
        // Every inter-segment gap is exactly digital silence.
        let stride = segment_frames + gap_frames;
        for index in 0..STI_FULL_SEGMENT_COUNT - 1 {
            let start = index * stride + segment_frames;
            assert!(
                signal[start..start + gap_frames].iter().all(|&v| v == 0.0),
                "gap {index} is not silent"
            );
        }
        // Active segments carry the target RMS; none are silent.
        let active: Vec<f32> = (0..STI_FULL_SEGMENT_COUNT)
            .flat_map(|index| {
                let start = index * stride;
                signal[start..start + segment_frames].iter().copied()
            })
            .collect();
        assert!((rms_of(&active) - f64::from(STI_TARGET_RMS)).abs() < 1e-4);
    }

    #[test]
    fn full_sti_segments_land_in_ascending_bands() {
        // Segment order is modulation-major: segment i belongs to octave
        // band (i mod 7). Wiring check via the shared octave filter.
        let signal = gen_full_sti_signal(SHORT_RATE, 0.5, 0.0);
        let segment_frames = frames_for(0.5, SHORT_RATE);
        for segment in [0, 48, 97] {
            let start = segment * segment_frames;
            let body = &signal[start..start + segment_frames];
            let own_band = segment % 7;
            let energies: Vec<f64> = STI_OCTAVE_CENTERS_HZ
                .iter()
                .map(|&center| {
                    bandpass_octave(body, center, SHORT_RATE)
                        .iter()
                        .map(|v| v.powi(2))
                        .sum()
                })
                .collect();
            let (peak_band, _) = energies
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .expect("seven band energies");
            assert_eq!(peak_band, own_band, "segment {segment} misbanded");
        }
    }

    #[test]
    fn full_sti_is_deterministic_and_validates_input() {
        assert_eq!(
            gen_full_sti_signal_seeded(SHORT_RATE, 0.1, 0.01, 3),
            gen_full_sti_signal_seeded(SHORT_RATE, 0.1, 0.01, 3)
        );
        assert_ne!(
            gen_full_sti_signal_seeded(SHORT_RATE, 0.1, 0.01, 3),
            gen_full_sti_signal_seeded(SHORT_RATE, 0.1, 0.01, 4)
        );
        assert!(gen_full_sti_signal(0, 0.1, 0.0).is_empty());
        assert!(gen_full_sti_signal(SHORT_RATE, 0.0, 0.0).is_empty());
        for gap in [f32::NAN, f32::INFINITY, -0.5] {
            assert!(
                gen_full_sti_signal(SHORT_RATE, 0.1, gap).is_empty(),
                "gap {gap} should yield no signal"
            );
        }
        // Zero gap concatenates segments back to back.
        let tight = gen_full_sti_signal(SHORT_RATE, 0.1, 0.0);
        assert_eq!(
            tight.len(),
            STI_FULL_SEGMENT_COUNT * frames_for(0.1, SHORT_RATE)
        );
    }

    #[test]
    fn stipa_bands_carry_sent_modulation_depth() {
        // Regression test for requirements §10: the generators once
        // multiplied a radians-per-sample omega by seconds time, which
        // froze the envelope. Demodulate every band of the emission and
        // check both sent depths against the 0.55 design value.
        const DURATION_S: f32 = 30.0;
        let signal = gen_stipa_signal_seeded(SHORT_RATE, DURATION_S, 11);
        let trim = SHORT_RATE as usize / 4;
        let body = &signal[trim..signal.len() - trim];
        for (band, &center) in STI_OCTAVE_CENTERS_HZ.iter().enumerate() {
            let squared: Vec<f64> = bandpass_octave(body, center, SHORT_RATE)
                .iter()
                .map(|v| v.powi(2))
                .collect();
            for &fm in &STIPA_MODULATION_FREQUENCIES_HZ[band] {
                let depth = fitted_depth(&squared, fm, SHORT_RATE);
                assert!(
                    (0.3..0.8).contains(&depth),
                    "band {center} Hz fm {fm}: depth {depth:.3}, expected ~0.55"
                );
            }
        }
    }

    #[test]
    fn full_sti_segments_carry_sent_modulation_depth() {
        // Full-STI segments send full modulation (intensity depth 1.0).
        // Segments are single-band by construction, so no test-side
        // bandpass is needed: square each segment and fit its sent fm.
        // Short windows make single-segment estimates noisy; the mean
        // over all 98 segments is the robust statistic.
        const SEGMENT_S: f32 = 1.0;
        let signal = gen_full_sti_signal_seeded(SHORT_RATE, SEGMENT_S, 0.0, 12);
        let segment_frames = frames_for(SEGMENT_S, SHORT_RATE);
        let trim = SHORT_RATE as usize / 10;
        let mut sum = 0.0_f64;
        for segment in 0..STI_FULL_SEGMENT_COUNT {
            let start = segment * segment_frames;
            let body = &signal[start + trim..start + segment_frames - trim];
            let squared: Vec<f64> = body.iter().map(|&v| f64::from(v).powi(2)).collect();
            let fm = STI_FULL_MODULATION_FREQUENCIES_HZ[segment / 7];
            sum += fitted_depth(&squared, fm, SHORT_RATE);
        }
        let mean = sum / STI_FULL_SEGMENT_COUNT as f64;
        assert!(
            (0.8..1.2).contains(&mean),
            "mean segment depth {mean:.3}, expected ~1.0"
        );
    }
}

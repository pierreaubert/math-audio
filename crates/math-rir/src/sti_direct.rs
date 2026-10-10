//! Direct-method STIPA analysis from a recorded test signal.
//!
//! Scores a recording of the standardized STIPA test signal played through
//! a transmission channel. Each octave band is isolated with a zero-phase
//! bandpass, its intensity envelope is lowpassed, and a least-squares fit
//! recovers the modulation depth at the band's two modulation frequencies.
//! Per-cell modulation transfer values map to transmission indices and
//! combine into STI with the Annex A weighting shared with the indirect
//! analysis in the `sti` module.
//!
//! Pass a loopback (electrical) capture of the sent signal as `reference`
//! when one is available; the loopback is the preferred denominator. Without
//! a reference, received depths are compared against the sent depth of the
//! `math-dsp` STIPA generator, so a recording of any other signal (or of a
//! chain whose processing alters modulation depth before the acoustic path)
//! scores against the wrong denominator. Scores reflect test-time conditions:
//! no ambient-noise, hearing-threshold, or masking corrections are applied.
//! Results do not constitute IEC instrument certification.
//!
//! The STIPA modulation table, sent depth, and rate minimum mirror the
//! `math-dsp` generator by hand; the two crates are siblings with no
//! dependency edge, so the copies are kept in sync by convention and by the
//! round-trip handoff values reported with this module.
//!
//! References: IEC 60268-16:2020; the open STIPA generator and measurement
//! example at <https://github.com/rajmic/Speech-Transmission-Index-STI>.

// Rust guideline compliant 2026-02-21

use crate::bands::{BandWidth, BandpassWorkspace};
use crate::sti::{STI_OCTAVE_CENTERS_HZ, combine_mti, transmission_index};
use math_audio_iir_fir::filtfilt;
use math_audio_iir_fir::{BiquadCoefficients, peq_butterworth_lowpass};
use std::fmt;

/// STIPA modulation pairs per octave band, ascending bands (Hz).
///
/// Each entry holds the low then high modulation frequency applied to one
/// band carrier. Exact hand-mirror of the `math-dsp` STIPA generator table;
/// the crates share no dependency edge, so a change there must be copied
/// here (and revalidated against the round-trip handoff values).
pub const STIPA_MODULATION_FREQUENCIES_HZ: [[f64; 2]; 7] = [
    [1.6, 8.0],
    [1.0, 5.0],
    [0.63, 3.15],
    [2.0, 10.0],
    [1.25, 6.25],
    [0.8, 4.0],
    [2.5, 12.5],
];

/// Sent intensity-modulation depth of the STIPA generator.
///
/// Mirrors the `math-dsp` generator's modulation depth: each band carrier
/// leaves the generator with this intensity depth at both of its modulation
/// frequencies. Denominator of the modulation transfer ratio when no
/// loopback reference is supplied; prefer a reference whenever the
/// playback chain is available for capture.
pub const STIPA_SENT_MODULATION_DEPTH: f64 = 0.55;

/// Minimum post-trim analysis window for STIPA scoring (s).
///
/// Ten seconds hold at least six cycles of the slowest (0.63 Hz) modulator,
/// which keeps least-squares cross-talk between the two modulators of a
/// band near one percent. Shorter windows are rejected outright rather than
/// scored with unknown conditioning.
pub const STIPA_MIN_ANALYSIS_DURATION_S: f64 = 10.0;

// The analysis loop indexes the octave centers and the modulation table by
// a shared band index; a length mismatch would silently misalign them, so
// the equality is enforced at compile time instead of trusted.
const _: () = assert!(STI_OCTAVE_CENTERS_HZ.len() == STIPA_MODULATION_FREQUENCIES_HZ.len());

/// Butterworth order per bandpass edge before forward/reverse filtering.
///
/// Sixth order matches the indirect analysis in the `sti` module, so both
/// STI paths isolate octaves with the same skirts. Lower orders leak
/// adjacent-band modulation into the fits; higher orders lengthen edge
/// ringing past the default trim.
const BANDPASS_ORDER: usize = 6;

/// Butterworth order of the intensity-envelope lowpass leg.
///
/// Fourth order (doubled by forward/reverse filtering) with the 30 Hz
/// cutoff below. Steeper skirts would sharpen carrier rejection but are
/// unnecessary: the lowest carrier residue sits at twice 88 Hz.
const ENVELOPE_LOWPASS_ORDER: usize = 4;

/// Intensity-envelope lowpass cutoff (Hz).
///
/// Engineering choice validated by the in-module tests, not a claim about
/// the IEC reference filter: 30 Hz keeps all modulation frequencies
/// (at most 12.5 Hz, attenuated under a percent) while removing carrier
/// ripple (at least 176 Hz after squaring the lowest octave).
const ENVELOPE_LOWPASS_CUTOFF_HZ: f64 = 30.0;

/// Minimum sample rate carrying all seven STI octaves (Hz).
///
/// The 8 kHz octave reaches `8000 * sqrt(2)` Hz; lower rates narrow it.
/// Matches the reference implementation's minimum. The Nyquist gate below
/// subsumes this bound in practice (it requires about 22856 Hz).
const MIN_SAMPLE_RATE_HZ: f64 = 22_050.0;

/// Default edge trim applied before scoring (s).
///
/// A quarter second exceeds the bandpass and envelope filter ringing at
/// every supported rate, so the analysis window holds settled filter
/// output only. Trimming less admits edge transients into the fits.
const DEFAULT_TRIM_S: f64 = 0.25;

/// Knobs for direct-method STIPA analysis.
///
/// Currently a single edge trim; more options may join without breaking
/// the [`analyze_stipa_direct`] signature.
///
/// # Examples
/// ```
/// use math_rir::sti_direct::StipaOptions;
///
/// let options = StipaOptions::default();
/// assert_eq!(options.trim_s, 0.25);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StipaOptions {
    /// Seconds trimmed from each end before scoring.
    ///
    /// Removes filter ringing so the fits see settled output. Negative or
    /// NaN values are treated as zero; whatever remains must still cover
    /// [`STIPA_MIN_ANALYSIS_DURATION_S`].
    pub trim_s: f64,
}

impl Default for StipaOptions {
    fn default() -> Self {
        Self {
            trim_s: DEFAULT_TRIM_S,
        }
    }
}

/// Direct-method STIPA score and intermediate results.
///
/// Depths, ratios, and indices are unitless; durations are in seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct StipaDirectResult {
    /// Speech Transmission Index on the interval [0, 1].
    pub sti: f64,
    /// Modulation transfer per band, indexed [band][low-fm, high-fm].
    pub modulation_transfer: [[f64; 2]; 7],
    /// Transmission indices after limiting apparent SNR to ±15 dB.
    pub transmission_indices: [[f64; 2]; 7],
    /// Mean transmission index over both modulations, per octave band.
    pub mti: [f64; 7],
    /// Analysis window after edge trim, in seconds.
    pub analysis_duration_s: f64,
}

/// Reasons a recording cannot be STIPA-scored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StipaDirectError {
    /// Recording is empty, non-finite, or carries no energy.
    InvalidRecording,
    /// Reference is empty, non-finite, or carries no energy.
    InvalidReference,
    /// Reference length differs from the recording length.
    ReferenceLengthMismatch,
    /// Sampling rate is non-finite or non-positive.
    InvalidSampleRate,
    /// Sample rate cannot cover the complete 8 kHz octave below Nyquist.
    InsufficientSampleRate,
    /// Post-trim analysis window is below the ten-second minimum.
    RecordingTooShort,
    /// An octave band holds no usable modulation energy.
    MissingBandEnergy,
    /// A reference octave band holds no usable modulation.
    DegenerateReferenceModulation,
}

impl fmt::Display for StipaDirectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidRecording => "STIPA direct needs a finite recording with nonzero energy",
            Self::InvalidReference => "STIPA direct needs a finite reference with nonzero energy",
            Self::ReferenceLengthMismatch => {
                "STIPA direct needs a reference as long as the recording"
            }
            Self::InvalidSampleRate => "STIPA direct sampling rate must be finite and positive",
            Self::InsufficientSampleRate => {
                "STIPA direct needs the full 8 kHz octave below Nyquist"
            }
            Self::RecordingTooShort => "STIPA direct needs at least 10 s after edge trim",
            Self::MissingBandEnergy => "STIPA direct needs usable modulation energy in every band",
            Self::DegenerateReferenceModulation => {
                "STIPA direct needs a modulated reference in every band"
            }
        })
    }
}
impl std::error::Error for StipaDirectError {}

/// Score a STIPA recording against an optional reference.
///
/// Bandpasses the recording into the seven STI octaves, trims filter
/// edges, extracts each band's intensity envelope with a 30 Hz lowpass,
/// and demodulates both modulation frequencies per band by least-squares
/// fit. With a loopback `reference` (an electrical capture of the sent
/// signal, same length and rate), each cell's modulation transfer is the
/// received-over-sent depth ratio; without one, received depths are
/// compared against [`STIPA_SENT_MODULATION_DEPTH`]. Transmission indices
/// and the final STI reuse the `sti` module's Annex A helpers. Gain of the
/// recording or reference does not affect the score: each signal is
/// peak-normalized before filtering.
///
/// # Examples
/// ```
/// // A one-second recording is finite and loud but far too short.
/// let recording = vec![0.5; 48_000];
/// let options = math_rir::sti_direct::StipaOptions::default();
/// let error = math_rir::sti_direct::analyze_stipa_direct(
///     &recording,
///     None,
///     48_000.0,
///     &options,
/// )
/// .unwrap_err();
/// assert_eq!(
///     error,
///     math_rir::sti_direct::StipaDirectError::RecordingTooShort
/// );
/// ```
///
/// # Errors
/// Returns [`StipaDirectError`] for invalid samples or rates, short
/// recordings, unusable octave bands, or a degenerate reference.
pub fn analyze_stipa_direct(
    recording: &[f32],
    reference: Option<&[f32]>,
    sample_rate_hz: f64,
    options: &StipaOptions,
) -> Result<StipaDirectResult, StipaDirectError> {
    if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        return Err(StipaDirectError::InvalidSampleRate);
    }
    if sample_rate_hz < MIN_SAMPLE_RATE_HZ {
        return Err(StipaDirectError::InsufficientSampleRate);
    }
    // Same 8 kHz-octave Nyquist gate as the indirect analysis: the top edge
    // 8000·√2 must clear 99% of Nyquist (effective minimum ≈ 22856 Hz).
    if 8000.0 * std::f64::consts::SQRT_2 >= sample_rate_hz * 0.5 * 0.99 {
        return Err(StipaDirectError::InsufficientSampleRate);
    }
    let Some(recording_peak) = signal_peak(recording) else {
        return Err(StipaDirectError::InvalidRecording);
    };
    if recording_peak <= 0.0 {
        return Err(StipaDirectError::InvalidRecording);
    }
    let normalized_reference = match reference {
        None => None,
        Some(signal) => {
            if signal.len() != recording.len() {
                return Err(StipaDirectError::ReferenceLengthMismatch);
            }
            let Some(peak) = signal_peak(signal) else {
                return Err(StipaDirectError::InvalidReference);
            };
            if peak <= 0.0 {
                return Err(StipaDirectError::InvalidReference);
            }
            Some(normalize_by_peak(signal, peak))
        }
    };
    let Some((trim_start, trim_end)) = trim_window(recording.len(), options.trim_s, sample_rate_hz)
    else {
        return Err(StipaDirectError::RecordingTooShort);
    };

    // Peak-normalize: modulation depth is a ratio against each signal's own
    // DC, so absolute gain carries no information and only risks float
    // overflow in the squared-envelope sums.
    let normalized_recording = normalize_by_peak(recording, recording_peak);

    let lowpass = filtfilt::peq_to_coefficients(&peq_butterworth_lowpass(
        ENVELOPE_LOWPASS_ORDER,
        ENVELOPE_LOWPASS_CUTOFF_HZ,
        sample_rate_hz,
    ));
    let mut workspace = BandpassWorkspace::new();
    let mut modulation_transfer = [[0.0; 2]; 7];
    for (band, &center) in STI_OCTAVE_CENTERS_HZ.iter().enumerate() {
        let modulations = STIPA_MODULATION_FREQUENCIES_HZ[band];
        let filtered = workspace.process(
            &normalized_recording,
            center,
            BandWidth::Octave,
            sample_rate_hz,
            BANDPASS_ORDER,
        );
        let envelope = intensity_envelope(&filtered[trim_start..trim_end], &lowpass);
        let Some(received) = band_depths(&envelope, modulations, sample_rate_hz) else {
            return Err(StipaDirectError::MissingBandEnergy);
        };
        modulation_transfer[band] = match &normalized_reference {
            None => received.map(|depth| (depth / STIPA_SENT_MODULATION_DEPTH).clamp(0.0, 1.0)),
            Some(normalized) => {
                let filtered = workspace.process(
                    normalized,
                    center,
                    BandWidth::Octave,
                    sample_rate_hz,
                    BANDPASS_ORDER,
                );
                let envelope = intensity_envelope(&filtered[trim_start..trim_end], &lowpass);
                let Some(sent) = band_depths(&envelope, modulations, sample_rate_hz) else {
                    return Err(StipaDirectError::DegenerateReferenceModulation);
                };
                let mut cell = [0.0; 2];
                for (slot, (&m_recv, &m_ref)) in cell.iter_mut().zip(received.iter().zip(&sent)) {
                    if m_ref <= 0.0 || !m_ref.is_finite() {
                        return Err(StipaDirectError::DegenerateReferenceModulation);
                    }
                    *slot = (m_recv / m_ref).clamp(0.0, 1.0);
                }
                cell
            }
        };
    }

    let transmission_indices = modulation_transfer.map(|row| row.map(transmission_index));
    let mti = std::array::from_fn(|band| {
        (transmission_indices[band][0] + transmission_indices[band][1]) / 2.0
    });
    Ok(StipaDirectResult {
        sti: combine_mti(&mti),
        modulation_transfer,
        transmission_indices,
        mti,
        analysis_duration_s: (trim_end - trim_start) as f64 / sample_rate_hz,
    })
}

/// Peak magnitude, or `None` for empty or non-finite input.
///
/// A finite signal has nonzero energy exactly when its peak is positive,
/// so this one pass serves both validation and normalization.
fn signal_peak(signal: &[f32]) -> Option<f64> {
    if signal.is_empty() {
        return None;
    }
    let mut peak = 0.0_f64;
    for &sample in signal {
        if !sample.is_finite() {
            return None;
        }
        peak = peak.max(f64::from(sample).abs());
    }
    Some(peak)
}

/// Scale a validated signal by its positive peak.
fn normalize_by_peak(signal: &[f32], peak: f64) -> Vec<f32> {
    signal
        .iter()
        .map(|&sample| (f64::from(sample) / peak) as f32)
        .collect()
}

/// Post-trim sample window, or `None` when too little remains.
///
/// A negative or NaN trim is meaningless and treated as zero; an infinite
/// trim removes everything and fails the duration check. Integer math is
/// ordered so no overflow precedes the checks.
fn trim_window(len: usize, trim_s: f64, sample_rate_hz: f64) -> Option<(usize, usize)> {
    let trim = if trim_s.is_nan() {
        0.0
    } else if !trim_s.is_finite() {
        if trim_s.is_sign_positive() {
            f64::INFINITY
        } else {
            0.0
        }
    } else {
        trim_s.max(0.0)
    };
    let trim_samples = (trim * sample_rate_hz).round() as usize;
    if trim_samples > len / 2 {
        return None;
    }
    let remaining = len - 2 * trim_samples;
    if remaining as f64 / sample_rate_hz < STIPA_MIN_ANALYSIS_DURATION_S {
        return None;
    }
    Some((trim_samples, trim_samples + remaining))
}

/// Intensity envelope of one trimmed octave band.
///
/// Squares the band (intensity) and removes carrier ripple with the shared
/// zero-phase lowpass, leaving the slow modulation plus the band's DC.
fn intensity_envelope(band: &[f32], lowpass: &[BiquadCoefficients<f64>]) -> Vec<f64> {
    let squared: Vec<f64> = band
        .iter()
        .map(|&sample| {
            let value = f64::from(sample);
            value * value
        })
        .collect();
    filtfilt::filtfilt(&squared, lowpass)
}

/// Least-squares `a·cos + b·sin + c` fit at one modulation frequency.
///
/// Returns the `(a, b, c)` amplitudes with time in seconds from the window
/// start. Inputs with fewer samples than parameters, or a singular normal
/// system (only possible for degenerate inputs), yield all zeros; the
/// caller then rejects the zero DC as missing energy.
fn fit_modulation(envelope: &[f64], fm_hz: f64, sample_rate_hz: f64) -> (f64, f64, f64) {
    if envelope.len() < 4 {
        return (0.0, 0.0, 0.0);
    }
    let omega = std::f64::consts::TAU * fm_hz / sample_rate_hz;
    let (mut scc, mut sss, mut scs) = (0.0, 0.0, 0.0);
    let (mut sc1, mut ss1) = (0.0, 0.0);
    let (mut bc, mut bs, mut b1) = (0.0, 0.0, 0.0);
    for (n, &sample) in envelope.iter().enumerate() {
        // Direct sin/cos per sample avoids oscillator drift over the
        // million-sample windows a high-rate recording can hold.
        let (sin, cos) = (omega * n as f64).sin_cos();
        scc += cos * cos;
        sss += sin * sin;
        scs += cos * sin;
        sc1 += cos;
        ss1 += sin;
        bc += sample * cos;
        bs += sample * sin;
        b1 += sample;
    }
    let frames = envelope.len() as f64;
    let det = det3(&[scc, scs, sc1, scs, sss, ss1, sc1, ss1, frames]);
    if !det.is_finite() || det == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let a = det3(&[bc, scs, sc1, bs, sss, ss1, b1, ss1, frames]) / det;
    let b = det3(&[scc, bc, sc1, scs, bs, ss1, sc1, b1, frames]) / det;
    let c = det3(&[scc, scs, bc, scs, sss, bs, sc1, ss1, b1]) / det;
    (a, b, c)
}

/// Determinant of a row-major 3×3 matrix.
fn det3(m: &[f64; 9]) -> f64 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

/// Modulation depths at both modulators of one band envelope.
///
/// Depth is the fitted cosine/sine magnitude over the fitted DC. That
/// quotient is exact for the fit basis: the factor of two found in
/// DFT-bin estimators belongs to single-sided spectra, where each cosine
/// splits across mirrored bins, while the least-squares amplitudes fitted
/// here already hold the full per-frequency content. `None` marks an
/// unusable envelope (non-positive or non-finite DC).
fn band_depths(
    envelope: &[f64],
    modulations_hz: [f64; 2],
    sample_rate_hz: f64,
) -> Option<[f64; 2]> {
    let mut depths = [0.0; 2];
    for (slot, &fm_hz) in depths.iter_mut().zip(&modulations_hz) {
        let (a, b, c) = fit_modulation(envelope, fm_hz, sample_rate_hz);
        if !c.is_finite() || c <= 0.0 {
            return None;
        }
        *slot = a.hypot(b) / c;
    }
    Some(depths)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Lowest standard rate clearing the octave gate (≈ 22856 Hz).
    const TEST_RATE_HZ: f64 = 24_000.0;
    const FIXTURE_SEED: u64 = 602_681_605;
    const IR_SEED_BASE: u64 = 77_000;

    /// Independent carrier/IR realizations averaged per test channel.
    ///
    /// One 10 s noise-carrier window estimates per-cell depth with
    /// σ ≈ 0.02-0.12 (narrow low bands worst), so a single realization
    /// cannot meet a 0.05 per-cell tolerance; averaging sixty-four cuts
    /// estimator noise eightfold while the fixed seeds keep the test
    /// deterministic and repeatable.
    const CHANNEL_REALIZATIONS: usize = 64;

    /// Speech-spectrum band levels (dB) for the STIPA-like fixtures.
    ///
    /// Fixture-local mirror of the generator's table; the analyzer never
    /// sees these values because each band normalizes to its own DC.
    const FIXTURE_BAND_LEVELS_DB: [f64; 7] = [-2.5, 0.5, 0.0, -6.0, -12.0, -18.0, -24.0];

    /// Deterministic uniform noise in [-1, 1). Test fixture only.
    fn fixture_noise(frames: usize, mut seed: u64) -> Vec<f32> {
        let mut out = Vec::with_capacity(frames);
        for _ in 0..frames {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let word = ((seed >> 33) ^ seed) as u32;
            out.push((word as f32 / u32::MAX as f32) * 2.0 - 1.0);
        }
        out
    }

    /// Apply one band's dual STIPA modulation and speech weighting in place.
    fn apply_band_modulation(
        signal: &mut [f32],
        carrier: &[f32],
        band: usize,
        sample_rate_hz: f64,
    ) {
        let gain = 10.0_f64.powf(FIXTURE_BAND_LEVELS_DB[band] / 20.0);
        let [low_fm, high_fm] = STIPA_MODULATION_FREQUENCIES_HZ[band];
        let omega_low = std::f64::consts::TAU * low_fm / sample_rate_hz;
        let omega_high = std::f64::consts::TAU * high_fm / sample_rate_hz;
        for (n, sample) in signal.iter_mut().enumerate() {
            let phase = n as f64;
            let beating = (omega_low * phase).sin() - (omega_high * phase).sin();
            let envelope = (0.5 * (1.0 + STIPA_SENT_MODULATION_DEPTH * beating).max(0.0)).sqrt();
            *sample += (f64::from(carrier[n]) * envelope * gain) as f32;
        }
    }

    /// STIPA-like fixture: seeded noise filtered into the seven STI
    /// octaves, dual-modulated per band, speech-weighted, summed.
    ///
    /// A test fixture, not a generator fork: it reuses this module's own
    /// modulation table and sent depth plus the crate's bandpass.
    fn stipa_like_fixture(sample_rate_hz: f64, duration_s: f64, seed: u64) -> Vec<f32> {
        let frames = (duration_s * sample_rate_hz).round() as usize;
        let noise = fixture_noise(frames, seed);
        let mut workspace = BandpassWorkspace::new();
        let mut signal = vec![0.0_f32; frames];
        for (band, &center) in STI_OCTAVE_CENTERS_HZ.iter().enumerate() {
            let carrier = workspace
                .process(
                    &noise,
                    center,
                    BandWidth::Octave,
                    sample_rate_hz,
                    BANDPASS_ORDER,
                )
                .to_vec();
            apply_band_modulation(&mut signal, &carrier, band, sample_rate_hz);
        }
        signal
    }

    /// Convolve with a noise-weighted exponential decay of the given RT60.
    ///
    /// The impulse response is white noise times `exp(−3·ln10·t/RT60)` over
    /// one RT60, so its energy decay matches the analytical modulation
    /// transfer shape used below.
    fn convolve_exponential_decay(
        signal: &[f32],
        sample_rate_hz: f64,
        rt60: f64,
        seed: u64,
    ) -> Vec<f32> {
        let ir_len = (rt60 * sample_rate_hz).round() as usize + 1;
        let noise = fixture_noise(ir_len, seed + (rt60 * 1000.0) as u64);
        let decay = 3.0 * 10.0_f64.ln() / rt60;
        let ir: Vec<f32> = noise
            .iter()
            .enumerate()
            .map(|(n, &tap)| {
                let time = n as f64 / sample_rate_hz;
                (f64::from(tap) * (-decay * time).exp()) as f32
            })
            .collect();
        let mut out = vec![0.0_f32; signal.len() + ir.len() - 1];
        for (i, &sample) in signal.iter().enumerate() {
            for (j, &tap) in ir.iter().enumerate() {
                out[i + j] += sample * tap;
            }
        }
        out
    }

    /// Add one realization's transfer cells into running totals.
    fn accumulate_mtf(total: &mut [[f64; 2]; 7], cells: &[[f64; 2]; 7]) {
        for (total_row, row) in total.iter_mut().zip(cells) {
            for (slot, &mtf) in total_row.iter_mut().zip(row) {
                *slot += mtf;
            }
        }
    }

    /// Analytical modulation transfer of an exponential energy decay.
    fn analytical_mtf(fm_hz: f64, rt60: f64) -> f64 {
        let scaled = std::f64::consts::TAU * fm_hz * rt60 / (6.0 * 10.0_f64.ln());
        1.0 / (1.0 + scaled * scaled).sqrt()
    }

    fn rms(signal: &[f32]) -> f64 {
        (signal
            .iter()
            .map(|&sample| {
                let value = f64::from(sample);
                value * value
            })
            .sum::<f64>()
            / signal.len() as f64)
            .sqrt()
    }

    #[test]
    fn stipa_demodulator_recovers_exact_depth() {
        // 2.5 s at 48 kHz; the synthetic envelopes sit exactly in the fit
        // span, so recovery is exact up to float rounding at any length.
        const FRAMES: usize = 120_000;
        const SAMPLE_RATE_HZ: f64 = 48_000.0;
        for (band, fms) in STIPA_MODULATION_FREQUENCIES_HZ.iter().enumerate() {
            for &fm in fms {
                for &depth in &[0.05, 0.3, 0.55, 1.0] {
                    for &phase in &[0.0, 0.7] {
                        let omega = std::f64::consts::TAU * fm / SAMPLE_RATE_HZ;
                        let envelope: Vec<f64> = (0..FRAMES)
                            .map(|n| 1.0 + depth * (omega * n as f64 + phase).cos())
                            .collect();
                        let (a, b, c) = fit_modulation(&envelope, fm, SAMPLE_RATE_HZ);
                        let recovered = a.hypot(b) / c;
                        assert!(
                            (recovered - depth).abs() < 1e-6,
                            "band {band} fm {fm}: recovered {recovered} vs {depth}"
                        );
                        assert!((c - 1.0).abs() < 1e-9, "band {band} fm {fm}: DC {c} vs 1");
                    }
                }
            }
        }
    }

    #[test]
    fn stipa_demodulator_rejects_degenerate_input() {
        let (a, b, c) = fit_modulation(&[], 2.0, 48_000.0);
        assert!(a == 0.0 && b == 0.0 && c == 0.0);
        let (a, b, c) = fit_modulation(&[1.0, 1.0], 2.0, 48_000.0);
        assert!(a == 0.0 && b == 0.0 && c == 0.0);
        assert!(band_depths(&[0.0; 64], [2.0, 10.0], 48_000.0).is_none());
    }

    #[test]
    fn stipa_self_reference_identity() {
        let recording = stipa_like_fixture(TEST_RATE_HZ, 10.5, FIXTURE_SEED);
        let options = StipaOptions::default();
        let result =
            analyze_stipa_direct(&recording, Some(&recording), TEST_RATE_HZ, &options).unwrap();
        for row in &result.modulation_transfer {
            for &mtf in row {
                assert_eq!(mtf, 1.0);
            }
        }
        assert!((result.sti - 1.0).abs() < 1e-9);
    }

    #[test]
    fn stipa_synthetic_channel_matches_analytical_mtf() {
        // About five minutes in release (naive convolution dominates); the
        // repo gates run release builds only.
        let options = StipaOptions::default();
        let crop_start = TEST_RATE_HZ.round() as usize;
        let crop_len = (10.5 * TEST_RATE_HZ).round() as usize;
        let sources: Vec<Vec<f32>> = (0..CHANNEL_REALIZATIONS)
            .map(|realization| {
                stipa_like_fixture(TEST_RATE_HZ, 12.0, FIXTURE_SEED + realization as u64)
            })
            .collect();
        let mut previous_noref_sti = f64::INFINITY;
        let mut previous_ref_sti = f64::INFINITY;
        // RT60 0 anchors the absolute calibration: no channel, so every
        // analytical cell is exactly 1.0 (the formula below agrees).
        for &rt60 in &[0.0, 0.3, 0.6] {
            let mut noref_sum = [[0.0; 2]; 7];
            let mut ref_sum = [[0.0; 2]; 7];
            let mut noref_sti_sum = 0.0;
            let mut ref_sti_sum = 0.0;
            for (realization, source) in sources.iter().enumerate() {
                let reference = source[crop_start..crop_start + crop_len].to_vec();
                let recording = if rt60 <= 0.0 {
                    reference.clone()
                } else {
                    let decayed = convolve_exponential_decay(
                        source,
                        TEST_RATE_HZ,
                        rt60,
                        IR_SEED_BASE + realization as u64,
                    );
                    decayed[crop_start..crop_start + crop_len].to_vec()
                };
                let absolute =
                    analyze_stipa_direct(&recording, None, TEST_RATE_HZ, &options).unwrap();
                let with_reference =
                    analyze_stipa_direct(&recording, Some(&reference), TEST_RATE_HZ, &options)
                        .unwrap();
                if realization == 0 {
                    assert!((absolute.analysis_duration_s - 10.0).abs() < 1e-3);
                }
                accumulate_mtf(&mut noref_sum, &absolute.modulation_transfer);
                accumulate_mtf(&mut ref_sum, &with_reference.modulation_transfer);
                noref_sti_sum += absolute.sti;
                ref_sti_sum += with_reference.sti;
            }
            let count = CHANNEL_REALIZATIONS as f64;
            let mean_noref_sti = noref_sti_sum / count;
            let mean_ref_sti = ref_sti_sum / count;
            assert!(
                mean_noref_sti < previous_noref_sti,
                "mean no-reference STI not decreasing at RT60 {rt60}"
            );
            assert!(
                mean_ref_sti < previous_ref_sti,
                "mean reference STI not decreasing at RT60 {rt60}"
            );
            previous_noref_sti = mean_noref_sti;
            previous_ref_sti = mean_ref_sti;
            // The reference path divides out generator coloration (envelope
            // clamp, square-root harmonics) and analyzer sideband cutting,
            // so its means track the analytical decay transfer directly.
            for (band, row) in ref_sum.iter().enumerate() {
                for (cell, &total) in row.iter().enumerate() {
                    let mtf = total / count;
                    let fm = STIPA_MODULATION_FREQUENCIES_HZ[band][cell];
                    let expected = analytical_mtf(fm, rt60);
                    assert!(
                        (mtf - expected).abs() < 0.05,
                        "band {band} fm {fm} RT60 {rt60}: {mtf} vs {expected}"
                    );
                }
            }
            if rt60 <= 0.0 {
                // No-reference clean-path characterization (not a precision
                // check): the absolute denominator cannot cancel generator
                // and sideband coloration, so narrow bands read up to ~12%
                // low. This pins the bias so a regression cannot silently
                // double it; prefer a loopback reference for accuracy.
                for (band, row) in noref_sum.iter().enumerate() {
                    for (cell, &total) in row.iter().enumerate() {
                        let mtf = total / count;
                        assert!(
                            mtf > 0.80,
                            "band {band} cell {cell}: clean no-reference MTF {mtf} too low"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn stipa_tone_transfer_matches_analytical() {
        // Deterministic demod-chain check: AM tones with reduced depths go
        // straight through square, lowpass, and fit (no carriers, no
        // convolution), pinning transfer linearity precisely and instantly.
        let lowpass = filtfilt::peq_to_coefficients(&peq_butterworth_lowpass(
            ENVELOPE_LOWPASS_ORDER,
            ENVELOPE_LOWPASS_CUTOFF_HZ,
            TEST_RATE_HZ,
        ));
        let frames = (10.0 * TEST_RATE_HZ).round() as usize;
        for &rt60 in &[0.3, 0.6] {
            for (band, &center) in STI_OCTAVE_CENTERS_HZ.iter().enumerate() {
                for &fm in &STIPA_MODULATION_FREQUENCIES_HZ[band] {
                    let expected = STIPA_SENT_MODULATION_DEPTH * analytical_mtf(fm, rt60);
                    let carrier_omega = std::f64::consts::TAU * center / TEST_RATE_HZ;
                    let mod_omega = std::f64::consts::TAU * fm / TEST_RATE_HZ;
                    let band_signal: Vec<f32> = (0..frames)
                        .map(|n| {
                            let phase = n as f64;
                            let carrier = (carrier_omega * phase).sin();
                            let am = (0.5 * (1.0 + expected * (mod_omega * phase).sin())).sqrt();
                            (carrier * am) as f32
                        })
                        .collect();
                    let envelope = intensity_envelope(&band_signal, &lowpass);
                    let (a, b, c) = fit_modulation(&envelope, fm, TEST_RATE_HZ);
                    let measured = a.hypot(b) / c;
                    assert!(
                        (measured - expected).abs() < 0.01,
                        "band {band} fm {fm} RT60 {rt60}: {measured} vs {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn stipa_added_noise_lowers_sti_monotonically() {
        let clean = stipa_like_fixture(TEST_RATE_HZ, 10.5, FIXTURE_SEED);
        let clean_rms = rms(&clean);
        let noise = fixture_noise(clean.len(), FIXTURE_SEED + 1);
        let noise_rms = rms(&noise);
        let options = StipaOptions::default();
        let mut previous = f64::INFINITY;
        for &ratio in &[0.0, 0.1, 0.3, 1.0] {
            let gain = ratio * clean_rms / noise_rms;
            let recording: Vec<f32> = clean
                .iter()
                .zip(&noise)
                .map(|(&sample, &white)| sample + (f64::from(white) * gain) as f32)
                .collect();
            let result = analyze_stipa_direct(&recording, None, TEST_RATE_HZ, &options).unwrap();
            assert!(
                result.sti < previous,
                "STI not decreasing at noise ratio {ratio}"
            );
            previous = result.sti;
        }
    }

    #[test]
    fn stipa_rejects_invalid_inputs() {
        let options = StipaOptions::default();
        let valid = stipa_like_fixture(48_000.0, 10.5, FIXTURE_SEED);
        assert_eq!(
            analyze_stipa_direct(&[], None, 48_000.0, &options),
            Err(StipaDirectError::InvalidRecording)
        );
        assert_eq!(
            analyze_stipa_direct(&vec![0.0; 1024], None, 48_000.0, &options),
            Err(StipaDirectError::InvalidRecording)
        );
        let mut nan = valid.clone();
        nan[100] = f32::NAN;
        assert_eq!(
            analyze_stipa_direct(&nan, None, 48_000.0, &options),
            Err(StipaDirectError::InvalidRecording)
        );
        for rate in [0.0, -48_000.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                analyze_stipa_direct(&valid, None, rate, &options),
                Err(StipaDirectError::InvalidSampleRate),
                "rate {rate}"
            );
        }
        for rate in [16_000.0, 22_050.0, 22_800.0] {
            assert_eq!(
                analyze_stipa_direct(&valid, None, rate, &options),
                Err(StipaDirectError::InsufficientSampleRate),
                "rate {rate}"
            );
        }
        // Just above the octave gate (≈ 22856 Hz): accepted. The rate only
        // sets filter edges and the time axis; content match is the
        // caller's responsibility, so the 48 kHz-shaped content suffices
        // to pin the boundary's upper side.
        assert!(analyze_stipa_direct(&valid, None, 23_000.0, &options).is_ok());
        let short = stipa_like_fixture(48_000.0, 10.0, FIXTURE_SEED);
        assert_eq!(
            analyze_stipa_direct(&short, None, 48_000.0, &options),
            Err(StipaDirectError::RecordingTooShort)
        );
        assert_eq!(
            analyze_stipa_direct(&valid, Some(&valid[..valid.len() - 1]), 48_000.0, &options),
            Err(StipaDirectError::ReferenceLengthMismatch)
        );
        assert_eq!(
            analyze_stipa_direct(&valid, Some(&[]), 48_000.0, &options),
            Err(StipaDirectError::ReferenceLengthMismatch)
        );
        assert_eq!(
            analyze_stipa_direct(&valid, Some(&vec![0.0; valid.len()]), 48_000.0, &options),
            Err(StipaDirectError::InvalidReference)
        );
        let mut nan_reference = valid.clone();
        nan_reference[7] = f32::NAN;
        assert_eq!(
            analyze_stipa_direct(&valid, Some(&nan_reference), 48_000.0, &options),
            Err(StipaDirectError::InvalidReference)
        );
    }

    #[test]
    fn stipa_trim_edge_cases() {
        assert_eq!(StipaOptions::default().trim_s, 0.25);
        let recording = stipa_like_fixture(48_000.0, 10.5, FIXTURE_SEED);
        let wide = StipaOptions { trim_s: 6.0 };
        assert_eq!(
            analyze_stipa_direct(&recording, None, 48_000.0, &wide),
            Err(StipaDirectError::RecordingTooShort)
        );
        for trim_s in [-1.0, f64::NAN] {
            let options = StipaOptions { trim_s };
            assert!(analyze_stipa_direct(&recording, None, 48_000.0, &options).is_ok());
        }
    }

    #[test]
    fn stipa_analysis_is_deterministic() {
        let recording = stipa_like_fixture(TEST_RATE_HZ, 10.5, FIXTURE_SEED);
        let options = StipaOptions::default();
        let first = analyze_stipa_direct(&recording, None, TEST_RATE_HZ, &options).unwrap();
        let second = analyze_stipa_direct(&recording, None, TEST_RATE_HZ, &options).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn stipa_error_display_covers_all_variants() {
        for error in [
            StipaDirectError::InvalidRecording,
            StipaDirectError::InvalidReference,
            StipaDirectError::ReferenceLengthMismatch,
            StipaDirectError::InvalidSampleRate,
            StipaDirectError::InsufficientSampleRate,
            StipaDirectError::RecordingTooShort,
            StipaDirectError::MissingBandEnergy,
            StipaDirectError::DegenerateReferenceModulation,
        ] {
            assert!(!format!("{error}").is_empty());
        }
    }
}

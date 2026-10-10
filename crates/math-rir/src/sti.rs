//! Full indirect Speech Transmission Index analysis from a measured mono impulse response.
//!
//! Uses the Schroeder energy-envelope MTF and IEC 60268-16:2020 Annex A
//! weighting/redundancy factors. All 14 modulation frequencies are evaluated
//! in all seven octave bands. This is IR-only STI: operational ambient noise,
//! absolute hearing threshold and auditory masking corrections are not applied.
//! The caller must provide a linear, time-invariant, full-band capture with
//! its complete decay and sufficient measurement SNR. No noise-tail subtraction,
//! onset crop, decay extrapolation or minimum-phase reconstruction is performed.
//! Results do not constitute IEC instrument certification.
//!
//! References: IEC 60268-16:2020, Annex A and indirect-method Annex C;
//! [IEC publication](https://webstore.iec.ch/en/publication/26771),
//! [REW STI panel](https://www.roomeqwizard.com/betahelp/help/html/sti.html),
//! and [Annex A/C validation examples](https://www.mathworks.com/help/audio/ug/speech-transmission-index-iec60268-16-edition-5-compliance.html).
//!
//! Octave analysis uses the crate's zero-phase Butterworth bandpass (sixth
//! order per edge, forward/reverse). Zero padding on both sides retains filter
//! ringing and prevents endpoint reflection from modifying the measured decay.

// Rust guideline compliant 2026-02-21
use crate::bands::{BandWidth, BandpassWorkspace};
use std::fmt;

/// Nominal IEC STI octave-band centers, in ascending order (Hz).
pub const STI_OCTAVE_CENTERS_HZ: [f64; 7] = [125., 250., 500., 1000., 2000., 4000., 8000.];
/// Nominal IEC full-STI modulation frequencies (Hz).
pub const STI_MODULATION_FREQUENCIES_HZ: [f64; 14] = [
    0.63, 0.8, 1., 1.25, 1.6, 2., 2.5, 3.15, 4., 5., 6.3, 8., 10., 12.5,
];
/// IEC 60268-16:2020 Table A.1 octave weighting factors.
pub const STI_ALPHA: [f64; 7] = [0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173];
/// IEC 60268-16:2020 Table A.1 adjacent-octave redundancy factors.
pub const STI_BETA: [f64; 6] = [0.085, 0.078, 0.065, 0.011, 0.047, 0.095];

// 250 ms exceeds 20 periods at the lowest octave edge; retains filter tails.
const FILTER_PADDING_S: f64 = 0.25;
// Reject bands below -120 dB of normalized input energy rather than divide by zero.
const MIN_RELATIVE_BAND_ENERGY: f64 = 1e-12;

/// Full indirect STI and its intermediate octave-band results.
#[derive(Debug, Clone, PartialEq)]
pub struct StiResult {
    /// Speech Transmission Index on the interval [0, 1].
    pub sti: f64,
    /// Modulation transfer matrix, indexed by modulation frequency then octave band.
    pub modulation_transfer: [[f64; 7]; 14],
    /// Transmission indices after limiting apparent SNR to ±15 dB.
    pub transmission_indices: [[f64; 7]; 14],
    /// Mean transmission index over all 14 modulations, per octave band.
    pub mti: [f64; 7],
    /// Original capture duration, excluding filter padding, in seconds.
    pub duration_s: f64,
}

/// Reasons an impulse response cannot support full indirect STI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StiError {
    /// Input is empty, nonfinite or has zero energy.
    InvalidImpulseResponse,
    /// Sampling rate is nonfinite, nonpositive or above the analysis limit.
    InvalidSampleRate,
    /// Sample rate cannot cover the complete 8 kHz octave below Nyquist.
    InsufficientSampleRate,
    /// An octave contains no numerically usable signal energy.
    MissingBandEnergy,
}

impl fmt::Display for StiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidSampleRate => "STI sampling rate must be finite and within (0, 768000] Hz",
            Self::InvalidImpulseResponse => "STI requires a finite, nonzero impulse response",
            Self::InsufficientSampleRate => "STI requires the full 8 kHz octave below Nyquist",
            Self::MissingBandEnergy => "STI requires usable energy in all seven octave bands",
        })
    }
}
impl std::error::Error for StiError {}

/// Compute full indirect STI from a measured mono impulse response.
///
/// `sample_rate_hz` is the native sampling rate, at most 768 kHz. The waveform must include
/// the complete measured decay; short or noisy captures can bias STI.
/// Gain and absolute time origin do not affect IR-only STI. No operational
/// speech/noise levels are inferred from arbitrary waveform amplitudes.
///
/// # Examples
/// ```
/// let mut ir = vec![0.0; 48000];
/// ir[2400] = 1.0;
/// let result = math_rir::sti::analyze_sti(&ir, 48000.0)?;
/// assert!(result.sti > 0.99);
/// # Ok::<(), math_rir::sti::StiError>(())
/// ```
///
/// # Errors
/// Returns [`StiError`] for invalid samples, inadequate Nyquist coverage,
/// or an octave without usable energy.
pub fn analyze_sti(rir: &[f32], sample_rate_hz: f64) -> Result<StiResult, StiError> {
    // Bound padding allocation at four times the supported native 192 kHz rate.
    if !sample_rate_hz.is_finite()
        || !(0.0..=768000.0).contains(&sample_rate_hz)
        || sample_rate_hz == 0.0
    {
        return Err(StiError::InvalidSampleRate);
    }
    if 8000.0 * std::f64::consts::SQRT_2 >= sample_rate_hz * 0.5 * 0.99 {
        return Err(StiError::InsufficientSampleRate);
    }
    if rir.is_empty() || rir.iter().any(|v| !v.is_finite()) {
        return Err(StiError::InvalidImpulseResponse);
    }
    let peak = rir.iter().map(|v| f64::from(*v).abs()).fold(0.0, f64::max);
    if peak == 0.0 {
        return Err(StiError::InvalidImpulseResponse);
    }
    let padding = (FILTER_PADDING_S * sample_rate_hz).ceil() as usize;
    let mut padded = vec![0.0; rir.len() + 2 * padding];
    for (out, &sample) in padded[padding..padding + rir.len()].iter_mut().zip(rir) {
        *out = (f64::from(sample) / peak) as f32;
    }
    let input_energy: f64 = padded.iter().map(|&v| f64::from(v).powi(2)).sum();
    let mut workspace = BandpassWorkspace::new();
    let mut mtf = [[0.0; 7]; 14];
    for (band, &center) in STI_OCTAVE_CENTERS_HZ.iter().enumerate() {
        let filtered = workspace.process(&padded, center, BandWidth::Octave, sample_rate_hz, 6);
        let energy: Vec<f64> = filtered.iter().map(|&v| f64::from(v).powi(2)).collect();
        let total: f64 = energy.iter().sum();
        if !total.is_finite() || total <= input_energy * MIN_RELATIVE_BAND_ENERGY {
            return Err(StiError::MissingBandEnergy);
        }
        for (row, &frequency) in mtf.iter_mut().zip(&STI_MODULATION_FREQUENCIES_HZ) {
            row[band] = energy_modulation(&energy, total, frequency, sample_rate_hz);
        }
    }
    let transmission_indices = mtf.map(|row| row.map(transmission_index));
    let mti = std::array::from_fn(|band| {
        transmission_indices
            .iter()
            .map(|row| row[band])
            .sum::<f64>()
            / 14.0
    });
    Ok(StiResult {
        sti: combine_mti(&mti),
        modulation_transfer: mtf,
        transmission_indices,
        mti,
        duration_s: rir.len() as f64 / sample_rate_hz,
    })
}

fn energy_modulation(energy: &[f64], total: f64, frequency: f64, sample_rate: f64) -> f64 {
    // Schroeder integral |sum(h² exp(-j 2π f t))| / sum(h²).
    // The common dt factor cancels. Direct sin/cos avoids oscillator drift.
    let omega = std::f64::consts::TAU * frequency / sample_rate;
    let (mut real, mut imag) = (0.0, 0.0);
    for (i, &value) in energy.iter().enumerate() {
        let (sin, cos) = (omega * i as f64).sin_cos();
        real += value * cos;
        imag += value * sin;
    }
    (real.hypot(imag) / total).clamp(0.0, 1.0)
}

pub(crate) fn transmission_index(m: f64) -> f64 {
    // IEC Annex A: apparent SNR = 10 log10(m/(1-m)), clipped at ±15 dB.
    if m <= 0.0 {
        return 0.0;
    }
    if m >= 1.0 {
        return 1.0;
    }
    ((10.0 * (m / (1.0 - m)).log10()).clamp(-15.0, 15.0) + 15.0) / 30.0
}

pub(crate) fn combine_mti(mti: &[f64; 7]) -> f64 {
    let weighted: f64 = STI_ALPHA.iter().zip(mti).map(|(a, m)| a * m).sum();
    let redundancy: f64 = STI_BETA
        .iter()
        .enumerate()
        .map(|(k, b)| b * (mti[k] * mti[k + 1]).sqrt())
        .sum();
    (weighted - redundancy).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_and_annex_a_weighting() {
        assert_eq!(transmission_index(0.0), 0.0);
        assert_eq!(transmission_index(1.0), 1.0);
        assert_eq!(transmission_index(0.5), 0.5);
        for value in [0.0, 0.2, 0.5, 0.8, 1.0] {
            assert!((combine_mti(&[value; 7]) - value).abs() < 1e-12);
        }
        // Adjacent perfect octaves: Table A.1's weights minus redundancy.
        let mut mti = [0.0; 7];
        mti[2] = 1.0;
        mti[3] = 1.0;
        assert!((combine_mti(&mti) - 0.398).abs() < 1e-12);
    }

    #[test]
    fn schroeder_exponential_matches_analytical_mtf() {
        let sr = 48000.0;
        for rt60 in [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0] {
            // h²(t) decays with exponent 6 ln(10)/RT60.
            let decay = 6.0 * 10.0_f64.ln() / rt60;
            let energy: Vec<f64> = (0..(sr * rt60 * 2.0) as usize)
                .map(|i| (-decay * i as f64 / sr).exp())
                .collect();
            let total = energy.iter().sum();
            for f in STI_MODULATION_FREQUENCIES_HZ {
                let expected = 1.0 / (1.0 + (std::f64::consts::TAU * f / decay).powi(2)).sqrt();
                assert!((energy_modulation(&energy, total, f, sr) - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn pure_impulse_is_perfect_and_shift_gain_invariant() {
        let mut ir = vec![0.0; 48000];
        ir[0] = 1.0;
        let first = analyze_sti(&ir, 48000.0).unwrap();
        ir[0] = 0.0;
        ir[12000] = -0.125;
        let shifted = analyze_sti(&ir, 48000.0).unwrap();
        assert!(first.sti > 0.99);
        assert!((first.sti - shifted.sti).abs() < 1e-6);
        for (a, b) in first
            .modulation_transfer
            .iter()
            .flatten()
            .zip(shifted.modulation_transfer.iter().flatten())
        {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn two_equal_arrivals_match_echo_mtf() {
        let mut ir = vec![0.0; 48000];
        ir[4800] = 1.0;
        let isolated = analyze_sti(&ir, 48000.0).unwrap();
        ir[4800 + 9600] = 1.0;
        let result = analyze_sti(&ir, 48000.0).unwrap();
        for (i, f) in STI_MODULATION_FREQUENCIES_HZ.iter().enumerate() {
            let expected = (std::f64::consts::PI * f * 0.2).cos().abs();
            for (band, &m) in result.modulation_transfer[i].iter().enumerate() {
                // Two separated filtered arrivals retain the filter's energy MTF.
                let filtered_expected = expected * isolated.modulation_transfer[i][band];
                assert!(
                    (m - filtered_expected).abs() < 1e-6,
                    "f={f}: {m} != {filtered_expected}"
                );
            }
        }
        assert!(result.sti < 0.9);
    }

    #[test]
    fn filtered_exponential_carriers_meet_modulation_error_limits() {
        let sr = 48000.0;
        let mut total_error = 0.0;
        let mut count = 0;
        let mut previous_sti = 1.0;
        for rt60 in [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0] {
            let decay = 6.0 * 10.0_f64.ln() / rt60;
            let ir: Vec<f32> = (0..(sr * (rt60 * 1.5 + 0.1)) as usize)
                .map(|i| {
                    let t = i as f64 / sr;
                    let carriers: f64 = STI_OCTAVE_CENTERS_HZ
                        .iter()
                        .map(|f| (std::f64::consts::TAU * f * t).sin())
                        .sum();
                    (carriers * (-decay * t / 2.0).exp()) as f32
                })
                .collect();
            let result = analyze_sti(&ir, sr).unwrap();
            let mut expected_mti = 0.0;
            for (row, &f) in result
                .modulation_transfer
                .iter()
                .zip(&STI_MODULATION_FREQUENCIES_HZ)
            {
                let expected = 1.0 / (1.0 + (std::f64::consts::TAU * f / decay).powi(2)).sqrt();
                expected_mti += transmission_index(expected) / 14.0;
                for &actual in row {
                    let error = (actual - expected).abs();
                    assert!(error < 0.05, "RT60={rt60}, f={f}: error={error}");
                    total_error += error;
                    count += 1;
                }
            }
            assert!(
                (result.sti - expected_mti).abs() < 0.01,
                "RT60={rt60}: STI {} vs {expected_mti}",
                result.sti
            );
            assert!(result.sti < previous_sti);
            previous_sti = result.sti;
        }
        assert!(total_error / (count as f64) < 0.01);
    }

    #[test]
    fn invalid_inputs_do_not_produce_scores() {
        for ir in [&[][..], &[0.0][..], &[f32::NAN][..], &[f32::INFINITY][..]] {
            assert_eq!(
                analyze_sti(ir, 48000.0),
                Err(StiError::InvalidImpulseResponse)
            );
        }
        for sr in [0.0, -48000.0, f64::NAN, f64::INFINITY, 1e100] {
            assert_eq!(analyze_sti(&[1.0], sr), Err(StiError::InvalidSampleRate));
        }
        for sr in [16000.0, 22050.0] {
            assert_eq!(
                analyze_sti(&[1.0], sr),
                Err(StiError::InsufficientSampleRate)
            );
        }
    }
}

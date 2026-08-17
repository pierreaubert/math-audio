use super::compute::compute_schroeder_decay;
use super::misc::next_power_of_two;
use super::misc::trim_impulse_to_noise_floor;
use super::plan::plan_fft_forward;
use super::plan::plan_fft_inverse;
use super::types::LagEstimate;
use super::types::Rt60Fit;
use super::types::Rt60FitMethod;
use super::types::fit_rt60_decay;
use rustfft::num_complex::Complex;

const MIN_NORMALIZED_LAG_PEAK: f32 = 0.15;
const MIN_LAG_CONFIDENCE: f32 = 0.05;

/// Estimate lag between reference and recorded signals using cross-correlation
///
/// Uses FFT-based cross-correlation for efficiency
///
/// # Arguments
/// * `reference` - Reference signal
/// * `recorded` - Recorded signal
///
/// # Returns
/// Estimated lag in samples (negative means recorded leads)
pub fn estimate_lag_with_confidence(
    reference: &[f32],
    recorded: &[f32],
) -> Result<LagEstimate, String> {
    if reference.is_empty() || recorded.is_empty() {
        return Err("Reference and recorded signals must be non-empty".to_string());
    }
    if reference
        .iter()
        .chain(recorded)
        .any(|sample| !sample.is_finite())
    {
        return Err("Reference and recorded signals must contain only finite samples".to_string());
    }

    // Zero-pad to avoid circular correlation artifacts
    let fft_size = next_power_of_two(reference.len() + recorded.len() - 1);

    // Correlation must use the original samples. Applying a window changes the
    // matched signal and can erase events at the buffer boundaries.
    let mut ref_fft = vec![Complex::new(0.0_f32, 0.0_f32); fft_size];
    let mut rec_fft = vec![Complex::new(0.0_f32, 0.0_f32); fft_size];
    for (dst, &sample) in ref_fft.iter_mut().zip(reference) {
        dst.re = sample;
    }
    for (dst, &sample) in rec_fft.iter_mut().zip(recorded) {
        dst.re = sample;
    }
    let fft = plan_fft_forward(fft_size);
    fft.process(&mut ref_fft);
    fft.process(&mut rec_fft);

    // Cross-correlation in frequency domain: conj(X) * Y
    let mut cross_corr_fft: Vec<Complex<f32>> = ref_fft
        .iter()
        .zip(rec_fft.iter())
        .map(|(x, y)| x.conj() * y)
        .collect();

    // IFFT to get cross-correlation in time domain
    let ifft = plan_fft_inverse(fft_size);
    ifft.process(&mut cross_corr_fft);
    // rustfft's inverse transform is intentionally unnormalised. Without
    // this scale factor every normalized correlation is multiplied by N and
    // then clamped to one, making the confidence diagnostics meaningless.
    let inverse_scale = 1.0 / fft_size as f32;
    for value in &mut cross_corr_fft {
        *value *= inverse_scale;
    }

    let reference_energy: f32 = reference.iter().map(|sample| sample * sample).sum();
    let recorded_energy: f32 = recorded.iter().map(|sample| sample * sample).sum();
    if reference_energy <= f32::EPSILON || recorded_energy <= f32::EPSILON {
        return Err("Unable to estimate lag: one signal has no energy".to_string());
    }

    // Normalize each valid lag by the energy in its overlap. This makes the
    // peak a correlation coefficient rather than a record-length-dependent
    // amplitude, and lets us distinguish a real match from a noise maximum.
    let mut reference_prefix = vec![0.0_f32; reference.len() + 1];
    for (index, sample) in reference.iter().enumerate() {
        reference_prefix[index + 1] = reference_prefix[index] + sample * sample;
    }
    let mut recorded_prefix = vec![0.0_f32; recorded.len() + 1];
    for (index, sample) in recorded.iter().enumerate() {
        recorded_prefix[index + 1] = recorded_prefix[index] + sample * sample;
    }
    let min_lag = -(reference.len() as isize - 1);
    let max_lag = recorded.len() as isize - 1;
    let mut candidates = Vec::with_capacity(reference.len() + recorded.len() - 1);
    for lag in min_lag..=max_lag {
        let (reference_start, recorded_start, length) = if lag >= 0 {
            let start = lag as usize;
            (0, start, reference.len().min(recorded.len() - start))
        } else {
            let start = lag.unsigned_abs();
            (start, 0, (reference.len() - start).min(recorded.len()))
        };
        if length == 0 {
            continue;
        }
        let ref_energy =
            reference_prefix[reference_start + length] - reference_prefix[reference_start];
        let rec_energy = recorded_prefix[recorded_start + length] - recorded_prefix[recorded_start];
        if ref_energy <= f32::EPSILON || rec_energy <= f32::EPSILON {
            continue;
        }
        let fft_index = if lag >= 0 {
            lag as usize
        } else {
            (fft_size as isize + lag) as usize
        };
        let correlation = cross_corr_fft[fft_index].re.abs();
        let normalized = (correlation / (ref_energy * rec_energy).sqrt()).clamp(0.0, 1.0);
        candidates.push((lag, normalized, length, correlation));
    }

    let Some(&(lag_samples, normalized_peak, peak_overlap, _)) = candidates.iter().max_by(
        |(_, normalized_a, _, correlation_a), (_, normalized_b, _, correlation_b)| {
            normalized_a
                .total_cmp(normalized_b)
                .then_with(|| correlation_a.total_cmp(correlation_b))
        },
    ) else {
        return Err("Unable to estimate lag: no overlapping signal energy".to_string());
    };

    // Ignore a small neighbourhood around the main peak when measuring the
    // sidelobe. A broad matched-filter peak should not fail its own quality
    // check merely because adjacent samples are also large.
    let guard = (reference.len().min(recorded.len()) / 64).max(2) as isize;
    let sidelobe = candidates
        .iter()
        .filter(|(candidate_lag, _, overlap, _)| {
            (*candidate_lag - lag_samples).abs() > guard
                && *overlap >= reference.len().min(recorded.len()) / 2
        })
        .map(|(_, value, _, _)| *value)
        .fold(0.0_f32, f32::max);
    let peak_to_sidelobe_db = if sidelobe <= f32::EPSILON {
        f32::INFINITY
    } else {
        20.0 * (normalized_peak / sidelobe).log10()
    };
    let sidelobe_ratio = if normalized_peak > f32::EPSILON {
        (sidelobe / normalized_peak).min(1.0)
    } else {
        1.0
    };
    // A perfect full-overlap match is itself decisive. Short deterministic
    // probes can have sidelobes nearly as large as the main lobe, so applying
    // the sidelobe penalty to an exact unit correlation would reject valid
    // zero-lag inputs purely because there are too few samples to estimate a
    // meaningful sidelobe floor.
    let full_overlap = peak_overlap == reference.len().min(recorded.len());
    let confidence = if full_overlap && normalized_peak >= 0.999 {
        normalized_peak
    } else {
        (normalized_peak * (1.0 - sidelobe_ratio)).clamp(0.0, 1.0)
    };

    if normalized_peak < MIN_NORMALIZED_LAG_PEAK || confidence < MIN_LAG_CONFIDENCE {
        return Err(format!(
            "Unable to estimate lag confidently: normalized peak {:.3}, confidence {:.3}, PSR {:.1} dB, overlap {} samples",
            normalized_peak, confidence, peak_to_sidelobe_db, peak_overlap
        ));
    }

    Ok(LagEstimate {
        lag_samples,
        normalized_peak,
        peak_to_sidelobe_db,
        confidence,
    })
}

/// In-module lag entry point carrying the confidence diagnostics.
/// [`estimate_lag_with_confidence`] is the public equivalent.
#[allow(dead_code)]
pub(super) fn estimate_lag(reference: &[f32], recorded: &[f32]) -> Result<LagEstimate, String> {
    estimate_lag_with_confidence(reference, recorded)
}

pub(super) fn estimate_rt60_broadband(impulse: &[f32], sample_rate: f32) -> Option<Rt60Fit> {
    if impulse.is_empty() || sample_rate <= 0.0 {
        return None;
    }

    let trimmed = trim_impulse_to_noise_floor(impulse, sample_rate);
    let decay = compute_schroeder_decay(trimmed);
    let decay_db: Vec<f32> = decay
        .iter()
        .map(|value| 10.0 * value.max(1e-12).log10())
        .collect();

    fit_rt60_decay(&decay_db, sample_rate, -5.0, -35.0, Rt60FitMethod::T30)
        .or_else(|| fit_rt60_decay(&decay_db, sample_rate, -5.0, -25.0, Rt60FitMethod::T20))
}

/// Estimate per-bin noise floor in dB from a silence window.
///
/// Takes the pre-silence samples captured before the sweep starts,
/// windows the FFT the same way the sweep analysis does, and returns
/// one dB value per positive-frequency bin (including DC and Nyquist,
/// i.e. `fft_size / 2 + 1` values). Result is reference-to-full-scale
/// (i.e., a silence bin at 0.001 linear amplitude maps to -60 dB).
///
/// The FFT size is taken as `silence.len().next_power_of_two()` so
/// the bin grid matches [`deconvolve_sweep`] when the silence window
/// is the same length as the sweep.
///
/// A Hann window is applied before the FFT to reduce spectral
/// leakage that would otherwise push DC noise into every other bin
/// and make bass SNR look healthier than it is.
pub fn estimate_noise_floor_db_from_silence(silence: &[f32], _sample_rate: u32) -> Vec<f32> {
    if silence.is_empty() {
        return Vec::new();
    }
    let n = silence.len();
    let fft_size = n.next_power_of_two();
    let spectrum_size = fft_size / 2 + 1;

    // Hann-window the silence before FFT.
    let mut buf: Vec<Complex<f32>> = silence
        .iter()
        .enumerate()
        .map(|(k, &s)| {
            let w = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * k as f32 / (n as f32 - 1.0)).cos());
            Complex::new(s * w, 0.0)
        })
        .collect();
    buf.resize(fft_size, Complex::new(0.0, 0.0));

    let fft = plan_fft_forward(fft_size);
    fft.process(&mut buf);

    // Windowed amplitude normalisation for a real sinusoid on a bin
    // centre. The FFT of `sin(2π·m·k/N)` has magnitude `N/2` at bin
    // `m`, and Hann windowing multiplies that by its coherent gain
    // of `0.5` — so the windowed peak is `N/4`. Multiply by `4/N` to
    // recover the underlying sinusoid amplitude (and let
    // `20·log10(mag)` match the tone's dBFS).
    let norm = 4.0 / n as f32;

    buf.into_iter()
        .take(spectrum_size)
        .map(|c| {
            let mag = c.norm() * norm;
            if mag > 1e-20 {
                20.0 * mag.log10()
            } else {
                -400.0 // effectively "nothing"; avoids -inf leaking downstream
            }
        })
        .collect()
}

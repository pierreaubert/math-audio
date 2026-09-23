//! C1–C2 drift-correction primitives: chirp TDOA and two-point clock skew.
//!
//! Consumer: room-EQ capture tooling. Deterministic DSP only — pure over
//! sample slices plus explicit rates; no audio I/O or session state.
//!
//! Capture-protocol constants below mirror `reviews/req-roomeq-capture.md`
//! (absent from this repo, so the assumed values are stated here and every
//! function takes them as explicit parameters): 48 kHz stimulus clock, one
//! logarithmic timing chirp at sweep start and one at sweep end, TDOA read
//! over the 2–8 kHz portion of the chirp where small-room SNR is usable.

use crate::analysis::{plan_fft_forward, plan_fft_inverse};
use rustfft::num_complex::Complex32;

/// Assumed timing-chirp analysis band (Hz). Below ~2 kHz room modes and
/// HVAC noise dominate; above ~8 kHz magnitude-only UMIK-class capsules
/// give no phase calibration and air absorption eats SNR.
pub const TIMING_CHIRP_LO_HZ: f64 = 2000.0;
/// See [`TIMING_CHIRP_LO_HZ`].
pub const TIMING_CHIRP_HI_HZ: f64 = 8000.0;
/// Minimum peak-to-sidelobe ratio for a usable C1 estimate.
pub const MIN_TDOA_CONFIDENCE_DB: f64 = 6.0;
/// Largest plausible crystal drift; larger two-chirp spreads indicate a
/// mis-associated chirp pair, not clock skew.
pub const MAX_PLAUSIBLE_SKEW_PPM: f64 = 5_000.0;

/// Cross-spectrum weighting for [`estimate_chirp_tdoa`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TdoaWeighting {
    /// Plain cross-correlation (matched filter). Optimal for a *known*
    /// sweep in white noise: the sweep's own spectrum shapes the pulse
    /// compression and sidelobes stay low. This is the default.
    #[default]
    Matched,
    /// PHAT (whitened) cross-correlation. Sharper main lobe under
    /// coloured noise or reverberation, but whitening destroys the
    /// sweep's sidelobe suppression, so expect lower `confidence_db`.
    Phat,
}

/// Configuration for [`estimate_chirp_tdoa`].
#[derive(Debug, Clone, Copy)]
pub struct TdoaConfig {
    /// Stimulus-clock sample rate in Hz.
    pub sample_rate_hz: f64,
    /// Lower edge of the analysis band in Hz.
    pub band_lo_hz: f64,
    /// Upper edge of the analysis band in Hz.
    pub band_hi_hz: f64,
    /// Minimum peak-to-sidelobe ratio (dB) for [`TdoaEstimate::valid`].
    pub min_confidence_db: f64,
    /// Cross-spectrum weighting; see [`TdoaWeighting`].
    pub weighting: TdoaWeighting,
}

impl Default for TdoaConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 48_000.0,
            band_lo_hz: TIMING_CHIRP_LO_HZ,
            band_hi_hz: TIMING_CHIRP_HI_HZ,
            min_confidence_db: MIN_TDOA_CONFIDENCE_DB,
            weighting: TdoaWeighting::Matched,
        }
    }
}

/// Fractional-sample delay of `recorded` relative to `reference`.
///
/// Positive `offset_samples` means the recording lags the stimulus (the
/// normal clock-offset case).
#[derive(Debug, Clone, Copy)]
pub struct TdoaEstimate {
    /// Delay of the recording vs the reference, in (fractional) samples.
    pub offset_samples: f64,
    /// Peak-to-sidelobe ratio of the PHAT correlation, in dB.
    pub confidence_db: f64,
    /// False when the inputs are degenerate or confidence is too low.
    pub valid: bool,
}

/// C1: delay between a known timing chirp and its recording.
///
/// Generalized cross-correlation restricted to `config`'s band, plus
/// parabolic peak interpolation for sub-sample resolution. The default
/// [`TdoaWeighting::Matched`] mode is the documented alternative to PHAT
/// the requirements allow: for a known sweep in white noise the matched
/// filter is optimal and preserves pulse-compression sidelobes.
/// Degenerate inputs (empty, all-silent band) yield `valid == false`
/// rather than a panic.
pub fn estimate_chirp_tdoa(
    reference: &[f32],
    recorded: &[f32],
    config: &TdoaConfig,
) -> TdoaEstimate {
    let invalid = TdoaEstimate {
        offset_samples: 0.0,
        confidence_db: f64::NEG_INFINITY,
        valid: false,
    };
    if reference.is_empty()
        || recorded.is_empty()
        || !config.sample_rate_hz.is_finite()
        || config.sample_rate_hz <= 0.0
        || !config.band_lo_hz.is_finite()
        || !config.band_hi_hz.is_finite()
        || config.band_hi_hz <= config.band_lo_hz
        || config.band_lo_hz < 0.0
        || config.band_hi_hz >= config.sample_rate_hz / 2.0
        || !config.min_confidence_db.is_finite()
        || reference
            .iter()
            .chain(recorded)
            .any(|sample| !sample.is_finite())
    {
        return invalid;
    }
    let size = (reference.len() + recorded.len()).next_power_of_two();
    let forward = plan_fft_forward(size);
    let inverse = plan_fft_inverse(size);

    let mut ref_buf: Vec<Complex32> = reference
        .iter()
        .map(|&s| Complex32::new(s, 0.0))
        .chain(std::iter::repeat_n(
            Complex32::new(0.0, 0.0),
            size - reference.len(),
        ))
        .collect();
    let mut rec_buf: Vec<Complex32> = recorded
        .iter()
        .map(|&s| Complex32::new(s, 0.0))
        .chain(std::iter::repeat_n(
            Complex32::new(0.0, 0.0),
            size - recorded.len(),
        ))
        .collect();
    forward.process(&mut ref_buf);
    forward.process(&mut rec_buf);

    // PHAT-weighted cross spectrum, restricted to the usable band.
    let bin_hz = config.sample_rate_hz / size as f64;
    let lo_bin = ((config.band_lo_hz / bin_hz).ceil() as usize).max(1);
    let hi_bin = ((config.band_hi_hz / bin_hz).floor() as usize).min(size / 2);
    if hi_bin <= lo_bin {
        return invalid;
    }
    let mut cross = vec![Complex32::new(0.0, 0.0); size];
    let mut band_energy = 0.0f64;
    for k in lo_bin..=hi_bin {
        // conj(reference) * recorded: with rec[n] = ref[n-D] the cross
        // phase is e^{-j2πkD/N}, so the inverse FFT peaks at +D, i.e. a
        // positive offset means the recording lags the reference.
        let c = rec_buf[k] * ref_buf[k].conj();
        let mag = c.norm() as f64;
        band_energy += mag * mag;
        // Symmetric bins (kept for clarity; the analytic correlation
        // below reads the non-negative half).
        cross[k] = match config.weighting {
            TdoaWeighting::Matched => c,
            TdoaWeighting::Phat => c / Complex32::new((mag + 1e-12) as f32, 0.0),
        };
        if k > 0 && k < size - k {
            cross[size - k] = cross[k].conj();
        }
    }
    if !band_energy.is_finite() || band_energy <= 0.0 {
        return invalid;
    }

    // Analytic correlation (one-sided spectrum): the envelope carries no
    // carrier cusps, so argmax and parabolic interpolation read the true
    // pulse peak. Interpolating the raw magnitude instead misplaces the
    // peak by ~1 sample whenever a carrier zero falls near the top — a
    // systematic bias, not noise.
    let mut analytic = vec![Complex32::new(0.0, 0.0); size];
    analytic[0] = cross[0];
    for k in 1..size / 2 {
        analytic[k] = cross[k] * Complex32::new(2.0, 0.0);
    }
    analytic[size / 2] = cross[size / 2];
    inverse.process(&mut analytic);
    let corr: Vec<f64> = analytic
        .iter()
        .map(|c| (c.norm() / size as f32) as f64)
        .collect();
    let peak = corr
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Peak-to-sidelobe outside the main lobe. The main lobe of a
    // rectangular-band correlation is a sinc with first null at ~1/BW, so
    // the exclusion half-width derives from the analysis bandwidth rather
    // than a fixed bin count; anything narrower mistakes main-lobe skirt
    // for sidelobes.
    let exclusion =
        ((config.sample_rate_hz / (config.band_hi_hz - config.band_lo_hz)).ceil() as usize).max(2);
    let mut sidelobe = 0.0f64;
    for (i, &v) in corr.iter().enumerate() {
        let dist = peak.abs_diff(i).min(size - peak.abs_diff(i));
        if dist > exclusion && v > sidelobe {
            sidelobe = v;
        }
    }
    let confidence_db = 20.0 * (corr[peak] / (sidelobe + 1e-30)).log10();

    // Parabolic interpolation on magnitudes for fractional resolution.
    let prev = corr[(peak + size - 1) % size];
    let next = corr[(peak + 1) % size];
    let denom = prev - 2.0 * corr[peak] + next;
    let delta = if denom.is_finite() && denom.abs() > 1e-12 {
        (0.5 * (prev - next) / denom).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    let mut offset = peak as f64 + delta;
    if offset > size as f64 / 2.0 {
        offset -= size as f64;
    }

    TdoaEstimate {
        offset_samples: offset,
        confidence_db,
        valid: confidence_db.is_finite() && confidence_db >= config.min_confidence_db,
    }
}

/// Per-device clock correction relative to the stimulus clock.
#[derive(Debug, Clone, Copy)]
pub struct ClockSkew {
    /// Offset at the first chirp, in samples (positive = mic lags).
    pub offset_samples: f64,
    /// Microphone samples per stimulus interval, in ppm (positive = faster).
    pub skew_ppm: f64,
    /// False when the estimate must not be used coherently.
    pub valid: bool,
    /// True when either chirp estimate was unusable.
    pub low_confidence: bool,
    /// True when the two-chirp spread is physically implausible.
    pub inconsistent_spacing: bool,
}

/// C2: clock skew from two C1 estimates over a known chirp spacing.
///
/// `chirp_spacing_samples` is the nominal stimulus-clock distance between
/// the start and end chirps. A spread implying more than
/// [`MAX_PLAUSIBLE_SKEW_PPM`] flags `inconsistent_spacing` (mis-associated
/// chirps, not drift) and invalidates the result.
pub fn estimate_clock_skew(
    first: &TdoaEstimate,
    second: &TdoaEstimate,
    chirp_spacing_samples: f64,
) -> ClockSkew {
    let invalid = |low_confidence: bool, inconsistent_spacing: bool| ClockSkew {
        offset_samples: first.offset_samples,
        skew_ppm: 0.0,
        valid: false,
        low_confidence,
        inconsistent_spacing,
    };
    if !chirp_spacing_samples.is_finite() || chirp_spacing_samples <= 0.0 {
        return invalid(false, true);
    }
    if !first.valid || !second.valid {
        return invalid(true, false);
    }
    let skew_ppm = (second.offset_samples - first.offset_samples) / chirp_spacing_samples * 1e6;
    if !skew_ppm.is_finite() || skew_ppm.abs() > MAX_PLAUSIBLE_SKEW_PPM {
        return invalid(false, true);
    }
    ClockSkew {
        offset_samples: first.offset_samples,
        skew_ppm,
        valid: true,
        low_confidence: false,
        inconsistent_spacing: false,
    }
}

/// C4: per-mic post-correction timing uncertainty in microseconds.
///
/// Adjudication input: downstream gates coherent use (pair sums, DOA) on
/// this bound. Returns [`f64::INFINITY`] — never zero — when the bound
/// cannot be computed (either chirp estimate or the skew invalid).
///
/// Model (stated, not hidden): each chirp's TDOA standard deviation follows
/// the delay-estimation Cramér–Rao form
/// `σ_chirp = sr / (2π·β·√(2·SNR))` with `SNR` from the estimate's
/// `confidence_db` (peak-to-sidelobe as SNR proxy — conservative for
/// matched chirps) and rectangular-band RMS bandwidth
/// `β = (band_hi − band_lo)/√12`. Two-point differencing gives
/// `σ_skew = √2·σ_chirp/spacing` (in ppm after scaling); the residual at
/// stimulus time `eval_time_samples` combines the anchor offset variance
/// with the skew-propagated variance as independent terms. All quantities
/// are worst-cased (max of the two chirps' σ), never best-cased. The
/// reported bound is `3·σ_total + 0.5` samples: 3σ coverage so downstream
/// gating rarely false-passes, plus a half-sample estimator-quantisation
/// floor.
/// The returned model envelope is three sigma plus half a sample for peak
/// interpolation. It assumes a linear clock and isolated correctly associated
/// chirps; it cannot certify arbitrary correlated reflections or nonlinear drift.
/// Peak-to-sidelobe confidence is a noise proxy, not a measured noise distribution.
pub fn post_correction_uncertainty_us(
    first: &TdoaEstimate,
    second: &TdoaEstimate,
    skew: &ClockSkew,
    chirp_spacing_samples: f64,
    eval_time_samples: f64,
    config: &TdoaConfig,
) -> f64 {
    if !first.valid
        || !second.valid
        || !skew.valid
        || !first.offset_samples.is_finite()
        || !second.offset_samples.is_finite()
        || !skew.offset_samples.is_finite()
        || !skew.skew_ppm.is_finite()
        || skew.skew_ppm.abs() > MAX_PLAUSIBLE_SKEW_PPM
        || !chirp_spacing_samples.is_finite()
        || chirp_spacing_samples <= 0.0
        || !eval_time_samples.is_finite()
        || !config.sample_rate_hz.is_finite()
        || config.sample_rate_hz <= 0.0
    {
        return f64::INFINITY;
    }
    let bandwidth = config.band_hi_hz - config.band_lo_hz;
    if !bandwidth.is_finite() || bandwidth <= 0.0 {
        return f64::INFINITY;
    }
    let sigma_chirp = |est: &TdoaEstimate| {
        let snr = 10.0f64.powf(est.confidence_db / 10.0);
        if !snr.is_finite() || snr <= 0.0 {
            return f64::INFINITY;
        }
        let beta = bandwidth / 12.0f64.sqrt();
        config.sample_rate_hz / (2.0 * std::f64::consts::PI * beta * (2.0 * snr).sqrt())
    };
    let (s1, s2) = (sigma_chirp(first), sigma_chirp(second));
    if !s1.is_finite() || !s2.is_finite() {
        return f64::INFINITY;
    }
    let sigma_chirp = s1.max(s2);
    let sigma_skew_ratio = std::f64::consts::SQRT_2 * sigma_chirp / chirp_spacing_samples;
    let sigma_total =
        (sigma_chirp * sigma_chirp + (eval_time_samples * sigma_skew_ratio).powi(2)).sqrt();
    if !sigma_total.is_finite() {
        return f64::INFINITY;
    }
    (3.0 * sigma_total + 0.5) / config.sample_rate_hz * 1e6
}

/// C2 extension: validity of a ≥3-chirp drift series.
///
/// Checks every estimate is usable and the per-interval skews are mutually
/// consistent (no sign flips beyond `tol_ppm`: non-monotonic drift) and
/// within [`MAX_PLAUSIBLE_SKEW_PPM`]. Returns the mean skew in ppm and
/// whether the series may be used coherently.
pub fn validate_drift_series(
    estimates: &[TdoaEstimate],
    chirp_spacing_samples: f64,
    tol_ppm: f64,
) -> (f64, bool) {
    if estimates.len() < 3
        || !chirp_spacing_samples.is_finite()
        || chirp_spacing_samples <= 0.0
        || estimates.iter().any(|e| !e.valid)
    {
        return (0.0, false);
    }
    let skews: Vec<f64> = estimates
        .windows(2)
        .map(|w| (w[1].offset_samples - w[0].offset_samples) / chirp_spacing_samples * 1e6)
        .collect();
    let mean = skews.iter().sum::<f64>() / skews.len() as f64;
    let consistent = skews.iter().all(|&s| {
        s.is_finite() && s.abs() <= MAX_PLAUSIBLE_SKEW_PPM && (s - mean).abs() <= tol_ppm
    });
    (mean, consistent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signals::gen_log_sweep;

    /// Fractional delay via Hann-windowed sinc FIR (test oracle, not C3).
    /// FFT-free by construction: integer part by index shift, fractional
    /// part by a 127-tap interpolator (≈ −90 dB error, far below what C1
    /// must resolve). Zero-padded edges.
    fn fractional_delay_fir(input: &[f32], delay_samples: f64) -> Vec<f32> {
        const TAPS: usize = 127;
        const CENTER: i64 = (TAPS / 2) as i64;
        let int = delay_samples.floor() as i64;
        let frac = delay_samples - delay_samples.floor();
        let hann = |i: usize| {
            0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (TAPS - 1) as f64).cos())
        };
        let sinc = |x: f64| {
            if x.abs() < 1e-12 {
                1.0
            } else {
                (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
            }
        };
        let mut taps = vec![0.0f64; TAPS];
        for (i, tap) in taps.iter_mut().enumerate() {
            *tap = sinc(i as f64 - CENTER as f64 - frac) * hann(i);
        }
        let gain: f64 = taps.iter().sum();
        for tap in taps.iter_mut() {
            *tap /= gain;
        }
        let at = |n: i64| {
            if n < 0 || n >= input.len() as i64 {
                0.0
            } else {
                input[n as usize] as f64
            }
        };
        (0..input.len())
            .map(|n| {
                taps.iter()
                    .enumerate()
                    .map(|(i, h)| h * at(n as i64 - int - (i as i64 - CENTER)))
                    .sum::<f64>() as f32
            })
            .collect()
    }

    fn noisy_chirp(delay: f64, noise_amp: f32, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let reference = gen_log_sweep(500.0, 12_000.0, 0.9, 48_000, 1.0);
        let shifted = fractional_delay_fir(&reference, delay);
        let noise = crate::signals::gen_white_noise_seeded(noise_amp, 48_000, 1.0, seed);
        let recorded: Vec<f32> = shifted
            .iter()
            .zip(noise.iter())
            .map(|(s, n)| s + n)
            .collect();
        (reference, recorded)
    }

    #[test]
    fn tdoa_recovers_fractional_delay() {
        let config = TdoaConfig::default();
        for &delay in &[0.0, 37.25, -123.75, 1000.5] {
            let (reference, recorded) = noisy_chirp(delay, 0.02, 0xC10C);
            let est = estimate_chirp_tdoa(&reference, &recorded, &config);
            assert!(est.valid, "delay {delay}: confidence {}", est.confidence_db);
            assert!(
                (est.offset_samples - delay).abs() < 0.05,
                "delay {delay}: got {}",
                est.offset_samples
            );
        }
    }

    #[test]
    fn tdoa_rejects_pure_noise() {
        let config = TdoaConfig::default();
        let reference = gen_log_sweep(500.0, 12_000.0, 0.9, 48_000, 1.0);
        let noise = crate::signals::gen_white_noise_seeded(0.5, 48_000, 1.0, 7);
        let est = estimate_chirp_tdoa(&reference, &noise, &config);
        assert!(!est.valid, "noise must not validate");
    }

    #[test]
    fn tdoa_rejects_degenerate_inputs() {
        let config = TdoaConfig::default();
        assert!(!estimate_chirp_tdoa(&[], &[1.0], &config).valid);
        let bad = TdoaConfig {
            band_lo_hz: 9000.0,
            band_hi_hz: 1000.0,
            ..config
        };
        let reference = gen_log_sweep(500.0, 12_000.0, 0.9, 48_000, 0.1);
        assert!(!estimate_chirp_tdoa(&reference, &reference, &bad).valid);
    }

    #[test]
    fn skew_recovers_known_ppm() {
        // 2.4-sample spread over 10 s @48 kHz = 5 ppm.
        let first = TdoaEstimate {
            offset_samples: 1000.0,
            confidence_db: 20.0,
            valid: true,
        };
        let second = TdoaEstimate {
            offset_samples: 1002.4,
            confidence_db: 20.0,
            valid: true,
        };
        let skew = estimate_clock_skew(&first, &second, 480_000.0);
        assert!(skew.valid);
        assert!((skew.skew_ppm - 5.0).abs() < 1e-9, "got {}", skew.skew_ppm);
        assert!((skew.offset_samples - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn skew_flags_invalid_inputs() {
        let good = TdoaEstimate {
            offset_samples: 10.0,
            confidence_db: 20.0,
            valid: true,
        };
        let bad = TdoaEstimate {
            valid: false,
            ..good
        };
        let s = estimate_clock_skew(&bad, &good, 480_000.0);
        assert!(!s.valid && s.low_confidence && !s.inconsistent_spacing);
        // 48000-sample spread over 10 s = 100 000 ppm: mis-associated pair.
        let far = TdoaEstimate {
            offset_samples: 48_010.0,
            ..good
        };
        let s = estimate_clock_skew(&good, &far, 480_000.0);
        assert!(!s.valid && s.inconsistent_spacing);
        assert!(!estimate_clock_skew(&good, &good, 0.0).valid);
    }

    #[test]
    fn uncertainty_bound_matches_hand_computation() {
        let config = TdoaConfig::default();
        let mk = |conf: f64| TdoaEstimate {
            offset_samples: 0.0,
            confidence_db: conf,
            valid: true,
        };
        let skew = ClockSkew {
            offset_samples: 0.0,
            skew_ppm: 5.0,
            valid: true,
            low_confidence: false,
            inconsistent_spacing: false,
        };
        // Hand computation of the documented model: conf 20/14 dB,
        // 10 s spacing @48 kHz, evaluated at the end chirp.
        let snr_worst = 10.0f64.powf(14.0 / 10.0);
        let beta = 6000.0 / 12.0f64.sqrt();
        let s = 48_000.0 / (2.0 * std::f64::consts::PI * beta * (2.0 * snr_worst).sqrt());
        let ss = std::f64::consts::SQRT_2 * s / 480_000.0;
        let expected = (3.0 * (s * s + (480_000.0 * ss).powi(2)).sqrt() + 0.5) / 48_000.0 * 1e6;
        let got = post_correction_uncertainty_us(
            &mk(20.0),
            &mk(14.0),
            &skew,
            480_000.0,
            480_000.0,
            &config,
        );
        assert!((got - expected).abs() < 1e-9, "got {got}, want {expected}");
        assert!(got > 0.0 && got < 100.0, "plausible magnitude: {got} µs");
    }

    #[test]
    fn uncertainty_is_infinite_when_uncomputable() {
        let config = TdoaConfig::default();
        let good = TdoaEstimate {
            offset_samples: 0.0,
            confidence_db: 20.0,
            valid: true,
        };
        let bad_est = TdoaEstimate {
            valid: false,
            ..good
        };
        let skew = ClockSkew {
            offset_samples: 0.0,
            skew_ppm: 0.0,
            valid: true,
            low_confidence: false,
            inconsistent_spacing: false,
        };
        let bad_skew = ClockSkew {
            valid: false,
            ..skew
        };
        assert!(
            post_correction_uncertainty_us(&bad_est, &good, &skew, 480_000.0, 0.0, &config)
                .is_infinite()
        );
        assert!(
            post_correction_uncertainty_us(&good, &good, &bad_skew, 480_000.0, 0.0, &config)
                .is_infinite()
        );
        assert!(
            post_correction_uncertainty_us(&good, &good, &skew, 0.0, 0.0, &config).is_infinite()
        );
        // Infinite reads as infinite — never as zero.
        let inf = post_correction_uncertainty_us(&good, &good, &bad_skew, 480_000.0, 0.0, &config);
        assert!(inf.is_infinite() && inf != 0.0);
    }

    #[test]
    fn drift_series_detects_non_monotonic() {
        let mk = |o: f64| TdoaEstimate {
            offset_samples: o,
            confidence_db: 20.0,
            valid: true,
        };
        let steady = [mk(0.0), mk(2.4), mk(4.8)];
        let (ppm, ok) = validate_drift_series(&steady, 480_000.0, 1.0);
        assert!(ok && (ppm - 5.0).abs() < 1e-9);
        // Direction flip mid-series: non-monotonic drift.
        let flip = [mk(0.0), mk(2.4), mk(0.5)];
        assert!(!validate_drift_series(&flip, 480_000.0, 1.0).1);
        assert!(!validate_drift_series(&steady[..2], 480_000.0, 1.0).1);
    }
}

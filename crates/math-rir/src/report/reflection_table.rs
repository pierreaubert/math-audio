//! Band-limited early-reflection table (report primitive M1).
//!
//! Feeds roomeq report section R1: a 1–8 kHz bandpassed envelope/peak pick
//! of reflections arriving < 15 ms after the direct sound, with gains vs
//! the direct sound (dBFS), time→distance conversion, first-dip frequency
//! `f = c / 2Δd`, and comb ripple from the relative gain.
//!
//! Conventions (stated next to the code per the cross-cutting requirement):
//! - Sound speed `c = 343 m/s` ([`REPORT_SOUND_SPEED_M_S`]).
//! - Bandpass: zero-phase Butterworth highpass at
//!   [`REFLECTION_BAND_LO_HZ`] cascaded with a lowpass at
//!   [`REFLECTION_BAND_HI_HZ`], each of order [`REFLECTION_FILTER_ORDER`]
//!   per side (`filtfilt` doubles the effective order, no group delay).
//!   Edges are clamped to `[1 Hz, 0.99 · Nyquist]` so the helper stays
//!   usable at any roomeq sample rate ≥ ~20 kHz.
//! - Direct-sound reference: the broadband SSIR direct TOA
//!   (`find_direct_sound_toa`); the direct peak is the maximum `|bandpassed|`
//!   within ±[`DIRECT_SEARCH_MS`] of that TOA. Gains are
//!   `20·log10(peak / direct_peak)` (dB, 0 dB = direct).
//! - Peak picking: envelope/peak — local maxima of a centred moving-average
//!   envelope of `|bandpassed|` ([`ENVELOPE_SMOOTH_MS`] wide) in
//!   `(direct, direct + window]` above `threshold_db` below the direct peak,
//!   with a minimum separation of `min_separation_ms` (greedy, loudest
//!   first), each refined to the nearest `|bandpassed|` maximum for timing
//!   (refinement never reaches back into the direct exclusion, and the
//!   separation gate applies to refined positions). A fixed prominence
//!   gate ([`ENVELOPE_PROMINENCE_DB`] over [`PROMINENCE_VALLEY_MS`])
//!   rejects smoothing-ripple crests on the decay tail before the
//!   greedy pass.
//! - Distance: extra path length `Δd = c·Δt`; first comb dip at
//!   `f = c / 2Δd = 1 / 2Δt`; peak-to-peak comb ripple for a relative
//!   linear gain `g` is `20·log10((1+g)/(1-g))`.

use crate::SsirConfig;
use crate::bands::BandpassWorkspace;
use crate::detection::find_direct_sound_toa;

/// Sound speed used for all time→distance conversions (m/s).
pub const REPORT_SOUND_SPEED_M_S: f64 = 343.0;

/// Lower edge of the reflection-picking bandpass (Hz).
pub const REFLECTION_BAND_LO_HZ: f64 = 1000.0;

/// Upper edge of the reflection-picking bandpass (Hz).
pub const REFLECTION_BAND_HI_HZ: f64 = 8000.0;

/// Butterworth order per side (highpass leg and lowpass leg) before the
/// forward-reverse pass doubles the effective order.
pub const REFLECTION_FILTER_ORDER: usize = 4;

/// Half-width of the direct-peak search around the broadband TOA (ms).
pub const DIRECT_SEARCH_MS: f64 = 1.0;

/// Samples before the direct peak excluded from reflection picking (ms).
/// Avoids re-picking the direct lobe / filter ring.
pub const DIRECT_EXCLUSION_MS: f64 = 0.5;

/// Default peak-picking threshold: reflections must be within this many dB
/// below the direct peak.
pub const DEFAULT_THRESHOLD_DB: f64 = 30.0;

/// Default minimum separation between picked reflections (ms).
pub const DEFAULT_MIN_SEPARATION_MS: f64 = 0.5;

/// Default post-direct picking window (ms).
pub const DEFAULT_WINDOW_MS: f64 = 15.0;

/// Envelope smoothing width (ms, full width): the envelope is a centred
/// moving average of `|bandpassed|` over this width — wide enough to cover
/// two carrier periods at the 1 kHz band edge (a moving maximum would leave
/// a staircase whose step edges fake peaks), narrow enough that the
/// subsequent `|bandpassed|` refinement recovers timing. Reflections closer
/// than ~1 ms merge in the envelope (resolution limit ≈ 1 / band edge).
pub const ENVELOPE_SMOOTH_MS: f64 = 1.0;

/// Minimum envelope prominence (dB): a candidate must rise this far above
/// the valley to its left. Smoothing leaves ±2 % ripple whose crests on a
/// decay tail have < 0.5 dB prominence; real reflections rise tens of dB
/// out of the tail. Fixed guard, not a config knob.
pub const ENVELOPE_PROMINENCE_DB: f64 = 6.0;

/// Prominence valley horizon (ms): the valley search extends left at most
/// this far. It must exceed one reflection bump width (~1.5 ms for the
/// 1–8 kHz band + smoothing) so merged bumps see the true tail valley;
/// the search stops early at higher ground, so ripple troughs still bound
/// ripple crests.
pub const PROMINENCE_VALLEY_MS: f64 = 3.0;

/// Configuration for [`early_reflection_table`].
#[derive(Debug, Clone, Copy)]
pub struct ReflectionTableConfig {
    /// Reflections this many dB (or less) below the direct peak are kept.
    pub threshold_db: f64,
    /// Minimum separation between picked reflections (ms).
    pub min_separation_ms: f64,
    /// Post-direct picking window `(0, window_ms]` (ms).
    pub window_ms: f64,
}

impl Default for ReflectionTableConfig {
    fn default() -> Self {
        Self {
            threshold_db: DEFAULT_THRESHOLD_DB,
            min_separation_ms: DEFAULT_MIN_SEPARATION_MS,
            window_ms: DEFAULT_WINDOW_MS,
        }
    }
}

/// One picked early reflection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarlyReflection {
    /// Delay after the direct peak (ms).
    pub delay_ms: f64,
    /// Gain relative to the direct peak (dB, ≤ 0).
    pub gain_db: f64,
    /// Extra path length `c·Δt` (m).
    pub path_difference_m: f64,
    /// First comb dip `c / 2Δd = 1 / 2Δt` (Hz).
    pub first_dip_hz: f64,
    /// Peak-to-peak comb ripple `20·log10((1+g)/(1-g))` (dB).
    pub comb_ripple_db: f64,
}

/// Result of [`early_reflection_table`].
#[derive(Debug, Clone)]
pub struct ReflectionTable {
    /// Sample index of the direct peak in the bandpassed signal.
    pub direct_sample: usize,
    /// Linear peak amplitude of the direct sound (bandpassed).
    pub direct_peak: f32,
    /// Picked reflections, ordered by ascending delay.
    pub reflections: Vec<EarlyReflection>,
}

/// Bandpass an IR to the 1–8 kHz reflection-picking band.
///
/// Zero-phase Butterworth HP(1 kHz) ∘ LP(8 kHz), order 4 per side.
/// Band edges are clamped to `[1 Hz, 0.99 · Nyquist]` so the helper works
/// at any roomeq sample rate. Returns the input unchanged when the band
/// is degenerate (sample rate too low) or the input is empty.
pub fn reflection_bandpass(rir: &[f32], sample_rate: f64) -> Vec<f32> {
    if rir.is_empty() || sample_rate <= 0.0 {
        return rir.to_vec();
    }
    let nyquist = sample_rate * 0.5;
    let f_low = REFLECTION_BAND_LO_HZ.max(1.0);
    let f_high = REFLECTION_BAND_HI_HZ.min(nyquist * 0.99);
    if f_high <= f_low {
        return rir.to_vec();
    }
    let mut coeffs = math_audio_iir_fir::filtfilt::peq_to_coefficients(
        &math_audio_iir_fir::peq_butterworth_highpass(REFLECTION_FILTER_ORDER, f_low, sample_rate),
    );
    coeffs.extend(math_audio_iir_fir::filtfilt::peq_to_coefficients(
        &math_audio_iir_fir::peq_butterworth_lowpass(REFLECTION_FILTER_ORDER, f_high, sample_rate),
    ));
    let input: Vec<f64> = rir.iter().map(|&s| f64::from(s)).collect();
    math_audio_iir_fir::filtfilt::filtfilt(&input, &coeffs)
        .into_iter()
        .map(|v| v as f32)
        .collect()
}

/// Pick band-limited early reflections from an IR.
///
/// `config` controls threshold, minimum separation, and window (see the
/// module docs for conventions). Empty input (or no detectable direct
/// sound) yields an empty table with `direct_sample = 0`.
#[allow(clippy::cast_precision_loss)]
pub fn early_reflection_table(
    rir: &[f32],
    sample_rate: f64,
    config: &ReflectionTableConfig,
) -> ReflectionTable {
    let empty = ReflectionTable {
        direct_sample: 0,
        direct_peak: 0.0,
        reflections: Vec::new(),
    };
    if rir.is_empty() || sample_rate <= 0.0 || config.window_ms <= 0.0 {
        return empty;
    }

    let cfg = SsirConfig::new(sample_rate);
    let toa = match find_direct_sound_toa(rir, &cfg) {
        Some(t) => t,
        None => return empty,
    };

    let filtered = reflection_bandpass(rir, sample_rate);
    if filtered.len() != rir.len() {
        return empty;
    }

    // Direct reference: max |bandpassed| within ±1 ms of the broadband TOA.
    let search = (DIRECT_SEARCH_MS * sample_rate / 1000.0).round() as usize;
    let lo = toa.saturating_sub(search);
    let hi = (toa + search + 1).min(filtered.len());
    let mut direct_sample = toa;
    let mut direct_peak = 0.0f32;
    for (i, &s) in filtered[lo..hi].iter().enumerate() {
        let a = s.abs();
        if a > direct_peak {
            direct_peak = a;
            direct_sample = lo + i;
        }
    }
    if direct_peak <= 0.0 {
        return empty;
    }

    let threshold_lin = 10.0f64.powf(-config.threshold_db / 20.0) * f64::from(direct_peak);
    let min_sep = ((config.min_separation_ms * sample_rate / 1000.0).round() as usize).max(1);
    let exclusion = (DIRECT_EXCLUSION_MS * sample_rate / 1000.0).round() as usize;
    let win_end = (direct_sample as f64 + config.window_ms * sample_rate / 1000.0).round() as usize;
    let win_end = win_end.min(filtered.len().saturating_sub(1));
    let win_start = (direct_sample + exclusion).min(filtered.len());

    // Envelope/peak pick: a centred moving average of |bandpassed| over
    // ENVELOPE_SMOOTH_MS smooths carrier sidelobes (which peak-picking
    // |signal| directly would report as reflections — and which a moving
    // maximum would turn into a staircase of fake step-edge peaks), then
    // each envelope peak is refined to the nearest |bandpassed| maximum.
    let env_w = ((ENVELOPE_SMOOTH_MS * sample_rate / 1000.0).round() as usize).max(1);
    let env_half = env_w / 2;
    let rect: Vec<f64> = filtered.iter().map(|s| f64::from(s.abs())).collect();
    // Prefix sums for the O(n) boxcar.
    let mut prefix = vec![0.0f64; rect.len() + 1];
    for (i, &v) in rect.iter().enumerate() {
        prefix[i + 1] = prefix[i] + v;
    }
    let boxcar = |i: usize| -> f64 {
        let a = i.saturating_sub(env_half);
        let b = (i + env_half + 1).min(rect.len());
        (prefix[b] - prefix[a]) / (b - a).max(1) as f64
    };
    let mut envelope = vec![0.0f64; filtered.len()];
    let e0 = win_start.saturating_sub(env_half);
    let e1 = (win_end + env_half).min(filtered.len());
    for (i, slot) in envelope.iter_mut().enumerate().take(e1).skip(e0) {
        *slot = boxcar(i);
    }
    // Local maxima of the envelope (plateau → first index) above threshold.
    let mut candidates: Vec<(usize, f64)> = Vec::new();
    if win_end > win_start + 1 {
        for i in win_start..win_end {
            let a = envelope[i];
            if a < threshold_lin {
                continue;
            }
            let left = if i > 0 { envelope[i - 1] } else { 0.0 };
            let right = if i + 1 < envelope.len() {
                envelope[i + 1]
            } else {
                0.0
            };
            if a >= left && a > right {
                candidates.push((i, a));
            }
        }
    }
    // Prominence gate: drop envelope ripple crests (a candidate must rise
    // ENVELOPE_PROMINENCE_DB above the valley in the min-separation window
    // to its left). Loudest first, greedy minimum-separation acceptance.
    // Each candidate is refined to the |bandpassed| maximum first (clamped
    // so refinement never reaches back into the direct exclusion); the
    // separation gate applies to the refined positions.
    candidates.retain(|&(idx, amp)| {
        // Topographic prominence: expand left to higher ground (at most
        // PROMINENCE_VALLEY_MS); the valley is the minimum over that
        // interval. Ripple crests meet higher ground at the previous
        // crest, so their valley is the adjacent trough (~0 dB
        // prominence); real reflections expand past the tail minimum.
        let valley_w = ((PROMINENCE_VALLEY_MS * sample_rate / 1000.0).round() as usize).max(1);
        let v0 = idx.saturating_sub(valley_w).max(win_start);
        let mut j = idx;
        while j > v0 && envelope[j - 1] <= amp {
            j -= 1;
        }
        if j >= idx {
            // Candidate at the exclusion edge with no room to establish
            // prominence against the direct lobe.
            return false;
        }
        let mut valley = f64::INFINITY;
        for &v in &envelope[j..idx] {
            if v < valley {
                valley = v;
            }
        }
        if !valley.is_finite() {
            return false;
        }
        if valley <= 0.0 {
            // Silent valley: unbounded prominence.
            return true;
        }
        20.0 * (amp / valley).log10() >= ENVELOPE_PROMINENCE_DB
    });
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut accepted: Vec<(usize, f32)> = Vec::new();
    for (idx, _) in candidates {
        let a = idx.saturating_sub(env_half).max(win_start);
        let b = (idx + env_half + 1).min(filtered.len());
        if a >= b {
            continue;
        }
        let mut best = a;
        let mut best_a = rect[a];
        for (k, &v) in rect[a..b].iter().enumerate() {
            if v > best_a {
                best_a = v;
                best = a + k;
            }
        }
        if accepted.iter().all(|&(j, _)| best.abs_diff(j) >= min_sep) {
            accepted.push((best, best_a as f32));
        }
    }
    accepted.sort_by_key(|&(idx, _)| idx);

    let direct_f = f64::from(direct_peak);
    let reflections = accepted
        .into_iter()
        .map(|(idx, amp)| {
            let dt_s = idx.saturating_sub(direct_sample) as f64 / sample_rate;
            let g = f64::from(amp) / direct_f;
            let path = REPORT_SOUND_SPEED_M_S * dt_s;
            let first_dip = if dt_s > 0.0 {
                1.0 / (2.0 * dt_s)
            } else {
                f64::INFINITY
            };
            let ripple = if g < 1.0 {
                20.0 * ((1.0 + g) / (1.0 - g)).log10()
            } else {
                f64::INFINITY
            };
            EarlyReflection {
                delay_ms: dt_s * 1000.0,
                gain_db: 20.0 * g.log10(),
                path_difference_m: path,
                first_dip_hz: first_dip,
                comb_ripple_db: ripple,
            }
        })
        .collect();

    ReflectionTable {
        direct_sample,
        direct_peak,
        reflections,
    }
}

/// Reusable workspace variant: mirrors [`early_reflection_table`] but keeps
/// the bandpass coefficient cache across calls.
pub fn early_reflection_table_with_workspace(
    rir: &[f32],
    sample_rate: f64,
    config: &ReflectionTableConfig,
    _ws: &mut BandpassWorkspace,
) -> ReflectionTable {
    // The wide 1–8 kHz bandpass uses its own coefficient set (not the
    // per-centre cache), so this currently delegates; the workspace
    // parameter keeps the batched call-site shape stable.
    early_reflection_table(rir, sample_rate, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_rir(sample_rate: f64, direct_at: usize, reflections: &[(usize, f32)]) -> Vec<f32> {
        let n = direct_at + (20.0 * sample_rate / 1000.0) as usize;
        // Ring each spike through the picking band so the bandpassed
        // envelope has resolvable local maxima: modulate with a 2 kHz
        // carrier (inside 1–8 kHz) decaying over ~1 ms. Cosine phase puts
        // each burst's peak exactly at its start sample, so planted delays
        // are exact (a sine start would hide a quarter-period ambiguity).
        let mut out = vec![0.0f32; n];
        let omega = 2.0 * std::f64::consts::PI * 2000.0 / sample_rate;
        let decay = (-1.0 / (0.001 * sample_rate)).exp();
        for &(off, amp) in std::iter::once(&(0usize, 1.0f32)).chain(reflections.iter()) {
            let start = direct_at + off;
            let mut env = f64::from(amp);
            let mut i = start;
            while i < n && env > 1e-4 {
                out[i] += (env * ((i - start) as f64 * omega).cos()) as f32;
                env *= decay;
                i += 1;
            }
        }
        out
    }

    #[test]
    fn picks_known_reflection_time_gain_dip() {
        let sr = 48000.0;
        let delay_samples = (5.0 * sr / 1000.0) as usize; // 5 ms
        let rir = synthetic_rir(sr, 480, &[(delay_samples, 0.5)]);
        let cfg = ReflectionTableConfig::default();
        let table = early_reflection_table(&rir, sr, &cfg);
        assert!(
            !table.reflections.is_empty(),
            "expected a picked reflection"
        );
        let r = &table.reflections[0];
        assert!(
            (r.delay_ms - 5.0).abs() < 0.25,
            "delay_ms = {} (expected ≈ 5)",
            r.delay_ms
        );
        assert!(
            (r.gain_db - (-6.02)).abs() < 1.0,
            "gain_db = {} (expected ≈ −6)",
            r.gain_db
        );
        // first dip: 1 / 2Δt = 100 Hz for Δt = 5 ms.
        assert!(
            (r.first_dip_hz - 100.0).abs() < 5.0,
            "first_dip_hz = {}",
            r.first_dip_hz
        );
        // path difference: 343 · 0.005 = 1.715 m.
        assert!(
            (r.path_difference_m - 1.715).abs() < 0.05,
            "path = {}",
            r.path_difference_m
        );
        // ripple for g = 0.5: 20·log10(3) ≈ 9.54 dB.
        assert!(
            (r.comb_ripple_db - 9.54).abs() < 0.6,
            "ripple = {}",
            r.comb_ripple_db
        );
    }

    #[test]
    fn delta_only_yields_no_reflections() {
        let sr = 48000.0;
        let mut rir = vec![0.0f32; 4800];
        rir[480] = 1.0;
        let table = early_reflection_table(&rir, sr, &ReflectionTableConfig::default());
        assert!(table.reflections.is_empty());
        assert_eq!(table.direct_sample, 480, "direct should be the delta");
    }

    #[test]
    fn empty_input_yields_empty_table() {
        let table = early_reflection_table(&[], 48000.0, &ReflectionTableConfig::default());
        assert!(table.reflections.is_empty());
    }

    #[test]
    fn silent_ir_yields_empty_table() {
        // All zeros: no direct sound detectable.
        let table = early_reflection_table(
            &vec![0.0f32; 4800],
            48000.0,
            &ReflectionTableConfig::default(),
        );
        assert!(table.reflections.is_empty());
        assert_eq!(table.direct_sample, 0);
    }

    #[test]
    fn degenerate_band_returns_input_unchanged() {
        // Sample rate too low for the 1–8 kHz band: passthrough.
        let input = vec![0.25f32, -0.5, 0.75];
        assert_eq!(reflection_bandpass(&input, 1000.0), input);
        assert_eq!(reflection_bandpass(&[], 48000.0), Vec::<f32>::new());
        assert_eq!(reflection_bandpass(&input, 0.0), input);
    }

    #[test]
    fn workspace_variant_matches_direct() {
        let sr = 48000.0;
        let rir = synthetic_rir(sr, 480, &[(240, 0.5)]);
        let cfg = ReflectionTableConfig::default();
        let mut ws = crate::bands::BandpassWorkspace::new();
        let a = early_reflection_table(&rir, sr, &cfg);
        let b = early_reflection_table_with_workspace(&rir, sr, &cfg, &mut ws);
        assert_eq!(a.direct_sample, b.direct_sample);
        assert_eq!(a.reflections.len(), b.reflections.len());
        for (x, y) in a.reflections.iter().zip(b.reflections.iter()) {
            assert!((x.delay_ms - y.delay_ms).abs() < 1e-9);
            assert!((x.gain_db - y.gain_db).abs() < 1e-6);
        }
    }

    #[test]
    fn close_pair_keeps_loudest_only() {
        // Two reflections 0.42 ms apart (< 0.5 ms minimum separation,
        // inside one envelope bump): the quieter one (−20 dB, still above
        // the picking threshold) must not survive as a separate entry,
        // and the merged timing/gain must follow the dominant burst.
        let sr = 48000.0;
        let rir = synthetic_rir(sr, 480, &[(240, 0.5), (260, 0.1)]);
        let table = early_reflection_table(&rir, sr, &ReflectionTableConfig::default());
        assert_eq!(table.reflections.len(), 1, "pair must merge");
        assert!((table.reflections[0].delay_ms - 5.0).abs() < 0.3);
        assert!((table.reflections[0].gain_db - (-6.02)).abs() < 1.5);
    }

    #[test]
    fn hotter_reflection_reports_positive_gain_infinite_ripple() {
        // Reflection stronger than the direct sound: g > 1 → the comb
        // formula has a pole, reported as +∞ ripple.
        let sr = 48000.0;
        let rir = synthetic_rir(sr, 480, &[(240, 1.5)]);
        let table = early_reflection_table(&rir, sr, &ReflectionTableConfig::default());
        assert!(!table.reflections.is_empty());
        let r = &table.reflections[0];
        assert!(r.gain_db > 3.0, "gain_db = {}", r.gain_db);
        assert!(
            r.comb_ripple_db.is_infinite(),
            "ripple = {}",
            r.comb_ripple_db
        );
        assert!(r.first_dip_hz.is_finite());
    }

    #[test]
    fn short_ir_clamps_window_at_signal_end() {
        // IR ends 14.5 ms after the direct sound: the picking window must
        // clamp instead of overrunning, and still find the 5 ms reflection.
        // The tail is faded to zero so the truncation itself does not ring
        // through the bandpass as a fake reflection.
        let sr = 48000.0;
        let mut rir = synthetic_rir(sr, 480, &[(240, 0.5)]);
        rir.truncate(480 + (14.5 * sr / 1000.0) as usize);
        let fade = (1.0 * sr / 1000.0) as usize;
        let n = rir.len();
        for (k, s) in rir[n - fade..].iter_mut().enumerate() {
            let t = k as f64 / fade as f64;
            *s = (*s as f64 * 0.5 * (1.0 + (std::f64::consts::PI * t).cos())) as f32;
        }
        let table = early_reflection_table(&rir, sr, &ReflectionTableConfig::default());
        assert_eq!(table.reflections.len(), 1);
        assert!((table.reflections[0].delay_ms - 5.0).abs() < 0.25);
    }
}

//! Batched octave-band T60 (report primitive M3).
//!
//! Feeds roomeq report section R3: one call maps an IR + sample rate to
//! nine octave-band T60 values (63 Hz–16 kHz) plus fit quality, reusing
//! [`crate::bands::analyze_iso3382_bands`].
//!
//! Conventions (stated next to the code per the cross-cutting requirement):
//! - Centres [`T60_OCTAVE_CENTERS_HZ`] = 63 … 16 000 Hz (9 bands).
//! - Each band is zero-phase Butterworth-filtered with
//!   [`T60_FILTER_ORDER`] = 4 per side before the Schroeder fit.
//! - Fit policy: T30 when its fit range exists and `r² ≥ min_r2`,
//!   else T20 under the same rule, else EDT; the chosen range is reported.
//! - Noise-cutoff policy: automatic per band via `estimate_noise_cutoff`
//!   (Chu seed + Lundeby refinement) inside the Schroeder curve.
//! - Validity: bands above Nyquist are skipped by the dispatcher and
//!   reported as invalid with reason `"above-nyquist"`; short/filtered IRs
//!   whose fit is missing or below `min_r2` are reported invalid with the
//!   fit range and `r²` that caused the rejection; fits that only resolve
//!   the filter ring (`B·T60 < 8`) are flagged `"filter-ring"`.

use crate::bands::{BandWidth, analyze_iso3382_bands};

/// Octave centres for the batched T60 (Hz): 63 Hz … 16 kHz, 9 bands.
pub const T60_OCTAVE_CENTERS_HZ: [f64; 9] = [
    63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0,
];

/// Butterworth order per side for each octave bandpass.
pub const T60_FILTER_ORDER: usize = 4;

/// Minimum bandwidth–reverberation-time product (`B·T60`) for validity.
///
/// A Schroeder fit can look clean (`r² ≥ min_r2`) while measuring only the
/// bandpass filter's own ring — e.g. a 50 ms IR reports T30 ≈ 61 ms at
/// 63 Hz with `r² = 0.97`. ISO 3382-grade practice requires `B·T ≫ 1`;
/// bands whose fitted T60 is not at least this many filter bandwidths long
/// are flagged `"filter-ring"`. The octave fractional bandwidth is
/// `2^(1/2) − 2^(−1/2) ≈ 0.707`, so at 63 Hz this rejects fitted decays
/// below ≈ 180 ms — the filter, not the room.
pub const MIN_T60_BANDWIDTH_PRODUCT: f64 = 8.0;

/// Octave-band fractional bandwidth (`f_high/f_c − f_low/f_c`).
pub const OCTAVE_FRACTIONAL_BW: f64 = std::f64::consts::SQRT_2 - std::f64::consts::FRAC_1_SQRT_2;

/// Default minimum fit `r²` for a T60 value to count as valid.
pub const DEFAULT_MIN_R2: f64 = 0.90;

/// Which Schroeder fit range produced a band T60.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum T60FitRange {
    T30,
    T20,
    Edt,
    None,
}

/// Configuration for [`analyze_t60_octaves`].
#[derive(Debug, Clone, Copy)]
pub struct T60BatchConfig {
    /// Minimum fit `r²` for validity.
    pub min_r2: f64,
    /// Butterworth order per side for each octave bandpass.
    pub filter_order: usize,
}

impl Default for T60BatchConfig {
    fn default() -> Self {
        Self {
            min_r2: DEFAULT_MIN_R2,
            filter_order: T60_FILTER_ORDER,
        }
    }
}

/// One octave band of the batched T60 result.
#[derive(Debug, Clone, PartialEq)]
pub struct OctaveT60 {
    /// Octave centre frequency (Hz).
    pub centre_hz: f64,
    /// Selected T60 value (s); NaN when invalid.
    pub t60_s: f64,
    /// Fit range that produced `t60_s`.
    pub fit_range: T60FitRange,
    /// `r²` of the selected fit (0.0 when invalid).
    pub r2: f64,
    /// Whether the value passed the fit-range + `min_r2` policy.
    pub valid: bool,
    /// Machine-readable reason when `!valid` (e.g. `"above-nyquist"`,
    /// `"no-fit"`, `"low-r2(Isnan)"`); empty when valid.
    pub reason: String,
}

/// Run octave-band Schroeder/T60 over 63 Hz–16 kHz in one call.
///
/// Returns one [`OctaveT60`] per centre in [`T60_OCTAVE_CENTERS_HZ`].
/// Bands the dispatcher drops (centre outside `[0, Nyquist]`, e.g. the
/// 16 kHz octave at 44.1 kHz) are still returned, flagged invalid with
/// reason `"above-nyquist"`, so callers always see all 9 bands — including
/// short/filtered IRs such as roomeq `pre_ir`/`post_ir`.
pub fn analyze_t60_octaves(
    rir: &[f32],
    sample_rate: f64,
    config: &T60BatchConfig,
) -> Vec<OctaveT60> {
    if rir.is_empty() || sample_rate <= 0.0 {
        return T60_OCTAVE_CENTERS_HZ
            .iter()
            .map(|&fc| OctaveT60 {
                centre_hz: fc,
                t60_s: f64::NAN,
                fit_range: T60FitRange::None,
                r2: 0.0,
                valid: false,
                reason: "empty-input".to_string(),
            })
            .collect();
    }

    let results = analyze_iso3382_bands(
        rir,
        sample_rate,
        &T60_OCTAVE_CENTERS_HZ,
        BandWidth::Octave,
        config.filter_order,
    );
    let by_fc: std::collections::HashMap<u64, _> = results
        .into_iter()
        .map(|(fc, m)| (fc.to_bits(), m))
        .collect();

    T60_OCTAVE_CENTERS_HZ
        .iter()
        .map(|&fc| {
            let Some(m) = by_fc.get(&fc.to_bits()) else {
                return OctaveT60 {
                    centre_hz: fc,
                    t60_s: f64::NAN,
                    fit_range: T60FitRange::None,
                    r2: 0.0,
                    valid: false,
                    reason: "above-nyquist".to_string(),
                };
            };
            // Fit-range policy: T30 → T20 → EDT, first finite value
            // whose r² clears min_r2.
            let chain = [
                (T60FitRange::T30, m.t30_s, m.t30_r2),
                (T60FitRange::T20, m.t20_s, m.t20_r2),
                (T60FitRange::Edt, m.edt_s, m.edt_r2),
            ];
            let mut chosen: Option<(T60FitRange, f64, f64)> = None;
            let mut best_fallback: Option<(T60FitRange, f64, f64)> = None;
            for (range, v, r2) in chain {
                if v.is_finite() && v > 0.0 {
                    if best_fallback.is_none() {
                        best_fallback = Some((range, v, r2));
                    }
                    if r2 >= config.min_r2 {
                        chosen = Some((range, v, r2));
                        break;
                    }
                }
            }
            match chosen {
                Some((range, v, r2)) => {
                    // Bandwidth–T60 guard: reject fits that only resolve the
                    // filter's own ring.
                    let product = v * fc * OCTAVE_FRACTIONAL_BW;
                    if product >= MIN_T60_BANDWIDTH_PRODUCT {
                        OctaveT60 {
                            centre_hz: fc,
                            t60_s: v,
                            fit_range: range,
                            r2,
                            valid: true,
                            reason: String::new(),
                        }
                    } else {
                        OctaveT60 {
                            centre_hz: fc,
                            t60_s: f64::NAN,
                            fit_range: range,
                            r2,
                            valid: false,
                            reason: format!("filter-ring(BT={product:.2})"),
                        }
                    }
                }
                None => {
                    let (range, r2) = best_fallback
                        .map(|(r, _, r2)| (r, r2))
                        .unwrap_or((T60FitRange::None, 0.0));
                    OctaveT60 {
                        centre_hz: fc,
                        t60_s: f64::NAN,
                        fit_range: range,
                        r2,
                        valid: false,
                        reason: if range == T60FitRange::None {
                            "no-fit".to_string()
                        } else {
                            format!("low-r2({range:?},r2={r2:.3})")
                        },
                    }
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic exponential-decay IR with known T60, in a mid band.
    fn decay_rir(sample_rate: f64, t60_s: f64, duration_s: f64, band_hz: f64) -> Vec<f32> {
        let n = (duration_s * sample_rate) as usize;
        let alpha = std::f64::consts::LN_10 * 6.0 / (2.0 * t60_s);
        // Band-limited-ish noise: sum of sines around the band centre with
        // random-ish phases from an LCG, under an exponential envelope.
        let mut rir = vec![0.0f32; n];
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next_rand = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 32) as i32 as f64) / (i32::MAX as f64)
        };
        let partials = [-0.5, -0.25, 0.0, 0.25, 0.5];
        let phases: Vec<f64> = (0..partials.len())
            .map(|_| next_rand() * 2.0 * std::f64::consts::PI)
            .collect();
        for (i, sample) in rir.iter_mut().enumerate() {
            let t = i as f64 / sample_rate;
            let mut v = 0.0;
            for (k, &off) in partials.iter().enumerate() {
                let f = band_hz * 2f64.powf(off / 2.0);
                v += (2.0 * std::f64::consts::PI * f * t + phases[k]).sin();
            }
            *sample = (v / partials.len() as f64 * (-alpha * t).exp()) as f32;
        }
        rir[0] += 1.0;
        rir
    }

    #[test]
    fn batch_reports_nine_bands_with_fit_quality() {
        let sr = 48000.0;
        let rir = decay_rir(sr, 0.6, 3.0, 1000.0);
        let out = analyze_t60_octaves(&rir, sr, &T60BatchConfig::default());
        assert_eq!(out.len(), 9, "must always return all 9 bands");
        // Mid bands should be valid and near the planted T60.
        let mut any_valid = false;
        for band in &out {
            if [500.0, 1000.0, 2000.0].contains(&band.centre_hz) {
                assert!(
                    band.valid,
                    "band {} Hz should be valid (reason: {})",
                    band.centre_hz, band.reason
                );
                any_valid = true;
                assert!(
                    (band.t60_s - 0.6).abs() < 0.25,
                    "band {} Hz: T60 = {:.3} (expected ≈ 0.6)",
                    band.centre_hz,
                    band.t60_s
                );
                assert!(band.r2 >= DEFAULT_MIN_R2);
            }
        }
        assert!(any_valid);
    }

    #[test]
    fn short_ir_flags_bands_invalid() {
        // 50 ms of decay: no band can fit a 60 dB decay.
        let sr = 48000.0;
        let rir = decay_rir(sr, 0.6, 0.05, 1000.0);
        let out = analyze_t60_octaves(&rir, sr, &T60BatchConfig::default());
        assert_eq!(out.len(), 9);
        assert!(
            out.iter().all(|b| !b.valid),
            "short IR must flag every band invalid"
        );
        assert!(out.iter().all(|b| !b.reason.is_empty()));
    }

    #[test]
    fn empty_and_silent_input_yield_nine_invalid() {
        for rir in [vec![], vec![0.0f32; 4800]] {
            let out = analyze_t60_octaves(&rir, 48000.0, &T60BatchConfig::default());
            assert_eq!(out.len(), 9, "must always return all 9 bands");
            for band in &out {
                assert!(!band.valid);
                assert!(!band.reason.is_empty());
                assert!(band.t60_s.is_nan());
            }
        }
    }

    #[test]
    fn low_sample_rate_flags_16k_above_nyquist() {
        let sr = 22050.0;
        let rir = decay_rir(sr, 0.5, 2.0, 1000.0);
        let out = analyze_t60_octaves(&rir, sr, &T60BatchConfig::default());
        let top = out.last().unwrap();
        assert_eq!(top.centre_hz, 16000.0);
        assert!(!top.valid);
        assert_eq!(top.reason, "above-nyquist");
    }
}

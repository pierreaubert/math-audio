//! Early/late split + 1/3-octave smoothed SPL (report primitive M2).
//!
//! Feeds roomeq report section R2: a 20 ms split of the IR energy into
//! early/late responses, FFT → 1/3-octave smoothed SPL, plus a subwoofer
//! path with a 120 Hz lowpass envelope-peak reference.
//!
//! Conventions (stated next to the code per the cross-cutting requirement):
//! - Split: [`EARLY_LATE_SPLIT_MS`] = 20 ms after the direct reference.
//!   [`split_early_late`] takes the direct sample explicitly; the `_auto`
//!   variant locates it with [`envelope_peak`] (max of the `|x|` envelope
//!   smoothed over [`ENVELOPE_WINDOW_MS`]).
//! - Subwoofer path: second-order Butterworth lowpass at
//!   [`SUB_LOWPASS_HZ`] = 120 Hz (forward only; phase is irrelevant for an
//!   envelope peak), then [`envelope_peak`] as the time reference.
//! - Third-octave SPL: new fixed-centre variant. Unlike
//!   [`crate::analysis::smooth_response_f64`] — a fractional-octave sliding
//!   average on the caller's grid in the caller's domain —
//!   [`third_octave_spl`] aggregates FFT power into the ISO third-octave
//!   centres [`THIRD_OCTAVE_CENTERS_HZ`] (energy domain) and reports dB
//!   relative to each segment's own broadband peak (0 dB = loudest band).
//!   FFT uses a Hann window, next-pow-2 size, power normalised by the FFT
//!   length so early/late segments of different lengths stay comparable.

use math_audio_iir_fir::{Biquad, BiquadFilterType};
use rustfft::{FftPlanner, num_complex::Complex};

/// Early/late energy split after the direct reference (ms).
pub const EARLY_LATE_SPLIT_MS: f64 = 20.0;

/// Envelope smoothing window for peak detection (ms).
pub const ENVELOPE_WINDOW_MS: f64 = 0.5;

/// Subwoofer-path lowpass corner (Hz).
pub const SUB_LOWPASS_HZ: f64 = 120.0;

/// ISO third-octave centres used for the fixed-centre SPL (Hz).
pub const THIRD_OCTAVE_CENTERS_HZ: [f64; 21] = [
    100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0,
    2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0,
];

/// dB floor for bands with no measurable energy.
pub const SPL_FLOOR_DB: f64 = -120.0;

/// Maximum FFT bin width (Hz): the FFT is zero-padded so `df ≤ 5 Hz`,
/// giving every third-octave band (narrowest ≈ 23 Hz at 100 Hz) several
/// bins. Coarser bins would leave narrow bands empty and pinned to the
/// floor even when energy is present.
pub const MAX_SPL_BIN_HZ: f64 = 5.0;

/// One third-octave SPL band.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThirdOctaveSpl {
    /// Band centre (Hz).
    pub centre_hz: f64,
    /// Early-segment level (dB rel. early broadband peak).
    pub early_db: f64,
    /// Late-segment level (dB rel. late broadband peak).
    pub late_db: f64,
}

/// Envelope-peak time reference: index of the maximum of `|x|` smoothed
/// with a [`ENVELOPE_WINDOW_MS`] moving average. Returns `(index, value)`.
/// Empty input → `(0, 0.0)`.
#[allow(clippy::cast_precision_loss)]
pub fn envelope_peak(signal: &[f32], sample_rate: f64) -> (usize, f32) {
    if signal.is_empty() || sample_rate <= 0.0 {
        return (0, 0.0);
    }
    let win = ((ENVELOPE_WINDOW_MS * sample_rate / 1000.0).round() as usize).max(1);
    let mut best_idx = 0usize;
    let mut best_val = 0.0f32;
    // Causal moving average of |x|; argmax is the envelope peak.
    let mut acc = 0.0f64;
    for (i, &s) in signal.iter().enumerate() {
        acc += f64::from(s.abs());
        if i >= win {
            acc -= f64::from(signal[i - win].abs());
        }
        let env = (acc / (win.min(i + 1) as f64)) as f32;
        if env > best_val {
            best_val = env;
            best_idx = i;
        }
    }
    (best_idx, best_val)
}

/// Subwoofer-path reference: second-order Butterworth lowpass at 120 Hz,
/// then [`envelope_peak`]. Returns `(index, envelope_value)`.
pub fn sub_lowpass_envelope_peak(ir: &[f32], sample_rate: f64) -> (usize, f32) {
    if ir.is_empty() || sample_rate <= 0.0 {
        return (0, 0.0);
    }
    let mut lp = Biquad::new(
        BiquadFilterType::Lowpass,
        SUB_LOWPASS_HZ,
        sample_rate,
        std::f64::consts::FRAC_1_SQRT_2,
        0.0,
    );
    let filtered: Vec<f32> = ir.iter().map(|&s| lp.process(s as f64) as f32).collect();
    envelope_peak(&filtered, sample_rate)
}

/// Split an IR into early/late segments around an explicit direct sample.
///
/// Early = `[direct, direct + split)` (clamped to the IR end);
/// late = `[direct + split, end)`. `split_ms` defaults to
/// [`EARLY_LATE_SPLIT_MS`] when non-positive. A `direct_sample` past the
/// end yields two empty segments.
pub fn split_early_late(
    ir: &[f32],
    direct_sample: usize,
    sample_rate: f64,
    split_ms: f64,
) -> (Vec<f32>, Vec<f32>) {
    if ir.is_empty() || sample_rate <= 0.0 || direct_sample >= ir.len() {
        return (Vec::new(), Vec::new());
    }
    let split = if split_ms > 0.0 {
        split_ms
    } else {
        EARLY_LATE_SPLIT_MS
    };
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let n_early = (split * sample_rate / 1000.0).round() as usize;
    let end_early = (direct_sample + n_early).min(ir.len());
    (
        ir[direct_sample..end_early].to_vec(),
        ir[end_early..].to_vec(),
    )
}

/// Split with an automatic [`envelope_peak`] direct reference.
pub fn split_early_late_auto(ir: &[f32], sample_rate: f64, split_ms: f64) -> (Vec<f32>, Vec<f32>) {
    let (direct, _) = envelope_peak(ir, sample_rate);
    split_early_late(ir, direct, sample_rate, split_ms)
}

fn hann(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()))
        .collect()
}

fn segment_band_power(segment: &[f32], sample_rate: f64) -> Vec<(f64, f64)> {
    if segment.is_empty() {
        return Vec::new();
    }
    // Zero-pad so df ≤ MAX_SPL_BIN_HZ (interpolation, not resolution —
    // the interpolated bins partition band energy correctly).
    let min_n = ((sample_rate / MAX_SPL_BIN_HZ).ceil() as usize)
        .next_power_of_two()
        .max(16);
    let n_fft = segment.len().next_power_of_two().max(16).max(min_n);
    let window = hann(segment.len());
    let mut buf: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_fft];
    let mut wsum = 0.0;
    for (i, &s) in segment.iter().enumerate() {
        buf[i].re = f64::from(s) * window[i];
        wsum += window[i] * window[i];
    }
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    fft.process(&mut buf);
    let df = sample_rate / n_fft as f64;
    // One-sided power spectrum, normalised by FFT length × window energy
    // so segments of different lengths stay comparable.
    let norm = n_fft as f64 * wsum.max(1e-30);
    let mut bands = vec![0.0f64; THIRD_OCTAVE_CENTERS_HZ.len()];
    let half = n_fft / 2;
    for (k, bin) in buf.iter().enumerate().take(half + 1) {
        let f = k as f64 * df;
        let mut p = (bin.norm_sqr()) / norm;
        if k > 0 && k < half {
            p *= 2.0; // fold negative frequencies
        }
        // Nearest-centre assignment in log frequency (fixed-centre variant).
        if f <= 0.0 {
            continue;
        }
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for (b, &fc) in THIRD_OCTAVE_CENTERS_HZ.iter().enumerate() {
            let d = (f.ln() - fc.ln()).abs();
            if d < best_d {
                best_d = d;
                best = b;
            }
        }
        // Only count bins inside the band's third-octave skirts.
        let edge = 2f64.powf(1.0 / 6.0);
        if f >= THIRD_OCTAVE_CENTERS_HZ[best] / edge && f <= THIRD_OCTAVE_CENTERS_HZ[best] * edge {
            bands[best] += p;
        }
    }
    THIRD_OCTAVE_CENTERS_HZ.iter().copied().zip(bands).collect()
}

/// FFT → fixed-centre 1/3-octave smoothed SPL for an early/late pair.
///
/// Each segment is levelled to its own broadband peak (0 dB = loudest
/// band), so the two curves compare spectral shape, not absolute energy.
/// Empty segments yield [`SPL_FLOOR_DB`] in every band.
pub fn third_octave_spl(early: &[f32], late: &[f32], sample_rate: f64) -> Vec<ThirdOctaveSpl> {
    fn to_db(bands: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let peak = bands.iter().map(|&(_, p)| p).fold(0.0f64, f64::max);
        bands
            .iter()
            .map(|&(fc, p)| {
                let db = if peak > 0.0 && p > 0.0 {
                    10.0 * (p / peak).log10()
                } else {
                    SPL_FLOOR_DB
                };
                (fc, db.max(SPL_FLOOR_DB))
            })
            .collect()
    }
    let e = to_db(&segment_band_power(early, sample_rate));
    let l = to_db(&segment_band_power(late, sample_rate));
    THIRD_OCTAVE_CENTERS_HZ
        .iter()
        .enumerate()
        .map(|(i, &fc)| ThirdOctaveSpl {
            centre_hz: fc,
            early_db: e.get(i).map_or(SPL_FLOOR_DB, |&(_, db)| db),
            late_db: l.get(i).map_or(SPL_FLOOR_DB, |&(_, db)| db),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decay_ir(sr: f64, t60: f64, dur: f64, hf_damping: f64) -> Vec<f32> {
        // Exponential noise decay; high bands decay faster (realistic room).
        let n = (dur * sr) as usize;
        let mut out = vec![0.0f32; n];
        let mut state: u64 = 0xABCD_1234_5678_9ABC;
        let mut lp = 0.0f64;
        for (i, s) in out.iter_mut().enumerate() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = ((state >> 32) as i32 as f64) / (i32::MAX as f64);
            lp += hf_damping * (noise - lp);
            let t = i as f64 / sr;
            let alpha = std::f64::consts::LN_10 * 3.0 / t60;
            *s = (lp * (-alpha * t).exp()) as f32;
        }
        out[0] += 1.0;
        out
    }

    #[test]
    fn envelope_peak_finds_delta() {
        let sr = 48000.0;
        let mut ir = vec![0.0f32; 4800];
        ir[123] = 1.0;
        let (idx, val) = envelope_peak(&ir, sr);
        assert!(
            (idx as i64 - 123).abs() <= 24,
            "peak at {idx}, expected ≈ 123"
        );
        assert!(val > 0.0);
    }

    #[test]
    fn split_twenty_ms_lengths() {
        let sr = 48000.0;
        let ir = decay_ir(sr, 0.5, 1.0, 0.2);
        let (early, late) = split_early_late(&ir, 0, sr, 20.0);
        assert_eq!(early.len(), (0.02 * sr) as usize);
        assert_eq!(early.len() + late.len(), ir.len());
    }

    #[test]
    fn third_octave_shape_early_vs_late() {
        // Early = 10 ms of decaying broadband noise (peak in the upper
        // bands); late = 100 Hz sine tail (peak at 100 Hz, HF at floor).
        // The two curves must show clearly different shapes.
        let sr = 48000.0;
        let mut early = vec![0.0f32; (0.01 * sr) as usize];
        let mut state: u64 = 0x1111_2222_3333_4444;
        for (i, s) in early.iter_mut().enumerate() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = ((state >> 32) as i32 as f64) / (i32::MAX as f64);
            let t = i as f64 / sr;
            *s = (noise * (-t * 50.0).exp()) as f32;
        }
        early[0] += 1.0;
        let mut late = vec![0.0f32; (0.5 * sr) as usize];
        for (i, s) in late.iter_mut().enumerate() {
            let t = i as f64 / sr;
            *s = (0.5 * (-t * 3.0).exp() * (2.0 * std::f64::consts::PI * 100.0 * t).sin()) as f32;
        }
        let spl = third_octave_spl(&early, &late, sr);
        assert_eq!(spl.len(), THIRD_OCTAVE_CENTERS_HZ.len());
        // Each curve normalises its own peak to 0 dB.
        let epeak = spl
            .iter()
            .map(|b| b.early_db)
            .fold(f64::NEG_INFINITY, f64::max);
        let lpeak = spl
            .iter()
            .map(|b| b.late_db)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((epeak - 0.0).abs() < 1e-9, "early peak = {epeak}");
        assert!((lpeak - 0.0).abs() < 1e-9, "late peak = {lpeak}");
        let early_peak_band = spl
            .iter()
            .max_by(|a, b| a.early_db.total_cmp(&b.early_db))
            .unwrap()
            .centre_hz;
        let late_peak_band = spl
            .iter()
            .max_by(|a, b| a.late_db.total_cmp(&b.late_db))
            .unwrap()
            .centre_hz;
        assert!(
            early_peak_band >= 1000.0,
            "broadband early should peak high, got {early_peak_band}"
        );
        assert!(
            late_peak_band <= 125.0,
            "100 Hz late tail should peak low, got {late_peak_band}"
        );
        let late_hf = spl.iter().find(|b| b.centre_hz == 8000.0).unwrap().late_db;
        assert!(
            late_hf <= -60.0,
            "sine tail must be at floor by 8 kHz, got {late_hf}"
        );
    }

    #[test]
    fn auto_split_matches_explicit_on_delta() {
        // The auto variant must locate the delta and split identically to
        // an explicit call with that reference.
        let sr = 48000.0;
        let mut ir = vec![0.0f32; (sr * 0.5) as usize];
        ir[100] = 1.0;
        let (auto_e, auto_l) = split_early_late_auto(&ir, sr, 20.0);
        let (exp_e, exp_l) = split_early_late(&ir, 100, sr, 20.0);
        assert_eq!(auto_e, exp_e);
        assert_eq!(auto_l, exp_l);
        assert_eq!(auto_e.len(), (0.02 * sr) as usize);
    }

    #[test]
    fn nonpositive_split_falls_back_to_default() {
        let sr = 48000.0;
        let ir = decay_ir(sr, 0.5, 1.0, 0.2);
        for split in [0.0, -5.0] {
            let (early, late) = split_early_late(&ir, 0, sr, split);
            assert_eq!(early.len(), (EARLY_LATE_SPLIT_MS * sr / 1000.0) as usize);
            assert_eq!(early.len() + late.len(), ir.len());
        }
    }

    #[test]
    fn empty_segments_and_references() {
        // Documented degenerate contracts: empty in → floor/zero out.
        let sr = 48000.0;
        assert_eq!(envelope_peak(&[], sr), (0, 0.0));
        assert_eq!(envelope_peak(&[1.0], 0.0), (0, 0.0));
        assert_eq!(sub_lowpass_envelope_peak(&[], sr), (0, 0.0));
        let (e, l) = split_early_late(&[1.0, 2.0], 99, sr, 20.0);
        assert!(e.is_empty() && l.is_empty());
        let spl = third_octave_spl(&[], &[], sr);
        assert_eq!(spl.len(), THIRD_OCTAVE_CENTERS_HZ.len());
        for b in &spl {
            assert_eq!(b.early_db, SPL_FLOOR_DB);
            assert_eq!(b.late_db, SPL_FLOOR_DB);
        }
    }

    #[test]
    fn sub_lowpass_reference_on_lf_burst() {
        let sr = 48000.0;
        let n = (sr * 0.2) as usize;
        let mut ir = vec![0.0f32; n];
        // 60 Hz burst at sample 2400 (inside the 120 Hz lowpass).
        for i in 0..2400 {
            let t = i as f64 / sr;
            ir[2400 + i % (n - 2400)] +=
                (0.5 * (-t * 20.0).exp() * (2.0 * std::f64::consts::PI * 60.0 * t).sin()) as f32;
        }
        let (idx, _) = sub_lowpass_envelope_peak(&ir, sr);
        assert!((idx as i64 - 2400).abs() < 480, "sub reference at {idx}");
    }
}

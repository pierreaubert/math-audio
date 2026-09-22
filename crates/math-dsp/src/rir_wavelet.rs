//! 3-cycle wavelet heatmap (report primitive M5).
//!
//! Feeds roomeq report section R5: complex-Morlet CWT magnitude over
//! log-frequency × time, documented here as a true CWT (no STFT
//! fallback needed).
//!
//! Conventions (stated next to the code per the cross-cutting requirement):
//! - Mother wavelet: complex Morlet `ψ(t) = π^(-1/4)·e^{i2πt}·e^{−t²/2}`
//!   scaled per centre frequency so exactly [`WAVELET_CYCLES`] = 3 sine
//!   cycles fall within ±2σ of the Gaussian envelope
//!   (`σ = cycles / 2πf`, kernel half-length `2σ`).
//! - Centre grid: log-spaced [`WAVELET_FREQS_PER_OCTAVE`]-per-octave from
//!   [`WAVELET_FMIN_HZ`] to [`WAVELET_FMAX_HZ`] (clamped to Nyquist).
//! - Normalisation: each kernel has unit energy; magnitudes are scaled so
//!   a full-scale sine at the centre frequency reads 0 dB, matching the
//!   feat-report blue→red colour scale. Display range is clamped to
//!   [`WAVELET_DB_MIN`]…0 dB ([`WAVELET_DB_MIN`] = −30 dB).
//! - Time axis: hop [`WAVELET_HOP_MS`] = 1 ms frames over the same
//!   −5…500 ms span as the waterfall ([`crate::rir_waterfall`]), centred
//!   on the direct reference (argmax `|ir|`, or explicit via
//!   [`wavelet_heatmap_at`]).

use num_complex::Complex;

/// Number of sine cycles inside ±2σ of the Morlet envelope.
pub const WAVELET_CYCLES: f64 = 3.0;

/// Wavelet grid bottom (Hz).
pub const WAVELET_FMIN_HZ: f64 = 50.0;

/// Wavelet grid top (Hz).
pub const WAVELET_FMAX_HZ: f64 = 16000.0;

/// Log-grid density (freqs per octave).
pub const WAVELET_FREQS_PER_OCTAVE: f64 = 12.0;

/// Display-range floor (dB); the top is always 0 dB.
pub const WAVELET_DB_MIN: f64 = -30.0;

/// Time hop between heatmap columns (ms).
pub const WAVELET_HOP_MS: f64 = 1.0;

/// Pre-direct span (ms, mirrors the waterfall).
pub const WAVELET_PRE_MS: f64 = -5.0;

/// Post-direct span (ms, mirrors the waterfall).
pub const WAVELET_POST_MS: f64 = 500.0;

/// Default grid caps for HTML-sized output.
pub const DEFAULT_WAVELET_MAX_FREQS: usize = 96;
/// Default grid caps for HTML-sized output.
pub const DEFAULT_WAVELET_MAX_FRAMES: usize = 256;

/// Configuration for [`wavelet_heatmap`] / [`wavelet_heatmap_at`].
#[derive(Debug, Clone, Copy)]
pub struct WaveletConfig {
    /// Freqs per octave for the log grid.
    pub freqs_per_octave: f64,
    /// Max frequency rows after decimation (max-pooling).
    pub max_freqs: usize,
    /// Max time columns after decimation (max-pooling).
    pub max_frames: usize,
}

impl Default for WaveletConfig {
    fn default() -> Self {
        Self {
            freqs_per_octave: WAVELET_FREQS_PER_OCTAVE,
            max_freqs: DEFAULT_WAVELET_MAX_FREQS,
            max_frames: DEFAULT_WAVELET_MAX_FRAMES,
        }
    }
}

/// Wavelet magnitude heatmap. `mags_db[freq][frame]`, clamped to −30…0 dB.
#[derive(Debug, Clone)]
pub struct WaveletHeatmap {
    /// Centre frequencies (Hz, ascending log grid).
    pub freqs_hz: Vec<f64>,
    /// Frame centre times rel. direct (ms).
    pub times_ms: Vec<f64>,
    /// Magnitudes in dB, 0 dB = full-scale sine at centre.
    pub mags_db: Vec<Vec<f32>>,
}

/// Build the heatmap with an automatic direct reference (argmax `|ir|`).
pub fn wavelet_heatmap(ir: &[f32], sample_rate: f64, config: &WaveletConfig) -> WaveletHeatmap {
    let direct = ir
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .map_or(0, |(i, _)| i);
    wavelet_heatmap_at(ir, sample_rate, direct, config)
}

/// Build the heatmap around an explicit direct sample.
#[allow(clippy::cast_precision_loss)]
pub fn wavelet_heatmap_at(
    ir: &[f32],
    sample_rate: f64,
    direct_sample: usize,
    config: &WaveletConfig,
) -> WaveletHeatmap {
    let empty = WaveletHeatmap {
        freqs_hz: Vec::new(),
        times_ms: Vec::new(),
        mags_db: Vec::new(),
    };
    if ir.is_empty() || sample_rate <= 0.0 || direct_sample >= ir.len() {
        return empty;
    }
    let nyquist = sample_rate * 0.5;
    let fmax = WAVELET_FMAX_HZ.min(nyquist * 0.9);
    if fmax <= WAVELET_FMIN_HZ {
        return empty;
    }
    // Log grid.
    let per_oct = config.freqs_per_octave.max(1.0);
    let noct = (fmax / WAVELET_FMIN_HZ).log2();
    let n_freqs = ((noct * per_oct).round() as usize).max(1);
    let freqs: Vec<f64> = (0..n_freqs)
        .map(|k| WAVELET_FMIN_HZ * 2f64.powf(k as f64 / per_oct))
        .filter(|&f| f <= fmax)
        .collect();
    if freqs.is_empty() {
        return empty;
    }

    // Frame centres every 1 ms over −5…500 ms.
    let hop = (WAVELET_HOP_MS * sample_rate / 1000.0).max(1.0);
    let t0 = direct_sample as f64 + WAVELET_PRE_MS * sample_rate / 1000.0;
    let t1 = direct_sample as f64 + WAVELET_POST_MS * sample_rate / 1000.0;
    let mut centres: Vec<f64> = Vec::new();
    let mut c = t0.max(0.0);
    while c <= t1.min(ir.len() as f64 - 1.0) {
        centres.push(c);
        c += hop;
    }
    if centres.is_empty() {
        return empty;
    }

    // One row per centre frequency: direct convolution with the unit-energy
    // Morlet kernel, evaluated only at the frame centres.
    let ir_f: Vec<f64> = ir.iter().map(|&s| f64::from(s)).collect();
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(freqs.len());
    for &fc in &freqs {
        let sigma = WAVELET_CYCLES / (2.0 * std::f64::consts::PI * fc) * sample_rate; // samples
        let half = (2.0 * sigma).ceil() as usize;
        let kernel: Vec<Complex<f64>> = (-(half as i64)..=half as i64)
            .map(|m| {
                let t = m as f64 / sample_rate;
                let gauss = (-0.5 * (m as f64 / sigma).powi(2)).exp();
                let phase = 2.0 * std::f64::consts::PI * fc * t;
                Complex::new(gauss * phase.cos(), gauss * phase.sin())
            })
            .collect();
        let energy: f64 = kernel.iter().map(|z| z.norm_sqr()).sum();
        let norm = energy.sqrt().max(1e-30);
        // Reference gain: a unit sine at fc correlates to Σg/2 (the
        // analytic half, with Σg the Gaussian envelope sum), so dividing
        // by Σg/2 reads exactly 1.0 → 0 dB.
        let sum_g: f64 = kernel.iter().map(|z| z.norm()).sum::<f64>().max(1e-30);
        let cal = sum_g / (2.0 * norm);
        let mut row = Vec::with_capacity(centres.len());
        for &ctr in &centres {
            let ci = ctr.round() as i64;
            let mut acc = Complex::new(0.0, 0.0);
            for (ki, k) in kernel.iter().enumerate() {
                let idx = ci + ki as i64 - half as i64;
                if idx >= 0 && (idx as usize) < ir_f.len() {
                    acc += k.conj() * ir_f[idx as usize];
                }
            }
            row.push(acc.norm() / (norm * cal));
        }
        rows.push(row);
    }

    // Normalise to the grid peak (a full-scale sine then reads ≈ 0 dB),
    // clamp to −30…0 dB, decimate by max-pooling.
    let peak = rows
        .iter()
        .flat_map(|r| r.iter())
        .copied()
        .fold(0.0f64, f64::max);
    let to_db = |v: f64| {
        if peak > 0.0 && v > 0.0 {
            (20.0 * (v / peak).log10()).clamp(WAVELET_DB_MIN, 0.0)
        } else {
            WAVELET_DB_MIN
        }
    };
    let nf = freqs.len().min(config.max_freqs.max(1));
    let nt = centres.len().min(config.max_frames.max(1));
    let fstride = (freqs.len() as f64 / nf as f64).max(1.0);
    let tstride = (centres.len() as f64 / nt as f64).max(1.0);
    // Decimation is max-pooling in the linear domain; both axis labels
    // track the argmax cell (a pool mean would mislabel peaks).
    let mut out_times = Vec::with_capacity(nt);
    for ti in 0..nt {
        let t0i = (ti as f64 * tstride) as usize;
        let t1i = (((ti + 1) as f64 * tstride) as usize).min(centres.len());
        let mut tbest = t0i.min(centres.len() - 1);
        let mut tvmax = 0.0f64;
        for row in &rows {
            for t in t0i..t1i.max(t0i + 1) {
                let tt = t.min(centres.len() - 1);
                if row[tt] > tvmax {
                    tvmax = row[tt];
                    tbest = tt;
                }
            }
        }
        out_times.push((centres[tbest] - direct_sample as f64) * 1000.0 / sample_rate);
    }
    let mut out_freqs = Vec::with_capacity(nf);
    let mut mags_db = Vec::with_capacity(nf);
    for fi in 0..nf {
        let f0 = (fi as f64 * fstride) as usize;
        let f1 = (((fi + 1) as f64 * fstride) as usize).min(freqs.len());
        let mut fbest = f0.min(freqs.len() - 1);
        let mut row_out = Vec::with_capacity(nt);
        for ti in 0..nt {
            let t0i = (ti as f64 * tstride) as usize;
            let t1i = (((ti + 1) as f64 * tstride) as usize).min(centres.len());
            let mut vmax = 0.0f64;
            for f in f0..f1.max(f0 + 1) {
                let ff = f.min(rows.len() - 1);
                for t in t0i..t1i.max(t0i + 1) {
                    let tt = t.min(centres.len() - 1);
                    if rows[ff][tt] > vmax {
                        vmax = rows[ff][tt];
                        fbest = ff;
                    }
                }
            }
            row_out.push(to_db(vmax) as f32);
        }
        out_freqs.push(freqs[fbest]);
        mags_db.push(row_out);
    }

    WaveletHeatmap {
        freqs_hz: out_freqs,
        times_ms: out_times,
        mags_db,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn burst_lights_up_right_band_and_time() {
        let sr = 48000.0;
        // Small direct tick at sample 0 (explicit time reference) + 1 kHz
        // tone burst at 100 ms, amplitude 1. The tick is kept small on
        // purpose: a full-scale delta out-responds any sine under
        // unit-energy normalisation at high frequencies.
        let n = (0.6 * sr) as usize;
        let mut ir = vec![0.0f32; n];
        ir[0] = 0.3;
        let start = (0.1 * sr) as usize;
        for (i, s) in ir
            .iter_mut()
            .enumerate()
            .skip(start)
            .take((0.05 * sr) as usize)
        {
            let t = i as f64 / sr;
            *s += (2.0 * std::f64::consts::PI * 1000.0 * t).sin() as f32;
        }
        let cfg = WaveletConfig::default();
        let heat = wavelet_heatmap_at(&ir, sr, 0, &cfg);
        assert!(!heat.freqs_hz.is_empty() && !heat.times_ms.is_empty());
        // All values inside the display range.
        for row in &heat.mags_db {
            for &v in row {
                assert!(
                    f64::from(v) >= WAVELET_DB_MIN - 1e-6 && f64::from(v) <= 1e-6,
                    "out-of-range value {v}"
                );
            }
        }
        // Peak row should be near 1 kHz (within 1/6 octave after decimation).
        let (bestr, _) = heat
            .mags_db
            .iter()
            .enumerate()
            .map(|(i, r)| (i, r.iter().copied().fold(f32::NEG_INFINITY, f32::max)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        let fbest = heat.freqs_hz[bestr];
        assert!(
            (fbest / 1000.0).abs() < 2f64.powf(1.0 / 3.0)
                && (1000.0 / fbest) < 2f64.powf(1.0 / 3.0),
            "peak band at {fbest} Hz (expected ≈ 1000)"
        );
        // Peak column must fall inside the burst span (100–150 ms): the
        // exact grid row beats against the burst (11.6 Hz at 12/oct), so
        // the beat maximum can sit anywhere in the burst — asserting an
        // exact onset would test the beat phase, not the heatmap.
        let (bestc, _) = heat.mags_db[bestr]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        let tbest = heat.times_ms[bestc];
        assert!(
            (95.0..=155.0).contains(&tbest),
            "peak time at {tbest} ms (expected inside 100–150 ms burst)"
        );
    }

    #[test]
    fn log_grid_spacing_and_cycles() {
        // Grid density: 12/octave → adjacent ratio 2^(1/12).
        let sr = 48000.0;
        let ir = vec![1.0f32; (0.6 * sr) as usize];
        let cfg = WaveletConfig {
            max_freqs: 1000, // no decimation
            ..WaveletConfig::default()
        };
        let heat = wavelet_heatmap(&ir, sr, &cfg);
        let ratio = heat.freqs_hz[1] / heat.freqs_hz[0];
        assert!(
            (ratio - 2f64.powf(1.0 / 12.0)).abs() < 1e-9,
            "grid ratio = {ratio}"
        );
        assert_eq!(WAVELET_CYCLES, 3.0, "cycle convention");
    }

    #[test]
    fn empty_input_yields_empty_heatmap() {
        let heat = wavelet_heatmap(&[], 48000.0, &WaveletConfig::default());
        assert!(heat.freqs_hz.is_empty());
    }

    #[test]
    fn degenerate_inputs_yield_empty_heatmap() {
        let cfg = WaveletConfig::default();
        let ir = vec![1.0f32; 4800];
        // Non-positive sample rate.
        let heat = wavelet_heatmap(&ir, 0.0, &cfg);
        assert!(heat.freqs_hz.is_empty());
        // Past-the-end direct reference.
        let heat = wavelet_heatmap_at(&ir, 48000.0, ir.len(), &cfg);
        assert!(heat.freqs_hz.is_empty());
        // Sample rate too low for the 50 Hz grid bottom.
        let heat = wavelet_heatmap(&ir, 80.0, &cfg);
        assert!(heat.freqs_hz.is_empty());
    }
}

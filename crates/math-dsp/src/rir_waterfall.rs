//! Waterfall decay grids + 60 ms resonance picking (report primitive M4).
//!
//! Feeds roomeq report section R4: STFT decay slices over −5…500 ms plus
//! prominent-resonance detection at the 60 ms slice with per-resonance
//! decay time.
//!
//! Conventions (stated next to the code per the cross-cutting requirement):
//! - Window: Hann, [`WATERFALL_WINDOW_MS`] = 32 ms, hop [`WATERFALL_HOP_MS`]
//!   = 2 ms (~94 % overlap), symmetric Hann weights. Frames are centred;
//!   frame time 0 = the direct reference (argmax `|ir|`, or caller-supplied
//!   index via [`waterfall_grid_at`]).
//! - Span: [`WATERFALL_PRE_MS`] = −5 ms … [`WATERFALL_POST_MS`] = 500 ms.
//! - Magnitudes in dB rel. the grid peak, floored at [`WATERFALL_FLOOR_DB`]
//!   = −100 dB.
//! - Decimation for HTML-sized grids: [`WaterfallConfig::max_frames`] /
//!   `max_bins` cap the grid; excess frames/bins are max-pooled in the
//!   linear-power domain (peaks survive, unlike averaging), with time and
//!   frequency labels tracking the argmax cell (not the pool mean).
//! - Resonance picking at the 60 ms slice ([`RESONANCE_SLICE_MS`]):
//!   local maxima above [`WaterfallConfig::threshold_db`] rel. the slice
//!   peak, minimum separation [`WaterfallConfig::min_separation_hz`]
//!   (greedy, loudest first). Per-resonance decay time comes from a
//!   least-squares line fit of that bin's dB decay over
//!   [`DECAY_FIT_FROM_MS`]…[`DECAY_FIT_TO_MS`]; `T60 = −60 / slope`.

use rustfft::{FftPlanner, num_complex::Complex};

/// STFT analysis window length (ms): 32 ms resolves room modes down to
/// ~30 Hz (an 8 ms window is shorter than one period at 110 Hz, so bass
/// content leaks into the DC bin and wins every slice). Time localisation
/// still comes from the 2 ms hop.
pub const WATERFALL_WINDOW_MS: f64 = 32.0;

/// STFT hop between frames (ms).
pub const WATERFALL_HOP_MS: f64 = 2.0;

/// Waterfall start rel. direct (ms, negative = pre-direct).
pub const WATERFALL_PRE_MS: f64 = -5.0;

/// Waterfall end rel. direct (ms).
pub const WATERFALL_POST_MS: f64 = 500.0;

/// dB floor rel. the grid peak.
pub const WATERFALL_FLOOR_DB: f64 = -100.0;

/// Slice time for resonance picking (ms after direct).
pub const RESONANCE_SLICE_MS: f64 = 60.0;

/// Decay-fit window start (ms after direct).
pub const DECAY_FIT_FROM_MS: f64 = 20.0;

/// Decay-fit window end (ms after direct).
pub const DECAY_FIT_TO_MS: f64 = 200.0;

/// Default resonance threshold rel. the 60 ms slice peak (dB).
pub const DEFAULT_RESONANCE_THRESHOLD_DB: f64 = 12.0;

/// Default minimum resonance separation (Hz).
pub const DEFAULT_RESONANCE_SEPARATION_HZ: f64 = 25.0;

/// Default grid caps for HTML-sized output.
pub const DEFAULT_MAX_FRAMES: usize = 256;
/// Default grid caps for HTML-sized output.
pub const DEFAULT_MAX_BINS: usize = 128;

/// Configuration for [`waterfall_grid_at`] / [`detect_resonances`].
#[derive(Debug, Clone, Copy)]
pub struct WaterfallConfig {
    /// Max frames after decimation (max-pooling).
    pub max_frames: usize,
    /// Max frequency bins after decimation (max-pooling).
    pub max_bins: usize,
    /// Resonance threshold rel. slice peak (dB).
    pub threshold_db: f64,
    /// Minimum resonance separation (Hz).
    pub min_separation_hz: f64,
}

impl Default for WaterfallConfig {
    fn default() -> Self {
        Self {
            max_frames: DEFAULT_MAX_FRAMES,
            max_bins: DEFAULT_MAX_BINS,
            threshold_db: DEFAULT_RESONANCE_THRESHOLD_DB,
            min_separation_hz: DEFAULT_RESONANCE_SEPARATION_HZ,
        }
    }
}

/// Decimated STFT decay grid. `mags_db[frame][bin]`, dB rel. grid peak.
#[derive(Debug, Clone)]
pub struct WaterfallGrid {
    /// Frame centre times rel. direct (ms).
    pub times_ms: Vec<f64>,
    /// Bin centre frequencies (Hz).
    pub freqs_hz: Vec<f64>,
    /// Magnitudes, dB rel. grid peak, floored.
    pub mags_db: Vec<Vec<f32>>,
}

/// One prominent resonance at the 60 ms slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Resonance {
    /// Frequency (Hz).
    pub freq_hz: f64,
    /// Level at the 60 ms slice (dB rel. grid peak).
    pub level_db: f64,
    /// Decay time from the bin's 20…200 ms fit (s); NaN if unfittable.
    pub decay_time_s: f64,
}

fn hann_symmetric(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let d = (n - 1) as f64;
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / d).cos()))
        .collect()
}

/// Build the waterfall grid with an automatic direct reference (argmax `|ir|`).
pub fn waterfall_grid(ir: &[f32], sample_rate: f64, config: &WaterfallConfig) -> WaterfallGrid {
    let direct = ir
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .map_or(0, |(i, _)| i);
    waterfall_grid_at(ir, sample_rate, direct, config)
}

/// Build the waterfall grid around an explicit direct sample.
#[allow(clippy::cast_precision_loss)]
pub fn waterfall_grid_at(
    ir: &[f32],
    sample_rate: f64,
    direct_sample: usize,
    config: &WaterfallConfig,
) -> WaterfallGrid {
    let empty = WaterfallGrid {
        times_ms: Vec::new(),
        freqs_hz: Vec::new(),
        mags_db: Vec::new(),
    };
    if ir.is_empty() || sample_rate <= 0.0 || direct_sample >= ir.len() {
        return empty;
    }
    let win_n = ((WATERFALL_WINDOW_MS * sample_rate / 1000.0).round() as usize).max(16);
    let hop_n = ((WATERFALL_HOP_MS * sample_rate / 1000.0).round() as usize).max(1);
    let n_fft = win_n.next_power_of_two();
    let window = hann_symmetric(win_n);

    let t0 = direct_sample as f64 + WATERFALL_PRE_MS * sample_rate / 1000.0;
    let t1 = direct_sample as f64 + WATERFALL_POST_MS * sample_rate / 1000.0;
    if t1 <= 0.0 {
        return empty;
    }
    // Frame centres from max(t0, 0): the first frame is centred exactly at
    // the span start (zero-padded where it overhangs the data) so the grid
    // honestly covers −5 ms.
    let mut centres: Vec<f64> = Vec::new();
    let mut c = t0.max(0.0);
    if c > t1 {
        return empty;
    }
    while c <= t1 {
        centres.push(c);
        c += hop_n as f64;
    }
    if centres.is_empty() {
        return empty;
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    let half = n_fft / 2;
    let df = sample_rate / n_fft as f64;
    let mut frames: Vec<Vec<f64>> = Vec::with_capacity(centres.len());
    let mut buf = vec![Complex::new(0.0, 0.0); n_fft];
    for &ctr in &centres {
        let start = (ctr - win_n as f64 / 2.0).round() as i64;
        for v in buf.iter_mut() {
            *v = Complex::new(0.0, 0.0);
        }
        for (i, &w) in window.iter().enumerate() {
            let idx = start + i as i64;
            if idx >= 0 && (idx as usize) < ir.len() {
                buf[i].re = f64::from(ir[idx as usize]) * w;
            }
        }
        fft.process(&mut buf);
        frames.push((0..=half).map(|k| buf[k].norm_sqr()).collect());
    }

    // dB rel. grid peak.
    let peak = frames
        .iter()
        .flat_map(|f| f.iter())
        .copied()
        .fold(0.0f64, f64::max);
    let to_db = |p: f64| {
        if peak > 0.0 && p > 0.0 {
            (10.0 * (p / peak).log10()).max(WATERFALL_FLOOR_DB)
        } else {
            WATERFALL_FLOOR_DB
        }
    };

    // Decimate (max-pool in power domain) to the HTML-sized caps.
    let nf = centres.len().min(config.max_frames.max(1));
    let nb = (half + 1).min(config.max_bins.max(1));
    let frame_stride = (centres.len() as f64 / nf as f64).max(1.0);
    let bin_stride = ((half + 1) as f64 / nb as f64).max(1.0);
    let mut times_ms = Vec::with_capacity(nf);
    let mut freqs_hz = Vec::with_capacity(nb);
    let mut mags_db = Vec::with_capacity(nf);
    for fi in 0..nf {
        let f0 = (fi as f64 * frame_stride) as usize;
        let f1 = (((fi + 1) as f64 * frame_stride) as usize).min(centres.len());
        let mut row = Vec::with_capacity(nb);
        for bi in 0..nb {
            let b0 = (bi as f64 * bin_stride) as usize;
            let b1 = (((bi + 1) as f64 * bin_stride) as usize).min(half + 1);
            let mut pmax = 0.0f64;
            let mut fbest = b0.min(half);
            for b in b0..b1.max(b0 + 1) {
                let bb = b.min(half);
                for f in f0..f1.max(f0 + 1) {
                    let ff = f.min(centres.len() - 1);
                    if frames[ff][bb] > pmax {
                        pmax = frames[ff][bb];
                        fbest = bb;
                    }
                }
            }
            if fi == 0 {
                // Frequency label = argmax bin (not the pool mean, which
                // mislabels the peak — e.g. 46.9 Hz for a 110 Hz mode).
                freqs_hz.push(fbest as f64 * df);
            }
            row.push(to_db(pmax) as f32);
        }
        // Time label = centre of the frame-group peak across all bins
        // (magnitudes are max-pooled, so a mean label would smear onsets).
        let mut tlabel = centres[f0.min(centres.len() - 1)];
        let mut pbest = 0.0f64;
        let f1c = f1.max(f0 + 1);
        for (fr, &tc) in frames
            .iter()
            .zip(centres.iter())
            .skip(f0)
            .take(f1c.saturating_sub(f0))
        {
            for &p in fr {
                if p > pbest {
                    pbest = p;
                    tlabel = tc;
                }
            }
        }
        times_ms.push((tlabel - direct_sample as f64) * 1000.0 / sample_rate);
        mags_db.push(row);
    }

    WaterfallGrid {
        times_ms,
        freqs_hz,
        mags_db,
    }
}

/// Detect prominent resonances at the 60 ms slice with per-resonance decay.
///
/// `grid` is typically built by [`waterfall_grid`] from the same IR. The
/// slice nearest [`RESONANCE_SLICE_MS`] is peak-picked; each candidate's
/// decay time comes from a least-squares fit of its bin over
/// [`DECAY_FIT_FROM_MS`]…[`DECAY_FIT_TO_MS`].
pub fn detect_resonances(grid: &WaterfallGrid, config: &WaterfallConfig) -> Vec<Resonance> {
    if grid.times_ms.is_empty() || grid.freqs_hz.is_empty() {
        return Vec::new();
    }
    // Nearest frame to the 60 ms slice.
    let slice = grid
        .times_ms
        .iter()
        .enumerate()
        .min_by(|a, b| {
            (*a.1 - RESONANCE_SLICE_MS)
                .abs()
                .total_cmp(&(*b.1 - RESONANCE_SLICE_MS).abs())
        })
        .map_or(0, |(i, _)| i);
    let row = &grid.mags_db[slice];
    let slice_peak = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !slice_peak.is_finite() {
        return Vec::new();
    }
    let floor = slice_peak - config.threshold_db as f32;
    let mut candidates: Vec<(usize, f32)> = Vec::new();
    for (b, &v) in row.iter().enumerate() {
        if v < floor {
            continue;
        }
        let left = if b > 0 { row[b - 1] } else { f32::NEG_INFINITY };
        let right = if b + 1 < row.len() {
            row[b + 1]
        } else {
            f32::NEG_INFINITY
        };
        if v >= left && v > right {
            candidates.push((b, v));
        }
    }
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut accepted: Vec<(usize, f32)> = Vec::new();
    for (b, v) in candidates {
        let f = grid.freqs_hz[b.min(grid.freqs_hz.len() - 1)];
        if accepted.iter().all(|&(ab, _)| {
            (grid.freqs_hz[ab.min(grid.freqs_hz.len() - 1)] - f).abs() >= config.min_separation_hz
        }) {
            accepted.push((b, v));
        }
    }

    // Fit window frames for the decay slope.
    let fit_frames: Vec<usize> = grid
        .times_ms
        .iter()
        .enumerate()
        .filter(|(_, t)| **t >= DECAY_FIT_FROM_MS && **t <= DECAY_FIT_TO_MS)
        .map(|(i, _)| i)
        .collect();

    let mut out: Vec<Resonance> = accepted
        .into_iter()
        .map(|(b, v)| {
            let freq_hz = grid.freqs_hz[b.min(grid.freqs_hz.len() - 1)];
            let mut xs: Vec<f64> = Vec::new();
            let mut ys: Vec<f64> = Vec::new();
            for &f in &fit_frames {
                let m = grid.mags_db[f][b.min(grid.mags_db[f].len() - 1)];
                if f64::from(m) > WATERFALL_FLOOR_DB + 1.0 {
                    xs.push(grid.times_ms[f] / 1000.0);
                    ys.push(f64::from(m));
                }
            }
            let decay_time_s = linear_t60(&xs, &ys);
            Resonance {
                freq_hz,
                level_db: f64::from(v),
                decay_time_s,
            }
        })
        .collect();
    out.sort_by(|a, b| b.level_db.total_cmp(&a.level_db));
    out
}

fn linear_t60(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return f64::NAN;
    }
    let n_f = n as f64;
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
    let denom = n_f * sxx - sx * sx;
    if denom.abs() < f64::EPSILON {
        return f64::NAN;
    }
    let slope = (n_f * sxy - sx * sy) / denom;
    if slope < -1e-9 {
        -60.0 / slope
    } else {
        f64::NAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn modal_ir(sr: f64, modes: &[(f64, f64, f64)]) -> Vec<f32> {
        // Sum of exponentially decaying sines: (freq, t60, amp).
        let n = (0.6 * sr) as usize;
        let mut ir = vec![0.0f32; n];
        ir[0] = 1.0;
        for &(f, t60, amp) in modes {
            let alpha = 3.0 * std::f64::consts::LN_10 / t60;
            for (i, s) in ir.iter_mut().enumerate().skip(1) {
                let t = i as f64 / sr;
                *s +=
                    (amp * (-alpha * t).exp() * (2.0 * std::f64::consts::PI * f * t).sin()) as f32;
            }
        }
        ir
    }

    #[test]
    fn grid_spans_minus5_to_500ms() {
        let sr = 48000.0;
        let ir = modal_ir(sr, &[(120.0, 0.5, 0.5)]);
        let grid = waterfall_grid(&ir, sr, &WaterfallConfig::default());
        assert!(!grid.times_ms.is_empty());
        assert!(
            grid.times_ms[0] <= 0.0,
            "first frame {:?}",
            grid.times_ms[0]
        );
        assert!(
            grid.times_ms.last().copied().unwrap_or(0.0) >= 400.0,
            "last frame {:?}",
            grid.times_ms.last()
        );
        assert!(grid.mags_db.len() <= DEFAULT_MAX_FRAMES);
        assert!(grid.freqs_hz.len() <= DEFAULT_MAX_BINS);
    }

    #[test]
    fn detects_planted_resonance_with_decay() {
        let sr = 48000.0;
        let ir = modal_ir(sr, &[(110.0, 0.6, 1.0), (440.0, 0.2, 0.3)]);
        let cfg = WaterfallConfig::default();
        let grid = waterfall_grid(&ir, sr, &cfg);
        let res = detect_resonances(&grid, &cfg);
        assert!(!res.is_empty(), "expected resonances");
        let top = &res[0];
        // Frequency resolution after decimation is coarse; allow one bin.
        assert!(
            (top.freq_hz - 110.0).abs() < 60.0,
            "top resonance at {} Hz (expected ≈ 110)",
            top.freq_hz
        );
        assert!(
            (top.decay_time_s - 0.6).abs() < 0.3,
            "decay = {:.3} s (expected ≈ 0.6)",
            top.decay_time_s
        );
    }

    #[test]
    fn empty_input_yields_empty_grid() {
        let grid = waterfall_grid(&[], 48000.0, &WaterfallConfig::default());
        assert!(grid.times_ms.is_empty());
        assert!(detect_resonances(&grid, &WaterfallConfig::default()).is_empty());
        // Past-the-end direct reference is equally empty.
        let ir = modal_ir(48000.0, &[(110.0, 0.5, 0.5)]);
        let grid = waterfall_grid_at(&ir, 48000.0, ir.len(), &WaterfallConfig::default());
        assert!(grid.times_ms.is_empty());
    }

    #[test]
    fn explicit_direct_matches_auto() {
        // Unambiguous direct (2.0 delta at sample 500): the explicit
        // reference must reproduce the automatic grid exactly.
        let sr = 48000.0;
        let mut ir = vec![0.0f32; (0.6 * sr) as usize];
        ir[500] = 2.0;
        for (i, s) in ir.iter_mut().enumerate().skip(501) {
            let t = i as f64 / sr;
            *s += (0.5 * (-t * 10.0).exp() * (2.0 * std::f64::consts::PI * 440.0 * t).sin()) as f32;
        }
        let cfg = WaterfallConfig::default();
        let auto_grid = waterfall_grid(&ir, sr, &cfg);
        let exp_grid = waterfall_grid_at(&ir, sr, 500, &cfg);
        assert_eq!(auto_grid.times_ms, exp_grid.times_ms);
        assert_eq!(auto_grid.freqs_hz, exp_grid.freqs_hz);
        assert_eq!(auto_grid.mags_db, exp_grid.mags_db);
    }

    #[test]
    fn silent_ir_sits_at_floor() {
        // All zeros: no peak to normalise against, every cell at the floor.
        let grid = waterfall_grid(&vec![0.0f32; 24000], 48000.0, &WaterfallConfig::default());
        assert!(!grid.mags_db.is_empty());
        for row in &grid.mags_db {
            for &v in row {
                assert_eq!(v, WATERFALL_FLOOR_DB as f32);
            }
        }
    }

    #[test]
    fn close_peaks_keep_loudest_only() {
        // Hand-built grid (no STFT interference): the 60 ms slice has
        // local maxima at 400 Hz (−18 dB) and 420 Hz (−19 dB), 20 Hz apart
        // (< 25 Hz separation), plus a distant 500 Hz peak. Only 400 Hz
        // and 500 Hz may survive; the decay fit must read T60 = 0.2 s.
        let times_ms = vec![0.0, 20.0, 60.0, 100.0, 200.0];
        let freqs_hz = vec![400.0, 410.0, 420.0, 500.0];
        // −0.3 dB/ms slope in every bin; slice peaks at 400/420/500.
        let peaks = [0.0f64, -10.0, -1.0, -0.5];
        let mags_db: Vec<Vec<f32>> = times_ms
            .iter()
            .map(|&t| {
                freqs_hz
                    .iter()
                    .zip(peaks.iter())
                    .map(|(_, &p)| (p - 0.3 * t) as f32)
                    .collect()
            })
            .collect();
        let grid = WaterfallGrid {
            times_ms,
            freqs_hz,
            mags_db,
        };
        let cfg = WaterfallConfig::default();
        let res = detect_resonances(&grid, &cfg);
        assert_eq!(res.len(), 2, "expected 400 + 500 Hz, got {res:?}");
        assert!((res[0].freq_hz - 400.0).abs() < 1e-9);
        assert!((res[1].freq_hz - 500.0).abs() < 1e-9);
        assert!((res[0].decay_time_s - 0.2).abs() < 0.01);
        // All-NaN slice: no resonances.
        let mut nan_grid = grid.clone();
        for row in nan_grid.mags_db.iter_mut() {
            for v in row.iter_mut() {
                *v = f32::NAN;
            }
        }
        assert!(detect_resonances(&nan_grid, &cfg).is_empty());
    }

    #[test]
    fn linear_t60_edge_cases() {
        // Fewer than two points: unfittable.
        assert!(linear_t60(&[0.0], &[-1.0]).is_nan());
        assert!(linear_t60(&[], &[]).is_nan());
        // Degenerate time base: unfittable.
        assert!(linear_t60(&[1.0, 1.0], &[0.0, -10.0]).is_nan());
        // Rising (non-decaying) slope: not a decay.
        assert!(linear_t60(&[0.0, 1.0], &[-20.0, -10.0]).is_nan());
        // Exact −60 dB/s slope: T60 = 1 s.
        assert!((linear_t60(&[0.0, 1.0], &[0.0, -60.0]) - 1.0).abs() < 1e-9);
    }
}

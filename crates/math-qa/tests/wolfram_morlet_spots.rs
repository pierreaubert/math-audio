//! Wolfram cross-check: M5 Morlet CWT magnitude at spot (freq, time)s.
//!
//! Oracle: `wolfram/morlet_spots.wls` (direct Morlet-correlation
//! summation, same 3-cycle unit-energy kernel convention). Both sides
//! peak-normalize, so dB *differences* between spots are compared.
//! Tolerance 1e-3 dB (f32 sample rounding dominates). Spots must stay
//! above the heatmap's -30 dB display floor: below it the Rust cells
//! clamp and no longer track the oracle.

use math_audio_dsp::rir_wavelet::{WaveletConfig, wavelet_heatmap_at};
use math_qa::{QaResult, assert_close_abs, emit_result, provenance, reference};

const CASE: &str = "morlet_spots";
const CASE_ID: &str = "math-qa.morlet-spots.v1";
const TOL_DB: f64 = 1e-3;

#[test]
fn wolfram_morlet_spots() {
    let Some(ref_json) = reference(CASE, "morlet_spots.wls") else {
        return;
    };
    assert_eq!(ref_json["case"], CASE_ID);
    let rows: Vec<f64> = serde_json::from_value(ref_json["rows_hz"].clone()).unwrap();
    let centers: Vec<usize> = serde_json::from_value(ref_json["centers"].clone()).unwrap();
    let magnitude: Vec<Vec<f64>> = serde_json::from_value(ref_json["magnitude"].clone()).unwrap();

    // Same signal as the oracle: 1 kHz tone burst, 100-150 ms.
    let sr = 48000.0;
    let n = (0.6 * sr) as usize;
    let mut ir = vec![0.0f32; n];
    let start = (0.1 * sr) as usize;
    for (i, s) in ir
        .iter_mut()
        .enumerate()
        .skip(start)
        .take((0.05 * sr) as usize)
    {
        let t = i as f64 / sr;
        *s = (2.0 * std::f64::consts::PI * 1000.0 * t).sin() as f32;
    }

    // Undecimated heatmap so spots map 1:1 to grid rows/columns.
    let cfg = WaveletConfig {
        max_freqs: usize::MAX,
        max_frames: usize::MAX,
        ..WaveletConfig::default()
    };
    let heat = wavelet_heatmap_at(&ir, sr, 0, &cfg);

    // Row labels are argmax (identity with stride 1); columns likewise.
    // Locate each oracle row/column by nearest grid value. Spots are the
    // in-range cells (row, time): (500 Hz, 99 ms), (500 Hz, 100 ms),
    // (1 kHz, 100 ms), (1 kHz, 125 ms, the reference); the 500 Hz
    // mid-burst cell at -51 dB sits below the display floor and is out
    // of comparison scope by design.
    let spots = [(0usize, 0usize), (0, 1), (1, 1), (1, 2)];
    let (ref_ri, ref_ci) = (1usize, 2usize);
    let ref_val = magnitude[ref_ri][ref_ci];
    let rust_ref = cell_db(&heat, &rows, &centers, sr, ref_ri, ref_ci);
    // The reference cell must itself be above the floor.
    assert!(
        rust_ref > math_audio_dsp::rir_wavelet::WAVELET_DB_MIN + 1.0,
        "reference spot clamped at floor: {rust_ref} dB"
    );
    let mut worst = 0.0f64;
    for (ri, ci) in spots {
        let ref_db = 20.0 * (magnitude[ri][ci] / ref_val).log10();
        let rust_db = cell_db(&heat, &rows, &centers, sr, ri, ci) - rust_ref;
        assert_close_abs(rust_db, ref_db, TOL_DB, &format!("spot ({ri}, {ci})"));
        worst = worst.max((rust_db - ref_db).abs());
    }
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: worst,
        tolerance: TOL_DB,
        provenance: provenance(),
    });
}

/// dB value of the heatmap cell nearest oracle row `ri` / centre `ci`.
fn cell_db(
    heat: &math_audio_dsp::rir_wavelet::WaveletHeatmap,
    rows: &[f64],
    centers: &[usize],
    sr: f64,
    ri: usize,
    ci: usize,
) -> f64 {
    let target_f = rows[ri];
    let r = heat
        .freqs_hz
        .iter()
        .enumerate()
        .min_by(|a, b| (*a.1 - target_f).abs().total_cmp(&(*b.1 - target_f).abs()))
        .map_or(0, |(i, _)| i);
    let target_t = centers[ci] as f64 * 1000.0 / sr;
    let c = heat
        .times_ms
        .iter()
        .enumerate()
        .min_by(|a, b| (*a.1 - target_t).abs().total_cmp(&(*b.1 - target_t).abs()))
        .map_or(0, |(i, _)| i);
    f64::from(heat.mags_db[r][c])
}

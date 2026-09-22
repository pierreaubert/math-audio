//! Wolfram cross-check: M4 waterfall STFT power at spot (frame, bin)s.
//!
//! Oracle: `wolfram/waterfall_spots.wls` (direct windowed-DFT summation,
//! same 32 ms symmetric-Hann / 2 ms hop / next-pow-2 convention, raw
//! power). Both sides peak-normalize, so dB *differences* between spots
//! are compared. Tolerance 1e-3 dB (f32 sample rounding dominates;
//! convention slips would read in whole dB).

use math_audio_dsp::rir_waterfall::{WaterfallConfig, waterfall_grid_at};
use math_qa::{QaResult, assert_close_abs, emit_result, provenance, reference};

const CASE: &str = "waterfall_spots";
const CASE_ID: &str = "math-qa.waterfall-spots.v1";
const TOL_DB: f64 = 1e-3;

#[test]
fn wolfram_waterfall_spots() {
    let Some(ref_json) = reference(CASE, "waterfall_spots.wls") else {
        return;
    };
    assert_eq!(ref_json["case"], CASE_ID);
    let sr: f64 = serde_json::from_value(ref_json["sample_rate_hz"].clone()).unwrap();
    let centers: Vec<usize> = serde_json::from_value(ref_json["frame_centres"].clone()).unwrap();
    let bins: Vec<usize> = serde_json::from_value(ref_json["bins"].clone()).unwrap();
    let power: Vec<Vec<f64>> = serde_json::from_value(ref_json["power"].clone()).unwrap();

    // Same signal as the oracle: decaying 440 Hz sine, T60 0.6 s.
    let t60 = 0.6;
    let alpha = 3.0 * std::f64::consts::LN_10 / t60;
    let n = (0.6 * sr) as usize;
    let ir: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / sr;
            ((-alpha * t).exp() * (2.0 * std::f64::consts::PI * 440.0 * t).sin()) as f32
        })
        .collect();

    // Undecimated grid so spots map 1:1 to frames/bins.
    let cfg = WaterfallConfig {
        max_frames: usize::MAX,
        max_bins: usize::MAX,
        ..WaterfallConfig::default()
    };
    let grid = waterfall_grid_at(&ir, sr, 0, &cfg);

    // .wls frames count from sample 0 with 96-sample hop: Rust frame fi
    // has centre fi * hop. Bins are raw FFT bins (no decimation here).
    // Both sides normalize to the first spot (frame 0, first bin).
    let hop = (2.0 * sr / 1000.0).round() as usize;
    let fi0 = centers[0] / hop;
    let rust00 = f64::from(grid.mags_db[fi0][bins[0]]);
    let mut worst = 0.0f64;
    for (ci, &center) in centers.iter().enumerate() {
        let fi = center / hop;
        for (bi, &bin) in bins.iter().enumerate() {
            // Reference dB differences against the first spot.
            let ref_db = 10.0 * (power[ci][bi] / power[0][0]).log10();
            let rust_db = f64::from(grid.mags_db[fi][bin]) - rust00;
            assert_close_abs(rust_db, ref_db, TOL_DB, &format!("spot ({center}, {bin})"));
            worst = worst.max((rust_db - ref_db).abs());
        }
    }
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: worst,
        tolerance: TOL_DB,
        provenance: provenance(),
    });
}

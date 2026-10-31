//! Long-record band and decay workspaces retain their output allocations.

// Rust guideline compliant 2026-02-21
use math_rir::bands::{BandWidth, BandpassWorkspace};
use math_rir::metrics::{DecayCurve, DecayWorkspace};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region};

#[global_allocator]
static GLOBAL: &stats_alloc::StatsAlloc<std::alloc::System> = &INSTRUMENTED_SYSTEM;

// Allocation regions observe the whole process, so measured tests must
// not run concurrently with each other.
static MEASURED: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn long_record_band_and_decay_outputs_are_reused() {
    let _measured = MEASURED.lock().unwrap();
    let input: Vec<f32> = (0..480000)
        .map(|n| (n as f32 * 0.13).sin() * (-(n as f32) / 48000.0).exp())
        .collect();
    let mut band = BandpassWorkspace::new();
    let mut decay = DecayWorkspace::new();
    let cold = Region::new(GLOBAL);
    let output = band.process(&input, 1000.0, BandWidth::Octave, 48000.0, 4);
    // Allocation reuse is independent of automatic measurement acceptance.
    // Integrate the full record explicitly to exercise maximum buffer sizes.
    decay.compute(output, 48000.0, 0, Some(output.len()));
    let cold = cold.change();
    // Two bandpass f64 buffers, f32 output, and two decay f64 buffers:
    // 36 bytes/sample, plus small coefficient/padding/allocator overhead.
    // Bound total requests too, catching geometric scratch growth during setup.
    assert!(
        cold.bytes_allocated <= input.len() * 36 + 1_048_576,
        "{cold:?}"
    );
    println!(
        "480k-sample retained workspace bytes={}, total allocation requests={} bytes",
        cold.bytes_allocated - cold.bytes_deallocated,
        cold.bytes_allocated
    );
    // The retained figure is asserted, not only printed: two bandpass f64
    // buffers, f32 output, and two decay f64 buffers (36 bytes/sample),
    // plus coefficient-cache slack.
    assert!(
        cold.bytes_allocated - cold.bytes_deallocated <= input.len() * 36 + 65_536,
        "retained workspace exceeds the 36 B/sample model: {cold:?}"
    );
    for length in [480000, 24000, 480000] {
        let region = Region::new(GLOBAL);
        let output = band.process(&input[..length], 1000.0, BandWidth::Octave, 48000.0, 4);
        let curve = decay.compute(output, 48000.0, 0, Some(output.len()));
        assert!(!curve.samples.is_empty());
        let stats = region.change();
        assert_eq!(stats.allocations + stats.reallocations, 0, "{stats:?}");
    }
    // The owning compatibility API needs integrals and one output, not an
    // additional retained-and-cloned output vector.
    let mut owned_workspace = DecayWorkspace::new();
    let region = Region::new(GLOBAL);
    let owned = DecayCurve::from_rir_with_workspace(
        &input,
        48000.0,
        0,
        Some(input.len()),
        &mut owned_workspace,
    );
    let stats = region.change();
    assert!(
        stats.bytes_allocated <= input.len() * 16 + 1024,
        "{stats:?}"
    );
    assert_eq!(owned.samples.len(), input.len());
}

#[test]
fn default_band_analysis_bounds_contexts_by_parallelism() {
    let _measured = MEASURED.lock().unwrap();
    use math_rir::bands::{BandWidth, ISO_THIRD_OCTAVE_CENTERS_HZ, analyze_iso3382_bands};
    let input: Vec<f32> = (0..24000)
        .map(|n| (n as f32 * 0.11).sin() * (-(n as f32) / 12000.0).exp())
        .collect();
    let region = Region::new(GLOBAL);
    let results = analyze_iso3382_bands(
        &input,
        48000.0,
        &ISO_THIRD_OCTAVE_CENTERS_HZ,
        BandWidth::ThirdOctave,
        4,
    );
    let stats = region.change();
    assert!(!results.is_empty());
    // One context retains ~36 bytes/sample. The default entry point must
    // bound live contexts by worker parallelism, not by the 21 bands.
    let bound = rayon::current_num_threads() * input.len() * 36 + 1_048_576;
    assert!(
        stats.bytes_allocated - stats.bytes_deallocated <= bound,
        "default band analysis retained {} bytes, bound {bound}: {stats:?}",
        stats.bytes_allocated - stats.bytes_deallocated
    );
}

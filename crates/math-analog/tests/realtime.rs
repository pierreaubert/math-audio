//! Realtime-contract tests that need a process-wide allocation counter.

use math_audio_analog::{AnalogModel, AnalogProcessor, ProcessSpec};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region};
mod audit_support;

#[global_allocator]
static GLOBAL: &stats_alloc::StatsAlloc<std::alloc::System> = &INSTRUMENTED_SYSTEM;

#[test]
fn representative_processing_does_not_allocate_after_prepare() {
    let mut cases = Vec::new();
    for channels in [1, 2, 6, 12] {
        for model_id in 0..=AnalogModel::TONE_STACK_ID {
            let spec = ProcessSpec::new(48_000.0, channels, 256);
            let mut model = AnalogModel::from_id(model_id).expect("stable model id");
            audit_support::drive_model(&mut model, 6.0);
            model.prepare(spec).expect("valid process spec");
            let block = vec![0.0_f32; channels * 256];
            // The very first callback after prepare is inside the measured region.
            cases.push((model, block, channels));
        }
    }

    for (index, (model, block, channels)) in cases.iter_mut().enumerate() {
        let region = Region::new(GLOBAL);
        for iteration in 0..8 {
            if iteration == 3 {
                audit_support::drive_model(model, 24.0);
            }
            if iteration == 4 {
                model.reset();
            }
            for frames in [32, 64, 128, 256] {
                let active = &mut block[..frames * *channels];
                audit_support::fill_input(active, *channels, iteration);
                model.process_interleaved(active, frames).unwrap();
            }
        }
        let stats = region.change();
        assert_eq!(
            stats.allocations, 0,
            "audio processing allocated in case {index}: {stats:?}"
        );
        assert_eq!(
            stats.reallocations, 0,
            "audio processing reallocated in case {index}: {stats:?}"
        );
    }
}

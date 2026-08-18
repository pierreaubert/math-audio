//! Realtime-contract tests that need a process-wide allocation counter.

use math_audio_analog::{AnalogModel, AnalogProcessor, ProcessSpec};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region};

#[global_allocator]
static GLOBAL: &stats_alloc::StatsAlloc<std::alloc::System> = &INSTRUMENTED_SYSTEM;

#[test]
fn representative_processing_does_not_allocate_after_prepare() {
    let mut cases = Vec::new();
    for channels in [1, 2, 6, 12] {
        for model_id in 0..=AnalogModel::CONSOLE_PREAMP_ID {
            let spec = ProcessSpec::new(48_000.0, channels, 512);
            let mut model = AnalogModel::from_id(model_id).expect("stable model id");
            model.prepare(spec).expect("valid process spec");
            let mut block = vec![0.0_f32; channels * 512];
            for (index, sample) in block.iter_mut().enumerate() {
                *sample = ((index as f32) * 0.017).sin() * 0.75;
            }
            // Warm the model and all branch/smoother paths before measuring.
            model.process_interleaved(&mut block, 512).unwrap();
            model.reset();
            cases.push((model, block));
        }
    }

    for (index, (model, block)) in cases.iter_mut().enumerate() {
        let region = Region::new(GLOBAL);
        for _ in 0..8 {
            model.process_interleaved(block, 512).unwrap();
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

use math_audio_analog::{AnalogModel, AnalogProcessor, ProcessSpec};
use std::time::Instant;

#[test]
fn worst_callback_report_covers_all_models_and_channel_widths() {
    let mut worst_ns = 0_u128;
    let mut worst_case = (0_u32, 0_usize);
    for channels in [1, 2, 6, 12] {
        for model_id in 0..=AnalogModel::TRANSFORMER_ID {
            let spec = ProcessSpec::new(48_000.0, channels, 2_048);
            let mut model = AnalogModel::from_id(model_id).unwrap();
            model.prepare(spec).unwrap();
            let mut block = vec![0.0_f32; channels * 2_048];
            let mut local_worst = 0_u128;
            for _ in 0..16 {
                let start = Instant::now();
                model.process_interleaved(&mut block, 2_048).unwrap();
                local_worst = local_worst.max(start.elapsed().as_nanos());
            }
            if local_worst > worst_ns {
                worst_ns = local_worst;
                worst_case = (model_id, channels);
            }
        }
    }
    println!(
        "math-analog worst callback: {worst_ns} ns, model={}, channels={}",
        worst_case.0, worst_case.1
    );
    assert!(worst_ns > 0);
}

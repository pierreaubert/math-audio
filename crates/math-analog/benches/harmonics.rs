use criterion::BenchmarkId;
use criterion::{Criterion, criterion_group, criterion_main};
use math_audio_analog::{AnalogModel, AnalogProcessor, HarmonicModel, ProcessSpec};
use std::hint::black_box;

fn harmonics_block(c: &mut Criterion) {
    let mut model = HarmonicModel::new();
    model.set_drive_db(12.0).expect("valid drive");
    model.set_h2_db(-18.0).expect("valid H2");
    model.set_h3_db(-24.0).expect("valid H3");
    model
        .prepare(ProcessSpec {
            sample_rate: 48_000.0,
            channels: 2,
            max_block_frames: 512,
        })
        .expect("valid process spec");
    let mut block = vec![0.125_f32; 512 * 2];

    c.bench_function("harmonics_2ch_512", |b| {
        b.iter(|| {
            block.fill(0.125);
            model
                .process_interleaved(black_box(block.as_mut_slice()), 512)
                .expect("prepared block");
            black_box(block.as_slice());
        });
    });
}

fn model_channel_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("analog_model_channel_matrix");
    for channels in [1, 2, 6, 12] {
        for model_id in 0..=AnalogModel::TRANSFORMER_ID {
            let spec = ProcessSpec::new(48_000.0, channels, 512);
            let mut model = AnalogModel::from_id(model_id).expect("stable model id");
            model.prepare(spec).expect("valid process spec");
            let mut block = vec![0.125_f32; channels * 512];
            group.bench_function(
                BenchmarkId::new(format!("model_{model_id}"), channels),
                |b| {
                    b.iter(|| {
                        block.fill(0.125);
                        model
                            .process_interleaved(black_box(block.as_mut_slice()), 512)
                            .unwrap();
                        black_box(block.as_slice());
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, harmonics_block, model_channel_matrix);
criterion_main!(benches);

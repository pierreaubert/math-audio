use math_audio_analog::{
    AnalogError, AnalogModel, AnalogProcessor, HarmonicModel, ProcessSpec, StaticColorModel,
    StaticCurve,
};

fn spec(channels: usize, max_block_frames: usize) -> ProcessSpec {
    ProcessSpec {
        sample_rate: 48_000.0,
        channels,
        max_block_frames,
    }
}

#[test]
fn invalid_prepare_specs_are_rejected() {
    for invalid in [
        ProcessSpec {
            sample_rate: 0.0,
            channels: 1,
            max_block_frames: 32,
        },
        ProcessSpec {
            sample_rate: f32::NAN,
            channels: 1,
            max_block_frames: 32,
        },
        spec(0, 32),
        spec(1, 0),
    ] {
        let mut model = HarmonicModel::new();
        assert!(model.prepare(invalid).is_err());
    }
}

#[test]
fn buffer_and_block_contracts_are_checked_before_processing() {
    let mut model = HarmonicModel::new();
    assert_eq!(
        model.process_interleaved(&mut [0.0], 1),
        Err(AnalogError::NotPrepared)
    );
    model.prepare(spec(2, 4)).unwrap();

    let mut too_large = [0.0; 10];
    assert_eq!(
        model.process_interleaved(&mut too_large, 5),
        Err(AnalogError::BlockTooLarge {
            requested: 5,
            maximum: 4,
        })
    );

    let mut wrong_length = [0.0; 3];
    assert_eq!(
        model.process_interleaved(&mut wrong_length, 2),
        Err(AnalogError::BufferLengthMismatch {
            expected: 4,
            actual: 3,
        })
    );
}

#[test]
fn callback_partitioning_produces_the_same_sample_stream() {
    let mut one_block = HarmonicModel::new();
    one_block.set_drive_db(12.0).unwrap();
    one_block.set_h2_db(-18.0).unwrap();
    one_block.set_h3_db(-24.0).unwrap();
    one_block
        .prepare(ProcessSpec {
            sample_rate: 44_100.0,
            channels: 2,
            max_block_frames: 256,
        })
        .unwrap();

    let mut partitioned = HarmonicModel::new();
    partitioned.set_drive_db(12.0).unwrap();
    partitioned.set_h2_db(-18.0).unwrap();
    partitioned.set_h3_db(-24.0).unwrap();
    partitioned
        .prepare(ProcessSpec {
            sample_rate: 44_100.0,
            channels: 2,
            max_block_frames: 256,
        })
        .unwrap();

    let input: Vec<f32> = (0..512)
        .map(|index| ((index as f32) * 0.17).sin() * 0.7)
        .collect();
    let mut reference = input.clone();
    let mut streamed = input;
    one_block.process_interleaved(&mut reference, 256).unwrap();
    for block in streamed.chunks_exact_mut(64) {
        partitioned.process_interleaved(block, 32).unwrap();
    }
    assert_eq!(reference, streamed);
}

#[test]
fn automation_is_partition_independent_at_block_boundaries() {
    let process_spec = ProcessSpec::new(48_000.0, 2, 128);
    let mut one_block = HarmonicModel::new();
    let mut partitioned = HarmonicModel::new();
    for model in [&mut one_block, &mut partitioned] {
        model.set_drive_db(-12.0).unwrap();
        model.set_h2_db(-18.0).unwrap();
        model.set_h3_db(-24.0).unwrap();
        model.prepare(process_spec).unwrap();
    }

    let first = vec![0.25_f32; 2 * 32];
    let second = vec![0.75_f32; 2 * 96];
    let mut one_block_first = first.clone();
    let mut partitioned_first = first;
    one_block
        .process_interleaved(&mut one_block_first, 32)
        .unwrap();
    partitioned
        .process_interleaved(&mut partitioned_first, 32)
        .unwrap();

    one_block.set_drive_db(24.0).unwrap();
    partitioned.set_drive_db(24.0).unwrap();
    let mut one_block_second = second.clone();
    let mut partitioned_second = second;
    one_block
        .process_interleaved(&mut one_block_second, 96)
        .unwrap();
    for block in partitioned_second.chunks_exact_mut(2 * 16) {
        partitioned.process_interleaved(block, 16).unwrap();
    }

    assert_eq!(one_block_first, partitioned_first);
    assert_eq!(one_block_second, partitioned_second);
}

#[test]
fn partial_final_callback_matches_one_block_reference() {
    let process_spec = ProcessSpec::new(48_000.0, 2, 64);
    let input: Vec<f32> = (0..2 * 37)
        .map(|index| ((index as f32) * 0.19).sin() * 0.7)
        .collect();
    let mut reference = HarmonicModel::new();
    let mut streamed = HarmonicModel::new();
    reference.prepare(process_spec).unwrap();
    streamed.prepare(process_spec).unwrap();
    let mut expected = input.clone();
    reference.process_interleaved(&mut expected, 37).unwrap();
    let mut actual = input;
    let mut offset = 0;
    for frames in [8, 8, 21] {
        let end = offset + frames * 2;
        streamed
            .process_interleaved(&mut actual[offset..end], frames)
            .unwrap();
        offset = end;
    }
    assert_eq!(expected, actual);
}

#[test]
fn model_ids_are_append_only_and_unknown_ids_fail_closed() {
    for id in [
        AnalogModel::HARMONICS_ID,
        AnalogModel::STATIC_ID,
        AnalogModel::HAMMERSTEIN_ID,
        AnalogModel::TAPE_ID,
        AnalogModel::TRANSFORMER_ID,
        AnalogModel::CONSOLE_PREAMP_ID,
    ] {
        assert_eq!(AnalogModel::from_id(id).unwrap().model_id(), id);
    }
    assert!(matches!(
        AnalogModel::from_id(999),
        Err(AnalogError::UnknownModelId(999))
    ));
}

#[test]
fn channel_state_isolation_holds_for_interleaved_processing() {
    let mut model = StaticColorModel::new(StaticCurve::TanhStyle);
    model.prepare(ProcessSpec::new(96_000.0, 6, 32)).unwrap();
    let mut samples = [0.75, 0.0, 0.0, 0.0, 0.0, 0.0];
    model.process_interleaved(&mut samples, 1).unwrap();
    assert_ne!(samples[0], 0.0);
    assert!(samples[1..].iter().all(|sample| *sample == 0.0));
}

#[test]
fn reprepare_rebuilds_channel_and_sample_rate_state() {
    let mut model = HarmonicModel::new();
    model.prepare(ProcessSpec::new(44_100.0, 1, 16)).unwrap();
    let mut mono = [0.5; 1];
    model.process_interleaved(&mut mono, 1).unwrap();

    model.prepare(ProcessSpec::new(192_000.0, 12, 16)).unwrap();
    let mut high_channel_count = [0.0_f32; 12];
    model
        .process_interleaved(&mut high_channel_count, 1)
        .unwrap();
    assert!(high_channel_count.iter().all(|sample| sample.is_finite()));
}

#[test]
fn every_model_is_finite_streamable_and_resettable() {
    for channels in [1, 2, 6, 12] {
        for model_id in 0..=AnalogModel::CONSOLE_PREAMP_ID {
            let process_spec = ProcessSpec::new(96_000.0, channels, 128);
            let mut one_block = AnalogModel::from_id(model_id).unwrap();
            one_block.prepare(process_spec).unwrap();
            let input: Vec<f32> = (0..128 * channels)
                .map(|index| ((index as f32) * 0.071).sin() * 4.0)
                .collect();
            let mut reference = input.clone();
            one_block.process_interleaved(&mut reference, 128).unwrap();
            assert!(reference.iter().all(|sample| sample.is_finite()));

            let mut partitioned = AnalogModel::from_id(model_id).unwrap();
            partitioned.prepare(process_spec).unwrap();
            let mut streamed = input;
            for block in streamed.chunks_exact_mut(channels * 32) {
                partitioned.process_interleaved(block, 32).unwrap();
            }
            assert_eq!(
                reference, streamed,
                "model {model_id} depends on callback partition at {channels} channels"
            );

            partitioned.reset();
            let mut after_reset = vec![0.125_f32; channels];
            partitioned
                .process_interleaved(&mut after_reset, 1)
                .unwrap();
            let mut fresh = AnalogModel::from_id(model_id).unwrap();
            fresh.prepare(process_spec).unwrap();
            let mut fresh_output = vec![0.125_f32; channels];
            fresh.process_interleaved(&mut fresh_output, 1).unwrap();
            assert_eq!(
                after_reset, fresh_output,
                "model {model_id} reset differs from fresh at {channels} channels"
            );

            let mut non_finite = vec![f32::MAX, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
            non_finite.resize(channels, 0.0);
            partitioned.process_interleaved(&mut non_finite, 1).unwrap();
            assert!(non_finite.iter().all(|sample| sample.is_finite()));
        }
    }
}

#[test]
fn maximum_plus_one_block_is_rejected_before_mutation() {
    let mut model = AnalogModel::default();
    model.prepare(ProcessSpec::new(48_000.0, 2, 4)).unwrap();
    let mut samples = [0.25_f32; 10];
    let before = samples;
    assert_eq!(
        model.process_interleaved(&mut samples, 5),
        Err(AnalogError::BlockTooLarge {
            requested: 5,
            maximum: 4,
        })
    );
    assert_eq!(samples, before);
}

fn set_drive(model: &mut AnalogModel, value: f32) {
    match model {
        AnalogModel::Harmonics(model) => model.set_drive_db(value).unwrap(),
        AnalogModel::Static(model) => model.set_drive_db(value).unwrap(),
        AnalogModel::Hammerstein(model) => model.set_drive_db(value).unwrap(),
        AnalogModel::Tape(model) => model.set_drive_db(value).unwrap(),
        AnalogModel::Transformer(model) => model.set_drive_db(value).unwrap(),
        AnalogModel::ConsolePreamp(model) => model.set_input_gain_db(value).unwrap(),
    }
}

#[test]
fn randomized_callback_partitions_are_independent_for_every_model() {
    let channels = 2;
    let process_spec = ProcessSpec::new(48_000.0, channels, 128);
    let input: Vec<f32> = (0..128 * channels)
        .map(|index| ((index as f32) * 0.037).sin() * 0.8)
        .collect();

    for model_id in 0..=AnalogModel::CONSOLE_PREAMP_ID {
        let mut reference = AnalogModel::from_id(model_id).unwrap();
        reference.prepare(process_spec).unwrap();
        let mut expected = input.clone();
        reference.process_interleaved(&mut expected, 128).unwrap();

        let mut partitioned = AnalogModel::from_id(model_id).unwrap();
        partitioned.prepare(process_spec).unwrap();
        let mut actual = input.clone();
        let mut offset = 0;
        let mut state = 0x1f12_3ab5_u32;
        while offset < actual.len() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let remaining_frames = (actual.len() - offset) / channels;
            let frames = (1 + (state as usize % 31)).min(remaining_frames);
            let end = offset + frames * channels;
            partitioned
                .process_interleaved(&mut actual[offset..end], frames)
                .unwrap();
            offset = end;
        }

        assert_eq!(expected, actual, "model {model_id} depends on partitions");
    }
}

#[test]
fn automation_is_partition_independent_for_every_model() {
    let channels = 2;
    let process_spec = ProcessSpec::new(48_000.0, channels, 128);
    let first = vec![0.25_f32; 32 * channels];
    let second = vec![0.75_f32; 96 * channels];
    let partitions = [7_usize, 13, 5, 29, 1, 41];

    for model_id in 0..=AnalogModel::CONSOLE_PREAMP_ID {
        let mut reference = AnalogModel::from_id(model_id).unwrap();
        set_drive(&mut reference, -12.0);
        reference.prepare(process_spec).unwrap();
        let mut expected_first = first.clone();
        reference
            .process_interleaved(&mut expected_first, 32)
            .unwrap();
        set_drive(&mut reference, 24.0);
        let mut expected_second = second.clone();
        reference
            .process_interleaved(&mut expected_second, 96)
            .unwrap();

        let mut partitioned = AnalogModel::from_id(model_id).unwrap();
        set_drive(&mut partitioned, -12.0);
        partitioned.prepare(process_spec).unwrap();
        let mut actual_first = first.clone();
        partitioned
            .process_interleaved(&mut actual_first, 32)
            .unwrap();
        set_drive(&mut partitioned, 24.0);
        let mut actual_second = second.clone();
        let mut offset = 0;
        for frames in partitions {
            let end = offset + frames * channels;
            partitioned
                .process_interleaved(&mut actual_second[offset..end], frames)
                .unwrap();
            offset = end;
        }

        assert_eq!(expected_first, actual_first, "model {model_id} first block");
        assert_eq!(
            expected_second, actual_second,
            "model {model_id} automation"
        );
    }
}

#[test]
fn deterministic_stimuli_remain_finite_for_every_model() {
    let process_spec = ProcessSpec::new(48_000.0, 1, 256);
    let mut stimuli = vec![vec![0.0_f32; 256], vec![0.25_f32; 256]];
    let mut impulse = vec![0.0_f32; 256];
    impulse[0] = 1.0;
    stimuli.push(impulse);
    let mut burst = vec![0.0_f32; 256];
    burst[..32].fill(0.5);
    stimuli.push(burst);

    for model_id in 0..=AnalogModel::CONSOLE_PREAMP_ID {
        for mut samples in stimuli.clone() {
            let mut model = AnalogModel::from_id(model_id).unwrap();
            model.prepare(process_spec).unwrap();
            model.process_interleaved(&mut samples, 256).unwrap();
            assert!(
                samples.iter().all(|sample| sample.is_finite()),
                "model {model_id} produced non-finite deterministic-stimulus output"
            );
        }
    }
}

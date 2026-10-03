use audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{
    Adjustable, Async, FixedAsync, PolynomialDegree, Resampler, Resizable,
    SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

fn endpoint(mut index: f64, mut step: f64, target: f64, frames: usize) -> f64 {
    if frames == 0 {
        return index;
    }
    let increment = (target.recip() - step) / frames as f64;
    for _ in 0..frames {
        step += increment;
        index += step;
    }
    index
}

fn process_checked(resampler: &mut Async<f64>, length: usize, fixed: FixedAsync, target: f64) {
    let input_frames = resampler.input_frames_next();
    let output_frames = resampler.output_frames_next();
    assert!(
        input_frames <= resampler.input_frames_max(),
        "input capacity exceeded"
    );
    assert!(
        output_frames <= resampler.output_frames_max(),
        "output capacity exceeded"
    );
    let start = resampler.last_input_index();
    let step = resampler.resample_ratio().recip();
    let expected_end = endpoint(start, step, target, output_frames);
    let positions: Vec<_> = resampler.input_positions_next().collect();
    assert_eq!(positions.len(), output_frames);
    assert_eq!(positions.last().copied().unwrap_or(start), expected_end);
    match fixed {
        FixedAsync::Input => {
            let boundary = input_frames as f64 - (length + 1) as f64;
            if resampler.output_frames_max() <= 1024 {
                let mut expected_count = 0;
                for count in 1..=resampler.output_frames_max() + 16 {
                    if endpoint(start, step, target, count) <= boundary {
                        expected_count = count;
                    }
                }
                assert_eq!(output_frames, expected_count, "brute force count mismatch");
            } else {
                assert!(endpoint(start, step, target, output_frames + 1) > boundary);
            }
        }
        FixedAsync::Output => assert_eq!(
            input_frames,
            (expected_end + (length + 1) as f64).ceil() as usize
        ),
    }
    // Independent support guard, before calling the unsafe polynomial kernels.
    if let Some(first) = positions.first() {
        assert!(
            first.floor() - (length / 2) as f64 + resampler.input_history_frames() as f64 >= 0.0,
            "insufficient history: first={first}, L={length}, start={start}, r0={}, target={target}, input={input_frames}, output={output_frames}",
            resampler.resample_ratio()
        );
    }
    if output_frames > 0 {
        assert!(expected_end <= input_frames as f64 - (length + 1) as f64);
    }
    let input = vec![vec![0.125; input_frames]; 2];
    let mut output = vec![vec![9876.0; resampler.output_frames_max() + 3]; 2];
    let capacity = output[0].len();
    let result = resampler
        .process_into_buffer(
            &SequentialSliceOfVecs::new(&input, 2, input_frames).unwrap(),
            &mut SequentialSliceOfVecs::new_mut(&mut output, 2, capacity).unwrap(),
            None,
        )
        .unwrap();
    assert_eq!(result, (input_frames, output_frames));
    assert_eq!(
        resampler.last_input_index(),
        expected_end - input_frames as f64
    );
    assert_eq!(resampler.resample_ratio(), target);
    for channel in output {
        assert!(channel[output_frames..].iter().all(|x| *x == 9876.0));
    }
}

#[test]
fn discrete_ramp_checkpoint_and_reversed_ramp() {
    for fixed in [FixedAsync::Input, FixedAsync::Output] {
        let mut r = Async::new_poly(1.0, 2.0, PolynomialDegree::Cubic, 256, 2, fixed).unwrap();
        for _ in 0..4 {
            process_checked(&mut r, 4, fixed, 1.0);
        }
        let initial = r.last_input_index();
        r.set_resample_ratio(0.5, true).unwrap();
        if matches!(fixed, FixedAsync::Input) {
            assert_eq!(r.output_frames_next(), 170);
        } else {
            assert_eq!(
                r.input_frames_next(),
                (initial + 384.5 + 5.0).ceil() as usize
            );
        }
        process_checked(&mut r, 4, fixed, 0.5);
        r.set_resample_ratio(2.0, true).unwrap();
        process_checked(&mut r, 4, fixed, 2.0);
    }
}

#[test]
fn sinc_modes_sizes_and_ramp_directions_have_valid_support_and_capacities() {
    for length in [8, 64, 128, 256] {
        for interpolation in [
            SincInterpolationType::Nearest,
            SincInterpolationType::Linear,
            SincInterpolationType::Quadratic,
            SincInterpolationType::Cubic,
        ] {
            for fixed in [FixedAsync::Input, FixedAsync::Output] {
                for chunk in [1, 7, 256] {
                    let params = SincInterpolationParameters {
                        sinc_len: length,
                        f_cutoff: Some(0.9),
                        oversampling_factor: 128,
                        interpolation,
                        window: WindowFunction::BlackmanHarris2,
                    };
                    let mut r = Async::new_sinc(1.0, 2.0, &params, chunk, 2, fixed).unwrap();
                    for index in 0..24 {
                        let target = [0.5, 1.0, 2.0, 0.7, 1.9, 1.0][index % 6];
                        r.set_resample_ratio(target, index % 3 != 0).unwrap();
                        process_checked(&mut r, length, fixed, target);
                    }
                    r.reset();
                    process_checked(&mut r, length, fixed, 1.0);
                }
            }
        }
    }
}

#[test]
fn polynomial_degrees_sizes_and_ramps_have_valid_support_and_capacities() {
    for degree in [
        PolynomialDegree::Nearest,
        PolynomialDegree::Linear,
        PolynomialDegree::Cubic,
        PolynomialDegree::Quintic,
        PolynomialDegree::Septic,
    ] {
        for fixed in [FixedAsync::Input, FixedAsync::Output] {
            for chunk in [1, 7, 256] {
                let mut r = Async::new_poly(1.0, 2.0, degree, chunk, 2, fixed).unwrap();
                for index in 0..24 {
                    let target = [0.5, 1.0, 2.0, 0.7, 1.9, 1.0][index % 6];
                    r.set_resample_ratio(target, index % 3 != 0).unwrap();
                    process_checked(&mut r, degree.nbr_points(), fixed, target);
                }
                r.reset();
                process_checked(&mut r, degree.nbr_points(), fixed, 1.0);
            }
        }
    }
}

#[test]
fn audio_rate_extremes_have_valid_support_and_capacities() {
    for (output_rate, input_rate) in [
        (8.0, 384.0),
        (8.0, 192.0),
        (44.1, 48.0),
        (48.0, 44.1),
        (192.0, 8.0),
        (384.0, 8.0),
    ] {
        let nominal = output_rate / input_rate;
        for length in [64, 128, 256] {
            for fixed in [FixedAsync::Input, FixedAsync::Output] {
                for chunk in [1, 7, 256] {
                    let params = SincInterpolationParameters {
                        sinc_len: length,
                        f_cutoff: Some(0.9),
                        oversampling_factor: 128,
                        interpolation: SincInterpolationType::Linear,
                        window: WindowFunction::BlackmanHarris2,
                    };
                    let mut r = Async::new_sinc(nominal, 2.0, &params, chunk, 2, fixed).unwrap();
                    for index in 0..32 {
                        let target = nominal * [0.5, 1.0, 2.0, 0.7, 1.9, 1.0][index % 6];
                        r.set_resample_ratio(target, index % 3 != 0).unwrap();
                        process_checked(&mut r, length, fixed, target);
                    }
                }
            }
        }
    }
}

#[test]
fn nonfinite_ratios_are_rejected_before_planning() {
    for ratio in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -1.0,
        f64::from_bits(1),
    ] {
        assert!(Async::<f64>::new_poly(
            ratio,
            2.0,
            PolynomialDegree::Cubic,
            256,
            2,
            FixedAsync::Input
        )
        .is_err());
    }
    for relative in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, 0.5] {
        assert!(Async::<f64>::new_poly(
            1.0,
            relative,
            PolynomialDegree::Cubic,
            256,
            2,
            FixedAsync::Input
        )
        .is_err());
    }
}

#[test]
fn extreme_downsample_jump_retains_deferred_input_history() {
    for (chunk, low_blocks) in [(1, 96), (7, 13), (256, 3)] {
        let params = SincInterpolationParameters {
            sinc_len: 64,
            f_cutoff: Some(0.9),
            oversampling_factor: 128,
            interpolation: SincInterpolationType::Linear,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut r = Async::new_sinc(1.0 / 48.0, 2.0, &params, chunk, 2, FixedAsync::Input).unwrap();
        r.set_resample_ratio(1.0 / 96.0, false).unwrap();
        for _ in 0..low_blocks {
            process_checked(&mut r, 64, FixedAsync::Input, 1.0 / 96.0);
        }
        r.set_resample_ratio(1.0 / 24.0, false).unwrap();
        process_checked(&mut r, 64, FixedAsync::Input, 1.0 / 24.0);
    }
}

#[test]
fn linear_signal_matches_independent_absolute_source_coordinate() {
    for fixed in [FixedAsync::Input, FixedAsync::Output] {
        let mut r = Async::new_poly(1.0, 8.0, PolynomialDegree::Linear, 37, 1, fixed).unwrap();
        let mut base = 0usize;
        for block in 0..60 {
            if block > 4 {
                r.set_chunk_size([1, 7, 37][block % 3]).unwrap();
                r.set_resample_ratio([0.125, 8.0, 0.7, 1.9][block % 4], block % 2 == 0)
                    .unwrap();
            }
            let frames = r.input_frames_next();
            let expected: Vec<_> = r.input_positions_next().map(|x| base as f64 + x).collect();
            let input = vec![(0..frames)
                .map(|index| (base + index) as f64)
                .collect::<Vec<_>>()];
            let mut output = vec![vec![9876.0; r.output_frames_max() + 3]];
            let capacity = output[0].len();
            let (_, written) = r
                .process_into_buffer(
                    &SequentialSliceOfVecs::new(&input, 1, frames).unwrap(),
                    &mut SequentialSliceOfVecs::new_mut(&mut output, 1, capacity).unwrap(),
                    None,
                )
                .unwrap();
            assert_eq!(written, expected.len());
            if block > 0 {
                for (actual, expected) in output[0][..written].iter().zip(expected) {
                    // One subtract, multiply and add in linear interpolation.
                    let rounding_bound = 4.0 * f64::EPSILON * expected.abs().max(1.0);
                    assert!(
                        (actual - expected).abs() <= rounding_bound,
                        "absolute source coordinate {actual} != {expected}"
                    );
                }
            }
            assert!(output[0][written..].iter().all(|x| *x == 9876.0));
            base += frames;
        }
    }
}

#[test]
fn offset_partial_input_reset_and_chunk_changes_match_fresh_state() {
    for fixed in [FixedAsync::Input, FixedAsync::Output] {
        let params = SincInterpolationParameters {
            sinc_len: 64,
            f_cutoff: Some(0.9),
            oversampling_factor: 128,
            interpolation: SincInterpolationType::Linear,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut used =
            Async::new_sinc_with_cutoff_bank(1.0, 2.0, &params, &[0.5, 0.75], 256, 1, fixed)
                .unwrap();
        let mut fresh =
            Async::new_sinc_with_cutoff_bank(1.0, 2.0, &params, &[0.5, 0.75], 256, 1, fixed)
                .unwrap();
        used.select_sinc_cutoff(1);
        used.set_resample_ratio(0.5, true).unwrap();
        let input = vec![vec![0.25; used.input_frames_next()]];
        let mut output = vec![vec![0.0; used.output_frames_max()]];
        let capacity = output[0].len();
        used.process_into_buffer(
            &SequentialSliceOfVecs::new(&input, 1, input[0].len()).unwrap(),
            &mut SequentialSliceOfVecs::new_mut(&mut output, 1, capacity).unwrap(),
            None,
        )
        .unwrap();
        used.set_chunk_size(7).unwrap();
        used.reset();
        assert_eq!(used.last_input_index(), fresh.last_input_index());
        assert_eq!(used.input_frames_next(), fresh.input_frames_next());
        for chunk in [7, 1, 256] {
            used.set_chunk_size(chunk).unwrap();
            fresh.set_chunk_size(chunk).unwrap();
            let input_count = used.input_frames_next();
            let prefix = 5;
            let output_prefix = 3;
            let input = vec![(0..input_count + prefix)
                .map(|i| (0.137 * i as f64).sin())
                .collect::<Vec<_>>()];
            let mut a = vec![vec![9876.0; used.output_frames_max() + output_prefix + 4]];
            let mut b = a.clone();
            let capacity = a[0].len();
            let indexing = rubato::Indexing {
                input_offset: prefix,
                output_offset: output_prefix,
                partial_len: Some(input_count.saturating_sub(1)),
                active_channels_mask: None,
            };
            let first = used
                .process_into_buffer(
                    &SequentialSliceOfVecs::new(&input, 1, input[0].len()).unwrap(),
                    &mut SequentialSliceOfVecs::new_mut(&mut a, 1, capacity).unwrap(),
                    Some(&indexing),
                )
                .unwrap();
            let second = fresh
                .process_into_buffer(
                    &SequentialSliceOfVecs::new(&input, 1, input[0].len()).unwrap(),
                    &mut SequentialSliceOfVecs::new_mut(&mut b, 1, capacity).unwrap(),
                    Some(&indexing),
                )
                .unwrap();
            assert_eq!(first, second);
            assert_eq!(a, b);
            assert!(a[0][..output_prefix].iter().all(|x| *x == 9876.0));
            assert!(a[0][output_prefix + first.1..].iter().all(|x| *x == 9876.0));
        }
    }
}

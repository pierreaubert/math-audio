use math_audio_analog::{AnalogModel, AnalogProcessor, ProcessSpec};
use math_audio_dsp::simd::{analog_simd_available, chebyshev_basis_scalar, chebyshev_basis_simd};
use std::{hint::black_box, time::Instant};

/// Keep the performance baseline scalar even when release LLVM auto-vectorizes
/// the production reference helper.  The opaque input load prevents LLVM from
/// turning this test-only loop into a second SIMD implementation.
#[inline(never)]
fn scalar_chebyshev_baseline(input: &[f32], output: &mut [f32], order: usize) {
    let length = input.len().min(output.len());
    for (z, result) in input[..length].iter().zip(output[..length].iter_mut()) {
        let z = black_box(*z);
        let mut previous = 1.0_f32;
        let mut current = z;
        for _ in 2..=order {
            let next = 2.0 * z * current - previous;
            previous = current;
            current = next;
        }
        *result = current;
    }
}

#[test]
fn worst_callback_report_covers_all_models_and_channel_widths() {
    let mut worst_ns = 0_u128;
    let mut worst_case = (0_u32, 0_usize);
    for channels in [1, 2, 6, 12] {
        for model_id in 0..=AnalogModel::CONSOLE_PREAMP_ID {
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
    // Provisional engineering guard from analog-20260818-roadmap.md:
    // no model may consume more than 25% of a 2,048-frame 48 kHz callback.
    assert!(worst_ns > 0);
    if cfg!(debug_assertions) {
        // Debug arithmetic is diagnostic only; the hard CI gate is the
        // release-mode branch below.
        eprintln!("debug timing is diagnostic; rerun this test with --release for the 25% gate");
    } else {
        let callback_budget_ns = 2_048_u128 * 1_000_000_000 / 48_000 * 25 / 100;
        assert!(
            worst_ns <= callback_budget_ns,
            "worst callback exceeded the provisional realtime budget: {worst_ns} ns > {callback_budget_ns} ns"
        );
    }
}

#[test]
fn stateful_models_survive_denormal_stress() {
    for model_id in [AnalogModel::TAPE_ID, AnalogModel::TRANSFORMER_ID] {
        let mut model = AnalogModel::from_id(model_id).unwrap();
        model.prepare(ProcessSpec::new(48_000.0, 2, 2_048)).unwrap();
        let mut block = vec![f32::MIN_POSITIVE * 0.25; 2 * 2_048];
        for _ in 0..32 {
            model.process_interleaved(&mut block, 2_048).unwrap();
        }
        assert!(block.iter().all(|sample| sample.is_finite()));
    }
}

#[test]
fn simd_chebyshev_criterion_covers_six_and_twelve_channels() {
    let input: Vec<f32> = (0..2_048)
        .map(|index| (index as f32 * 0.0137).sin() * 0.9)
        .collect();
    for channels in [6_usize, 12] {
        let mut scalar = vec![0.0_f32; input.len()];
        let mut simd = vec![0.0_f32; input.len()];
        for _ in 0..8 {
            for _ in 0..channels {
                scalar_chebyshev_baseline(&input, &mut scalar, 5);
                chebyshev_basis_simd(&input, &mut simd, 5);
            }
        }
        let scalar_start = Instant::now();
        let mut scalar_checksum = 0.0_f32;
        for _ in 0..64 {
            for _ in 0..channels {
                scalar_chebyshev_baseline(&input, &mut scalar, 5);
                scalar_checksum += scalar[17];
            }
        }
        let scalar_ns = scalar_start.elapsed().as_nanos();
        let simd_start = Instant::now();
        let mut simd_checksum = 0.0_f32;
        for _ in 0..64 {
            for _ in 0..channels {
                chebyshev_basis_simd(&input, &mut simd, 5);
                simd_checksum += simd[17];
            }
        }
        let simd_ns = simd_start.elapsed().as_nanos();
        let mut scalar_reference = vec![0.0_f32; input.len()];
        chebyshev_basis_scalar(&input, &mut scalar_reference, 5);
        assert_eq!(
            scalar_reference, simd,
            "SIMD changed the order-5 recurrence"
        );
        assert_eq!(scalar, simd, "SIMD changed the order-5 recurrence");
        assert!((scalar_checksum - simd_checksum).abs() < 1e-3);
        println!(
            "math-analog SIMD criterion: channels={channels} scalar_ns={scalar_ns} simd_ns={simd_ns} available={}",
            analog_simd_available()
        );
        if !cfg!(debug_assertions) && analog_simd_available() {
            assert!(
                simd_ns < scalar_ns,
                "SIMD kernel did not beat scalar reference at {channels} channels: {simd_ns} >= {scalar_ns}"
            );
        }
    }
}

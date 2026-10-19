use math_audio_analog::{AnalogModel, AnalogProcessor, ProcessSpec};
use math_audio_dsp::simd::{analog_simd_available, chebyshev_basis_scalar, chebyshev_basis_simd};
use std::{hint::black_box, time::Instant};
mod audit_support;

// These tests measure CPU work; running them concurrently contaminates their
// own latency and speed comparisons even on an otherwise idle runner.
static TIMING_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn thread_cpu_ns() -> Option<u128> {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: the clock ID is valid for this platform, and the OS receives a
    // live, properly aligned, writable timespec for the duration of the call.
    let status = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) };
    if status == 0 && time.tv_sec >= 0 && time.tv_nsec >= 0 {
        Some(time.tv_sec as u128 * 1_000_000_000 + time.tv_nsec as u128)
    } else {
        None
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn thread_cpu_ns() -> Option<u128> {
    None
}

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
    let _timing = TIMING_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut worst_ns = 0_u128;
    let mut worst_cpu_ns = 0_u128;
    let mut cpu_available = true;
    let mut worst_case = (0_u32, 0_usize);
    for channels in [1, 2, 6, 12] {
        for model_id in 0..=AnalogModel::TONE_STACK_ID {
            let spec = ProcessSpec::new(48_000.0, channels, 2_048);
            let mut model = AnalogModel::from_id(model_id).unwrap();
            model.prepare(spec).unwrap();
            let mut block = vec![0.0_f32; channels * 2_048];
            let mut local_worst = 0_u128;
            for _ in 0..16 {
                let cpu_start = thread_cpu_ns();
                let start = Instant::now();
                model.process_interleaved(&mut block, 2_048).unwrap();
                local_worst = local_worst.max(start.elapsed().as_nanos());
                if let (Some(before), Some(after)) = (cpu_start, thread_cpu_ns()) {
                    worst_cpu_ns = worst_cpu_ns.max(after.saturating_sub(before));
                } else {
                    cpu_available = false;
                }
            }
            if local_worst > worst_ns {
                worst_ns = local_worst;
                worst_case = (model_id, channels);
            }
        }
    }
    println!(
        "math-analog worst callback: wall={worst_ns} ns cpu={worst_cpu_ns} ns, model={}, channels={}",
        worst_case.0, worst_case.1
    );
    // Provisional engineering guard from analog-20260818-roadmap.md:
    // no model may consume more than 25% of a 2,048-frame 48 kHz callback.
    assert!(worst_ns > 0);
    let under_coverage = std::env::var_os("LLVM_PROFILE_FILE").is_some();
    if cfg!(debug_assertions) || under_coverage {
        eprintln!(
            "debug/instrumented timing is diagnostic; the 25% gate requires an uninstrumented release build"
        );
    } else {
        let callback_budget_ns = 2_048_u128 * 1_000_000_000 / 48_000 * 25 / 100;
        // CPU time enforces the work budget on shared runners. The wall clock
        // additionally includes scheduling pauses outside the DSP's control.
        let work_ns = if cpu_available {
            worst_cpu_ns
        } else {
            worst_ns
        };
        assert!(
            work_ns > 0 && work_ns <= callback_budget_ns,
            "worst callback work exceeded the 25% budget: {work_ns} ns > {callback_budget_ns} ns"
        );
        if std::env::var_os("MATH_AUDIO_ENFORCE_REALTIME").is_some() {
            assert!(
                worst_ns <= callback_budget_ns,
                "isolated-runner wall deadline exceeded"
            );
        }
    }
}

#[test]
fn stateful_models_survive_denormal_stress() {
    let _timing = TIMING_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for model_id in [
        AnalogModel::TAPE_ID,
        AnalogModel::TRANSFORMER_ID,
        AnalogModel::DIODE_CLIPPER_ID,
        AnalogModel::TRIODE_STAGE_ID,
        AnalogModel::TONE_STACK_ID,
    ] {
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
    let _timing = TIMING_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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
        let mut portable = vec![0.0_f32; input.len()];
        let portable_start = Instant::now();
        for _ in 0..64 {
            for _ in 0..channels {
                chebyshev_basis_scalar(black_box(&input), black_box(&mut portable), 5);
            }
        }
        let portable_ns = portable_start.elapsed().as_nanos();
        println!(
            "portable compiled baseline channels={channels} portable_ns={portable_ns} explicit_simd_ns={simd_ns}"
        );
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
        // Timing assertions are meaningless under coverage instrumentation
        // (-C instrument-coverage), where counter writes dominate and the
        // SIMD speedup disappears; llvm-cov sets LLVM_PROFILE_FILE there.
        let under_coverage = std::env::var_os("LLVM_PROFILE_FILE").is_some();
        if !cfg!(debug_assertions) && !under_coverage && analog_simd_available() {
            assert!(
                simd_ns < scalar_ns,
                "SIMD kernel did not beat scalar reference at {channels} channels: {simd_ns} >= {scalar_ns}"
            );
        }
    }
}

#[test]
fn driven_small_callbacks_report_latency_distributions_and_budget() {
    let _timing = TIMING_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let enforce = std::env::var_os("MATH_AUDIO_ENFORCE_REALTIME").is_some();
    for channels in [1, 2, 6, 12] {
        for model_id in 0..=AnalogModel::TONE_STACK_ID {
            for frames in [32, 64, 128, 256] {
                let mut model = AnalogModel::from_id(model_id).unwrap();
                audit_support::drive_model(&mut model, 6.0);
                model
                    .prepare(ProcessSpec::new(48000.0, channels, 256))
                    .unwrap();
                let mut block = vec![0.0; channels * frames];
                let mut times = [0_u128; 32];
                let mut cpu_times = [0_u128; 32];
                let mut cpu_available = true;
                for (iteration, elapsed) in times.iter_mut().enumerate() {
                    audit_support::fill_input(&mut block, channels, iteration % 8);
                    let cpu_start = thread_cpu_ns();
                    let start = Instant::now();
                    if iteration == 8 {
                        audit_support::drive_model(&mut model, 24.0);
                    }
                    if iteration == 16 {
                        model.reset();
                    }
                    model
                        .process_interleaved(black_box(&mut block), frames)
                        .unwrap();
                    *elapsed = start.elapsed().as_nanos();
                    if let (Some(before), Some(after)) = (cpu_start, thread_cpu_ns()) {
                        cpu_times[iteration] = after.saturating_sub(before);
                    } else {
                        cpu_available = false;
                    }
                }
                let worst_cpu_iteration = cpu_times
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, elapsed)| **elapsed)
                    .map(|(index, _)| index)
                    .unwrap();
                times.sort_unstable();
                cpu_times.sort_unstable();
                let budget = frames as u128 * 1_000_000_000 / 48000 / 4;
                println!(
                    "small callback model={model_id} channels={channels} frames={frames} wall_median_ns={} wall_p95_ns={} wall_worst_ns={} cpu_median_ns={} cpu_p95_ns={} cpu_worst_ns={} cpu_available={cpu_available} worst_cpu_iteration={worst_cpu_iteration} budget25pct_ns={budget}",
                    times[16], times[30], times[31], cpu_times[16], cpu_times[30], cpu_times[31]
                );
                assert!(times[31] > 0);
                let under_coverage = std::env::var_os("LLVM_PROFILE_FILE").is_some();
                if !cfg!(debug_assertions) && !under_coverage {
                    let work = if cpu_available {
                        cpu_times[31]
                    } else {
                        times[31]
                    };
                    assert!(
                        work > 0 && work <= budget,
                        "small callback work exceeded 25% budget: {work} > {budget}"
                    );
                    // Isolated runners can additionally enforce the wall deadline.
                    if enforce {
                        assert!(times[31] <= budget, "small callback wall deadline exceeded");
                    }
                }
            }
        }
    }
}

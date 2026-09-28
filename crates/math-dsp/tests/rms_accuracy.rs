//! Direct-window numerical, lifecycle, and realtime checks for the sum tree.

// Rust guideline compliant 2026-02-21
use math_audio_dsp::detector as tree;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::f64::consts::TAU;

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static FREES: Cell<usize> = const { Cell::new(0) };
}
struct CallbackAllocator;
// SAFETY: Pointer/layout contracts are forwarded unchanged to System; tracking
// uses nonallocating, constant-initialized thread-local cells.
unsafe impl GlobalAlloc for CallbackAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = TRACKING.try_with(|tracking| {
            if tracking.get() {
                ALLOCATIONS.with(|n| n.set(n.get() + 1));
            }
        });
        // SAFETY: The original allocation contract is forwarded unchanged.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        let _ = TRACKING.try_with(|tracking| {
            if tracking.get() {
                FREES.with(|n| n.set(n.get() + 1));
            }
        });
        // SAFETY: This pointer and layout come from the original allocation.
        unsafe { System.dealloc(pointer, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: CallbackAllocator = CallbackAllocator;

fn window(rate: u32, millis: f32) -> usize {
    (millis * 0.001 * rate as f32).round().max(1.0) as usize
}

fn reference(signal: &[f32], frame: usize, window: usize) -> f64 {
    let start = (frame + 1).saturating_sub(window);
    // Positive direct f64 sum is independent of tree shape and update order.
    let energy: f64 = signal[start..=frame]
        .iter()
        .map(|&sample| f64::from(sample).powi(2))
        .sum();
    (energy / window as f64).sqrt()
}

#[test]
fn normal_matrix_matches_independent_f64_windows() {
    let mut total = 0;
    let mut max_oracle_relative = 0.0_f64;
    for rate in [8_000, 44_100, 48_000, 96_000, 192_000] {
        for millis in [0.01, 1.0, 10.0, 50.0] {
            let count = window(rate, millis);
            for pattern in 0..5 {
                let mut seed = 0x5a17_u32;
                let signal: Vec<_> = (0..count * 3 + 511)
                    .map(|frame| {
                        seed ^= seed << 13;
                        seed ^= seed >> 17;
                        seed ^= seed << 5;
                        let value = match pattern {
                            0 => 0.1,
                            1 => -0.875,
                            2 => 0.8 * (TAU * 997.0 * frame as f64 / rate as f64).sin(),
                            3 => {
                                0.4 * (TAU * 123.0 * frame as f64 / rate as f64).sin()
                                    + 0.3 * (TAU * 2713.0 * frame as f64 / rate as f64).cos()
                            }
                            _ => (f64::from(seed) / f64::from(u32::MAX) * 2.0 - 1.0) * 0.9,
                        };
                        value as f32
                    })
                    .collect();
                let mut actual =
                    tree::LevelDetector::new(tree::DetectionMode::Rms { window_ms: millis }, rate);
                for (frame, &sample) in signal.iter().enumerate() {
                    let output = actual.process_linear(sample);
                    total += 1;
                    if frame % 113 == 0
                        || frame + 1 == signal.len()
                        || [count - 1, count, count * 2].contains(&frame)
                    {
                        let expected = reference(&signal, frame, count);
                        if expected > 0.0 {
                            max_oracle_relative = max_oracle_relative
                                .max((f64::from(output) - expected).abs() / expected);
                        }
                        assert!(
                            output.to_bits().abs_diff((expected as f32).to_bits()) <= 1,
                            "rate={rate} ms={millis} pattern={pattern} frame={frame}: {output} vs {expected}"
                        );
                    }
                }
            }
        }
    }
    assert_eq!(total, 406300);
    assert!(max_oracle_relative <= f64::from(f32::EPSILON));
}

fn check_all_samples(signal: &[f32], rate: u32, millis: f32) {
    let count = window(rate, millis);
    let mut detector =
        tree::LevelDetector::new(tree::DetectionMode::Rms { window_ms: millis }, rate);
    for (frame, &sample) in signal.iter().enumerate() {
        let actual = detector.process_linear(sample);
        let expected = reference(signal, frame, count) as f32;
        assert!(actual.is_finite());
        assert!(
            actual.to_bits().abs_diff(expected.to_bits()) <= 1,
            "rate={rate} ms={millis} frame={frame}, actual={actual:e}, expected={expected:e}"
        );
        if expected == 0.0 {
            assert_eq!(actual, 0.0, "zero window must have zero root");
        }
    }
}

#[test]
fn repeated_overlapping_maximum_pulses_and_backgrounds_match_direct_windows() {
    let mut configurations = 0;
    for rate in [8_000, 44_100, 48_000, 96_000, 192_000] {
        let count = window(rate, 10.0);
        for amplitude in [1.0e20_f32, f32::MAX] {
            for pulses in [1, 2, 3, 17] {
                for background in [0.0, 0.1, 1.0e-5] {
                    let mut signal = vec![background; count * 4];
                    for index in 0..pulses {
                        signal[count + index * (count / 19).max(1)] = if index % 2 == 0 {
                            amplitude
                        } else {
                            -amplitude
                        };
                    }
                    check_all_samples(&signal, rate, 10.0);
                    configurations += 1;
                }
            }
        }
        let signal = vec![f32::MAX; count * 3];
        check_all_samples(&signal, rate, 10.0);
    }
    println!(
        "tree extreme matrix: {configurations} pulse/background configs +5 constant maximum configs, every output ≤1ULP from direct f64 window"
    );
}

#[test]
fn magnitude_ladders_reveal_retained_smaller_pulses_after_dominant_ones_exit() {
    let ladder = [
        f32::MAX,
        1.0e30,
        1.0e20,
        1.0e10,
        1.0,
        1.0e-10,
        1.0e-20,
        1.0e-30,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
    ];
    for rate in [8_000, 44_100, 48_000, 192_000] {
        let count = window(rate, 10.0);
        for reversed in [false, true] {
            for phase in [0, count - 1] {
                let mut signal = vec![0.0; count * 4];
                for index in 0..ladder.len() {
                    let magnitude = ladder[if reversed {
                        ladder.len() - 1 - index
                    } else {
                        index
                    }];
                    signal[phase + index * (count / 13).max(1)] = if index % 2 == 0 {
                        magnitude
                    } else {
                        -magnitude
                    };
                }
                check_all_samples(&signal, rate, 10.0);
            }
        }
    }
    println!(
        "tree magnitude ladder:16 configurations, full finite f32 exponent span, every output ≤1ULP from direct f64 window"
    );
}

#[test]
fn reset_and_mode_changes_restore_fresh_state_across_window_sizes() {
    for rate in [8_000, 44_100, 48_000, 96_000, 192_000] {
        let mut detector = tree::LevelDetector::new(tree::DetectionMode::Peak, rate);
        for sample in [-f32::MAX, -0.2, 0.0, 0.75, f32::MAX] {
            assert_eq!(detector.process_linear(sample), sample.abs());
        }
        for millis in [0.01, 1.0, 10.0, 50.0, 1.0] {
            detector.set_mode(tree::DetectionMode::Rms { window_ms: millis });
            let count = window(rate, millis);
            for _ in 0..count {
                detector.process_linear(f32::MAX);
            }
            detector.reset();
            let mut fresh =
                tree::LevelDetector::new(tree::DetectionMode::Rms { window_ms: millis }, rate);
            assert_eq!(detector.sample_rate(), rate);
            assert_eq!(detector.mode(), fresh.mode());
            for frame in 0..count * 2 + 17 {
                let sample = (frame as f64 * 0.13).sin() as f32 * 0.7;
                assert_eq!(
                    detector.process_linear(sample),
                    fresh.process_linear(sample)
                );
            }
        }
        detector.set_mode(tree::DetectionMode::Peak);
        assert_eq!(detector.process_linear(-0.25), 0.25);
    }
}

#[test]
fn cold_processing_and_reset_have_zero_allocations_and_frees() {
    for (rate, millis) in [(8_000, 0.01), (48_000, 10.0), (192_000, 50.0)] {
        let mut detector =
            tree::LevelDetector::new(tree::DetectionMode::Rms { window_ms: millis }, rate);
        let count = window(rate, millis);
        let counts = std::thread::spawn(move || {
            ALLOCATIONS.with(|n| n.set(0));
            FREES.with(|n| n.set(0));
            TRACKING.with(|tracking| tracking.set(true));
            for frame in 0..count * 4 {
                std::hint::black_box(detector.process_linear(if frame == count {
                    f32::MAX
                } else {
                    0.1
                }));
            }
            detector.reset();
            for _ in 0..count {
                std::hint::black_box(detector.process_linear(0.2));
            }
            TRACKING.with(|tracking| tracking.set(false));
            (ALLOCATIONS.with(Cell::get), FREES.with(Cell::get))
        })
        .join()
        .unwrap();
        assert_eq!(counts, (0, 0));
        println!(
            "tree cold rate={rate} window={count}: allocations={} frees={}",
            counts.0, counts.1
        );
    }
}

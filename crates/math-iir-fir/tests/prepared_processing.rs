//! Preparation must cover first, changing-size, oversized, and reset callbacks.

// Rust guideline compliant 2026-02-21
use math_audio_iir_fir::filtfilt::{
    FiltfiltWorkspace, filtfilt, filtfilt_into, peq_to_coefficients,
};
use math_audio_iir_fir::{FilterFloat, Fir};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region};

#[global_allocator]
static GLOBAL: &stats_alloc::StatsAlloc<std::alloc::System> = &INSTRUMENTED_SYSTEM;

fn fir_contract<T: FilterFloat>() {
    for symmetric in [true, false] {
        for taps in [31, 128, 512] {
            let mut coefficients = vec![T::from_f64(1.0 / taps as f64).unwrap(); taps];
            if !symmetric {
                coefficients[0] *= T::from_f64(0.9).unwrap();
            }
            let mut filter = Fir::new_custom(coefficients.clone(), T::from_f64(48000.0).unwrap());
            filter.prepare(256);
            let source: Vec<T> = (0..2048)
                .map(|n| T::from_f64((n as f64 * 0.137).sin()).unwrap())
                .collect();
            let expected: Vec<f64> = (0..source.len())
                .map(|n| {
                    coefficients
                        .iter()
                        .enumerate()
                        .take(n + 1)
                        .map(|(tap, coefficient)| {
                            coefficient.to_f64().unwrap() * source[n - tap].to_f64().unwrap()
                        })
                        .sum()
                })
                .collect();
            let mut output = source.clone();
            for reset in [false, true] {
                if reset {
                    filter.reset();
                }
                output.copy_from_slice(&source);
                let region = Region::new(GLOBAL);
                let mut offset = 0;
                for length in [32, 64, 128, 256, 1024, 544] {
                    filter.process_block(&mut output[offset..offset + length]);
                    offset += length;
                }
                let stats = region.change();
                assert_eq!(stats.allocations + stats.reallocations, 0, "{stats:?}");
                let tolerance = 100.0 * T::epsilon().to_f64().unwrap();
                for (&actual, &expected) in output.iter().zip(&expected) {
                    assert!((actual.to_f64().unwrap() - expected).abs() < tolerance);
                }
            }
        }
    }
}

#[test]
fn prepared_fir_and_filtfilt_contracts() {
    fir_contract::<f32>();
    fir_contract::<f64>();
    let sections = peq_to_coefficients(&math_audio_iir_fir::peq_butterworth_lowpass(
        4, 4000.0, 48000.0,
    ));
    let input: Vec<_> = (0..48000).map(|n| (n as f64 * 0.123).sin()).collect();
    let expected = filtfilt(&input, &sections);
    let mut output = vec![0.0; input.len()];
    let mut workspace = FiltfiltWorkspace::new();
    workspace.prepare(input.len(), sections.len());
    let region = Region::new(GLOBAL);
    filtfilt_into(&input, &sections, &mut output, &mut workspace);
    let stats = region.change();
    assert_eq!(stats.allocations + stats.reallocations, 0, "{stats:?}");
    assert_eq!(output, expected);
}

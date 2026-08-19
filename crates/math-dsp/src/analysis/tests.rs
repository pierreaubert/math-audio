use super::apply::apply_hann_window;
use super::compute::compute_coherence_from_realizations;
use super::compute::compute_impulse_response_from_fr;
use super::compute::compute_single_fft_spectrum_internal;
use super::compute::compute_spectrogram;
use super::compute::compute_thd_from_ir;
use super::compute::compute_welch_spectrum_internal;
use super::compute::compute_windowed_fr;
use super::estimate::estimate_lag;
use super::estimate::estimate_noise_floor_db_from_silence;
use super::misc::next_power_of_two;
use super::plan::cross_correlate_envelope;
use super::plan::deconvolve_sweep;
use super::*;
use rustfft::num_complex::Complex;
use std::f32::consts::PI;

mod misc;

fn bin_centered_sine(amplitude: f32, bin: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|n| amplitude * (2.0 * PI * bin as f32 * n as f32 / len as f32).sin())
        .collect()
}

#[test]
fn single_fft_reports_peak_amplitude_independent_of_fft_size() {
    let amplitude = 0.25_f32;
    let expected_db = 20.0 * amplitude.log10();

    for (fft_size, bin) in [(1024, 32), (2048, 64)] {
        let signal = bin_centered_sine(amplitude, bin, fft_size);
        for no_window in [false, true] {
            let (_, magnitude_db, _) =
                compute_single_fft_spectrum_internal(&signal, 48_000, fft_size, no_window).unwrap();
            assert!(
                (magnitude_db[bin] - expected_db).abs() < 0.05,
                "fft_size={fft_size}, no_window={no_window}: expected {expected_db:.3} dBFS, got {:.3}",
                magnitude_db[bin]
            );
        }
    }
}

#[test]
fn welch_reports_peak_amplitude_independent_of_fft_size() {
    let amplitude = 0.25_f32;
    let expected_db = 20.0 * amplitude.log10();

    for (fft_size, bin) in [(1024, 32), (2048, 64)] {
        let signal = bin_centered_sine(amplitude, bin * 8, fft_size * 8);
        let (_, magnitude_db, _) =
            compute_welch_spectrum_internal(&signal, 48_000, fft_size, 0.5).unwrap();
        assert!(
            (magnitude_db[bin] - expected_db).abs() < 0.05,
            "fft_size={fft_size}: expected {expected_db:.3} dBFS, got {:.3}",
            magnitude_db[bin]
        );
    }
}

#[test]
fn welch_includes_a_trailing_partial_frame() {
    let fft_size = 1024;
    let bin = 32;
    let mut signal = vec![0.0_f32; fft_size + 3 * fft_size / 4];
    let tail_start = fft_size + fft_size / 2;
    for (n, sample) in signal[tail_start..].iter_mut().enumerate() {
        *sample = 0.5 * (2.0 * PI * bin as f32 * n as f32 / fft_size as f32).sin();
    }

    let (_, magnitude_db, _) =
        compute_welch_spectrum_internal(&signal, 48_000, fft_size, 0.5).unwrap();
    assert!(
        magnitude_db[bin] > -80.0,
        "the only non-zero tail must contribute to Welch averaging, got {:.1} dB",
        magnitude_db[bin]
    );
}

#[cfg(test)]
mod gd_1c_tests {
    use super::*;

    use std::f32::consts::PI;

    #[test]
    fn coherence_single_realization_is_unity() {
        // Issue #7: N < 4 now returns an error instead of misleading γ² = 1.
        let h = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.5, 0.5),
            Complex::new(0.0, 1.0),
        ];
        let result = compute_coherence_from_realizations(&[h]);
        assert!(result.is_err(), "N=1 should error, not return γ² = 1");
        let err = result.unwrap_err();
        assert!(
            err.contains("at least 4 realizations"),
            "error should mention N≥4 requirement, got: {err}"
        );
    }

    #[test]
    fn coherence_too_few_realizations_errors() {
        let r = vec![Complex::new(1.0, 0.0)];
        for n in [2usize, 3] {
            let realizations: Vec<_> = (0..n).map(|_| r.clone()).collect();
            let result = compute_coherence_from_realizations(&realizations);
            assert!(result.is_err(), "N={n} should error, not return γ² = 1");
        }
    }

    #[test]
    fn coherence_identical_realizations_is_unity() {
        let r = vec![
            Complex::new(0.8, 0.2),
            Complex::new(0.0, 1.0),
            Complex::new(-0.5, 0.5),
        ];
        let realizations = vec![r.clone(), r.clone(), r.clone(), r];
        let coh = compute_coherence_from_realizations(&realizations).unwrap();
        for c in coh {
            assert!(
                (c - 1.0).abs() < 1e-6,
                "identical realizations → γ² = 1, got {c}"
            );
        }
    }

    #[test]
    fn coherence_random_realizations_is_zero() {
        // Four realizations whose phases cancel out on average:
        // ±1 and ±i. The complex mean is 0, so γ² = 0.
        let bins = 3;
        let r0: Vec<Complex<f32>> = (0..bins).map(|_| Complex::new(1.0, 0.0)).collect();
        let r1: Vec<Complex<f32>> = (0..bins).map(|_| Complex::new(-1.0, 0.0)).collect();
        let r2: Vec<Complex<f32>> = (0..bins).map(|_| Complex::new(0.0, 1.0)).collect();
        let r3: Vec<Complex<f32>> = (0..bins).map(|_| Complex::new(0.0, -1.0)).collect();
        let coh = compute_coherence_from_realizations(&[r0, r1, r2, r3]).unwrap();
        for c in coh {
            assert!(c < 1e-6, "canceling-phase realizations → γ² ≈ 0, got {c}");
        }
    }

    #[test]
    fn coherence_rejects_mismatched_lengths() {
        let r0 = vec![Complex::new(1.0_f32, 0.0); 3];
        let r1 = vec![Complex::new(1.0_f32, 0.0); 4];
        let r2 = vec![Complex::new(1.0_f32, 0.0); 3];
        let r3 = vec![Complex::new(1.0_f32, 0.0); 4];
        let err = compute_coherence_from_realizations(&[r0, r1, r2, r3]).unwrap_err();
        assert!(err.contains("has 4 bins, expected 3"), "got: {err}");
    }

    #[test]
    fn coherence_empty_input_errors() {
        let err = compute_coherence_from_realizations(&[]).unwrap_err();
        assert!(err.contains("empty"), "got: {err}");
    }

    #[test]
    fn deconvolve_matches_unity_system() {
        // If the recorded signal IS the emitted sweep, H should be
        // approximately 1 across the passband.
        let n: usize = 1024;
        let sr = 48_000_u32;
        let sweep: Vec<f32> = (0..n)
            .map(|k| {
                let t = k as f32 / sr as f32;
                let f = 100.0 * (10.0_f32).powf(3.0 * t / (n as f32 / sr as f32));
                (2.0 * PI * f * t).sin() * 0.5
            })
            .collect();
        let recording = sweep.clone();
        let h = deconvolve_sweep(&recording, &sweep, sr).unwrap();
        assert_eq!(h.len(), n.next_power_of_two() / 2 + 1);
        // Mid-band bins should be ≈ 1 (within the regularisation
        // floor). Check bins 10..50 — avoids DC where the sweep has
        // no energy and the Nyquist edge where the log sweep dies out.
        let mid_slice = &h[10..50];
        for (i, c) in mid_slice.iter().enumerate() {
            let mag = c.norm();
            assert!(
                mag > 0.1 && mag < 10.0,
                "bin {} magnitude {mag} out of expected range",
                i + 10
            );
        }
    }

    #[test]
    fn deconvolve_rejects_length_mismatch() {
        let a = vec![0.0_f32; 10];
        let b = vec![0.0_f32; 11];
        let err = deconvolve_sweep(&a, &b, 48_000).unwrap_err();
        assert!(err.contains("!="), "got: {err}");
    }

    #[test]
    fn deconvolve_accepts_and_retains_a_recording_tail() {
        let reference = crate::signals::gen_log_sweep(100.0, 8_000.0, 0.5, 48_000, 0.04);
        let mut recording = vec![0.0_f32; reference.len() + 512];
        for (index, &sample) in reference.iter().enumerate() {
            recording[index] = sample;
        }
        recording[reference.len() + 300] = 0.2;
        let response = deconvolve_sweep(&recording, &reference, 48_000).unwrap();
        assert_eq!(response.len(), recording.len().next_power_of_two() / 2 + 1);
    }

    #[test]
    fn canonical_ess_returns_transfer_ir_and_harmonics() {
        let sample_rate = 48_000;
        let reference = crate::signals::gen_log_sweep(100.0, 8_000.0, 0.5, sample_rate, 0.08);
        let delay = 37;
        let mut recording = vec![0.0_f32; delay + reference.len() + 256];
        for (index, &sample) in reference.iter().enumerate() {
            recording[delay + index] = sample * 0.5;
        }
        recording[delay + reference.len() + 120] = 0.1;
        let result = super::measurement::analyze_log_sweep_recording(
            &recording,
            &reference,
            sample_rate,
            (100.0, 8_000.0),
        )
        .unwrap();
        assert!((result.lag.lag_samples - delay as isize).abs() <= 1);
        assert_eq!(result.harmonic_impulse_responses.len(), 4);
        assert_eq!(result.frequency_response.len(), result.frequencies.len());
        assert!(result.thd_percent.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn noise_floor_pure_silence_is_very_low() {
        let silence = vec![0.0_f32; 4096];
        let nf = estimate_noise_floor_db_from_silence(&silence, 48_000);
        assert_eq!(nf.len(), 4096 / 2 + 1);
        for (i, v) in nf.iter().enumerate() {
            assert!(
                *v < -200.0,
                "pure silence bin {i} should report extremely low dB, got {v}",
            );
        }
    }

    #[test]
    fn noise_floor_tone_peaks_at_exact_bin() {
        // Pick a frequency that lands exactly on an FFT bin centre
        // so there's no inter-bin leakage. Hann windowing still
        // splits ~half the peak energy into the two adjacent bins by
        // design; at the exact centre the main-lobe peak returns
        // within ~1 dB of the target.
        let sr = 48_000_u32;
        let n: usize = 4096;
        let target_bin = 100_usize;
        let freq = (target_bin as f32 * sr as f32) / n as f32; // 1171.875 Hz
        let amp_db = -40.0_f32;
        let amp = 10.0_f32.powf(amp_db / 20.0);
        let tone: Vec<f32> = (0..n)
            .map(|k| amp * (2.0 * PI * freq * k as f32 / sr as f32).sin())
            .collect();
        let nf = estimate_noise_floor_db_from_silence(&tone, sr);
        // Find the peak bin in a small bracket around the target.
        let mut peak_db = f32::NEG_INFINITY;
        let mut peak_bin = 0;
        for (k, v) in nf
            .iter()
            .enumerate()
            .take(target_bin + 3)
            .skip(target_bin - 2)
        {
            if *v > peak_db {
                peak_db = *v;
                peak_bin = k;
            }
        }
        assert_eq!(
            peak_bin, target_bin,
            "peak bin should be at the tone frequency"
        );
        assert!(
            (peak_db - amp_db).abs() < 1.5,
            "peak dB {peak_db} should be within ±1.5 dB of target {amp_db}"
        );
    }
}

#[test]
fn test_next_power_of_two() {
    assert_eq!(next_power_of_two(1), 1);
    assert_eq!(next_power_of_two(2), 2);
    assert_eq!(next_power_of_two(3), 4);
    assert_eq!(next_power_of_two(1000), 1024);
    assert_eq!(next_power_of_two(1024), 1024);
    assert_eq!(next_power_of_two(1025), 2048);
}

#[test]
fn test_hann_window() {
    let signal = vec![1.0; 100];
    let windowed = apply_hann_window(&signal);

    // First and last samples should be near zero
    assert!(windowed[0].abs() < 0.01);
    assert!(windowed[99].abs() < 0.01);

    // Middle sample should be near 1.0
    assert!((windowed[50] - 1.0).abs() < 0.01);
}

#[test]
fn test_estimate_lag_zero() {
    // Identical signals should have zero lag
    let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let lag = estimate_lag(&signal, &signal).unwrap();
    assert_eq!(lag, 0);
}

#[test]
fn lag_confidence_rejects_silence() {
    let silence = vec![0.0_f32; 128];
    let error = super::estimate::estimate_lag_with_confidence(&silence, &silence).unwrap_err();
    assert!(error.contains("no energy"), "unexpected error: {error}");
}

#[test]
fn microphone_compensation_interpolates_on_log_frequency() {
    let compensation =
        MicrophoneCompensation::new(vec![10.0, 100.0, 1000.0], vec![0.0, 10.0, 20.0]).unwrap();
    assert!((compensation.interpolate_at(100.0) - 10.0).abs() < 1e-6);
    assert!((compensation.interpolate_at((10.0_f32 * 1000.0).sqrt()) - 10.0).abs() < 1e-5);
    let corrected = compensation.apply_to_response(&[10.0, 100.0, 1000.0], &[1.0, 11.0, 21.0]);
    assert_eq!(corrected, vec![1.0, 1.0, 1.0]);
}

#[test]
fn linear_phase_fit_recovers_delay_from_wrapped_phase() {
    let delay_seconds = 0.0025_f32;
    let frequencies = [100.0_f32, 200.0, 300.0, 400.0];
    let phases: Vec<f32> = frequencies
        .iter()
        .map(|frequency| {
            (-2.0 * PI * *frequency * delay_seconds)
                .to_degrees()
                .rem_euclid(360.0)
        })
        .collect();
    let fitted = super::measurement::fit_linear_phase_delay_seconds(&frequencies, &phases).unwrap();
    assert!((fitted - delay_seconds as f64).abs() < 1e-6);
}

#[test]
fn robust_average_rejects_a_bad_capture() {
    let good = vec![Complex::new(1.0_f32, 0.0); 4];
    let bad = vec![Complex::new(10.0_f32, 0.0); 4];
    let averaged =
        super::measurement::average_deconvolved_sweeps(&[good.clone(), good.clone(), good, bad])
            .unwrap();
    assert_eq!(averaged.accepted_indices, vec![0, 1, 2]);
    assert_eq!(averaged.rejected_indices, vec![3]);
    assert!(
        averaged
            .response
            .iter()
            .all(|value| (value.re - 1.0).abs() < 1e-6)
    );
    assert!(
        averaged.coherence.is_empty(),
        "three accepted captures do not support coherence"
    );
}

#[test]
fn lag_confidence_is_normalized_and_selects_the_normalized_peak() {
    let mut reference = vec![0.0_f32; 128];
    for (index, sample) in reference[12..44].iter_mut().enumerate() {
        *sample = ((index as f32 + 1.0) * 0.17).sin();
    }
    let delay = 19;
    let mut recorded = vec![0.0_f32; reference.len() + delay];
    recorded[delay..].copy_from_slice(&reference);

    let estimate = super::estimate::estimate_lag_with_confidence(&reference, &recorded)
        .expect("delayed copy must have a confident lag");
    assert_eq!(estimate.lag_samples, delay as isize);
    assert!(estimate.normalized_peak > 0.99, "{estimate:?}");
    assert!(estimate.confidence > 0.05, "{estimate:?}");
}

#[test]
fn envelope_confidence_uses_the_envelope_peak() {
    let sample_rate = 48_000_u32;
    let probe = crate::signals::gen_narrowband_probe(1024, sample_rate, 0.5, 42, 800.0, 2_000.0);
    let delay = 73;
    let mut recorded = vec![0.0_f32; probe.len() + delay + 32];
    for (index, &sample) in probe.iter().enumerate() {
        recorded[index + delay] = sample * 0.4;
    }

    let result = cross_correlate_envelope(&probe, &recorded, sample_rate).unwrap();
    assert!((result.peak_sample as isize - delay as isize).abs() <= 2);
    assert!(result.normalized_peak > 0.5, "{result:?}");
    assert!(result.confidence > 0.05, "{result:?}");
}

#[test]
fn repeated_ess_recordings_align_before_averaging() {
    let reference: Vec<f32> = (0..64)
        .map(|index| ((index as f32 + 1.0) * 0.23).sin() * 0.5)
        .collect();
    let delays = [7_usize, 11, 13, 17];
    let recordings: Vec<Vec<f32>> = delays
        .iter()
        .map(|&delay| {
            let mut recording = vec![0.0_f32; delay + reference.len() + 16];
            recording[delay..delay + reference.len()].copy_from_slice(&reference);
            recording
        })
        .collect();

    let result = super::measurement::average_ess_recordings(&recordings, &reference, 48_000)
        .expect("repeated captures should be aligned and averaged");
    assert_eq!(
        result
            .lag_estimates
            .iter()
            .map(|estimate| estimate.lag_samples as usize)
            .collect::<Vec<_>>(),
        delays
    );
    assert_eq!(result.averaged.accepted_indices, vec![0, 1, 2, 3]);
    assert_eq!(result.averaged.rejected_indices, Vec::<usize>::new());
    assert_eq!(result.averaged.response.len(), 65);
}

#[test]
fn h1_transfer_estimator_averages_cross_spectra() {
    let inputs = vec![
        vec![Complex::new(1.0_f32, 0.0), Complex::new(2.0, 0.0)],
        vec![Complex::new(0.5, 0.0), Complex::new(1.5, 0.0)],
    ];
    let outputs = inputs
        .iter()
        .map(|spectrum| spectrum.iter().map(|value| *value * 2.0).collect())
        .collect::<Vec<Vec<Complex<f32>>>>();
    let estimate = super::measurement::compute_h1_transfer_response(&inputs, &outputs).unwrap();
    assert!(estimate.iter().all(|value| (value.re - 2.0).abs() < 1e-6));
}

#[test]
fn quality_report_exposes_missing_data_and_rejects_non_finite_recordings() {
    let lag = LagEstimate {
        lag_samples: 0,
        normalized_peak: 1.0,
        peak_to_sidelobe_db: f32::INFINITY,
        confidence: 1.0,
    };
    let partial = super::measurement::assess_measurement_quality(
        &[0.0, 0.1],
        &lag,
        None,
        None,
        None,
        MeasurementQualityConfig::default(),
    );
    assert!(partial.trustworthy);
    assert!(!partial.quality_data_complete);
    assert!(partial.missing_metrics.contains(&"coherence".to_string()));

    let required = MeasurementQualityConfig {
        require_coherence: true,
        require_snr: true,
        ..MeasurementQualityConfig::default()
    };
    let rejected = super::measurement::assess_measurement_quality(
        &[f32::NAN, 0.1],
        &lag,
        None,
        None,
        None,
        required,
    );
    assert!(!rejected.trustworthy);
    assert!(rejected.score == 0.0);
    assert!(
        rejected
            .issues
            .iter()
            .any(|issue| issue.contains("non-finite"))
    );
}

#[test]
fn quality_report_from_silence_derives_snr_data() {
    let lag = LagEstimate {
        lag_samples: 0,
        normalized_peak: 1.0,
        peak_to_sidelobe_db: f32::INFINITY,
        confidence: 1.0,
    };
    let silence = vec![0.0_f32; 1024];
    let measured = vec![0.0_f32; 1024 / 2 + 1];
    let report = super::measurement::assess_measurement_quality_from_silence(
        &[0.0, 0.1],
        &lag,
        Some(&vec![1.0_f32; 1024 / 2 + 1]),
        Some(&measured),
        &silence,
        48_000,
        MeasurementQualityConfig::default(),
    );

    assert!(report.quality_data_complete);
    assert_eq!(report.snr_db.len(), measured.len());
    assert!(report.median_snr_db.is_some());
}

#[test]
fn clock_drift_correction_resamples_around_the_measured_lag() {
    let mut recording = vec![0.0_f32; 2_000];
    recording[1_010] = 1.0;
    let estimate = ClockDriftEstimate {
        ppm: 1_000.0,
        start_lag_samples: 10,
        end_lag_samples: 11,
        confidence: 1.0,
    };
    let corrected = super::measurement::correct_clock_drift(&recording, &estimate).unwrap();
    assert!(corrected[1_009] > 0.9, "peak was not moved earlier");
    assert!(corrected[1_010] < 0.2, "uncorrected peak remains dominant");
}

#[test]
fn clock_drift_estimator_reports_zero_for_a_stable_clock() {
    let reference: Vec<f32> = (0..256)
        .map(|index| ((index as f32 * 0.17).sin() + (index as f32 * 0.043).cos()) * 0.4)
        .collect();
    let delay = 23;
    let mut recording = vec![0.0_f32; delay + reference.len() + 32];
    recording[delay..delay + reference.len()].copy_from_slice(&reference);

    let estimate = super::measurement::estimate_clock_drift(&reference, &recording, 48_000)
        .expect("a stable delayed capture should produce a drift estimate");
    assert!(estimate.ppm.abs() < 1.0, "{estimate:?}");
    assert!(estimate.confidence > 0.05, "{estimate:?}");
}

#[test]
fn clock_drift_estimator_reports_sample_domain_ppm() {
    let sample_rate = 44_100_u32;
    let target_ppm = 10_000.0_f64;
    let scale = 1.0 + target_ppm / 1_000_000.0;
    let delay = 17_usize;
    let reference_len = 1_024_usize;

    let reference: Vec<f32> = (0..reference_len)
        .map(|index| {
            let index = index as f32;
            (0.7 * (0.031 * index + 0.00001 * index * index).sin() + 0.3 * (0.097 * index).cos())
        })
        .collect();
    let recording_len = delay + ((reference_len - 1) as f64 * scale).ceil() as usize + 32;
    let mut recording = vec![0.0_f32; recording_len];
    for (recording_index, sample) in recording.iter_mut().enumerate() {
        let reference_position = (recording_index as f64 - delay as f64) / scale;
        if !(0.0..=(reference_len - 1) as f64).contains(&reference_position) {
            continue;
        }
        let left = reference_position.floor() as usize;
        let right = (left + 1).min(reference_len - 1);
        let fraction = (reference_position - left as f64) as f32;
        *sample = reference[left] + fraction * (reference[right] - reference[left]);
    }

    let estimate = super::measurement::estimate_clock_drift(&reference, &recording, sample_rate)
        .expect("a stretched capture should produce a drift estimate");
    let elapsed_samples = (reference_len - reference_len / 4) as f64;
    let lag_change =
        (estimate.end_lag_samples - estimate.start_lag_samples) as f64 - elapsed_samples;
    let expected_ppm = lag_change / elapsed_samples * 1e6;

    assert!(lag_change > 3.0 && lag_change < 15.0, "{estimate:?}");
    assert!(
        (estimate.ppm - expected_ppm).abs() < 1e-6,
        "sample-domain ppm expected {expected_ppm}, got {estimate:?}"
    );
}

#[test]
fn group_delay_recovers_a_linear_phase_delay() {
    let frequencies = [100.0_f32, 200.0, 300.0, 400.0];
    let delay_ms = 2.5_f32;
    let phase: Vec<f32> = frequencies
        .iter()
        .map(|frequency| -360.0 * *frequency * delay_ms / 1_000.0)
        .collect();
    let group_delay = super::compute::compute_group_delay(&frequencies, &phase);
    assert!(
        group_delay
            .iter()
            .all(|value| (*value - delay_ms).abs() < 1e-4)
    );
}

#[test]
fn mls_deconvolution_recovers_a_short_circular_ir() {
    let mls = crate::signals::gen_mls(8, 0.5);
    let ir = [0.4_f32, -0.15, 0.08];
    let mut recording = vec![0.0_f32; mls.len()];
    for (index, output) in recording.iter_mut().enumerate() {
        for (tap, &coefficient) in ir.iter().enumerate() {
            *output += coefficient * mls[(index + mls.len() - tap) % mls.len()];
        }
    }
    let recovered = super::measurement::deconvolve_mls(&recording, &mls).unwrap();
    for (index, &expected) in ir.iter().enumerate() {
        assert!(
            (recovered[index] - expected).abs() < 2e-3,
            "tap {index}: {} != {expected}",
            recovered[index]
        );
    }
}

#[test]
fn test_estimate_lag_positive() {
    // Reference leads recorded (recorded is delayed)
    // Use longer signals for reliable FFT-based cross-correlation
    let mut reference = vec![0.0; 100];
    let mut recorded = vec![0.0; 100];

    // Create a pulse pattern that will correlate well
    for (j, val) in reference[10..20].iter_mut().enumerate() {
        *val = j as f32 / 10.0;
    }
    // Same pattern but delayed by 5 samples
    for (j, val) in recorded[15..25].iter_mut().enumerate() {
        *val = j as f32 / 10.0;
    }

    let lag = estimate_lag(&reference, &recorded).unwrap();
    assert_eq!(lag, 5, "Recorded signal is delayed by 5 samples");
}

#[test]
fn test_estimate_lag_does_not_window_away_an_edge_impulse() {
    let mut reference = vec![0.0_f32; 64];
    reference[0] = 1.0;
    let mut recorded = vec![0.0_f32; 96];
    recorded[17] = 1.0;

    let lag = estimate_lag(&reference, &recorded).unwrap();
    assert_eq!(lag, 17);
}

#[test]
fn test_identical_signals_have_zero_lag() {
    // When signals are truly identical (like in the bug case),
    // lag should be exactly zero
    let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let lag = estimate_lag(&signal, &signal).unwrap();
    assert_eq!(lag, 0, "Identical signals should have zero lag");
}

#[test]
fn test_cross_correlate_envelope_known_delay() {
    // Generate a narrowband probe, delay it, and verify detection
    let n = 4096;
    let sr = 48000_u32;
    let probe = crate::signals::gen_narrowband_probe(n, sr, 0.5, 42, 800.0, 2000.0);

    // Simulate recording: delay by 240 samples (~5ms) + attenuation
    let delay = 240_usize;
    let attenuation = 0.3;
    let mut recorded = vec![0.0_f32; n + delay + 1000];
    for (i, &s) in probe.iter().enumerate() {
        recorded[i + delay] += s * attenuation;
    }

    let result = cross_correlate_envelope(&probe, &recorded, sr).unwrap();

    // Peak should be near the known delay
    let detected_samples = result.peak_sample;
    assert!(
        (detected_samples as isize - delay as isize).unsigned_abs() <= 2,
        "Expected delay ~{} samples, got {}",
        delay,
        detected_samples
    );

    // Arrival time should be ~5ms
    assert!(
        (result.arrival_ms - 5.0).abs() < 0.1,
        "Expected ~5.0 ms, got {:.3} ms",
        result.arrival_ms
    );
}

#[test]
fn test_cross_correlate_envelope_with_noise() {
    // Probe detection should work even with additive noise
    let n = 4096;
    let sr = 48000_u32;
    let probe = crate::signals::gen_narrowband_probe(n, sr, 0.5, 42, 800.0, 2000.0);

    let delay = 480_usize; // 10ms
    let mut recorded = vec![0.0_f32; n + delay + 1000];
    for (i, &s) in probe.iter().enumerate() {
        recorded[i + delay] += s * 0.5;
    }
    // Add noise
    let noise = crate::signals::gen_white_noise(0.1, sr, recorded.len() as f32 / sr as f32);
    for (r, &n_s) in recorded.iter_mut().zip(noise.iter()) {
        *r += n_s;
    }

    let result = cross_correlate_envelope(&probe, &recorded, sr).unwrap();

    assert!(
        (result.peak_sample as isize - delay as isize).unsigned_abs() <= 2,
        "Expected delay ~{}, got {} (with noise)",
        delay,
        result.peak_sample
    );
}

#[test]
fn test_cross_correlate_envelope_ignores_wrapped_negative_lag_tail() {
    // This short asymmetric pair has a larger analytic-envelope value at FFT
    // index 3, outside the recording's valid positive-lag range 0..2. Searching
    // fft_size/2 used to select that wrapped/zero-padded alias.
    let probe = [0.023_056_554, -0.123_002_43, -0.681_590_6];
    let recorded = [-1.188_907_5, 0.152_694_91];
    let result = cross_correlate_envelope(&probe, &recorded, 48_000).unwrap();
    assert_eq!(result.peak_sample, 1);
    assert!(result.peak_sample < recorded.len());
}

#[test]
fn test_windowed_fr_synthetic() {
    // Create synthetic IR: impulse at sample 0 + delayed impulse at sample 240 (5ms)
    // Direct window [0, 240) should show flat response
    // Early window [240, 1920) should show the reflection's response
    let sr = 48000;
    let mut ir = vec![0.0f32; 4096];
    ir[0] = 1.0; // direct sound
    ir[240] = 0.5; // reflection at 5ms, -6dB

    let result = compute_windowed_fr(&ir, 240, 1920, sr, 200).unwrap();

    // Direct window should have content
    assert!(!result.direct_sound_spl.is_empty());
    assert!(!result.early_reflections_spl.is_empty());
    assert!(!result.late_reverb_spl.is_empty());

    // All frequency vectors should have the requested number of points
    assert_eq!(result.direct_sound_freq.len(), 200);
    assert_eq!(result.early_reflections_freq.len(), 200);
    assert_eq!(result.late_reverb_freq.len(), 200);

    // Time boundaries should match
    assert!((result.direct_end_ms - 5.0).abs() < 0.01);
    assert!((result.early_end_ms - 40.0).abs() < 0.01);

    // Direct sound should be roughly flat above the resolution limit.
    // Short window = poor LF resolution, but mid-HF should be flat.
    // Filter to frequencies above 500 Hz where the 240-sample window has resolution
    let mid_hf: Vec<f32> = result
        .direct_sound_freq
        .iter()
        .zip(result.direct_sound_spl.iter())
        .filter(|&(&f, _)| f > 500.0 && f < 18000.0)
        .map(|(_, &spl)| spl)
        .collect();
    if mid_hf.len() > 2 {
        let max = mid_hf.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min = mid_hf.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let range = max - min;
        assert!(
            range < 12.0,
            "Direct sound mid-HF range too large: {:.1} dB",
            range
        );
    }
}

#[test]
fn test_windowed_fr_empty_window() {
    // If early_end == direct_end (no early reflections), that window should be empty/silent
    let sr = 48000;
    let mut ir = vec![0.0f32; 2048];
    // Place impulse away from window edges so fading doesn't zero it out
    ir[50] = 1.0;

    let result = compute_windowed_fr(&ir, 200, 200, sr, 200).unwrap();

    // Early reflections window is zero-length — SPL should be very low
    assert_eq!(result.early_reflections_spl.len(), 200);
    for &spl in &result.early_reflections_spl {
        assert!(
            spl <= -199.0,
            "Expected silent early reflections, got {:.1} dB",
            spl
        );
    }

    // Direct and late should still have content
    let direct_max = result
        .direct_sound_spl
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    assert!(
        direct_max > -100.0,
        "Direct sound should have content, max was {:.1} dB",
        direct_max
    );
}

#[test]
fn test_thd_window_min_is_frequency_dependent() {
    // Issue #6: for a low start_freq the old hardcoded 256-sample floor
    // could be far smaller than 3 periods of the harmonic.  With the fix
    // the window must be at least 3 periods.
    let sr = 48000.0f32;
    let start_freq = 20.0f32;
    let end_freq = 20000.0f32;
    let duration = 10.0f32; // 10-second sweep
    let n = 65536usize;
    let mut ir = vec![0.0f32; n];
    // Synthetic peak at the centre
    ir[n / 2] = 1.0;

    let freqs = vec![1000.0f32];
    let fund_db = vec![0.0f32];
    let (thd, _harmonics) =
        compute_thd_from_ir(&ir, sr, &freqs, &fund_db, start_freq, end_freq, duration);
    // Should complete without panicking; THD of a pure impulse is high
    // because there is no harmonic structure, but the point is that the
    // window sizing does not overflow or underflow.
    assert!(
        thd[0] >= 0.0 && thd[0] <= 100.0,
        "THD should be in [0, 100], got {}",
        thd[0]
    );
}

#[test]
fn test_thd_harmonic_ir_uses_transfer_function_amplitude() {
    let sample_rate = 48_000.0_f32;
    let start_freq = 20.0_f32;
    let end_freq = 20_000.0_f32;
    let duration = 1.0_f32;
    let n = 65_536;
    let peak_idx = 20_000;
    let mut ir = vec![0.0_f32; n];
    ir[peak_idx] = 1.0;

    let h2_delay =
        (duration * 2.0_f32.ln() / (end_freq / start_freq).ln() * sample_rate).round() as usize;
    ir[peak_idx - h2_delay] = 0.1;

    let (thd, harmonics) = compute_thd_from_ir(
        &ir,
        sample_rate,
        &[1000.0],
        &[0.0],
        start_freq,
        end_freq,
        duration,
    );
    assert!(
        (harmonics[0][0] + 20.0).abs() < 0.1,
        "a 0.1 harmonic IR must be -20 dB, got {:.2} dB",
        harmonics[0][0]
    );
    assert!(
        (thd[0] - 10.0).abs() < 0.2,
        "a single 0.1 second harmonic must produce 10% THD, got {:.3}%",
        thd[0]
    );
}

#[test]
fn deconvolve_silent_reference_is_finite_and_zero() {
    let recording = vec![0.25_f32; 1024];
    let reference = vec![0.0_f32; 1024];
    let response = deconvolve_sweep(&recording, &reference, 48_000).unwrap();
    assert!(
        response
            .iter()
            .all(|bin| bin.re.is_finite() && bin.im.is_finite())
    );
    assert!(response.iter().all(|bin| bin.norm() == 0.0));
}

#[test]
fn impulse_response_resolution_scales_with_input_grid() {
    let frequencies: Vec<f32> = (0..=2048).map(|i| 20.0 + i as f32 * 10.0).collect();
    let magnitude_db = vec![0.0_f32; frequencies.len()];
    let phase_deg = vec![0.0_f32; frequencies.len()];

    let (times, impulse) =
        compute_impulse_response_from_fr(&frequencies, &magnitude_db, &phase_deg, 48_000.0);
    assert_eq!(times.len(), 4096);
    assert_eq!(impulse.len(), 4096);
}

#[test]
fn spectrogram_reports_peak_amplitude_and_includes_exact_frame() {
    let amplitude = 0.25_f32;
    let expected_db = 20.0 * amplitude.log10();

    for (window_size, bin) in [(1024, 32), (2048, 64)] {
        let signal = bin_centered_sine(amplitude, bin, window_size);
        let (matrix, _, times) =
            compute_spectrogram(&signal, 48_000.0, window_size, window_size / 4);
        assert_eq!(matrix.len(), 1);
        assert_eq!(times.len(), 1);
        assert!(
            (matrix[0][bin] - expected_db).abs() < 0.05,
            "window_size={window_size}: expected {expected_db:.3} dBFS, got {:.3}",
            matrix[0][bin]
        );
    }
}

#[test]
fn test_unwrap_phase_deg_basic() {
    let phase = vec![0.0f32, 90.0, 180.0, -90.0, 0.0];
    let unwrapped = super::misc::unwrap_phase_deg(&phase);
    assert!((unwrapped[0] - 0.0).abs() < 1e-6);
    // 180 to -90 is a -270 jump; wrap correction is +360 -> 270
    assert!((unwrapped[3] - 270.0).abs() < 1e-6);
    assert!((unwrapped[4] - 360.0).abs() < 1e-6);
}

#[test]
fn test_unwrap_phase_deg_empty() {
    assert!(super::misc::unwrap_phase_deg(&[]).is_empty());
}

#[test]
fn test_find_db_point_basic() {
    let freqs = vec![100.0f32, 200.0, 300.0, 400.0];
    let mag = vec![-10.0f32, -6.0, -3.0, 0.0];
    let f = super::misc::find_db_point(&freqs, &mag, -4.5, true);
    assert!(f.is_some());
    let f = f.unwrap();
    assert!(f > 200.0 && f < 300.0, "got {f}");
}

#[test]
fn test_find_db_point_from_end() {
    let freqs = vec![100.0f32, 200.0, 300.0, 400.0];
    let mag = vec![0.0f32, -3.0, -6.0, -10.0];
    let f = super::misc::find_db_point(&freqs, &mag, -4.5, false);
    assert!(f.is_some());
    let f = f.unwrap();
    assert!(f > 200.0 && f < 300.0, "got {f}");
}

#[test]
fn test_find_db_point_mismatched_lengths() {
    assert!(super::misc::find_db_point(&[100.0], &[1.0, 2.0], 0.0, true).is_none());
}

#[test]
fn test_generate_log_frequencies_bounds() {
    let freqs = super::misc::generate_log_frequencies(10, 20.0, 20000.0);
    assert_eq!(freqs.len(), 10);
    assert!((freqs[0] - 20.0).abs() < 1e-3);
    assert!((freqs[9] - 20000.0).abs() < 1e-1);
    for w in freqs.windows(2) {
        assert!(w[1] > w[0]);
    }
}

#[test]
fn test_wav_next_power_of_two() {
    assert_eq!(super::misc::wav_next_power_of_two(0), 1);
    assert_eq!(super::misc::wav_next_power_of_two(3), 4);
    assert_eq!(super::misc::wav_next_power_of_two(1024), 1024);
    assert_eq!(super::misc::wav_next_power_of_two(1048577), 1048576);
}

#[test]
fn test_compute_group_delay_flat_phase() {
    let freqs = vec![100.0f32, 200.0, 300.0, 400.0];
    let phase = vec![0.0f32; 4];
    let gd = compute_group_delay(&freqs, &phase);
    assert_eq!(gd.len(), 4);
    for &v in &gd {
        assert!(v.abs() < 1e-6);
    }
}

#[test]
fn test_compute_group_delay_short_input() {
    let freqs = vec![100.0f32];
    let phase = vec![0.0f32];
    let gd = compute_group_delay(&freqs, &phase);
    assert_eq!(gd.len(), 1);
    assert_eq!(gd[0], 0.0);
}

#[test]
fn test_compute_clarity_broadband_dirac() {
    // ISO 3382: a Dirac at t=0 has all energy inside the early windows,
    // so both clarity values peg at the positive cap.
    let mut ir = vec![0.0f32; 48000];
    ir[0] = 1.0;
    let (c50, c80) = compute_clarity_broadband(&ir, 48000.0);
    assert_eq!(c50, 60.0);
    assert_eq!(c80, 60.0);
}

#[test]
fn test_compute_clarity_broadband_mid_impulse() {
    // Impulse at 60 ms: late for the 50 ms window, early for the 80 ms one.
    let mut ir = vec![0.0f32; 48000];
    ir[(0.060 * 48000.0) as usize] = 1.0;
    let (c50, c80) = compute_clarity_broadband(&ir, 48000.0);
    assert_eq!(c50, -60.0);
    assert_eq!(c80, 60.0);
}

#[test]
fn test_compute_clarity_broadband_late_impulse() {
    // Place impulse after 100 ms so both early windows miss it.
    let mut ir = vec![0.0f32; 48000];
    ir[5000] = 1.0;
    let (c50, c80) = compute_clarity_broadband(&ir, 48000.0);
    assert_eq!(c50, -60.0);
    assert_eq!(c80, -60.0);
}

#[test]
fn test_compute_clarity_broadband_empty() {
    let ir: Vec<f32> = vec![];
    let (c50, c80) = compute_clarity_broadband(&ir, 48000.0);
    assert_eq!(c50, -60.0);
    assert_eq!(c80, -60.0);
}

#[test]
fn test_compute_rt60_broadband_empty() {
    assert_eq!(compute_rt60_broadband(&[], 48000.0), 0.0);
}

#[test]
fn test_compute_rt60_spectrum_returns_milliseconds() {
    // Synthetic IR with a known T60 of 0.5 s, concentrated in the 1 kHz
    // octave band so the band-passed Schroeder decay is clean:
    // amplitude envelope exp(-6.908 t / 0.5) modulating a 1 kHz carrier.
    // The result feeds AnalysisResult.rt60_ms and the rt60_ms CSV column,
    // so the spectrum must come back in milliseconds (~500 ms here).
    let sample_rate = 48_000.0_f32;
    let t60 = 0.5_f32;
    let len = (2.0 * sample_rate) as usize;
    let ir: Vec<f32> = (0..len)
        .map(|i| {
            let t = i as f32 / sample_rate;
            (-6.907_755 * t / t60).exp() * (2.0 * PI * 1000.0 * t).sin()
        })
        .collect();
    let rt60 = compute_rt60_spectrum(&ir, sample_rate, &[1000.0]);
    assert_eq!(rt60.len(), 1);
    assert!(
        (rt60[0] - 500.0).abs() < 50.0,
        "rt60_ms must be in milliseconds (~500 ms), got {}",
        rt60[0]
    );
}

#[test]
fn rt60_clarity_shared_pass_matches_separate_public_paths() {
    // Characterization test for the shared RT60+clarity band-filtering pass:
    // compute_rt60_clarity_spectra only changes buffer management (one f64
    // conversion + reused scratch instead of per-band temporaries), so its
    // outputs must be bit-identical to the two separate public functions.
    let sample_rate = 48_000.0_f32;
    // Deterministic exponentially decaying noise IR (multi-band content).
    let n = 24_000_usize;
    let mut state = 0x1234_5678_u32;
    let ir: Vec<f32> = (0..n)
        .map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
            noise * (-3.0 * i as f32 / n as f32).exp()
        })
        .collect();
    let freqs: Vec<f32> = (0..32).map(|i| 100.0 * 1.1f32.powi(i)).collect();

    let rt60_ref = compute_rt60_spectrum(&ir, sample_rate, &freqs);
    let (c50_ref, c80_ref) = compute_clarity_spectrum(&ir, sample_rate, &freqs);
    let (rt60, c50, c80) = super::compute::compute_rt60_clarity_spectra(&ir, sample_rate, &freqs);

    assert_eq!(rt60, rt60_ref, "rt60_ms must be bit-identical");
    assert_eq!(c50, c50_ref, "c50_db must be bit-identical");
    assert_eq!(c80, c80_ref, "c80_db must be bit-identical");
}

#[test]
fn welch_dc_bin_matches_single_fft_dc_bin() {
    // DC is a one-sided bin: it must not receive the 2x folding applied to
    // interior bins. A constant 0.5 signal must read ~-6.02 dB in both paths.
    let fft_size = 1024;
    let signal = vec![0.5_f32; fft_size * 4];
    let (_, welch_db, _) = compute_welch_spectrum_internal(&signal, 48_000, fft_size, 0.5).unwrap();
    let (_, single_db, _) =
        compute_single_fft_spectrum_internal(&signal, 48_000, fft_size, false).unwrap();
    let expected = 20.0 * 0.5_f32.log10();
    assert!(
        (welch_db[0] - expected).abs() < 0.1,
        "Welch DC bin: expected {expected:.3} dBFS, got {:.3}",
        welch_db[0]
    );
    assert!(
        (welch_db[0] - single_db[0]).abs() < 0.1,
        "Welch DC bin {:.3} must match single-FFT DC bin {:.3}",
        welch_db[0],
        single_db[0]
    );
}

#[test]
fn welch_skips_trailing_frame_with_no_new_data() {
    // With len == fft_size and 50% overlap, the second frame re-analyzes the
    // back half of the same data and must not enter the average: here that
    // back half is silence, so counting it would halve the measured power.
    let fft_size = 1024;
    let bin = 32;
    let mut signal = vec![0.0_f32; fft_size];
    for (n, sample) in signal[..fft_size / 2].iter_mut().enumerate() {
        *sample = 0.5 * (2.0 * PI * bin as f32 * n as f32 / fft_size as f32).sin();
    }
    let (_, reference_db, _) =
        compute_welch_spectrum_internal(&signal, 48_000, fft_size, 0.0).unwrap();
    let (_, overlapped_db, _) =
        compute_welch_spectrum_internal(&signal, 48_000, fft_size, 0.5).unwrap();
    assert!(
        (overlapped_db[bin] - reference_db[bin]).abs() < 0.1,
        "trailing no-new-data frame must not dilute the average: overlap=0.0 gave {:.3} dB, overlap=0.5 gave {:.3} dB",
        reference_db[bin],
        overlapped_db[bin]
    );
}

#[test]
fn test_windowed_fr_tiny_windows_do_not_panic() {
    // 1- and 2-sample windows have no frequency resolution; the function must
    // return a finite flat spectrum instead of panicking on bin clamping.
    let sr = 48000;
    let mut ir = vec![0.0f32; 64];
    ir[0] = 1.0;
    ir[1] = 0.5;
    ir[2] = 0.25;

    let result = compute_windowed_fr(&ir, 1, 3, sr, 32).unwrap();
    assert_eq!(result.direct_sound_spl.len(), 32);
    assert_eq!(result.early_reflections_spl.len(), 32);
    assert!(result.direct_sound_spl.iter().all(|v| v.is_finite()));
    assert!(result.early_reflections_spl.iter().all(|v| v.is_finite()));
    // A one-sample window [1.0] has a flat |H(f)| = 1 -> 0 dB everywhere.
    assert!(
        result.direct_sound_spl.iter().all(|&v| v.abs() < 0.1),
        "one-sample direct window should be flat 0 dB, got {:?}",
        result.direct_sound_spl
    );
}

#[test]
fn test_thd_spread_harmonic_ir_compensates_window_coherent_gain() {
    // A harmonic IR spread across the whole extraction window is attenuated
    // by the Hann window's coherent gain (~0.5); the measured level must be
    // compensated or THD reads ~6 dB low.
    let sample_rate = 48_000.0_f32;
    let start_freq = 20.0_f32;
    let end_freq = 20_000.0_f32;
    let duration = 1.0_f32;
    let n = 65_536;
    let peak_idx = 20_000;
    let mut ir = vec![0.0_f32; n];
    ir[peak_idx] = 1.0;

    let sweep_ratio = end_freq / start_freq;
    let h2_delay = (duration * 2.0_f32.ln() / sweep_ratio.ln() * sample_rate).round() as usize;
    let center = peak_idx - h2_delay;
    // Window sizing replicated from compute_thd_from_ir for the 2nd harmonic.
    let dt_next_rel = duration * (3.0_f32.ln() - 2.0_f32.ln()) / sweep_ratio.ln();
    let min_win_len = (3.0 * sample_rate / (2.0 * start_freq)).max(16.0);
    let win_len = ((dt_next_rel * sample_rate * 0.8).max(min_win_len) as usize).min(n / 2);
    let fft_size = next_power_of_two(win_len);

    // Bin-centered tone filling the whole harmonic window, amplitude 0.1.
    // Its true transfer-function magnitude is amp * win_len / 2.
    let bin = 170;
    let freq = bin as f32 * sample_rate / fft_size as f32;
    let amp = 0.1_f32;
    for (k, idx) in (center - win_len / 2..center + win_len / 2).enumerate() {
        ir[idx] = amp * (2.0 * PI * freq * k as f32 / sample_rate).sin();
    }

    let (_thd, harmonics) = compute_thd_from_ir(
        &ir,
        sample_rate,
        &[freq],
        &[0.0],
        start_freq,
        end_freq,
        duration,
    );
    let expected = 20.0 * (amp * win_len as f32 / 2.0).log10();
    assert!(
        (harmonics[0][0] - expected).abs() < 1.5,
        "spread harmonic IR of true magnitude {expected:.2} dB must not be attenuated by the window coherent gain, got {:.2} dB",
        harmonics[0][0]
    );
}

#[test]
fn noise_floor_single_sample_returns_empty() {
    // A Hann window is undefined for n < 2 (division by n - 1); a single
    // sample carries no estimable spectrum.
    let nf = estimate_noise_floor_db_from_silence(&[0.5], 48_000);
    assert!(
        nf.is_empty(),
        "single-sample silence must yield no bins, got {nf:?}"
    );
}

#[test]
fn noise_floor_dc_and_nyquist_are_not_folded() {
    // DC and Nyquist are one-sided bins: they must use 2/N scaling, not the
    // 4/N folding applied to interior bins (which would read +6 dB high).
    let n = 1024;
    let amplitude = 0.01_f32;
    let expected = 20.0 * amplitude.log10();

    let dc = vec![amplitude; n];
    let nf = estimate_noise_floor_db_from_silence(&dc, 48_000);
    assert!(
        (nf[0] - expected).abs() < 0.1,
        "DC bin: expected {expected:.3} dB, got {:.3}",
        nf[0]
    );

    let nyquist: Vec<f32> = (0..n)
        .map(|k| amplitude * if k % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let nf = estimate_noise_floor_db_from_silence(&nyquist, 48_000);
    assert!(
        (nf[n / 2] - expected).abs() < 0.1,
        "Nyquist bin: expected {expected:.3} dB, got {:.3}",
        nf[n / 2]
    );
}

#[test]
fn test_generate_log_frequencies_single_point() {
    // A single point degenerates to the geometric midpoint of the range.
    let freqs = super::misc::generate_log_frequencies(1, 20.0, 20000.0);
    assert_eq!(freqs.len(), 1);
    let expected = (20.0_f32 * 20000.0).sqrt();
    assert!(
        (freqs[0] - expected).abs() < 0.5,
        "expected geometric midpoint {expected}, got {}",
        freqs[0]
    );
}

#[test]
fn read_analysis_csv_rejects_malformed_rows() {
    let dir = std::env::temp_dir().join(format!("sotf_test_csv_parse_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // Non-numeric field must error, not silently parse as 0.0.
    let bad_value = dir.join("bad_value.csv");
    std::fs::write(
        &bad_value,
        "frequency_hz,spl_db,phase_deg\n100.0,bogus,0.0\n",
    )
    .unwrap();
    assert!(
        super::types::read_analysis_csv(&bad_value).is_err(),
        "non-numeric field must return Err"
    );

    // Short row in the extended format must error, not leave the extended
    // vectors shorter than the frequency vector.
    let short_row = dir.join("short_row.csv");
    std::fs::write(
        &short_row,
        "frequency_hz,spl_db,phase_deg,thd_percent,rt60_ms,c50_db,c80_db,group_delay_ms\n\
         100.0,-6.0,0.0,1.0,500.0\n",
    )
    .unwrap();
    assert!(
        super::types::read_analysis_csv(&short_row).is_err(),
        "short row in extended format must return Err"
    );

    // A valid legacy CSV still parses.
    let good = dir.join("good.csv");
    std::fs::write(&good, "frequency_hz,spl_db,phase_deg\n100.0,-6.0,3.0\n").unwrap();
    let result = super::types::read_analysis_csv(&good).unwrap();
    assert_eq!(result.frequencies, vec![100.0]);
    assert_eq!(result.spl_db, vec![-6.0]);
    assert_eq!(result.phase_deg, vec![3.0]);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_compute_average_response_basic() {
    let freqs = vec![100.0f32, 200.0, 400.0, 800.0];
    let mag = vec![0.0f32, 0.0, 0.0, 0.0];
    let avg = compute_average_response(&freqs, &mag, None);
    assert!(avg.abs() < 1e-6);
}

#[test]
fn test_compute_average_response_mismatched() {
    let avg = compute_average_response(&[100.0, 200.0], &[0.0], None);
    assert_eq!(avg, 0.0);
}

#[test]
fn test_analyze_wav_buffer_empty_errors() {
    let config = WavAnalysisConfig::default();
    let result = analyze_wav_buffer(&[], 48000, &config);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("empty"));
}

#[test]
fn align_signals_rejects_lag_overflow() {
    let reference = vec![0.0_f32; 10];
    let recorded = vec![0.0_f32; 10];
    let result = super::analyze::align_signals(isize::MIN, &reference, &recorded);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("overflow"));
}

#[test]
fn test_coherence_bins_zero() {
    let r0: Vec<Complex<f32>> = vec![];
    let r1: Vec<Complex<f32>> = vec![];
    let r2: Vec<Complex<f32>> = vec![];
    let r3: Vec<Complex<f32>> = vec![];
    let coh = compute_coherence_from_realizations(&[r0, r1, r2, r3]).unwrap();
    assert!(coh.is_empty());
}

#[test]
fn test_coherence_partial_zero_energy() {
    let r0 = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
    let r1 = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
    let r2 = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
    let r3 = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
    let coh = compute_coherence_from_realizations(&[r0, r1, r2, r3]).unwrap();
    assert_eq!(coh.len(), 2);
    assert!((coh[0] - 1.0).abs() < 1e-6);
    assert_eq!(coh[1], 0.0);
}

#[test]
fn test_coherence_intermediate_value() {
    let mag = 1.0_f32;
    let phases = [-22.5_f32, -7.5, 7.5, 22.5];
    let realizations: Vec<Vec<Complex<f32>>> = phases
        .iter()
        .map(|&deg| {
            let rad = deg.to_radians();
            vec![Complex::new(mag * rad.cos(), mag * rad.sin())]
        })
        .collect();
    let coh = compute_coherence_from_realizations(&realizations).unwrap();
    assert_eq!(coh.len(), 1);
    // Mean real part = (cos(-22.5) + cos(-7.5) + cos(7.5) + cos(22.5)) / 4
    //                 = (cos(22.5) + cos(7.5)) / 2
    // Imag part cancels by symmetry. mean_sq = 1.
    let expected = ((22.5_f32.to_radians().cos() + 7.5_f32.to_radians().cos()) / 2.0).powi(2);
    assert!(
        (coh[0] - expected).abs() < 1e-4,
        "expected {expected}, got {}",
        coh[0]
    );
}

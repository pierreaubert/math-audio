//! C3 common-clock resampler: fractional-ratio polyphase correction.
//!
//! Applies a C2 [`ClockSkew`] to bring one drifted microphone channel onto
//! the stimulus clock. Deterministic DSP only — pure over sample slices.

use crate::capture_tdoa::ClockSkew;

/// Polyphase table resolution: phases per sample interval.
pub const RESAMPLE_PHASES: usize = 1024;
/// Prototype FIR length in taps (odd → exact integer centre, constant
/// group delay; mirrors the validated C1 fractional-delay oracle).
pub const RESAMPLE_TAPS: usize = 129;
/// Kaiser window beta for the prototype (≈ −90 dB stopband).
pub const RESAMPLE_KAISER_BETA: f64 = 9.0;
/// Quality bar: 0–20 kHz at 48 kHz within 0.01 dB, with at least 80 dB
/// rejection at 23.98 kHz for the worst allowed positive clock skew.
/// At other rates the useful band scales as 0–(5/12)*sample_rate. Adjacent
/// phase-table rows are interpolated to avoid nearest-phase timing steps.
pub const RESAMPLE_PASSBAND_HZ: f64 = 20_000.0;

/// Zero-order modified Bessel function I₀ (series; ~20 terms suffice).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half_sq = (x * 0.5) * (x * 0.5);
    for k in 1..32 {
        term *= half_sq / (k as f64 * k as f64);
        sum += term;
        if term < 1e-16 * sum {
            break;
        }
    }
    sum
}

/// Polyphase table: `table[phase][tap]`, taps symmetric around centre so
/// the group delay is constant and compensated by construction (the tap
/// window is centred on the exact fractional read position — see
/// [`resample_to_common_clock`]).
fn prototype_table(ratio: f64) -> Vec<Vec<f64>> {
    // Leave a transition band below both clocks' Nyquist limits.
    let cutoff = 0.94 / ratio.max(1.0);
    // Exact integer centre; tap t reads `base - (t - CENTER)`.
    //
    // Sign note: weights `sinc(t - CENTER - frac)` evaluate the
    // interpolation at `(base - frac)` — exactly what a delay *synthesis*
    // oracle needs (cf. the C1 tests), but resampling must evaluate at
    // `(base + frac)`, hence the `+ frac` below.
    let center = ((RESAMPLE_TAPS - 1) / 2) as f64;
    let i0_beta = bessel_i0(RESAMPLE_KAISER_BETA);
    let mut table = vec![vec![0.0; RESAMPLE_TAPS]; RESAMPLE_PHASES + 1];
    for (p, row) in table.iter_mut().enumerate() {
        let frac = p as f64 / RESAMPLE_PHASES as f64;
        for (t, tap) in row.iter_mut().enumerate() {
            let x = t as f64 - center + frac;
            let sinc = if x.abs() < 1e-12 {
                cutoff
            } else {
                (std::f64::consts::PI * cutoff * x).sin() / (std::f64::consts::PI * x)
            };
            let u = 2.0 * t as f64 / (RESAMPLE_TAPS - 1) as f64 - 1.0;
            let kaiser = bessel_i0(RESAMPLE_KAISER_BETA * (1.0 - u * u).max(0.0).sqrt()) / i0_beta;
            *tap = sinc * kaiser;
        }
        // Unity DC gain per phase so level is exact for slow signals.
        let gain: f64 = row.iter().sum();
        for tap in row.iter_mut() {
            *tap /= gain;
        }
    }
    table
}

/// C3: resample one drifted mic channel onto the stimulus clock.
///
/// Clock model (see C2): marker lags are measured against stimulus index `m`.
/// Therefore `n(m) = offset_samples + m * (1 + skew_ppm/1e6)` is the
/// microphone index to read. Positive skew means more microphone samples per
/// stimulus-clock interval. The offset must be referenced to stimulus sample
/// zero; subtract the first marker's stimulus index times the skew ratio when
/// converting a C2 fit anchored at a nonzero marker position.
///
/// Latency / edges: the symmetric FIR has constant group delay, absorbed
/// by centring the tap window on `n(m)` — the IR time origin stays exact
/// with no post-hoc shift. Input is zero-padded, so the first and last
/// `RESAMPLE_TAPS/2` output samples are edge transients; downstream IR
/// processing must exclude them (or supply context samples). Returns
/// `None` when `skew` is invalid — never silently resamples on a refused
/// clock estimate.
pub fn resample_to_common_clock(
    input: &[f32],
    skew: &ClockSkew,
    out_len: usize,
) -> Option<Vec<f32>> {
    if !skew.valid
        || !skew.offset_samples.is_finite()
        || !skew.skew_ppm.is_finite()
        || skew.skew_ppm.abs() > crate::capture_tdoa::MAX_PLAUSIBLE_SKEW_PPM
    {
        return None;
    }
    if input.is_empty()
        || input.iter().any(|sample| !sample.is_finite())
        || skew.offset_samples.abs() > input.len() as f64 + out_len as f64
    {
        return None;
    }
    let ratio = 1.0 + skew.skew_ppm / 1e6;
    let table = prototype_table(ratio);
    let center = ((RESAMPLE_TAPS - 1) / 2) as i64;
    let at = |n: i64| {
        if n < 0 || n >= input.len() as i64 {
            0.0
        } else {
            input[n as usize] as f64
        }
    };
    let mut out = Vec::with_capacity(out_len);
    for m in 0..out_len {
        let n = m as f64 * ratio + skew.offset_samples;
        let i0 = n.floor();
        let frac = n - i0;
        let phase_position = frac * RESAMPLE_PHASES as f64;
        let phase = (phase_position.floor() as usize).min(RESAMPLE_PHASES - 1);
        let mix = phase_position - phase as f64;
        let base = i0 as i64;
        let mut acc = 0.0;
        for (t, (lo, hi)) in table[phase].iter().zip(&table[phase + 1]).enumerate() {
            acc += (lo * (1.0 - mix) + hi * mix) * at(base - t as i64 + center);
        }
        out.push(acc as f32);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture_tdoa::TdoaConfig;
    use crate::signals::{gen_log_sweep, gen_white_noise_seeded};

    fn valid_skew(offset: f64, ppm: f64) -> ClockSkew {
        ClockSkew {
            offset_samples: offset,
            skew_ppm: ppm,
            valid: true,
            low_confidence: false,
            inconsistent_spacing: false,
        }
    }

    /// Independent variable-delay oracle (direct per-sample windowed sinc,
    /// no polyphase table) applying mic-clock drift to a stimulus signal.
    ///
    /// Sign note: with floored base `i0`, minus-form weights would evaluate
    /// at `s − 2·frac`; plus-form weights evaluate at `i0 + frac = s(n)`.
    fn apply_drift_oracle(input: &[f32], offset: f64, ppm: f64) -> Vec<f32> {
        const TAPS: usize = 129;
        const CENTER: i64 = (TAPS / 2) as i64;
        let ratio = ppm / 1e6;
        let sinc = |x: f64| {
            if x.abs() < 1e-12 {
                1.0
            } else {
                (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
            }
        };
        let at = |n: i64| {
            if n < 0 || n >= input.len() as i64 {
                0.0
            } else {
                input[n as usize] as f64
            }
        };
        (0..input.len())
            .map(|n| {
                // Independent microphone sampling of the continuous stimulus clock.
                let s = (n as f64 - offset) / (1.0 + ratio);
                let i0 = s.floor() as i64;
                let frac = s - s.floor();
                let mut acc = 0.0;
                let mut gain = 0.0;
                for t in 0..TAPS {
                    let x = t as f64 - CENTER as f64 + frac;
                    let u = 2.0 * t as f64 / (TAPS - 1) as f64 - 1.0;
                    let hann = 0.5 * (1.0 + (std::f64::consts::PI * u).cos());
                    let h = sinc(x) * hann;
                    gain += h;
                    acc += h * at(i0 - (t as i64 - CENTER));
                }
                (acc / gain) as f32
            })
            .collect()
    }

    fn middle_rms(a: &[f32], b: &[f32], end: usize) -> f64 {
        let margin = RESAMPLE_TAPS * 2;
        let (xs, ys) = (&a[margin..end], &b[margin..end]);
        let sum: f64 = xs
            .iter()
            .zip(ys.iter())
            .map(|(x, y)| ((x - y) as f64).powi(2))
            .sum();
        (sum / xs.len() as f64).sqrt()
    }

    #[test]
    fn identity_resampling_is_near_transparent() {
        let sweep = gen_log_sweep(20.0, 20_000.0, 0.9, 48_000, 1.0);
        let out = resample_to_common_clock(&sweep, &valid_skew(0.0, 0.0), sweep.len())
            .expect("valid skew resamples");
        assert_eq!(out.len(), sweep.len());
        let rms = middle_rms(&sweep, &out, sweep.len() - RESAMPLE_TAPS * 2);
        assert!(rms < 0.01, "identity rms {rms}");
    }

    #[test]
    fn skew_round_trip_recovers_stimulus() {
        let stimulus = gen_log_sweep(20.0, 20_000.0, 0.9, 48_000, 2.0);
        let noise = gen_white_noise_seeded(0.005, 48_000, 2.0, 0xD21F);
        // Microphone sample origin lags by 500 samples; its rate is 5 ppm faster.
        let mic: Vec<f32> = apply_drift_oracle(&stimulus, 500.0, 5.0)
            .iter()
            .zip(noise.iter())
            .map(|(s, n)| s + n)
            .collect();
        let fixed = resample_to_common_clock(&mic, &valid_skew(500.0, 5.0), stimulus.len())
            .expect("valid skew resamples");
        // Valid region: C3 reads mic at n(m) = (m + 500) / (1 − 5e−6), so
        // outputs past m ≈ 95467 need mic context past the recording end
        // (zero-padded edge transient, excluded here by construction).
        let rms = middle_rms(&stimulus, &fixed, 95_400);
        assert!(rms < 0.02, "round-trip rms {rms}");
    }

    #[test]
    fn identity_and_round_trip_hold_across_sample_rates() {
        // The correction is sample-domain, but the signals live in real
        // time: cheap-device, 44.1 kHz-family and 96 kHz clocks.
        for sr in [6_000.0, 44_100.0, 96_000.0] {
            let hi_hz = 20_000.0f64.min(0.4 * sr);
            let sweep = gen_log_sweep(20.0, hi_hz as f32, 0.9, sr as u32, 1.0);
            let out = resample_to_common_clock(&sweep, &valid_skew(0.0, 0.0), sweep.len())
                .expect("valid skew resamples");
            let rms = middle_rms(&sweep, &out, sweep.len() - RESAMPLE_TAPS * 2);
            assert!(rms < 0.01, "identity rms {rms} at {sr} Hz");
        }
        // Full drift round-trip at 44.1 kHz: 500-sample lag, 5 ppm fast.
        let stimulus = gen_log_sweep(20.0, 17_640.0, 0.9, 44_100, 2.0);
        let noise = gen_white_noise_seeded(0.005, 44_100, 2.0, 0x4410);
        let mic: Vec<f32> = apply_drift_oracle(&stimulus, 500.0, 5.0)
            .iter()
            .zip(noise.iter())
            .map(|(s, n)| s + n)
            .collect();
        let fixed = resample_to_common_clock(&mic, &valid_skew(500.0, 5.0), stimulus.len())
            .expect("valid skew resamples");
        let rms = middle_rms(&stimulus, &fixed, mic.len() - 700);
        assert!(rms < 0.02, "round-trip rms {rms} at 44100 Hz");
    }

    #[test]
    fn invalid_skew_refuses() {
        let sweep = gen_log_sweep(20.0, 20_000.0, 0.9, 48_000, 0.1);
        let bad = ClockSkew {
            valid: false,
            ..valid_skew(0.0, 0.0)
        };
        assert!(resample_to_common_clock(&sweep, &bad, sweep.len()).is_none());
        let wild = valid_skew(0.0, 100_000.0);
        assert!(resample_to_common_clock(&sweep, &wild, sweep.len()).is_none());
        // TdoaConfig import is used to keep the C2→C3 type link explicit.
        let _ = TdoaConfig::default();
    }
}

#[cfg(test)]
mod clock_mapping_regression {
    use super::*;
    use crate::capture_tdoa::{TdoaEstimate, estimate_clock_skew};

    #[test]
    fn estimated_clock_mapping_recovers_an_analytic_tone() {
        let rate = 48_000.0;
        let spacing = 16_000.0;
        let offset = 1_200.0;
        let ratio = 1.001;
        let first = TdoaEstimate {
            offset_samples: offset,
            confidence_db: 40.0,
            valid: true,
        };
        let second = TdoaEstimate {
            offset_samples: offset + spacing * (ratio - 1.0),
            ..first
        };
        let fit = estimate_clock_skew(&first, &second, spacing);
        // Independent continuous-time oracle: microphone sample n occurs at
        // stimulus time (n - offset) / ratio, not n * (1 - ppm) - offset.
        let phase = |time: f64| (std::f64::consts::TAU * 3000.0 * time / rate).sin();
        let input: Vec<f32> = (0..20_000)
            .map(|n| phase((n as f64 - offset) / ratio) as f32)
            .collect();
        let output = resample_to_common_clock(&input, &fit, 16_000).unwrap();
        let rms = (output
            .iter()
            .enumerate()
            .map(|(m, value)| (f64::from(*value) - phase(m as f64)).powi(2))
            .sum::<f64>()
            / output.len() as f64)
            .sqrt();
        assert!(rms < 0.002, "C2/C3 clock convention mismatch: rms={rms}");
    }
}

#[cfg(test)]
mod quality_regression {
    use super::*;
    #[test]
    fn measurement_passband_and_alias_rejection_at_maximum_skew() {
        let rate = 48_000.0;
        let skew = ClockSkew {
            offset_samples: 256.0,
            skew_ppm: 5000.0,
            valid: true,
            low_confidence: false,
            inconsistent_spacing: false,
        };
        for (frequency, maximum_error) in [(20_000.0, 0.001), (23_980.0, 0.0001)] {
            let input: Vec<f32> = (0..20_000)
                .map(|n| (std::f64::consts::TAU * frequency * n as f64 / rate).sin() as f32)
                .collect();
            let output = resample_to_common_clock(&input, &skew, 16_000).unwrap();
            let rms = (output
                .iter()
                .enumerate()
                .map(|(n, sample)| {
                    let expected = if frequency < 21_000.0 {
                        (std::f64::consts::TAU * frequency * (n as f64 * 1.005 + 256.0) / rate)
                            .sin()
                    } else {
                        0.0
                    };
                    (f64::from(*sample) - expected).powi(2)
                })
                .sum::<f64>()
                / output.len() as f64)
                .sqrt();
            assert!(rms < maximum_error, "frequency {frequency}: rms {rms}");
        }
    }
    #[test]
    fn invalid_sample_data_and_extreme_offsets_are_refused() {
        let skew = ClockSkew {
            offset_samples: 0.0,
            skew_ppm: 0.0,
            valid: true,
            low_confidence: false,
            inconsistent_spacing: false,
        };
        assert!(resample_to_common_clock(&[f32::NAN], &skew, 1).is_none());
        assert!(resample_to_common_clock(&[], &skew, 1).is_none());
        assert!(
            resample_to_common_clock(
                &[0.0],
                &ClockSkew {
                    offset_samples: f64::MAX,
                    ..skew
                },
                1
            )
            .is_none()
        );
    }
}

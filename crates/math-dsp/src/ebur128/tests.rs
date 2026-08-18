use super::ebu_r128::EbuR128;
use super::misc::energy_to_loudness;
use super::misc::loudness_to_energy;
use super::mode::Mode;
use std::f64::consts::PI;

#[cfg(test)]
pub(super) const MAX_GATING_BLOCKS: usize = 100;

#[test]
fn silence_returns_neg_inf() {
    let mut meter = EbuR128::new(2, 48000, Mode::all()).unwrap();
    let silence = vec![0.0f32; 48000 * 2]; // 1 second stereo
    meter.add_frames_f32(&silence).unwrap();
    let lufs = meter.loudness_global().unwrap();
    assert!(lufs == f64::NEG_INFINITY || lufs < -100.0);
}

#[test]
fn sine_1khz_loudness() {
    let sr = 48000;
    let duration_s = 5;
    let num_frames = sr * duration_s;
    let mut samples = vec![0.0f32; num_frames * 2];

    // 0 dBFS 1 kHz sine, both channels
    for i in 0..num_frames {
        let t = i as f64 / sr as f64;
        let s = (2.0 * PI * 1000.0 * t).sin() as f32;
        samples[i * 2] = s;
        samples[i * 2 + 1] = s;
    }

    let mut meter = EbuR128::new(2, sr as u32, Mode::all()).unwrap();
    meter.add_frames_f32(&samples).unwrap();

    let lufs = meter.loudness_global().unwrap();
    // Stereo 0 dBFS 1 kHz sine through K-weighting: ~-0.3 LUFS
    // (K pre-filter adds ~+0.2 dB at 1 kHz, 2 channels × 0.5 RMS² ≈ 1.0)
    assert!(
        lufs > -2.0 && lufs < 1.0,
        "Expected ~-0.3 LUFS for 0dBFS stereo 1kHz sine, got {lufs}"
    );
}

#[test]
fn sample_peak_tracking() {
    let mut meter = EbuR128::new(1, 48000, Mode::SAMPLE_PEAK).unwrap();
    let mut samples = vec![0.0f32; 4800]; // 100ms mono
    samples[100] = 0.75;
    samples[200] = -0.9;
    meter.add_frames_f32(&samples).unwrap();

    let peak = meter.sample_peak(0).unwrap();
    assert!((peak - 0.9).abs() < 1e-6, "Expected peak ~0.9, got {peak}");
}

#[test]
fn true_peak_non_reference_sample_rate_is_allowed() {
    let meter = EbuR128::new(2, 44_100, Mode::TRUE_PEAK).unwrap();
    assert!(meter.true_peak_detector.is_some());
}

#[test]
fn reset_clears_state() {
    let mut meter = EbuR128::new(2, 48000, Mode::all()).unwrap();
    let samples = vec![0.5f32; 48000 * 2];
    meter.add_frames_f32(&samples).unwrap();

    meter.reset();

    let lufs = meter.loudness_global().unwrap();
    assert!(lufs == f64::NEG_INFINITY || lufs < -100.0);
    assert_eq!(meter.sample_peak(0).unwrap(), 0.0);
}

#[test]
fn energy_to_loudness_roundtrip() {
    let lufs = -23.0;
    let energy = loudness_to_energy(lufs);
    let back = energy_to_loudness(energy);
    assert!((back - lufs).abs() < 1e-10);
}

#[test]
fn gating_block_count_and_energy() {
    let sr = 48000;
    let duration_s = 5;
    let num_frames = sr * duration_s;
    let mut samples = vec![0.0f32; num_frames * 2];

    for i in 0..num_frames {
        let t = i as f64 / sr as f64;
        let s = (2.0 * PI * 440.0 * t).sin() as f32 * 0.5;
        samples[i * 2] = s;
        samples[i * 2 + 1] = s;
    }

    let mut meter = EbuR128::new(2, sr as u32, Mode::I).unwrap();
    meter.add_frames_f32(&samples).unwrap();

    let result = meter.gating_block_count_and_energy();
    assert!(result.is_some());
    let (count, energy) = result.unwrap();
    assert!(count > 0);
    assert!(energy > 0.0);

    // Verify: energy/count should give similar loudness to global
    let album_lufs = energy_to_loudness(energy / count as f64);
    let global_lufs = meter.loudness_global().unwrap();
    assert!(
        (album_lufs - global_lufs).abs() < 0.5,
        "Album LUFS {album_lufs} should match global {global_lufs}"
    );
}

#[test]
fn gating_blocks_overflow_oldest_dropped() {
    // Issue #1: when gating_blocks exceeds MAX_GATING_BLOCKS the oldest
    // block must be dropped.  With the old Vec::remove(0) this was O(n);
    // with VecDeque::pop_front() it is O(1).
    let sr = 48000;
    // Need > MAX_GATING_BLOCKS sub-blocks.  Each sub-block is 100 ms,
    // so 101 blocks = 10.1 s → 48000 * 10.1 ≈ 484800 frames.
    let duration_s = 11;
    let num_frames = sr * duration_s;
    let mut samples = vec![0.0f32; num_frames * 2];

    for i in 0..num_frames {
        let t = i as f64 / sr as f64;
        let s = (2.0 * PI * 440.0 * t).sin() as f32 * 0.5;
        samples[i * 2] = s;
        samples[i * 2 + 1] = s;
    }

    let mut meter = EbuR128::new(2, sr as u32, Mode::I).unwrap();
    meter.add_frames_f32(&samples).unwrap();

    // Gating must still produce a valid result after overflow.
    let lufs = meter.loudness_global().unwrap();
    assert!(
        lufs > -30.0 && lufs < 5.0,
        "global loudness should be reasonable after overflow, got {lufs}"
    );

    let result = meter.gating_block_count_and_energy();
    assert!(
        result.is_some(),
        "gating_block_count_and_energy should work after overflow"
    );
}

#[test]
fn add_frames_f32_error_on_non_multiple_channels() {
    let mut meter = EbuR128::new(2, 48000, Mode::SAMPLE_PEAK).unwrap();
    // 3 samples for 2 channels → error
    let result = meter.add_frames_f32(&[0.5, 0.5, 0.5]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("multiple of channel count"));
}

#[test]
fn add_frames_f32_empty() {
    let mut meter = EbuR128::new(2, 48000, Mode::SAMPLE_PEAK).unwrap();
    let result = meter.add_frames_f32(&[]);
    assert!(result.is_ok());
    assert_eq!(meter.sample_peak(0).unwrap(), 0.0);
}

#[test]
fn add_frames_f32_mono_peak_tracking() {
    let mut meter = EbuR128::new(1, 48000, Mode::SAMPLE_PEAK).unwrap();
    let samples = vec![0.3f32, -0.8, 0.5, -0.9, 0.1];
    meter.add_frames_f32(&samples).unwrap();

    let peak = meter.sample_peak(0).unwrap();
    assert!((peak - 0.9).abs() < 1e-6, "Expected peak ~0.9, got {peak}");
}

#[test]
fn add_frames_f32_multichannel_peaks() {
    let mut meter = EbuR128::new(4, 48000, Mode::SAMPLE_PEAK).unwrap();
    // Interleaved 4 channels: ch0=0.1, ch1=0.2, ch2=0.3, ch3=0.4
    let samples = vec![0.1f32, 0.2, 0.3, 0.4, -0.5, -0.6, -0.7, -0.8];
    meter.add_frames_f32(&samples).unwrap();

    assert!((meter.sample_peak(0).unwrap() - 0.5).abs() < 1e-6);
    assert!((meter.sample_peak(1).unwrap() - 0.6).abs() < 1e-6);
    assert!((meter.sample_peak(2).unwrap() - 0.7).abs() < 1e-6);
    assert!((meter.sample_peak(3).unwrap() - 0.8).abs() < 1e-6);
}

#[test]
fn add_frames_f32_prev_sample_peak_snapshot() {
    let mut meter = EbuR128::new(1, 48000, Mode::SAMPLE_PEAK).unwrap();
    let samples1 = vec![0.5f32];
    meter.add_frames_f32(&samples1).unwrap();

    let prev = meter.prev_sample_peak(0).unwrap();
    assert!((prev - 0.5).abs() < 1e-6);

    // After reading, prev should be reset
    let prev2 = meter.prev_sample_peak(0).unwrap();
    assert_eq!(prev2, 0.0);
}

#[test]
fn add_frames_f32_silence_momentary() {
    let mut meter = EbuR128::new(2, 48000, Mode::M).unwrap();
    // 400ms silence = 19200 frames stereo
    let silence = vec![0.0f32; 19200 * 2];
    meter.add_frames_f32(&silence).unwrap();

    let lufs = meter.loudness_momentary().unwrap();
    assert!(
        lufs == f64::NEG_INFINITY || lufs < -100.0,
        "silence momentary should be -inf, got {lufs}"
    );
}

#[test]
fn relative_gate_is_exactly_ten_lu_below_mean() {
    // BS.1770 / libebur128: the relative gate factor is 10^(-10/10) = 0.1
    // exactly. A block between mean-10 LU and mean-9.309 LU must be included.
    // The old code multiplied by loudness_to_energy(-10.0) = 0.11725, which
    // leaks the -0.691 absolute offset into the relative factor.
    let mut meter = EbuR128::new(2, 48000, Mode::I).unwrap();
    // 10 loud blocks (energy 1.0) + 10 quiet blocks (energy 0.057).
    // mean_above_abs = 0.5285.
    // Correct gate: 0.05285  -> quiet blocks (0.057) are included.
    // Buggy gate:   0.06197  -> quiet blocks are excluded.
    for _ in 0..10 {
        meter.gating_blocks.push_back(1.0);
        meter.gating_blocks.push_back(0.057);
    }
    let expected = energy_to_loudness(0.5285);
    let got = meter.loudness_global().unwrap();
    assert!(
        (got - expected).abs() < 1e-9,
        "quiet blocks above mean-10 LU must pass the relative gate: \
         expected {expected}, got {got}"
    );

    // gating_block_count_and_energy must use the same gate.
    let (count, energy) = meter.gating_block_count_and_energy().unwrap();
    assert_eq!(count, 20, "all 20 blocks must pass the relative gate");
    assert!((energy - 10.57).abs() < 1e-12);
}

#[test]
fn channel_weights_match_bs1770_default_map() {
    use super::misc::channel_weight;
    // BS.1770-4 §5.1 general rule with libebur128's default channel map:
    // L, R, C -> weight 1.0; LFE excluded (0.0); surrounds 1.41 (+1.5 dB).
    let w = |n: usize| (0..n).map(|ch| channel_weight(ch, n)).collect::<Vec<_>>();
    assert_eq!(w(1), vec![1.0]); // mono
    assert_eq!(w(2), vec![1.0, 1.0]); // stereo
    assert_eq!(w(3), vec![1.0, 1.0, 1.0]); // L, R, C
    // 4ch = L, R, Ls, Rs (libebur128 special case: no centre, no LFE)
    assert_eq!(w(4), vec![1.0, 1.0, 1.41, 1.41]);
    // 5ch = L, R, C, Ls, Rs
    assert_eq!(w(5), vec![1.0, 1.0, 1.0, 1.41, 1.41]);
    // 6ch = L, R, C, LFE, Ls, Rs
    assert_eq!(w(6), vec![1.0, 1.0, 1.0, 0.0, 1.41, 1.41]);
    // 8ch: libebur128's default map leaves channels beyond 5.1 unused
    assert_eq!(w(8), vec![1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 0.0, 0.0]);
}

#[test]
fn compute_coeffs_48k_matches_hardcoded_table() {
    use super::kweight_filter::KWeightFilter;
    // The generic bilinear-transform path must reproduce the hardcoded
    // BS.1770-4 48 kHz table (which libebur128 also produces at 48 kHz).
    let (g1, g2) = KWeightFilter::compute_coeffs(48000.0);
    let (h1, h2) = KWeightFilter::coeffs_48k();
    for ((g, h), stage) in [((g1, h1), "pre-filter"), ((g2, h2), "RLB high-pass")] {
        for (gv, hv, name) in [
            (g.b0, h.b0, "b0"),
            (g.b1, h.b1, "b1"),
            (g.b2, h.b2, "b2"),
            (g.a1, h.a1, "a1"),
            (g.a2, h.a2, "a2"),
        ] {
            assert!(
                (gv - hv).abs() < 1e-12,
                "{stage} {name}: compute_coeffs(48k) {gv} != hardcoded {hv}"
            );
        }
    }
}

#[test]
fn k_weighting_1khz_gain_is_about_0_7_db() {
    use super::kweight_filter::KWeightFilter;
    // Known BS.1770 reference: K-weighting gain at 1 kHz is ~+0.7 dB.
    // Build the filter from the generic bilinear path (compute_coeffs) so
    // this also covers non-hardcoded sample rates.
    let (s1, s2) = KWeightFilter::compute_coeffs(48000.0);
    let mut f = KWeightFilter {
        stage1: s1,
        stage2: s2,
    };
    let sr = 48000;
    let n = sr * 5;
    let mut sum_in = 0.0;
    let mut sum_out = 0.0;
    for i in 0..n {
        let x = (2.0 * PI * 1000.0 * i as f64 / sr as f64).sin();
        let y = f.process(x);
        if i >= sr {
            // skip the filter transient
            sum_in += x * x;
            sum_out += y * y;
        }
    }
    let db = 10.0 * (sum_out / sum_in).log10();
    assert!(
        (db - 0.6977).abs() < 0.02,
        "K-weighting gain at 1 kHz should be ~+0.7 dB, got {db}"
    );
}

#[test]
fn invalid_sample_rate_rejected() {
    // libebur128 bounds: 16..=2822400. Below 16 the 100 ms sub-block would
    // have 0 frames and produce NaN energies.
    assert!(EbuR128::new(2, 0, Mode::all()).is_err());
    assert!(EbuR128::new(2, 15, Mode::all()).is_err());
    assert!(EbuR128::new(2, 2_822_401, Mode::all()).is_err());
    assert!(EbuR128::new(2, 16, Mode::all()).is_ok());
    assert!(EbuR128::new(2, 2_822_400, Mode::all()).is_ok());

    // A low-but-valid rate must not produce NaN loudness.
    let mut meter = EbuR128::new(2, 16, Mode::all()).unwrap();
    let samples = vec![0.5f32; 16 * 10 * 2]; // 10 s stereo at 16 Hz
    meter.add_frames_f32(&samples).unwrap();
    assert!(!meter.loudness_momentary().unwrap().is_nan());
    assert!(!meter.loudness_global().unwrap().is_nan());
}

#[test]
fn nan_input_is_reported_as_nan_not_neg_inf() {
    // A NaN sample contaminates the K-weighting filter state, so all
    // subsequent block energies are NaN. That must surface as NaN, not be
    // silently reported as -inf.
    let mut meter = EbuR128::new(2, 48000, Mode::all()).unwrap();
    let mut samples = vec![0.0f32; 48000 * 2]; // 1 s stereo
    samples[12345] = f32::NAN;
    meter.add_frames_f32(&samples).unwrap();

    assert!(
        meter.loudness_momentary().unwrap().is_nan(),
        "NaN-contaminated momentary loudness must be NaN"
    );
    assert!(
        meter.loudness_shortterm().unwrap().is_nan(),
        "NaN-contaminated short-term loudness must be NaN"
    );
    assert!(
        meter.loudness_global().unwrap().is_nan(),
        "NaN-contaminated integrated loudness must be NaN"
    );
}

#[test]
fn peak_queries_require_their_mode() {
    // Querying peaks without the corresponding mode bit must be an error,
    // not a silent 0.0.
    let mut meter = EbuR128::new(2, 48000, Mode::M).unwrap();
    assert!(meter.sample_peak(0).is_err());
    assert!(meter.prev_sample_peak(0).is_err());
    assert!(meter.prev_true_peak(0).is_err());

    // With the modes enabled the same queries succeed.
    let mut meter = EbuR128::new(2, 48000, Mode::SAMPLE_PEAK | Mode::TRUE_PEAK).unwrap();
    assert_eq!(meter.sample_peak(0).unwrap(), 0.0);
    assert_eq!(meter.prev_sample_peak(0).unwrap(), 0.0);
    assert_eq!(meter.prev_true_peak(0).unwrap(), 0.0);
}

#[test]
fn pooled_album_gating_applies_single_relative_gate() {
    // Strict BS.1770 album gating: pool all blocks above the ABSOLUTE gate
    // across tracks, then apply ONE relative gate to the pool.
    //
    // Track A: 10 blocks at energy 1.0. Track B: 8 at 0.5 + 2 at 0.058.
    // Per-track gating includes B's 0.058 blocks (B's own gate is 0.0412),
    // but the pooled gate is 0.0706, so they must drop out of the album pool.
    let mut a = EbuR128::new(2, 48000, Mode::I).unwrap();
    let mut b = EbuR128::new(2, 48000, Mode::I).unwrap();
    for _ in 0..10 {
        a.gating_blocks.push_back(1.0);
    }
    for _ in 0..8 {
        b.gating_blocks.push_back(0.5);
    }
    for _ in 0..2 {
        b.gating_blocks.push_back(0.058);
    }
    // A block below the absolute gate (-70 LUFS) must be excluded up front.
    b.gating_blocks.push_back(1e-9);

    let blocks_a = a.gating_blocks_above_absolute_gate();
    let blocks_b = b.gating_blocks_above_absolute_gate();
    assert_eq!(blocks_a.len(), 10);
    assert_eq!(blocks_b.len(), 10, "the 1e-9 block is below the absolute gate");

    let tracks = vec![(0.9, blocks_a), (0.5, blocks_b)];
    let (gain, peak) =
        crate::replaygain::compute_album_gain_pooled(&tracks).expect("pooled gain");

    // Reference: pool = 10×1.0 + 8×0.5 + 2×0.058, mean = 0.7058,
    // relative gate = 0.07058 -> the two 0.058 blocks drop out.
    // gated mean = 14.0 / 18.
    let expected = -18.0 - energy_to_loudness(14.0 / 18.0);
    assert!(
        (gain - expected).abs() < 1e-9,
        "pooled album gain: expected {expected}, got {gain}"
    );
    assert_eq!(peak, 0.9);

    // The legacy per-track pre-gated path must give a different (louder-gated
    // differently) result, demonstrating the deviation.
    let legacy = crate::replaygain::compute_album_gain(&[(0.9, 10, 10.0), (0.5, 10, 4.116)]);
    assert!(legacy.is_some());
    assert!(
        (legacy.unwrap().0 - gain).abs() > 0.1,
        "pooled gating must differ from per-track pre-gated pooling"
    );
}

#[test]
fn true_peak_detector_matches_shift_register_reference() {
    // Characterization test for the circular-buffer rewrite of
    // TruePeakDetector::process_frame. The reference below is the original
    // shift-register algorithm (copy_within + linear history); the optimized
    // implementation must reproduce it exactly (same coefficients, same
    // oldest-to-newest summation order, so bit-identical).
    use super::consts::TRUE_PEAK_FIR_LEN;
    use super::consts::TRUE_PEAK_FIR_PHASES;

    // Deterministic pseudo-random input (LCG), stereo with different data
    // per channel, longer than the FIR length so the ring wraps many times.
    let n = 5000;
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // map to [-1, 1]
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let input: Vec<[f32; 2]> = (0..n).map(|_| [next() as f32, next() as f32]).collect();

    // Reference implementation: original shift-register detector.
    let mut ref_peak = [0.0f64; 2];
    let mut ref_hist = [[0.0f64; TRUE_PEAK_FIR_LEN]; 2];
    for frame in &input {
        for ch in 0..2 {
            let h = &mut ref_hist[ch];
            h.copy_within(1.., 0);
            h[TRUE_PEAK_FIR_LEN - 1] = frame[ch] as f64;
            for phase in &TRUE_PEAK_FIR_PHASES {
                let mut sum = 0.0;
                for (i, &coeff) in phase.iter().enumerate() {
                    sum += coeff * h[i];
                }
                ref_peak[ch] = ref_peak[ch].max(sum.abs());
            }
        }
    }

    let mut meter = EbuR128::new(2, 48000, Mode::TRUE_PEAK).unwrap();
    let flat: Vec<f32> = input.iter().flat_map(|f| f.iter().copied()).collect();
    meter.add_frames_f32(&flat).unwrap();

    let tp = meter.true_peak_detector.as_ref().unwrap();
    for (ch, ((&peak, &prev), &refp)) in tp
        .peak
        .iter()
        .zip(&tp.prev_peak)
        .zip(&ref_peak)
        .enumerate()
    {
        assert_eq!(
            peak, refp,
            "channel {ch}: true peak must match the shift-register reference bit-exactly"
        );
        assert_eq!(prev, refp);
    }
}

#[test]
fn add_frames_f32_short_burst() {
    let mut meter = EbuR128::new(1, 48000, Mode::SAMPLE_PEAK).unwrap();
    // Very short burst (less than 100ms)
    let burst = vec![1.0f32; 100];
    meter.add_frames_f32(&burst).unwrap();

    let peak = meter.sample_peak(0).unwrap();
    assert!((peak - 1.0).abs() < 1e-6);
}

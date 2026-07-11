use math_audio_autodiff::signals::{SignalType, signal_gallery};

const FS: f64 = 48_000.0;
const N_SAMPLES: usize = 1024;
const N_CHANNELS: usize = 2;

#[test]
fn impulse_has_energy_only_at_first_sample() {
    let signal = signal_gallery(SignalType::Impulse, N_SAMPLES, N_CHANNELS, FS);
    assert_eq!(signal.data.shape(), &[1, N_SAMPLES, N_CHANNELS]);
    for ch in 0..N_CHANNELS {
        assert_eq!(signal.data[[0, 0, ch]].re, 1.0);
        assert_eq!(signal.data[[0, 0, ch]].im, 0.0);
        for n in 1..N_SAMPLES {
            assert_eq!(signal.data[[0, n, ch]].re, 0.0);
            assert_eq!(signal.data[[0, n, ch]].im, 0.0);
        }
    }
}

#[test]
fn sine_peak_amplitude_is_one() {
    let signal = signal_gallery(SignalType::Sine { freq_hz: 1_000.0 }, N_SAMPLES, 1, FS);
    let max_abs = signal.data.iter().map(|x| x.re.abs()).fold(0.0, f64::max);
    assert!((max_abs - 1.0).abs() < 1e-12, "max_abs = {}", max_abs);
}

#[test]
fn exp_decay_starts_at_one_and_decreases() {
    let signal = signal_gallery(SignalType::ExpDecay { rate: 10.0 }, N_SAMPLES, 1, FS);
    assert_eq!(signal.data[[0, 0, 0]].re, 1.0);
    for n in 1..N_SAMPLES {
        assert!(
            signal.data[[0, n, 0]].re < signal.data[[0, n - 1, 0]].re,
            "decay not monotonic at sample {}: {} >= {}",
            n,
            signal.data[[0, n, 0]].re,
            signal.data[[0, n - 1, 0]].re
        );
    }
}

#[test]
fn velvet_noise_has_expected_impulse_count() {
    let density = 100.0; // impulses per second
    let signal = signal_gallery(
        SignalType::VelvetNoise { density },
        N_SAMPLES,
        N_CHANNELS,
        FS,
    );
    for ch in 0..N_CHANNELS {
        let count = (0..N_SAMPLES)
            .filter(|&n| signal.data[[0, n, ch]].re != 0.0)
            .count();
        let expected = (N_SAMPLES as f64 * density / FS).floor() as usize;
        // Allow ±1 due to jitter and clamping.
        assert!(
            count.abs_diff(expected) <= 1,
            "channel {}: count={} expected≈{}",
            ch,
            count,
            expected
        );
    }
}

#[test]
#[should_panic(expected = "must not exceed the sample rate")]
fn velvet_noise_rejects_super_nyquist_density() {
    let _ = signal_gallery(
        SignalType::VelvetNoise { density: 96_000.0 },
        128,
        1,
        48_000.0,
    );
}

#[test]
fn white_noise_is_non_zero_and_bounded() {
    let signal = signal_gallery(SignalType::WhiteNoise, N_SAMPLES, N_CHANNELS, FS);
    for sample in signal.data.iter() {
        assert!(sample.re.abs() <= 1.0);
        assert_eq!(sample.im, 0.0);
    }
    // Extremely unlikely to be all zeros.
    let total_energy: f64 = signal.data.iter().map(|x| x.norm_sqr()).sum();
    assert!(total_energy > 0.0);
}

#[test]
fn sweep_covers_frequency_range() {
    let signal = signal_gallery(
        SignalType::Sweep {
            f0_hz: 100.0,
            f1_hz: 1_000.0,
        },
        N_SAMPLES,
        1,
        FS,
    );
    // Just a smoke test: signal is non-trivial.
    let energy: f64 = signal.data.iter().map(|x| x.norm_sqr()).sum();
    assert!(energy > 0.0);
}

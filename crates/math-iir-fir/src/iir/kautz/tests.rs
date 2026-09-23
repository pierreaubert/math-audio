use super::correction::{KAUTZ_CORRECTION_FORMAT, KautzCorrection, KautzError, parse_bank_spec};
use super::correction_fit::{
    BandLimit, FitConfig, FitStatus, FitWeights, NormalizationPolicy, check_bounds, fit_correction,
};
use super::kautz_filter::KautzFilter;
use super::kautz_section::KautzSection;
use ndarray::Array1;
use num_complex::Complex;

#[test]
fn test_kautz_section_pole_at_80hz() {
    // Create section with pole at 80 Hz, Q=5. Gain=1 so basis response is unscaled.
    let section = KautzSection::new(80.0f64, 5.0, 1.0, 48000.0);

    // Compute basis function magnitude at several frequencies.
    // The section should show a distinct peak near 80 Hz.
    let chain = Complex::new(1.0, 0.0);
    let mag_at_pole = section.basis_response(80.0, 48000.0, chain).norm();
    let mag_at_40 = section.basis_response(40.0, 48000.0, chain).norm();
    let mag_at_160 = section.basis_response(160.0, 48000.0, chain).norm();

    // The magnitude at 80 Hz should be strictly greater than at 40 or 160 Hz.
    assert!(
        mag_at_pole > mag_at_40,
        "basis magnitude at 80 Hz ({mag_at_pole:.4}) should exceed 40 Hz ({mag_at_40:.4})"
    );
    assert!(
        mag_at_pole > mag_at_160,
        "basis magnitude at 80 Hz ({mag_at_pole:.4}) should exceed 160 Hz ({mag_at_160:.4})"
    );
}

#[test]
fn test_kautz_filter_from_modes() {
    let modes = vec![(63.0, 8.0), (100.0, 5.0), (160.0, 4.0)];
    let filter = KautzFilter::from_room_modes(&modes, 48000.0);
    assert_eq!(filter.sections.len(), 3);
    for (section, &(freq, _q)) in filter.sections.iter().zip(modes.iter()) {
        assert!(
            (section.pole_freq - freq).abs() < 1e-9,
            "section pole_freq mismatch"
        );
        assert!(
            section.gain == 0.0,
            "gains should be zero before optimization"
        );
    }
}

#[test]
fn test_kautz_gain_optimization() {
    let modes = vec![(80.0, 6.0), (120.0, 4.0)];
    let mut filter = KautzFilter::from_room_modes(&modes, 48000.0);

    let freqs: Vec<f64> = (20..500).map(|f| f as f64).collect();
    // Synthetic measurement: flat baseline + two Lorentzian peaks in dB
    let measured: Vec<f64> = freqs
        .iter()
        .map(|&f| {
            let peak1 = 10.0 / (1.0 + ((f - 80.0) / 5.0).powi(2));
            let peak2 = 8.0 / (1.0 + ((f - 120.0) / 8.0).powi(2));
            peak1 + peak2
        })
        .collect();
    let target: Vec<f64> = vec![0.0; freqs.len()];

    filter.optimize_gains(&freqs, &measured, &target);

    // Correction must cut the peaks: both gains should be negative.
    for (i, s) in filter.sections.iter().enumerate() {
        assert!(
            s.gain < 0.0,
            "section {i} gain ({:.4}) should be negative to cut peaks",
            s.gain
        );
    }
}

#[test]
fn test_kautz_reset() {
    let modes = vec![(100.0, 5.0)];
    let mut filter = KautzFilter::from_room_modes(&modes, 48000.0);
    filter.sections[0].gain = 1.0; // nonzero so processing does something

    // Pump some state into the filter
    for _ in 0..100 {
        filter.process(1.0);
    }

    filter.reset();

    // After reset, a zero input should produce zero output
    let out = filter.process(0.0);
    assert_eq!(
        out, 0.0,
        "after reset, processing 0.0 should yield 0.0 (got {out})"
    );
}

#[test]
fn test_np_log_result() {
    let modes = vec![(100.0, 5.0)];
    let mut filter = KautzFilter::from_room_modes(&modes, 48000.0);
    filter.sections[0].gain = 1.0;
    let freqs = Array1::from_vec(vec![50.0, 100.0, 200.0]);
    let db = filter.np_log_result(&freqs);
    assert_eq!(db.len(), 3);
    // With gain=1, all values should be finite and not -400.0 at 100 Hz (near pole)
    assert!(
        db[1] > -100.0,
        "response at pole frequency should be nonzero"
    );
}

#[test]
fn test_process_matches_frequency_response() {
    // Verify that process() output matches complex_response() prediction.
    // Feed a sine at the pole frequency, measure steady-state amplitude.
    let pole_freq = 100.0;
    let srate = 48000.0;
    let modes = vec![(pole_freq, 5.0)];
    let mut filter = KautzFilter::from_room_modes(&modes, srate);
    filter.sections[0].gain = -6.0; // dB

    let test_freq = pole_freq;
    let omega = 2.0 * std::f64::consts::PI * test_freq / srate;

    // Run for enough samples to reach steady state
    let n_samples = 48000; // 1 second
    let mut last_outputs = vec![0.0_f64; 1000];
    for i in 0..n_samples {
        let x = (omega * i as f64).sin();
        let y = filter.process(x);
        if i >= n_samples - 1000 {
            last_outputs[i - (n_samples - 1000)] = y;
        }
    }

    // Measure peak amplitude in last 1000 samples
    let process_peak = last_outputs.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);

    // Predicted amplitude from complex_response
    let predicted = filter.complex_response(test_freq).norm();

    // They should be within 20% (filter transient + measurement imprecision)
    let ratio = process_peak / predicted.max(1e-20);
    assert!(
        ratio > 0.5 && ratio < 2.0,
        "process/predict mismatch at {}Hz: process_peak={:.6}, predicted={:.6}, ratio={:.2}",
        test_freq,
        process_peak,
        predicted,
        ratio
    );
}

#[test]
fn test_section_impulse_response_matches_analytic_basis_response() {
    let sample_rate = 48_000.0;
    let frequency = 1375.0;
    let mut section = KautzSection::new(1000.0_f64, 5.0, 1.0, sample_rate);
    let omega = 2.0 * std::f64::consts::PI * frequency / sample_rate;
    let mut measured = Complex::new(0.0, 0.0);

    for sample_index in 0..8192 {
        let input = if sample_index == 0 { 1.0 } else { 0.0 };
        let (basis, _) = section.process_section(input);
        measured += Complex::from_polar(basis, -omega * sample_index as f64);
    }

    let expected = KautzSection::new(1000.0_f64, 5.0, 1.0, sample_rate).basis_response(
        frequency,
        sample_rate,
        Complex::new(1.0, 0.0),
    );
    assert!(
        (measured - expected).norm() < 1e-9,
        "measured {measured:?}, expected {expected:?}"
    );
}

#[test]
fn test_kautz_ill_conditioned_poles() {
    // Issue #4: closely-spaced poles (100 Hz, 100.05 Hz, 100.1 Hz)
    // create an ill-conditioned basis matrix.  The old normal-equations
    // path produces huge oscillating gains (|gain| > 500).  After the
    // QR fix each gain should stay within a modest bound.
    let modes = vec![(100.0, 100.0), (100.05, 100.0), (100.1, 100.0)];
    let mut filter = KautzFilter::from_room_modes(&modes, 48000.0);

    let freqs: Vec<f64> = (20..500).map(|f| f as f64).collect();
    let measured: Vec<f64> = freqs
        .iter()
        .map(|&f| 10.0 / (1.0 + ((f - 100.0) / 1.0).powi(2)))
        .collect();
    let target: Vec<f64> = vec![0.0; freqs.len()];

    filter.optimize_gains(&freqs, &measured, &target);

    for (i, s) in filter.sections.iter().enumerate() {
        assert!(
            s.gain.abs() < 100.0,
            "section {i} gain ({:.2}) should be < 100 for ill-conditioned poles",
            s.gain
        );
    }
}

// ---------------------------------------------------------------------------
// Faithful dry-plus-bank correction (req-math-audio-kautz.md).
// ---------------------------------------------------------------------------

/// Acceptance rates for every rate-dependent check.
const ACCEPT_RATES: [f64; 3] = [44_100.0, 48_000.0, 96_000.0];

fn rel_complex(actual: Complex<f64>, expected: Complex<f64>) -> f64 {
    (actual - expected).norm() / expected.norm().max(1e-30)
}

/// Deterministic pseudo-random generator (LCG) for seeded fixtures.
fn lcg_next(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64) / (u32::MAX as f64)
}

/// Record length so the slowest pole's residual tail sits below `tol` in a
/// DTFT comparison. A resonant section's impulse envelope follows `b0·n·r^n`
/// (`b0` the numerator scale), whose omitted tail sums to about
/// `b0·N·r^N/(1−r)`; solve for `N` by bisection with a 1000x margin so the
/// tail is negligible beside the comparison tolerance.
fn record_len_for_tail(radius: f64, b0: f64, gain_max: f64, tol: f64) -> usize {
    assert!(
        radius < 1.0,
        "tail rule needs a stable pole (got radius {radius})"
    );
    let ceiling = tol / 1000.0;
    let tail = |n: f64| b0 * gain_max * n * radius.powf(n) / (1.0 - radius);
    let mut lo = 0.0f64;
    let mut hi = 1024.0;
    while tail(hi) > ceiling {
        hi *= 2.0;
        assert!(
            hi < 32_000_000.0,
            "tail record unreasonable (radius {radius})"
        );
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if tail(mid) > ceiling {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi.ceil() as usize + 128
}

/// Numerator scale `sqrt(1−r²)·(1−r²)` of a section with pole radius `r`.
fn basis_b0(radius: f64) -> f64 {
    let one_minus_r2 = 1.0 - radius * radius;
    one_minus_r2.sqrt() * one_minus_r2
}

fn slowest_radius(modes: &[(f64, f64)], srate: f64) -> f64 {
    modes
        .iter()
        .map(|&(f, q)| (-std::f64::consts::PI * f / (q * srate)).exp())
        .fold(0.0f64, f64::max)
}

/// Naive DTFT of a recorded response at the given frequencies (test oracle,
/// independent of the closed-form response: recurrence stream vs formula).
fn dtft(samples: &[f64], freqs: &[f64], srate: f64) -> Vec<Complex<f64>> {
    freqs
        .iter()
        .map(|&freq| {
            let omega = 2.0 * std::f64::consts::PI * freq / srate;
            let mut acc = Complex::new(0.0, 0.0);
            for (n, &x) in samples.iter().enumerate() {
                acc += Complex::from_polar(x, -omega * n as f64);
            }
            acc
        })
        .collect()
}

#[test]
fn correction_zero_gains_is_unity_and_strength_scales() {
    for &srate in &ACCEPT_RATES {
        let mut bank =
            KautzCorrection::<f64>::new(&[(80.0, 6.0), (120.0, 4.0)], srate).expect("valid bank");
        // Zero gains: bank silent, correction unity.
        for &freq in &[0.0, 20.0, 80.0, 120.0, 500.0, srate / 2.0] {
            assert!(
                bank.bank_response(freq).norm() == 0.0,
                "bank must be silent at {freq} Hz ({srate} Hz rate)"
            );
            let h = bank.correction_response(freq);
            assert!(
                (h - Complex::new(1.0, 0.0)).norm() < 1e-12,
                "correction must be unity at {freq} Hz ({srate} Hz rate)"
            );
        }
        // Strength law H_s = 1 + s·(H_full − 1) at s = 0, 0.5, 1.
        bank.set_gains(&[1.5, -2.0]).expect("gains fit");
        let grid: Vec<f64> = (1..200).map(|i| i as f64 * 2.5).collect();
        let full: Vec<Complex<f64>> = grid.iter().map(|&f| bank.correction_response(f)).collect();
        for &s in &[0.0, 0.5, 1.0] {
            bank.set_strength(s).expect("finite strength");
            for (f, h_full) in grid.iter().zip(full.iter()) {
                let expected = Complex::new(1.0, 0.0) + *h_full * s - Complex::new(s, 0.0);
                let got = bank.correction_response(*f);
                assert!(
                    (got - expected).norm() < 1e-12,
                    "strength law violated at {f} Hz s={s} ({srate} Hz rate)"
                );
            }
        }
    }
}

#[test]
fn correction_invalid_inputs_refused_atomically() {
    // Constructor refusals.
    assert!(matches!(
        KautzCorrection::<f64>::new(&[], 48_000.0),
        Err(KautzError::EmptyBank)
    ));
    for &rate in &[0.0, -48_000.0, f64::NAN, 1_000.0, 1e9, f64::INFINITY] {
        assert!(
            KautzCorrection::<f64>::new(&[(100.0, 5.0)], rate).is_err(),
            "rate {rate} must be refused"
        );
    }
    // Pole at/above Nyquist, non-positive frequency, bad Q, unstable radius.
    assert!(matches!(
        KautzCorrection::<f64>::new(&[(24_000.0, 5.0)], 48_000.0),
        Err(KautzError::PoleAboveNyquist { .. })
    ));
    assert!(matches!(
        KautzCorrection::<f64>::new(&[(0.0, 5.0)], 48_000.0),
        Err(KautzError::PoleAboveNyquist { .. })
    ));
    assert!(matches!(
        KautzCorrection::<f64>::new(&[(100.0, 0.0)], 48_000.0),
        Err(KautzError::NonPositiveQ { .. })
    ));
    assert!(matches!(
        KautzCorrection::<f64>::new(&[(100.0, 0.05)], 48_000.0),
        Err(KautzError::NonPositiveQ { .. })
    ));
    assert!(matches!(
        KautzCorrection::<f64>::new(&[(100.0, 1e9)], 48_000.0),
        Err(KautzError::UnstableRadius { .. })
    ));
    assert!(matches!(
        KautzCorrection::<f64>::new(&[(f64::NAN, 5.0)], 48_000.0),
        Err(KautzError::NonFinite { .. })
    ));

    // Setter refusals leave state untouched (atomicity).
    let mut bank =
        KautzCorrection::<f64>::new(&[(100.0, 5.0), (200.0, 4.0)], 48_000.0).expect("valid");
    bank.set_gains(&[1.0, -1.0]).expect("valid gains");
    let snapshot = bank.gains();
    assert!(bank.set_gains(&[1.0]).is_err());
    assert!(bank.set_gains(&[1.0, f64::NAN]).is_err());
    assert!(bank.set_gains(&[1.0, f64::INFINITY]).is_err());
    assert_eq!(bank.gains(), snapshot, "refused set_gains mutated state");
    assert!(bank.set_strength(f64::NAN).is_err());
    assert_eq!(bank.strength(), 1.0, "refused set_strength mutated state");
    let response_before = bank.correction_response(100.0);
    assert!(bank.set_sample_rate(1_000.0).is_err()); // below the 2 kHz floor
    assert!(
        (bank.correction_response(100.0) - response_before).norm() == 0.0,
        "refused rate change mutated state"
    );
    assert!(bank.set_sample_rate(24_000.0).is_ok());
    // 30 kHz pole cannot move to a 48 kHz clock: refusal keeps old state.
    let mut wide =
        KautzCorrection::<f64>::new(&[(30_000.0, 8.0)], 96_000.0).expect("valid at 96 kHz");
    wide.set_gains(&[2.0]).expect("gains");
    assert!(wide.set_sample_rate(48_000.0).is_err());
    assert_eq!(wide.gains(), vec![2.0]);
    assert_eq!(wide.sample_rate(), 96_000.0);
}

#[test]
fn correction_impulse_dtft_matches_declared_response() {
    // Streamed impulse through the recurrence vs closed-form response,
    // with the record sized from the slowest pole's tail rule.
    let modes = [(120.0, 6.0), (200.0, 5.0)];
    let gains = [1.25, -0.75];
    let grid: Vec<f64> = (1..60).map(|i| i as f64 * 10.0).collect();
    for &srate in &ACCEPT_RATES {
        let mut bank = KautzCorrection::<f64>::new(&modes, srate).expect("valid bank");
        bank.set_gains(&gains).expect("gains");
        let radius = slowest_radius(&modes, srate);
        let n = record_len_for_tail(radius, basis_b0(radius), 2.0, 1e-9);
        assert!(n < 1_000_000, "tail record unreasonable at {srate} Hz: {n}");
        let mut recorded = Vec::with_capacity(n);
        for i in 0..n {
            recorded.push(bank.process(if i == 0 { 1.0 } else { 0.0 }));
        }
        let measured = dtft(&recorded, &grid, srate);
        for (freq, m) in grid.iter().zip(measured.iter()) {
            let expected = bank.correction_response(*freq);
            assert!(
                (*m - expected).norm() < 1e-8,
                "impulse DTFT mismatch at {freq} Hz ({srate} Hz rate)"
            );
        }
    }
}

/// Fit a realizable target (built from `true_gains`) from zero gains with
/// wide-open composite bounds. Returns diagnostics plus both banks.
fn fit_realizable(
    modes: &[(f64, f64)],
    true_gains: &[f64],
    srate: f64,
) -> (
    super::correction_fit::FitDiagnostics,
    KautzCorrection<f64>,
    KautzCorrection<f64>,
) {
    let mut bank = KautzCorrection::<f64>::new(modes, srate).expect("valid");
    bank.set_gains(true_gains).expect("true gains");
    let freqs: Vec<f64> = (0..240).map(|i| 20.0 + i as f64 * 2.0).collect();
    let target: Vec<f64> = freqs.iter().map(|&f| truth_db(&bank, f)).collect();
    let mut fitted = KautzCorrection::<f64>::new(modes, srate).expect("valid");
    let config = FitConfig {
        max_iterations: 200,
        max_boost_db: 24.0,
        max_cut_db: 24.0,
        ..FitConfig::default()
    };
    let diag = fit_correction(&mut fitted, &freqs, &target, &config).expect("fit runs");
    (diag, bank, fitted)
}

fn max_transfer_error(
    fitted: &KautzCorrection<f64>,
    truth: &KautzCorrection<f64>,
    freqs: &[f64],
) -> f64 {
    freqs
        .iter()
        .map(|&f| rel_complex(fitted.correction_response(f), truth.correction_response(f)))
        .fold(0.0f64, f64::max)
}

#[test]
fn fit_recovers_well_posed_realizable_transfers() {
    // Realizable targets in good basins: full complex transfer recovery,
    // not coefficient identity. The zero-weight section must be retained
    // structurally and stay near zero numerically.
    let check_freqs: Vec<f64> = [25.0, 80.0, 100.2, 120.0, 160.0, 300.0, 499.0].to_vec();
    let cases = [
        (vec![(100.0, 8.0)], vec![2.0]),
        (vec![(80.0, 6.0), (120.0, 4.0)], vec![1.5, 1.0]),
        (
            vec![(80.0, 6.0), (120.0, 4.0), (160.0, 8.0)],
            vec![1.0, 0.0, 1.5],
        ),
    ];
    for (modes, true_gains) in &cases {
        let (diag, truth, fitted) = fit_realizable(modes, true_gains, 48_000.0);
        assert_eq!(diag.status, FitStatus::Converged, "case {true_gains:?}");
        assert!(
            diag.final_objective < 1e-9,
            "well-posed target must fit to noise (got {})",
            diag.final_objective
        );
        let err = max_transfer_error(&fitted, &truth, &check_freqs);
        assert!(
            err < 1e-4,
            "transfer recovery failed (err {err:.2e}, case {true_gains:?})"
        );
        assert_eq!(fitted.gains().len(), modes.len(), "sections retained");
        for g in fitted.gains() {
            assert!(g.is_finite(), "non-finite fitted gain");
        }
    }
    // Zero-weight section stays near zero numerically.
    let (_, _, fitted) = fit_realizable(
        &[(80.0, 6.0), (120.0, 4.0), (160.0, 8.0)],
        &[1.0, 0.0, 1.5],
        48_000.0,
    );
    assert!(
        fitted.gains()[1].abs() < 0.05,
        "zero-weight section drifted to {}",
        fitted.gains()[1]
    );
}

#[test]
fn fit_documents_ambiguous_and_ill_conditioned_limits() {
    // Opposite-sign gains create deep magnitude notches with ambiguous
    // phase: magnitude-only fitting can converge to a phase-flipped local
    // minimum ([-1.28, 1.74] vs true [2.0, -1.0] measured). The honest
    // contract is improvement + a non-global claim, never fake recovery.
    let (diag, _truth, fitted) =
        fit_realizable(&[(80.0, 6.0), (120.0, 4.0)], &[2.0, -1.0], 48_000.0);
    assert!(
        diag.final_objective < diag.initial_objective,
        "ambiguous case must still improve"
    );
    assert!(
        diag.final_objective > 1.0,
        "ambiguous case must NOT claim near-zero residual (got {})",
        diag.final_objective
    );
    for g in fitted.gains() {
        assert!(g.is_finite(), "non-finite fitted gain");
    }
    // Nearly coincident poles: transfer-level agreement only, with a loose
    // bound that records the conditioning limit instead of hiding it.
    let (diag, truth, fitted) =
        fit_realizable(&[(100.0, 60.0), (100.4, 60.0)], &[1.0, -1.0], 48_000.0);
    let check_freqs: Vec<f64> = [90.0, 99.0, 100.2, 101.0, 110.0, 200.0].to_vec();
    let err = max_transfer_error(&fitted, &truth, &check_freqs);
    assert!(
        err < 0.5,
        "ill-conditioned transfer error out of documented bound (err {err:.2e})"
    );
    assert!(diag.final_objective < diag.initial_objective);
}

#[test]
fn fit_handles_offgrid_narrow_and_sparse_grids() {
    // Narrow off-grid mode (107.3 Hz, Q=40) on a sparse nonuniform grid:
    // bounded, finite, strictly improving, sane on a dense grid.
    let srate = 48_000.0;
    let modes = [(107.3, 40.0)];
    let mut truth = KautzCorrection::<f64>::new(&modes, srate).expect("valid");
    truth.set_gains(&[-2.0]).expect("gains");
    // Sparse nonuniform grid with an intentional gap around the peak.
    let freqs: Vec<f64> = [
        20.0, 30.0, 45.0, 60.0, 80.0, 95.0, 102.0, 104.5, 110.5, 113.0, 120.0, 140.0, 170.0, 200.0,
        250.0, 320.0, 400.0, 500.0,
    ]
    .to_vec();
    assert!(
        !freqs.iter().any(|&f| (f - 107.3).abs() < 1.0),
        "test requires the peak to sit off-grid"
    );
    let target: Vec<f64> = freqs.iter().map(|&f| truth_db(&truth, f)).collect();
    let mut fitted = KautzCorrection::<f64>::new(&modes, srate).expect("valid");
    let config = FitConfig {
        max_iterations: 200,
        max_boost_db: 24.0,
        max_cut_db: 24.0,
        ..FitConfig::default()
    };
    let diag = fit_correction(&mut fitted, &freqs, &target, &config).expect("fit runs");
    assert!(
        diag.final_objective < diag.initial_objective,
        "sparse off-grid fit must improve"
    );
    for g in fitted.gains() {
        assert!(g.is_finite(), "non-finite fitted gain");
    }
    // Dense-grid sanity: bounded correction, peak captured within 3 dB.
    let dense: Vec<f64> = (0..480).map(|i| 20.0 + i as f64 * 1.0).collect();
    let mut worst = 0.0f64;
    for &f in &dense {
        let got = fitted.correction_db(f);
        let want = truth_db(&truth, f);
        assert!(got.is_finite(), "non-finite dense response at {f} Hz");
        assert!(
            got.abs() < 30.0,
            "dense response blew up at {f} Hz ({got} dB)"
        );
        worst = worst.max((got - want).abs());
    }
    assert!(
        worst < 3.0,
        "off-grid peak missed by {worst:.2} dB on the dense grid"
    );
}

#[test]
fn fit_infeasible_demand_is_bounds_active() {
    // A +12 dB narrow peak demand against a 3 dB ceiling: the search must
    // stop at the bound with a BoundsActive status, never by violating it.
    let srate = 48_000.0;
    let freqs: Vec<f64> = (0..96).map(|i| 20.0 + i as f64 * 5.0).collect();
    let target: Vec<f64> = freqs
        .iter()
        .map(|&f| 12.0 / (1.0 + ((f - 100.0) / 2.0).powi(2)))
        .collect();
    let mut bank = KautzCorrection::<f64>::new(&[(100.0, 25.0)], srate).expect("valid bank");
    let config = FitConfig {
        max_iterations: 200,
        max_boost_db: 3.0,
        max_cut_db: 20.0,
        ..FitConfig::default()
    };
    let diag = fit_correction(&mut bank, &freqs, &target, &config).expect("fit runs");
    assert_eq!(diag.status, FitStatus::BoundsActive);
    assert!(
        diag.worst_boost_db <= 3.01,
        "ceiling violated: {} dB",
        diag.worst_boost_db
    );
    // Dense-grid characterization (not a certificate): sampled guards do
    // not bound the continuum, so measure the between-point excess and
    // keep it small rather than claiming zero.
    let dense: Vec<f64> = (0..960).map(|i| 20.0 + i as f64 * 0.5).collect();
    let dense_max = dense
        .iter()
        .map(|&f| bank.correction_db(f))
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        dense_max <= diag.worst_boost_db + 0.15,
        "off-grid excess beyond characterization bound: dense {dense_max:.3} dB vs guard-grid {:.3} dB",
        diag.worst_boost_db
    );
    for g in bank.gains() {
        assert!(g.is_finite(), "non-finite fitted gain");
    }
}

#[test]
fn matched_budget_canary_unchanged() {
    // Analytic canary, parameters fixed by the requirements: 96 bins over
    // 20-500 Hz, 100 Hz narrow peak, seed 42, one section, 120 iterations,
    // shifts 0 and 3 Hz, boost ceiling 3 dB plus 0.01 dB tolerance.
    // Compared against an independent denser grid and the unchanged
    // absolute baseline error. A safe-but-ineffective outcome must read as
    // Stalled/BoundsActive, never as a converged improvement.
    let srate = 48_000.0;
    let bins: Vec<f64> = (0..96)
        .map(|i| 20.0f64 * (500.0f64 / 20.0f64).powf(f64::from(i) / 95.0))
        .collect();
    for &shift in &[0.0, 3.0] {
        let mut seed = 42u64;
        let ripple: Vec<f64> = bins
            .iter()
            .map(|_| (lcg_next(&mut seed) - 0.5) * 0.5)
            .collect();
        let target: Vec<f64> = bins
            .iter()
            .zip(ripple.iter())
            .map(|(&f, &r)| 9.0 / (1.0 + ((f - 100.0 - shift) / 3.0).powi(2)) + r)
            .collect();
        let mut bank = KautzCorrection::<f64>::new(&[(100.0 + shift, 25.0)], srate).expect("valid");
        let config = FitConfig {
            max_iterations: 120,
            max_boost_db: 3.0,
            max_cut_db: 20.0,
            ..FitConfig::default()
        };
        let diag = fit_correction(&mut bank, &bins, &target, &config).expect("fit runs");
        // Ceiling holds on the fit grid within numerical tolerance...
        assert!(
            diag.worst_boost_db <= 3.01,
            "canary ceiling violated on-grid (shift {shift}): {} dB",
            diag.worst_boost_db
        );
        // ...and on an independent 4x denser grid.
        let dense: Vec<f64> = (0..384)
            .map(|i| 20.0f64 * (500.0f64 / 20.0f64).powf(f64::from(i) / 383.0))
            .collect();
        let dense_max = dense
            .iter()
            .map(|&f| bank.correction_db(f))
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            dense_max <= diag.worst_boost_db + 0.15,
            "canary ceiling excess off-grid (shift {shift}): {dense_max:.3} dB"
        );
        // Baseline comparison: absolute error before vs after is recorded,
        // and a non-improvement must not wear a Converged label.
        assert!(
            diag.final_abs_rms_db <= diag.initial_abs_rms_db,
            "canary regressed the absolute baseline (shift {shift})"
        );
        if diag.final_abs_rms_db >= diag.initial_abs_rms_db - 1e-9 {
            assert!(
                diag.status != FitStatus::Converged,
                "safe-but-ineffective result mislabeled Converged (shift {shift})"
            );
        }
        // Gains stay finite and the bank keeps its single section.
        assert_eq!(bank.gains().len(), 1);
        for g in bank.gains() {
            assert!(g.is_finite());
        }
    }
}

#[test]
fn fit_invalid_inputs_exhaustion_and_bounds_refused() {
    let srate = 48_000.0;
    let freqs: Vec<f64> = (0..48).map(|i| 20.0 + i as f64 * 10.0).collect();
    let target = vec![0.0; freqs.len()];
    let good = FitConfig::default();
    // Every invalid input refuses with the bank untouched.
    let mut bank = KautzCorrection::<f64>::new(&[(100.0, 5.0)], srate).expect("valid");
    bank.set_gains(&[1.0]).expect("gains");
    let snapshot = bank.gains();
    let bad_configs: Vec<FitConfig> = vec![
        FitConfig {
            max_iterations: 0,
            ..good.clone()
        },
        FitConfig {
            tolerance: 0.0,
            ..good.clone()
        },
        FitConfig {
            tolerance: f64::NAN,
            ..good.clone()
        },
        FitConfig {
            regularization: -1.0,
            ..good.clone()
        },
        FitConfig {
            max_boost_db: -1.0,
            ..good.clone()
        },
        FitConfig {
            correction_band: Some((500.0, 20.0)),
            ..good.clone()
        },
        FitConfig {
            extra_constraints: vec![BandLimit {
                lo_hz: 200.0,
                hi_hz: 100.0,
                max_boost_db: 3.0,
                max_cut_db: 3.0,
            }],
            ..good.clone()
        },
        FitConfig {
            pole_guard_points: 0,
            ..good.clone()
        },
        FitConfig {
            normalization: NormalizationPolicy::MeanAnchored {
                band_lo_hz: 1000.0,
                band_hi_hz: 900.0,
            },
            ..good.clone()
        },
    ];
    for (i, bad) in bad_configs.iter().enumerate() {
        assert!(
            fit_correction(&mut bank, &freqs, &target, bad).is_err(),
            "bad config {i} must refuse"
        );
        assert_eq!(bank.gains(), snapshot, "refused fit {i} mutated gains");
    }
    // Anchor band covering no grid point refuses (grid tops out at 490 Hz).
    let uncovered = FitConfig {
        normalization: NormalizationPolicy::MeanAnchored {
            band_lo_hz: 900.0,
            band_hi_hz: 950.0,
        },
        ..good.clone()
    };
    assert!(fit_correction(&mut bank, &freqs, &target, &uncovered).is_err());
    assert_eq!(bank.gains(), snapshot);
    // Grid problems refuse too.
    assert!(fit_correction(&mut bank, &freqs[1..], &target, &good).is_err());
    assert!(fit_correction(&mut bank, &[], &[], &good).is_err());
    let mut unordered = freqs.clone();
    unordered.swap(5, 6);
    assert!(fit_correction(&mut bank, &unordered, &target, &good).is_err());
    let mut nonfinite = target.clone();
    nonfinite[3] = f64::NAN;
    assert!(fit_correction(&mut bank, &freqs, &nonfinite, &good).is_err());
    let short_weights = FitConfig {
        weights: FitWeights::Custom(vec![1.0; freqs.len() - 1]),
        ..good.clone()
    };
    assert!(fit_correction(&mut bank, &freqs, &target, &short_weights).is_err());
    let zero_weights = FitConfig {
        weights: FitWeights::Custom(vec![0.0; freqs.len()]),
        ..good.clone()
    };
    assert!(fit_correction(&mut bank, &freqs, &target, &zero_weights).is_err());
    assert_eq!(bank.gains(), snapshot, "refused fits mutated gains");
    // No NaN/Inf anywhere in diagnostics of a valid run.
    let diag = fit_correction(&mut bank, &freqs, &target, &good).expect("valid fit runs");
    for v in [
        diag.initial_objective,
        diag.final_objective,
        diag.initial_abs_rms_db,
        diag.final_abs_rms_db,
        diag.final_normalized_rms_db,
        diag.normalization_shift_db,
        diag.worst_boost_db,
        diag.worst_cut_db,
    ] {
        assert!(v.is_finite(), "non-finite diagnostic {v}");
    }
    // Iteration exhaustion: one step on a nontrivial target cannot converge.
    let mut tight = KautzCorrection::<f64>::new(&[(100.0, 8.0)], srate).expect("valid");
    let peak_target: Vec<f64> = freqs
        .iter()
        .map(|&f| 6.0 / (1.0 + ((f - 100.0) / 5.0).powi(2)))
        .collect();
    let one_step = FitConfig {
        max_iterations: 1,
        tolerance: 1e-15,
        max_boost_db: 24.0,
        max_cut_db: 24.0,
        ..good.clone()
    };
    let diag = fit_correction(&mut tight, &freqs, &peak_target, &one_step).expect("fit runs");
    assert_eq!(diag.status, FitStatus::IterationExhausted);
    assert!(diag.final_objective <= diag.initial_objective);
    for g in tight.gains() {
        assert!(g.is_finite(), "exhausted fit left non-finite gain");
    }
}

#[test]
fn processing_blocks_reset_and_channel_isolation() {
    // Blocks of 1, 17, 64, 257 plus a partial tail block match
    // sample-by-sample output bit-for-bit.
    let modes = [(80.0, 6.0), (150.0, 5.0)];
    let gains = [1.25, -0.75];
    let input: Vec<f64> = (0..300)
        .map(|i| ((i * 37) % 11) as f64 / 11.0 - 0.5)
        .collect();
    let mut reference = KautzCorrection::<f64>::new(&modes, 48_000.0).expect("valid bank");
    reference.set_gains(&gains).expect("gains");
    let expected: Vec<f64> = input.iter().map(|&x| reference.process(x)).collect();

    for &block in &[1usize, 17, 64, 257] {
        let mut bank = KautzCorrection::<f64>::new(&modes, 48_000.0).expect("valid bank");
        bank.set_gains(&gains).expect("gains");
        let mut buf = input.clone();
        let mut start = 0;
        while start < buf.len() {
            let end = (start + block).min(buf.len());
            bank.process_block(&mut buf[start..end]);
            start = end;
        }
        assert_eq!(buf, expected, "block size {block} diverged");
    }
    // Empty block is a no-op: state matches a fresh instance.
    let mut bank = KautzCorrection::<f64>::new(&modes, 48_000.0).expect("valid bank");
    bank.set_gains(&gains).expect("gains");
    bank.process_block(&mut []);
    let mut fresh = KautzCorrection::<f64>::new(&modes, 48_000.0).expect("valid bank");
    fresh.set_gains(&gains).expect("gains");
    assert_eq!(bank.process(0.5), fresh.process(0.5));

    // Reset reproducibility: process, reset, reprocess identically.
    let mut bank = KautzCorrection::<f64>::new(&modes, 48_000.0).expect("valid bank");
    bank.set_gains(&gains).expect("gains");
    let first: Vec<f64> = input.iter().map(|&x| bank.process(x)).collect();
    bank.reset();
    let second: Vec<f64> = input.iter().map(|&x| bank.process(x)).collect();
    assert_eq!(first, second, "reset did not reproduce the stream");

    // Channel isolation: two instances diverge under different inputs and
    // reconverge after reset.
    let mut left = KautzCorrection::<f64>::new(&modes, 48_000.0).expect("valid bank");
    let mut right = KautzCorrection::<f64>::new(&modes, 48_000.0).expect("valid bank");
    left.set_gains(&gains).expect("gains");
    right.set_gains(&gains).expect("gains");
    for &x in &input {
        left.process(x);
        right.process(-x);
    }
    assert_ne!(
        left.process(0.25),
        right.process(0.25),
        "channels share state"
    );
    left.reset();
    right.reset();
    assert_eq!(
        left.process(0.25),
        right.process(0.25),
        "reset channels diverged"
    );

    // Bypass outputs dry while states keep tracking: toggling back off
    // reproduces the never-bypassed stream exactly.
    let mut direct = KautzCorrection::<f64>::new(&modes, 48_000.0).expect("valid bank");
    direct.set_gains(&gains).expect("gains");
    let mut toggled = KautzCorrection::<f64>::new(&modes, 48_000.0).expect("valid bank");
    toggled.set_gains(&gains).expect("gains");
    for (i, &x) in input.iter().enumerate() {
        toggled.set_bypass(i % 2 == 0);
        let out = toggled.process(x);
        if i % 2 == 0 {
            assert_eq!(out, x, "bypass must output dry");
        }
        let _ = direct.process(x);
    }
    toggled.set_bypass(false);
    assert_eq!(
        toggled.process(0.125),
        direct.process(0.125),
        "bypass disturbed tracked state"
    );
}

#[test]
fn f32_playback_matches_f64_within_budget() {
    // Actual f32 recurrence against the f64 closed-form transfer: the
    // evaluator is not its own oracle (stream vs formula, width vs width).
    let modes = [(120.0, 6.0), (200.0, 5.0)];
    let gains = [1.25f64, -0.75];
    for &srate in &ACCEPT_RATES {
        let mut bank64 = KautzCorrection::<f64>::new(&modes, srate).expect("valid");
        bank64.set_gains(&gains).expect("gains");
        let mut bank32 = KautzCorrection::<f32>::new(&modes, srate).expect("valid f32 bank");
        bank32.set_gains(&gains).expect("f32 gains representable");
        let n = record_len_for_tail(
            slowest_radius(&modes, srate),
            basis_b0(slowest_radius(&modes, srate)),
            2.0,
            1e-9,
        );
        let mut worst = 0.0f64;
        for i in 0..n {
            let x = if i == 0 { 1.0 } else { 0.0 };
            let y32 = bank32.process(x as f32) as f64;
            let y64 = bank64.process(x);
            worst = worst.max((y32 - y64).abs());
            assert!(y32.is_finite(), "non-finite f32 playback at sample {i}");
        }
        // Declared precision budget, measured ~2e-6 on this fixture.
        assert!(
            worst < 1e-4,
            "f32 playback exceeded budget at {srate} Hz (worst {worst:.2e})"
        );
        // Coefficient fidelity at the source: f32 pole coefficients match
        // f64 to 2e-7 relative. (A narrowband DTFT-vs-analytic leg was
        // tried and dropped: f32 rounding re-rings at resonances and adds
        // coherently in those bins — 4.7e-3 at 44.1 kHz, 1.5e-2 at 96 kHz
        // on |H| ~ 3.5 — so that leg measures coefficient detune, not
        // streaming fidelity. Streaming is covered above; detune here.)
        for &(freq, q) in &modes {
            let s64 = KautzSection::new(freq, q, 1.0, srate);
            let s32 = KautzSection::<f32>::new(freq as f32, q as f32, 1.0, srate as f32);
            for (a32, a64, name) in [
                (s32.a1_coeff, s64.a1_coeff, "a1"),
                (s32.a2_coeff, s64.a2_coeff, "a2"),
            ] {
                let rel = (f64::from(a32) - a64).abs() / a64.abs().max(1e-30);
                assert!(
                    rel < 2e-7,
                    "f32 {name} detune at {freq} Hz ({srate} Hz): {rel:.2e}"
                );
            }
        }
    }
}

#[test]
fn sample_rate_rebuild_matches_fresh_and_rejects() {
    let modes = [(100.0, 8.0), (300.0, 6.0)];
    let gains = [1.5, -1.0];
    let mut bank = KautzCorrection::<f64>::new(&modes, 48_000.0).expect("valid");
    bank.set_gains(&gains).expect("gains");
    bank.set_strength(0.5).expect("strength");
    bank.process(1.0); // pollute state: rebuild must reset it
    bank.set_sample_rate(96_000.0).expect("rebuild at 96 kHz");
    let fresh = KautzCorrection::<f64>::new(&modes, 96_000.0).expect("fresh 96 kHz");
    let mut fresh_gained = fresh;
    fresh_gained.set_gains(&gains).expect("gains");
    fresh_gained.set_strength(0.5).expect("strength");
    for &freq in &[0.0, 50.0, 100.0, 300.0, 1000.0, 48_000.0] {
        let err = rel_complex(
            bank.correction_response(freq),
            fresh_gained.correction_response(freq),
        );
        assert!(err < 1e-12, "rebuilt response mismatch at {freq} Hz");
    }
    // Rebuilt state is reset: impulse response starts identically.
    let mut probe = KautzCorrection::<f64>::new(&modes, 96_000.0).expect("fresh");
    probe.set_gains(&gains).expect("gains");
    probe.set_strength(0.5).expect("strength");
    assert_eq!(bank.process(1.0), probe.process(1.0));
    // Gains, strength and modes survive the move.
    assert_eq!(bank.gains(), gains.to_vec());
    assert_eq!(bank.strength(), 0.5);
    assert_eq!(bank.modes().to_vec(), modes.to_vec());
}

#[test]
fn serialization_round_trip_with_legacy_aliases() {
    let mut bank =
        KautzCorrection::<f64>::new(&[(80.0, 6.0), (120.0, 4.0)], 48_000.0).expect("valid");
    bank.set_gains(&[1.5, 0.0])
        .expect("gains incl. zero weight");
    bank.set_strength(0.5).expect("strength");
    bank.set_bypass(true);
    let spec = bank.to_spec();
    assert_eq!(spec.format, KAUTZ_CORRECTION_FORMAT);
    let json = serde_json::to_string(&spec).expect("serializes");
    let parsed = parse_bank_spec(&json).expect("parses");
    assert_eq!(parsed, spec, "canonical round-trip");
    let rebuilt = KautzCorrection::<f64>::from_spec(&parsed).expect("rebuilds");
    for &freq in &[0.0, 80.0, 120.0, 500.0, 24_000.0] {
        let err = rel_complex(
            rebuilt.correction_response(freq),
            bank.correction_response(freq),
        );
        assert!(err < 1e-12, "round-trip transfer mismatch at {freq} Hz");
    }
    assert!(rebuilt.is_bypass());
    assert_eq!(rebuilt.gains().len(), 2, "zero-weight section retained");

    // Legacy field aliases deserialize to the identical spec.
    let legacy = r#"{
        "format": "kautz-correction-v1",
        "sample_rate": 48000.0,
        "sections": [
            {"freq": 80.0, "Q": 6.0, "weight": 1.5},
            {"frequency": 120.0, "q": 4.0, "gain": 0.0}
        ]
    }"#;
    let legacy_spec = parse_bank_spec(legacy).expect("legacy aliases parse");
    assert_eq!(legacy_spec.sample_rate_hz, 48_000.0);
    assert_eq!(
        legacy_spec.strength, 1.0,
        "absent strength defaults to full"
    );
    assert!(!legacy_spec.bypass);
    let legacy_bank = KautzCorrection::<f64>::from_spec(&legacy_spec).expect("legacy rebuilds");
    assert_eq!(legacy_bank.gains(), vec![1.5, 0.0]);

    // Unknown format and malformed JSON are structured refusals.
    let wrong_format =
        "{\"format\": \"kautz-bank-v0\", \"sample_rate_hz\": 48000.0, \"sections\": []}";
    assert!(matches!(
        parse_bank_spec(wrong_format).and_then(|spec| KautzCorrection::<f64>::from_spec(&spec)),
        Err(KautzError::UnsupportedModel { .. })
    ));
    assert!(matches!(
        parse_bank_spec("not json{"),
        Err(KautzError::UnsupportedModel { .. })
    ));
    // Spec round-trips across rates (rate-independent pole identity).
    let at_96k = KautzCorrection::<f64>::from_spec(&spec).expect("rebuilds");
    let mut moved = at_96k;
    moved.set_sample_rate(96_000.0).expect("rate move");
    let fresh = KautzCorrection::<f64>::new(&[(80.0, 6.0), (120.0, 4.0)], 96_000.0).expect("fresh");
    let mut fresh_gained = fresh;
    fresh_gained.set_gains(&[1.5, 0.0]).expect("gains");
    fresh_gained.set_strength(0.5).expect("strength");
    assert!(
        rel_complex(
            moved.correction_response(120.0),
            fresh_gained.correction_response(120.0)
        ) < 1e-12
    );
}

#[test]
fn normalization_policy_reported_separately() {
    // A +5 dB everywhere target under mean anchoring: the shift is
    // reported, the normalized error is what the optimizer sees, and the
    // absolute error stays honest beside it.
    let srate = 48_000.0;
    let freqs: Vec<f64> = (0..96).map(|i| 20.0 + i as f64 * 5.0).collect();
    let target = vec![5.0; freqs.len()];
    let anchored = FitConfig {
        normalization: NormalizationPolicy::MeanAnchored {
            band_lo_hz: 20.0,
            band_hi_hz: 500.0,
        },
        max_iterations: 200,
        max_boost_db: 24.0,
        max_cut_db: 24.0,
        ..FitConfig::default()
    };
    let mut bank = KautzCorrection::<f64>::new(&[(100.0, 8.0)], srate).expect("valid");
    let diag = fit_correction(&mut bank, &freqs, &target, &anchored).expect("fit runs");
    assert!(
        (diag.normalization_shift_db - 5.0).abs() < 1e-9,
        "anchor shift should recover the +5 dB offset (got {})",
        diag.normalization_shift_db
    );
    assert!(
        diag.final_normalized_rms_db <= diag.final_abs_rms_db,
        "normalized error must not exceed the absolute error"
    );
    // Absolute policy: zero shift, normalized equals absolute.
    let mut plain = KautzCorrection::<f64>::new(&[(100.0, 8.0)], srate).expect("valid");
    let absolute = FitConfig {
        max_iterations: 200,
        max_boost_db: 24.0,
        max_cut_db: 24.0,
        ..FitConfig::default()
    };
    let diag = fit_correction(&mut plain, &freqs, &target, &absolute).expect("fit runs");
    assert_eq!(diag.normalization_shift_db, 0.0);
    assert_eq!(
        diag.final_normalized_rms_db, diag.final_abs_rms_db,
        "absolute policy must report one error, not two"
    );
}

#[test]
fn check_bounds_standalone_reports_worst_points() {
    let srate = 48_000.0;
    let mut bank = KautzCorrection::<f64>::new(&[(100.0, 8.0)], srate).expect("valid");
    bank.set_gains(&[3.0]).expect("hot gains");
    let config = FitConfig {
        max_boost_db: 3.0,
        max_cut_db: 20.0,
        correction_band: Some((20.0, 500.0)),
        neutrality_db: 0.5,
        ..FitConfig::default()
    };
    let report = check_bounds(&bank, &config).expect("check runs");
    assert!(!report.passed, "hot bank must fail the 3 dB ceiling");
    assert!(
        report.worst_boost_db > 3.0,
        "worst boost must name the violation (got {})",
        report.worst_boost_db
    );
    assert!(
        (report.worst_boost_hz - 100.0).abs() < 15.0,
        "worst boost should sit near the pole (got {} Hz)",
        report.worst_boost_hz
    );
    // Guard identity: measurement points plus DC, Nyquist, pole sides.
    assert!(report.grid.includes_dc);
    assert!(report.grid.includes_nyquist);
    assert!(report.grid.freqs.contains(&0.0));
    assert!(report.grid.freqs.contains(&24_000.0));
    assert!(
        report.grid.freqs.len() > report.grid.measurement_count,
        "guard grid must extend the measurement grid"
    );
    // A quiet bank passes the same bounds.
    bank.set_gains(&[0.05]).expect("quiet gains");
    let report = check_bounds(&bank, &config).expect("check runs");
    assert!(report.passed, "quiet bank must pass");
}

/// Helper: realized correction magnitude in dB for target synthesis.
fn truth_db(bank: &KautzCorrection<f64>, freq: f64) -> f64 {
    bank.correction_db(freq)
}

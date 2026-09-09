use math_audio_iir_fir::{FirDesignConfig, generate_kirkeby_correction};
// Upstream regression for math-iir-fir/src/fir_design/tests.rs.
// Uses that module's FirDesignConfig and generate_kirkeby_correction imports.
#[test]
fn kirkeby_excess_phase_uses_frequency_coordinates_after_minimum_phase_fft() {
    use num_complex::Complex64;
    let sample_rate = 48_000.0;
    let pole = (-std::f64::consts::TAU * 150.0 / sample_rate).exp();
    let frequencies: Vec<f64> = (1..=2400).map(|i| i as f64 * 10.0).collect();
    let allpass = |f: f64| {
        let z = Complex64::from_polar(1.0, -std::f64::consts::TAU * f / sample_rate);
        (z - pole) / (1.0 - pole * z)
    };
    let phase: Vec<f64> = frequencies
        .iter()
        .map(|&f| allpass(f).arg().to_degrees())
        .collect();
    let magnitude = vec![0.0; frequencies.len()];
    let config = FirDesignConfig {
        n_taps: 8192,
        sample_rate,
        min_freq: 20.0,
        max_freq: 1000.0,
        correct_excess_phase: true,
        phase_smoothing_octaves: 0.0,
        pre_ringing: None,
        ..Default::default()
    };
    let taps =
        generate_kirkeby_correction(&frequencies, &magnitude, Some(&phase), &magnitude, &config);
    let response = |f: f64| -> Complex64 {
        taps.iter()
            .enumerate()
            .map(|(n, &h)| {
                h * Complex64::from_polar(1.0, -std::f64::consts::TAU * f * n as f64 / sample_rate)
            })
            .sum()
    };
    let mut raw_delays = Vec::new();
    let mut corrected_delays = Vec::new();
    for f in [40.0, 100.0, 200.0, 400.0] {
        let step = 0.01;
        let lower = allpass(f - step);
        let upper = allpass(f + step);
        let scale = -sample_rate / (std::f64::consts::TAU * 2.0 * step);
        raw_delays.push((upper / lower).arg() * scale);
        corrected_delays
            .push((upper * response(f + step) / (lower * response(f - step))).arg() * scale);
    }
    let spread = |values: &[f64]| {
        assert!(
            values.iter().all(|value| value.is_finite()),
            "nonfinite group delay: {values:?}"
        );
        values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - values.iter().copied().fold(f64::INFINITY, f64::min)
    };
    assert!(
        spread(&corrected_delays) < spread(&raw_delays) * 0.1,
        "phase dispersion: raw {:?}, corrected {:?}",
        raw_delays,
        corrected_delays
    );
}

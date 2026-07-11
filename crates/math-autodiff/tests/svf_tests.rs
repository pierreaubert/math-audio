#![allow(
    non_snake_case,
    reason = "`R` is the damping coefficient naming used in the task brief and SVF literature"
)]

use math_audio_autodiff::iir::svf::{SvFilter, SvfType};
use math_audio_autodiff::module::DiffModule;
use math_audio_autodiff::tensor::DiffTensor;
use math_audio_iir_fir::svf::{SvfFilter as RefSvfFilter, SvfFilterType};
use ndarray::{Array3, ArrayD, IxDyn};
use num_complex::Complex;
use std::f64::consts::PI;

const FS: f64 = 48_000.0;
const NFFT: usize = 512;
const ALIAS_DECAY_DB: f64 = 0.0;

fn ones_spectrum(shape: &[usize]) -> DiffTensor<f64> {
    let n = shape.iter().product();
    let data: Vec<Complex<f64>> = std::iter::repeat_n(Complex::new(1.0, 0.0), n).collect();
    DiffTensor::from_array(ArrayD::from_shape_vec(IxDyn(shape), data).unwrap())
}

fn complex_spectrum(shape: &[usize]) -> DiffTensor<f64> {
    let n = shape.iter().product();
    let data = (0..n)
        .map(|i| {
            let phase = i as f64 * 0.17;
            Complex::new(0.5 + phase.sin(), phase.cos() - 0.25)
        })
        .collect();
    DiffTensor::from_array(ArrayD::from_shape_vec(IxDyn(shape), data).unwrap())
}

fn mse_loss(output: &DiffTensor<f64>, target: &DiffTensor<f64>) -> f64 {
    output
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(o, t)| {
            let diff = o - t;
            diff.norm_sqr()
        })
        .sum::<f64>()
}

fn set_svf_param(filter: &mut SvFilter, fc_hz: f64, R: f64, gain_db: f64) {
    filter.param.fill(0.0);
    let shape = filter.param.shape();
    let n_out = shape[2];
    let n_in = shape[3];
    for out in 0..n_out {
        for inp in 0..n_in {
            filter.param[[0, 0, out, inp]] = fc_hz;
            filter.param[[0, 1, out, inp]] = R;
            if filter.filter_type.n_params() > 2 {
                filter.param[[0, 2, out, inp]] = gain_db;
            }
        }
    }
}

#[test]
fn svf_forward_matches_hand_built_sos() {
    let n_bins = NFFT / 2 + 1;

    for (filter_type, fc_hz, R, gain_db) in [
        (SvfType::Lowpass, 1_000.0, 1.0 / 0.707, 0.0),
        (SvfType::Highpass, 1_000.0, 1.0 / 0.707, 0.0),
        (SvfType::Bandpass, 1_000.0, 1.0 / 0.707, 0.0),
        (SvfType::Notch, 1_000.0, 1.0 / 5.0, 0.0),
        (SvfType::Allpass, 1_000.0, 1.0 / 0.707, 0.0),
        (SvfType::Peak, 1_000.0, 1.0 / 2.0, 6.0),
        (SvfType::Lowshelf, 1_000.0, 1.0 / 0.707, 6.0),
        (SvfType::Highshelf, 1_000.0, 1.0 / 0.707, 6.0),
    ] {
        let mut svf =
            SvFilter::new(NFFT, FS, 1, 1, filter_type, ALIAS_DECAY_DB).expect("valid svf filter");
        set_svf_param(&mut svf, fc_hz, R, gain_db);

        let input = ones_spectrum(&[1, n_bins, 1]);
        let svf_output = svf.forward(&input).expect("svf forward should succeed");

        // Build an equivalent SOS filter using the SVF's computed biquad coefficients.
        let mut sos_filter =
            math_audio_autodiff::iir::sos_filter::SosFilter::new(NFFT, 1, 1, 1, ALIAS_DECAY_DB)
                .expect("valid sos filter");
        let (b, a) = svf.coefficients();
        for tap in 0..3 {
            sos_filter.param[[0, tap, 0, 0]] = b[tap];
            sos_filter.param[[0, 3 + tap, 0, 0]] = a[tap];
        }
        let sos_output = sos_filter
            .forward(&input)
            .expect("sos forward should succeed");

        for bin in [0, 10, 50, 100, 200, n_bins - 1] {
            assert!(
                (svf_output.data[[0, bin, 0]].norm() - sos_output.data[[0, bin, 0]].norm()).abs()
                    < 1e-9,
                "filter={:?} bin={}",
                filter_type,
                bin
            );
        }
    }
}

#[test]
fn svf_default_parameters_are_inside_the_trainable_region() {
    let filter =
        SvFilter::new(NFFT, FS, 1, 1, SvfType::Lowpass, ALIAS_DECAY_DB).expect("valid svf");
    assert!(filter.param[[0, 0, 0, 0]] > 1.0);
    assert!(filter.param[[0, 0, 0, 0]] < FS * 0.499);
    assert!(filter.param[[0, 1, 0, 0]] > 1e-6);
}

#[test]
fn svf_gradient_finite_difference_lowpass() {
    let n_bins = NFFT / 2 + 1;
    let mut filter =
        SvFilter::new(NFFT, FS, 1, 1, SvfType::Lowpass, ALIAS_DECAY_DB).expect("valid svf filter");
    set_svf_param(&mut filter, 1_000.0, 1.0 / 0.707, 0.0);

    let target = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, 1), Complex::new(0.3, -0.2)).into_dyn(),
    );

    let input = complex_spectrum(&[1, n_bins, 1]);
    let output = filter.forward(&input).expect("forward should succeed");
    let grad_output_data: Vec<Complex<f64>> = output
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(o, t)| 2.0 * (o - t))
        .collect();
    let grad_output =
        DiffTensor::from_array(Array3::from_shape_vec((1, n_bins, 1), grad_output_data).unwrap());
    filter.zero_grad();
    let _grad_input = filter
        .backward(&input, &output, &grad_output)
        .expect("backward should succeed");
    let analytical = filter.gradients()[0].clone();

    let epsilon = 1e-5;
    let n_params = filter.filter_type.n_params();
    for p in 0..n_params {
        let mut param_plus = filter.param.clone();
        param_plus[[0, p, 0, 0]] += epsilon;
        let mut filter_plus = filter.clone();
        filter_plus.param = param_plus;
        let out_plus = filter_plus.forward(&input).expect("forward should succeed");
        let loss_plus = mse_loss(&out_plus, &target);

        let mut param_minus = filter.param.clone();
        param_minus[[0, p, 0, 0]] -= epsilon;
        let mut filter_minus = filter.clone();
        filter_minus.param = param_minus;
        let out_minus = filter_minus
            .forward(&input)
            .expect("forward should succeed");
        let loss_minus = mse_loss(&out_minus, &target);

        let finite_diff = (loss_plus - loss_minus) / (2.0 * epsilon);
        let analytical_val = analytical[[0, p, 0, 0]];

        let denom = finite_diff.abs().max(1e-8);
        let relative_error = (analytical_val - finite_diff).abs() / denom;
        assert!(
            relative_error < 1e-4 || (analytical_val - finite_diff).abs() < 1e-6,
            "p={}: analytical={} finite_diff={} rel_err={}",
            p,
            analytical_val,
            finite_diff,
            relative_error
        );
    }
}

#[test]
fn svf_gradient_finite_difference_peak() {
    let n_bins = NFFT / 2 + 1;
    let mut filter =
        SvFilter::new(NFFT, FS, 1, 1, SvfType::Peak, ALIAS_DECAY_DB).expect("valid svf filter");
    set_svf_param(&mut filter, 1_000.0, 1.0 / 2.0, 6.0);

    let target = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, 1), Complex::new(0.3, -0.2)).into_dyn(),
    );

    let input = complex_spectrum(&[1, n_bins, 1]);
    let output = filter.forward(&input).expect("forward should succeed");
    let grad_output_data: Vec<Complex<f64>> = output
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(o, t)| 2.0 * (o - t))
        .collect();
    let grad_output =
        DiffTensor::from_array(Array3::from_shape_vec((1, n_bins, 1), grad_output_data).unwrap());
    filter.zero_grad();
    let _grad_input = filter
        .backward(&input, &output, &grad_output)
        .expect("backward should succeed");
    let analytical = filter.gradients()[0].clone();

    let epsilon = 1e-5;
    let n_params = filter.filter_type.n_params();
    for p in 0..n_params {
        let mut param_plus = filter.param.clone();
        param_plus[[0, p, 0, 0]] += epsilon;
        let mut filter_plus = filter.clone();
        filter_plus.param = param_plus;
        let out_plus = filter_plus.forward(&input).expect("forward should succeed");
        let loss_plus = mse_loss(&out_plus, &target);

        let mut param_minus = filter.param.clone();
        param_minus[[0, p, 0, 0]] -= epsilon;
        let mut filter_minus = filter.clone();
        filter_minus.param = param_minus;
        let out_minus = filter_minus
            .forward(&input)
            .expect("forward should succeed");
        let loss_minus = mse_loss(&out_minus, &target);

        let finite_diff = (loss_plus - loss_minus) / (2.0 * epsilon);
        let analytical_val = analytical[[0, p, 0, 0]];

        let denom = finite_diff.abs().max(1e-8);
        let relative_error = (analytical_val - finite_diff).abs() / denom;
        assert!(
            relative_error < 1e-4 || (analytical_val - finite_diff).abs() < 1e-6,
            "p={}: analytical={} finite_diff={} rel_err={}",
            p,
            analytical_val,
            finite_diff,
            relative_error
        );
    }
}

fn map_svf_type(ty: SvfType) -> SvfFilterType {
    match ty {
        SvfType::Lowpass => SvfFilterType::Lowpass,
        SvfType::Highpass => SvfFilterType::Highpass,
        SvfType::Bandpass => SvfFilterType::Bandpass,
        SvfType::Notch => SvfFilterType::Notch,
        SvfType::Peak => SvfFilterType::Peak,
        SvfType::Lowshelf => SvfFilterType::Lowshelf,
        SvfType::Highshelf => SvfFilterType::Highshelf,
        SvfType::Allpass => SvfFilterType::Allpass,
    }
}

/// Measure the steady-state magnitude of `math-iir-fir`'s SVF at `freq` by
/// processing a sinusoid. This uses the time-domain `process` implementation
/// rather than the analytic `response_at`, which has known sign errors for
/// some filter types.
fn process_magnitude(
    filter_type: SvfFilterType,
    fc_hz: f64,
    fs: f64,
    q: f64,
    gain_db: f64,
    freq: f64,
) -> f64 {
    let mut filter = RefSvfFilter::new(filter_type, fc_hz, fs, q, gain_db);

    // Warm up long enough for any resonant transient to decay, then measure
    // over a generous window so that a full cycle of low-frequency signals is
    // captured.
    let skip = (fs * 0.5).ceil() as usize;
    let measure = (fs * 0.2).ceil() as usize;

    if freq < 1.0 {
        // DC / near-DC: use a constant input.
        let mut sum_y = 0.0f64;
        for i in 0..skip + measure {
            let y = filter.process(1.0);
            if i >= skip {
                sum_y += y;
            }
        }
        (sum_y / measure as f64).abs()
    } else {
        // Fit the steady-state output to A*sin(omega*n) + B*cos(omega*n).
        // This removes finite-window bias for arbitrary frequencies.
        let omega = 2.0 * PI * freq / fs;
        let mut s_ss = 0.0f64;
        let mut s_cc = 0.0f64;
        let mut s_sc = 0.0f64;
        let mut s_sy = 0.0f64;
        let mut s_cy = 0.0f64;
        for i in 0..skip + measure {
            let s = (omega * i as f64).sin();
            let c = (omega * i as f64).cos();
            let y = filter.process(s);
            if i >= skip {
                s_ss += s * s;
                s_cc += c * c;
                s_sc += s * c;
                s_sy += s * y;
                s_cy += c * y;
            }
        }
        let det = s_ss * s_cc - s_sc * s_sc;
        let a = (s_sy * s_cc - s_sc * s_cy) / det;
        let b = (s_ss * s_cy - s_sy * s_sc) / det;
        (a * a + b * b).sqrt()
    }
}

#[test]
fn svf_forward_matches_math_iir_fir_process() {
    let n_bins = NFFT / 2 + 1;

    for (filter_type, fc_hz, R, gain_db) in [
        (SvfType::Lowpass, 1_000.0, 1.0 / 0.707, 0.0),
        (SvfType::Highpass, 1_000.0, 1.0 / 0.707, 0.0),
        (SvfType::Bandpass, 1_000.0, 1.0 / 0.707, 0.0),
        (SvfType::Notch, 1_000.0, 1.0 / 5.0, 0.0),
        (SvfType::Allpass, 1_000.0, 1.0 / 0.707, 0.0),
        (SvfType::Peak, 1_000.0, 1.0 / 2.0, 6.0),
        (SvfType::Lowshelf, 1_000.0, 1.0 / 0.707, 6.0),
        (SvfType::Highshelf, 1_000.0, 1.0 / 0.707, 6.0),
    ] {
        let mut svf =
            SvFilter::new(NFFT, FS, 1, 1, filter_type, ALIAS_DECAY_DB).expect("valid svf filter");
        set_svf_param(&mut svf, fc_hz, R, gain_db);

        let input = ones_spectrum(&[1, n_bins, 1]);
        let output = svf.forward(&input).expect("svf forward should succeed");

        let q = 1.0 / R;
        for bin in [0, 1, 10, 50, 100, 200] {
            let freq = bin as f64 * FS / NFFT as f64;
            let expected =
                process_magnitude(map_svf_type(filter_type), fc_hz, FS, q, gain_db, freq);
            let actual = output.data[[0, bin, 0]].norm();
            let scale = expected.max(1.0);
            assert!(
                (actual - expected).abs() < 1e-4 || (actual - expected).abs() / scale < 1e-4,
                "filter={:?} bin={} freq={}: actual={} expected={}",
                filter_type,
                bin,
                freq,
                actual,
                expected
            );
        }
    }
}

#[test]
fn svf_gradient_remains_accurate_near_the_top_of_the_audio_band() {
    let n_bins = NFFT / 2 + 1;
    let mut filter = SvFilter::new(NFFT, FS, 1, 1, SvfType::Peak, ALIAS_DECAY_DB).unwrap();
    set_svf_param(&mut filter, 20_000.0, 0.05, 12.0);
    let input = complex_spectrum(&[1, n_bins, 1]);
    let output = filter.forward(&input).unwrap();
    let grad_output = DiffTensor::from_array(output.data.clone() * 2.0);
    filter.zero_grad();
    filter.backward(&input, &output, &grad_output).unwrap();

    for (p, eps) in [(0, 1e-2), (1, 1e-6), (2, 1e-5)] {
        let mut plus = filter.clone();
        plus.param[[0, p, 0, 0]] += eps;
        let plus_loss = plus
            .forward(&input)
            .unwrap()
            .data
            .iter()
            .map(Complex::norm_sqr)
            .sum::<f64>();
        let mut minus = filter.clone();
        minus.param[[0, p, 0, 0]] -= eps;
        let minus_loss = minus
            .forward(&input)
            .unwrap()
            .data
            .iter()
            .map(Complex::norm_sqr)
            .sum::<f64>();
        let numerical = (plus_loss - minus_loss) / (2.0 * eps);
        let analytical = filter.param_grad[[0, p, 0, 0]];
        let scale = numerical.abs().max(1.0);
        assert!(
            (analytical - numerical).abs() / scale < 2e-5,
            "p={p}: analytical={analytical}, numerical={numerical}"
        );
    }
}

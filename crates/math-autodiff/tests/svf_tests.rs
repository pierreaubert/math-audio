#![allow(
    non_snake_case,
    reason = "`R` is the damping coefficient naming used in the task brief and SVF literature"
)]

use math_audio_autodiff::iir::svf::{SvFilter, SvfType};
use math_audio_autodiff::module::DiffModule;
use math_audio_autodiff::tensor::DiffTensor;
use ndarray::{Array3, ArrayD, IxDyn};
use num_complex::Complex;

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
        let mut svf = SvFilter::new(NFFT, FS, 1, 1, filter_type, ALIAS_DECAY_DB)
            .expect("valid svf filter");
        set_svf_param(&mut svf, fc_hz, R, gain_db);

        let input = ones_spectrum(&[1, n_bins, 1]);
        let svf_output = svf.forward(&input).expect("svf forward should succeed");

        // Build an equivalent SOS filter using the SVF's computed biquad coefficients.
        let mut sos_filter = math_audio_autodiff::iir::sos_filter::SosFilter::new(
            NFFT,
            1,
            1,
            1,
            ALIAS_DECAY_DB,
        )
        .expect("valid sos filter");
        let (b, a) = svf.coefficients();
        for tap in 0..3 {
            sos_filter.param[[0, tap, 0, 0]] = b[tap];
            sos_filter.param[[0, 3 + tap, 0, 0]] = a[tap];
        }
        let sos_output = sos_filter.forward(&input).expect("sos forward should succeed");

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
fn svf_gradient_finite_difference_lowpass() {
    let n_bins = NFFT / 2 + 1;
    let mut filter = SvFilter::new(NFFT, FS, 1, 1, SvfType::Lowpass, ALIAS_DECAY_DB)
        .expect("valid svf filter");
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
        let out_minus = filter_minus.forward(&input).expect("forward should succeed");
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
    let mut filter = SvFilter::new(NFFT, FS, 1, 1, SvfType::Peak, ALIAS_DECAY_DB)
        .expect("valid svf filter");
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
        let out_minus = filter_minus.forward(&input).expect("forward should succeed");
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

use approx::assert_relative_eq;
use math_audio_autodiff::{
    delay::{Delay, ParallelDelay},
    module::DiffModule,
    tensor::DiffTensor,
};
use ndarray::{Array3, ArrayD, IxDyn};
use num_complex::Complex;

const NFFT: usize = 512;

fn random_spectrum(shape: &[usize]) -> DiffTensor<f64> {
    let n = shape.iter().product();
    let data: Vec<Complex<f64>> = (0..n)
        .map(|_| Complex::new(rand::random::<f64>() - 0.5, rand::random::<f64>() - 0.5))
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

#[test]
fn parallel_delay_forward_matches_phase_rotation() {
    let n_bins = NFFT / 2 + 1;
    let mut delay = ParallelDelay::new(NFFT, 1, 0.0).unwrap();
    delay.param[[0]] = 0.0; // raw -> tau = 0

    let input = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, 1), Complex::new(1.0, 0.0)).into_dyn(),
    );
    let output = delay.forward(&input).unwrap();
    assert_relative_eq!(output.data[[0, 10, 0]].re, 1.0, epsilon = 1e-9);
    assert_relative_eq!(output.data[[0, 10, 0]].im, 0.0, epsilon = 1e-9);
}

#[test]
fn delay_mapping_is_bounded_and_finite() {
    let n_bins = NFFT / 2 + 1;
    let input = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, 1), Complex::new(1.0, 0.0)).into_dyn(),
    );
    for raw in [-1_000.0, -2.0, 0.0] {
        let mut delay = ParallelDelay::new(NFFT, 1, 0.0).unwrap();
        delay.param[[0]] = raw;
        let output = delay.forward(&input).unwrap();
        assert!(output.data.iter().all(|value| value.is_finite()));
        // A non-negative delay has a non-positive phase at the first bin.
        assert!(
            output.data[[0, 1, 0]].im <= 1e-12,
            "raw={raw} mapped below tau_min"
        );
    }
    let mut delay = ParallelDelay::new(NFFT, 1, 0.0).unwrap();
    delay.param[[0]] = 1_000.0;
    let output = delay.forward(&input).unwrap();
    assert!(output.data.iter().all(|value| value.is_finite()));
}

#[test]
fn delay_backward_rejects_mismatched_batch_shape() {
    let n_bins = NFFT / 2 + 1;
    let mut delay = Delay::new(NFFT, 1, 1, 0.0).unwrap();
    let input = random_spectrum(&[2, n_bins, 1]);
    let output = delay.forward(&input).unwrap();
    let grad = random_spectrum(&[1, n_bins, 1]);
    assert!(delay.backward(&input, &output, &grad).is_err());
}

#[test]
fn parallel_delay_backward_rejects_mismatched_channel_shape() {
    let n_bins = NFFT / 2 + 1;
    let mut delay = ParallelDelay::new(NFFT, 1, 0.0).unwrap();
    let input = random_spectrum(&[1, n_bins, 2]);
    let grad = random_spectrum(&[1, n_bins, 2]);
    let output = DiffTensor::zeros(IxDyn(&[1, n_bins, 2]));
    assert!(delay.backward(&input, &output, &grad).is_err());
}

#[test]
fn delay_forward_uses_the_current_public_parameter_shape() {
    let n_bins = NFFT / 2 + 1;
    let mut delay = Delay::new(NFFT, 1, 1, 0.0).unwrap();
    delay.param = ArrayD::zeros(IxDyn(&[2, 1]));
    let input = random_spectrum(&[1, n_bins, 1]);
    let output = delay.forward(&input).unwrap();
    assert_eq!(output.data.shape(), &[1, n_bins, 2]);
}

#[test]
fn parallel_delay_gradient_matches_finite_difference() {
    let n_bins = NFFT / 2 + 1;
    let mut delay = ParallelDelay::new(NFFT, 1, 0.0).unwrap();
    delay.param[[0]] = 1.0; // some non-zero raw

    let input = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, 1), Complex::new(1.0, 0.1)).into_dyn(),
    );
    let target = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, 1), Complex::new(0.5, -0.2)).into_dyn(),
    );

    let eps = 1e-6;
    let loss_plus = {
        delay.param[[0]] += eps;
        let out = delay.forward(&input).unwrap();
        let diff = &out.data - &target.data;
        diff.iter().map(|x| x.norm_sqr()).sum::<f64>()
    };
    let loss_minus = {
        delay.param[[0]] -= 2.0 * eps;
        let out = delay.forward(&input).unwrap();
        let diff = &out.data - &target.data;
        diff.iter().map(|x| x.norm_sqr()).sum::<f64>()
    };
    let numeric_grad = (loss_plus - loss_minus) / (2.0 * eps);

    delay.param[[0]] += eps; // restore
    delay.zero_grad();
    let out = delay.forward(&input).unwrap();
    let diff = &out.data - &target.data;
    let grad = DiffTensor::from_array(diff.into_owned() * 2.0);
    delay.backward(&input, &out, &grad).unwrap();

    assert_relative_eq!(delay.param_grad[[0]], numeric_grad, epsilon = 1e-5);
}

#[test]
fn delay_gradient_matches_finite_difference() {
    let n_bins = NFFT / 2 + 1;
    let n_out = 2;
    let n_in = 3;
    let mut delay = Delay::new(NFFT, n_out, n_in, 0.0).unwrap();
    delay.param =
        ArrayD::from_shape_vec(IxDyn(&[n_out, n_in]), vec![0.2, -0.5, 1.0, -0.7, 0.9, 0.1])
            .unwrap();

    let input = random_spectrum(&[1, n_bins, n_in]);
    let output = delay.forward(&input).expect("forward should succeed");

    let target = DiffTensor::zeros(IxDyn(output.data.shape()));
    let grad_output_data: Vec<Complex<f64>> = output
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(o, t)| 2.0 * (o - t))
        .collect();
    let grad_output = DiffTensor::from_array(
        ArrayD::from_shape_vec(IxDyn(output.data.shape()), grad_output_data).unwrap(),
    );

    delay.zero_grad();
    let _grad_input = delay
        .backward(&input, &output, &grad_output)
        .expect("backward should succeed");
    let analytical = delay.param_grad.clone();

    let epsilon = 1e-5;
    for out in 0..n_out {
        for inp in 0..n_in {
            let mut param_plus = delay.param.clone();
            param_plus[[out, inp]] += epsilon;
            let mut delay_plus = delay.clone();
            delay_plus.param = param_plus;
            delay_plus.param_grad.fill(0.0);
            let out_plus = delay_plus.forward(&input).expect("forward should succeed");
            let loss_plus = mse_loss(&out_plus, &target);

            let mut param_minus = delay.param.clone();
            param_minus[[out, inp]] -= epsilon;
            let mut delay_minus = delay.clone();
            delay_minus.param = param_minus;
            delay_minus.param_grad.fill(0.0);
            let out_minus = delay_minus.forward(&input).expect("forward should succeed");
            let loss_minus = mse_loss(&out_minus, &target);

            let finite_diff = (loss_plus - loss_minus) / (2.0 * epsilon);
            let analytical_val = analytical[[out, inp]];

            let denom = finite_diff.abs().max(1e-8);
            let relative_error = (analytical_val - finite_diff).abs() / denom;
            assert!(
                relative_error < 1e-4 || (analytical_val - finite_diff).abs() < 1e-6,
                "out={} inp={}: analytical={} finite_diff={} rel_err={}",
                out,
                inp,
                analytical_val,
                finite_diff,
                relative_error
            );
        }
    }
}

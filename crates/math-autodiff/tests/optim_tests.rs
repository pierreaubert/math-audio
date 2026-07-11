//! Tests for loss functions and the SGD optimizer.

use approx::assert_abs_diff_eq;
use math_audio_autodiff::gain::Gain;
use math_audio_autodiff::loss::{
    magnitude_mse_loss, magnitude_mse_loss_backward, mse_loss, mse_loss_backward,
};
use math_audio_autodiff::module::DiffModule;
use math_audio_autodiff::optim::Sgd;
use math_audio_autodiff::tensor::DiffTensor;
use ndarray::{Array, ArrayD, IxDyn};
use num_complex::Complex;

fn complex_tensor(values: &[Complex<f64>], shape: &[usize]) -> DiffTensor<f64> {
    let arr = Array::from_shape_vec(IxDyn(shape), values.to_vec()).unwrap();
    DiffTensor::from_array(arr)
}

#[test]
fn mse_loss_matches_manual() {
    let pred = complex_tensor(
        &[
            Complex::new(1.0, 2.0),
            Complex::new(3.0, -1.0),
            Complex::new(0.0, 1.0),
        ],
        &[3],
    );
    let target = complex_tensor(
        &[
            Complex::new(0.0, 1.0),
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 0.0),
        ],
        &[3],
    );

    let expected = ((1.0_f64.powi(2) + 1.0_f64.powi(2))
        + (2.0_f64.powi(2) + 2.0_f64.powi(2))
        + (2.0_f64.powi(2) + 1.0_f64.powi(2)))
        / 3.0;

    assert_abs_diff_eq!(mse_loss(&pred, &target).unwrap(), expected, epsilon = 1e-12);
}

#[test]
fn mse_loss_backward_matches_finite_diff() {
    let values = vec![
        Complex::new(1.0, 2.0),
        Complex::new(-0.5, 1.5),
        Complex::new(2.0, -1.0),
    ];
    let target_values = vec![
        Complex::new(0.5, 1.0),
        Complex::new(1.0, -0.5),
        Complex::new(0.0, 0.0),
    ];
    let pred = complex_tensor(&values, &[3]);
    let target = complex_tensor(&target_values, &[3]);

    let grad = mse_loss_backward(&pred, &target).unwrap();
    let eps = 1e-6;

    for i in 0..values.len() {
        // Real-part finite difference.
        let mut plus = values.clone();
        plus[i].re += eps;
        let mut minus = values.clone();
        minus[i].re -= eps;
        let loss_plus = mse_loss(&complex_tensor(&plus, &[3]), &target).unwrap();
        let loss_minus = mse_loss(&complex_tensor(&minus, &[3]), &target).unwrap();
        let numerical = (loss_plus - loss_minus) / (2.0 * eps);
        assert_abs_diff_eq!(grad.data[i].re, numerical, epsilon = 1e-5);

        // Imaginary-part finite difference.
        let mut plus = values.clone();
        plus[i].im += eps;
        let mut minus = values.clone();
        minus[i].im -= eps;
        let loss_plus = mse_loss(&complex_tensor(&plus, &[3]), &target).unwrap();
        let loss_minus = mse_loss(&complex_tensor(&minus, &[3]), &target).unwrap();
        let numerical = (loss_plus - loss_minus) / (2.0 * eps);
        assert_abs_diff_eq!(grad.data[i].im, numerical, epsilon = 1e-5);
    }
}

#[test]
fn magnitude_mse_loss_backward_matches_finite_diff() {
    let values = vec![
        Complex::new(1.0, 2.0),
        Complex::new(-0.5, 1.5),
        Complex::new(2.0, -1.0),
    ];
    let target_values = vec![
        Complex::new(0.5, 1.0),
        Complex::new(1.0, -0.5),
        Complex::new(0.0, 0.0),
    ];
    let pred = complex_tensor(&values, &[3]);
    let target = complex_tensor(&target_values, &[3]);

    let grad = magnitude_mse_loss_backward(&pred, &target).unwrap();
    let eps = 1e-6;

    for i in 0..values.len() {
        // Real-part finite difference.
        let mut plus = values.clone();
        plus[i].re += eps;
        let mut minus = values.clone();
        minus[i].re -= eps;
        let loss_plus = magnitude_mse_loss(&complex_tensor(&plus, &[3]), &target).unwrap();
        let loss_minus = magnitude_mse_loss(&complex_tensor(&minus, &[3]), &target).unwrap();
        let numerical = (loss_plus - loss_minus) / (2.0 * eps);
        assert_abs_diff_eq!(grad.data[i].re, numerical, epsilon = 1e-5);

        // Imaginary-part finite difference.
        let mut plus = values.clone();
        plus[i].im += eps;
        let mut minus = values.clone();
        minus[i].im -= eps;
        let loss_plus = magnitude_mse_loss(&complex_tensor(&plus, &[3]), &target).unwrap();
        let loss_minus = magnitude_mse_loss(&complex_tensor(&minus, &[3]), &target).unwrap();
        let numerical = (loss_plus - loss_minus) / (2.0 * eps);
        assert_abs_diff_eq!(grad.data[i].im, numerical, epsilon = 1e-5);
    }
}

#[test]
fn sgd_reduces_mse_loss() {
    let nfft = 32;
    let n_bins = nfft / 2 + 1;

    // Target gain value to recover.
    let target_gain = 2.5_f64;

    // Build a single-channel input spectrum (batch=1, n_bins, 1 channel).
    let input_values: Vec<Complex<f64>> = (0..n_bins)
        .map(|i| Complex::new((i as f64 * 0.1).cos(), (i as f64 * 0.1).sin()))
        .collect();
    let input = DiffTensor::from_array(
        Array::from_shape_vec(IxDyn(&[1, n_bins, 1]), input_values.clone()).unwrap(),
    );

    // Target is input multiplied by the target gain.
    let target_values: Vec<Complex<f64>> = input_values.iter().map(|x| x * target_gain).collect();
    let target = DiffTensor::from_array(
        Array::from_shape_vec(IxDyn(&[1, n_bins, 1]), target_values).unwrap(),
    );

    let mut gain = Gain::new(nfft, 1, 1).unwrap();
    // Initialize gain to a wrong value.
    gain.param.fill(0.5);

    let optimizer = Sgd::new(0.05);

    let initial_loss = {
        let output = gain.forward(&input).unwrap();
        mse_loss(&output, &target).unwrap()
    };

    for _ in 0..200 {
        gain.zero_grad();
        let output = gain.forward(&input).unwrap();
        let grad_output = mse_loss_backward(&output, &target).unwrap();
        gain.backward(&input, &output, &grad_output).unwrap();

        let grads: Vec<&ArrayD<f64>> = gain.gradients();
        let grads_owned: Vec<ArrayD<f64>> = grads.iter().map(|g| (**g).clone()).collect();
        let mut params: Vec<&mut ArrayD<f64>> = gain.parameters_mut();
        // SGD expects owned slices; clone parameters for the update.
        let mut params_owned: Vec<ArrayD<f64>> = params.iter_mut().map(|p| (**p).clone()).collect();
        optimizer.step(&mut params_owned, &grads_owned).unwrap();
        for (p, p_owned) in params.iter_mut().zip(params_owned) {
            **p = p_owned;
        }
    }

    let final_loss = {
        let output = gain.forward(&input).unwrap();
        mse_loss(&output, &target).unwrap()
    };

    assert!(
        final_loss < initial_loss,
        "SGD did not reduce MSE loss: initial {initial_loss}, final {final_loss}"
    );
    assert_abs_diff_eq!(gain.param[[0, 0]], target_gain, epsilon = 1e-2);
}

#[test]
fn losses_reject_mismatched_and_empty_tensors() {
    let one = complex_tensor(&[Complex::new(1.0, 0.0)], &[1]);
    let two = complex_tensor(&[Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)], &[2]);
    let empty = complex_tensor(&[], &[0]);

    assert!(mse_loss(&one, &two).is_err());
    assert!(mse_loss_backward(&one, &two).is_err());
    assert!(magnitude_mse_loss(&one, &two).is_err());
    assert!(magnitude_mse_loss_backward(&one, &two).is_err());
    assert!(mse_loss(&empty, &empty).is_err());
    assert!(magnitude_mse_loss(&empty, &empty).is_err());
}

#[test]
fn sgd_rejects_incompatible_gradients_before_updating() {
    let optimizer = Sgd::new(0.1);
    let original = ArrayD::from_elem(IxDyn(&[2]), 1.0);
    let mut params = vec![original.clone()];
    let wrong_count: Vec<ArrayD<f64>> = Vec::new();
    assert!(optimizer.step(&mut params, &wrong_count).is_err());
    assert_eq!(params[0], original);

    let wrong_shape = vec![ArrayD::zeros(IxDyn(&[1]))];
    assert!(optimizer.step(&mut params, &wrong_shape).is_err());
    assert_eq!(params[0], original);
}

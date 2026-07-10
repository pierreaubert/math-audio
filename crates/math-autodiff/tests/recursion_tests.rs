use approx::assert_relative_eq;
use math_audio_autodiff::{
    gain::Gain, module::DiffModule, recursion::Recursion, tensor::DiffTensor,
};
use ndarray::Array3;
use num_complex::Complex;

const NFFT: usize = 512;

#[test]
fn recursion_forward_is_finite_with_nonzero_feedback() {
    let n = 2;
    let mut feedforward = Gain::new(NFFT, n, n).unwrap();
    feedforward.param[[0, 0]] = 0.5;
    feedforward.param[[1, 1]] = -0.3;
    let mut feedback = Gain::new(NFFT, n, n).unwrap();
    feedback.param[[0, 0]] = 0.1;
    feedback.param[[1, 1]] = 0.05;

    let recursion = Recursion::new(Box::new(feedforward), Box::new(feedback)).unwrap();

    let n_bins = NFFT / 2 + 1;
    let input = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, n), Complex::new(1.0, 0.0)).into_dyn(),
    );
    let output = recursion.forward(&input).unwrap();

    assert_eq!(output.data.shape()[2], n);
    assert!(output.data.iter().all(|x| x.is_finite()));
}

#[test]
fn recursion_with_zero_feedback_reduces_to_feedforward() {
    // If feedback H_fb = 0, Recursion forward equals feedforward forward,
    // and the parameter gradient must equal the feedforward-only gradient.
    let n = 2;
    let n_bins = NFFT / 2 + 1;

    let mut feedforward = Gain::new(NFFT, n, n).unwrap();
    feedforward.param[[0, 0]] = 0.5;
    feedforward.param[[1, 0]] = -0.2;
    feedforward.param[[0, 1]] = 0.3;
    feedforward.param[[1, 1]] = -0.1;

    let feedback = Gain::new(NFFT, n, n).unwrap(); // all zeros
    let mut recursion = Recursion::new(Box::new(feedforward.clone()), Box::new(feedback)).unwrap();

    let input = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, n), Complex::new(0.5, 0.1)).into_dyn(),
    );
    let target = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, n), Complex::new(0.3, -0.2)).into_dyn(),
    );

    // Reference: standalone feedforward gradient.
    let mut standalone = feedforward.clone();
    standalone.zero_grad();
    let out_standalone = standalone.forward(&input).unwrap();
    let diff_standalone = &out_standalone.data - &target.data;
    let grad_standalone = DiffTensor::from_array(diff_standalone.into_owned() * 2.0);
    standalone
        .backward(&input, &out_standalone, &grad_standalone)
        .unwrap();

    // Recursion gradient.
    recursion.zero_grad();
    let out_recursion = recursion.forward(&input).unwrap();
    let diff_recursion = &out_recursion.data - &target.data;
    let grad_recursion = DiffTensor::from_array(diff_recursion.into_owned() * 2.0);
    recursion
        .backward(&input, &out_recursion, &grad_recursion)
        .unwrap();

    let standalone_grads = standalone.gradients();
    let recursion_grads = recursion.gradients();
    for i in 0..n {
        for j in 0..n {
            assert_relative_eq!(
                recursion_grads[0][[i, j]],
                standalone_grads[0][[i, j]],
                epsilon = 1e-8
            );
        }
    }
}

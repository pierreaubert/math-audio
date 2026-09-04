//! Tests for the spectral/perceptual loss terms, PEQ priors, and the
//! Series/Parallel forward-cache behavior.

use approx::assert_abs_diff_eq;
use math_audio_autodiff::gain::Gain;
use math_audio_autodiff::iir::peq::{
    peq_smoothness_penalty, peq_smoothness_penalty_backward, peq_sparsity_penalty,
    peq_sparsity_penalty_backward,
};
use math_audio_autodiff::loss::{
    bark_band_index, bark_weighted_loss, bark_weighted_loss_backward, bark_weights,
    bin_frequencies, erb_weighted_loss, erb_weighted_loss_backward, erb_weights,
    log_magnitude_loss, log_magnitude_loss_backward, magnitude_mse_loss, mse_loss,
    mse_loss_backward, multi_scale_spectral_loss, multi_scale_spectral_loss_backward,
    spectral_convergence_loss, spectral_convergence_loss_backward, weighted_mse_loss,
    weighted_mse_loss_backward,
};
use math_audio_autodiff::module::DiffModule;
use math_audio_autodiff::system::{Parallel, Series};
use math_audio_autodiff::tensor::DiffTensor;
use ndarray::{Array, ArrayD, IxDyn};
use num_complex::Complex;

const NFFT: usize = 16;
const FS: f64 = 48_000.0;

fn complex_tensor(values: &[Complex<f64>], shape: &[usize]) -> DiffTensor<f64> {
    let arr = Array::from_shape_vec(IxDyn(shape), values.to_vec()).unwrap();
    DiffTensor::from_array(arr)
}

fn sample_spectra() -> (DiffTensor<f64>, DiffTensor<f64>) {
    let pred = complex_tensor(
        &[
            Complex::new(1.0, 2.0),
            Complex::new(3.0, -1.0),
            Complex::new(0.0, 1.0),
            Complex::new(-2.0, 0.5),
        ],
        &[4],
    );
    let target = complex_tensor(
        &[
            Complex::new(0.0, 1.0),
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 0.0),
            Complex::new(-1.0, -0.5),
        ],
        &[4],
    );
    (pred, target)
}

/// Central finite difference of `loss` w.r.t. the real/imag part of `pred[idx]`.
fn fd_grad(
    loss: &dyn Fn(&DiffTensor<f64>, &DiffTensor<f64>) -> f64,
    pred: &DiffTensor<f64>,
    target: &DiffTensor<f64>,
    idx: usize,
    imag: bool,
) -> f64 {
    let eps = 1e-6;
    let shape = pred.data.shape().to_vec();
    let mut plus_vals: Vec<Complex<f64>> = pred.data.iter().copied().collect();
    let mut minus_vals = plus_vals.clone();
    if imag {
        plus_vals[idx].im += eps;
        minus_vals[idx].im -= eps;
    } else {
        plus_vals[idx].re += eps;
        minus_vals[idx].re -= eps;
    }
    let lp = loss(&complex_tensor(&plus_vals, &shape), target);
    let lm = loss(&complex_tensor(&minus_vals, &shape), target);
    (lp - lm) / (2.0 * eps)
}

fn check_backward_matches_fd(
    loss: &dyn Fn(&DiffTensor<f64>, &DiffTensor<f64>) -> f64,
    grad: &DiffTensor<f64>,
    pred: &DiffTensor<f64>,
    target: &DiffTensor<f64>,
) {
    for i in 0..pred.data.len() {
        assert_abs_diff_eq!(
            grad.data[i].re,
            fd_grad(loss, pred, target, i, false),
            epsilon = 1e-5
        );
        assert_abs_diff_eq!(
            grad.data[i].im,
            fd_grad(loss, pred, target, i, true),
            epsilon = 1e-5
        );
    }
}

#[test]
fn uniform_weighted_mse_matches_mse() {
    let (pred, target) = sample_spectra();
    let weights = vec![1.0; 4];
    assert_abs_diff_eq!(
        weighted_mse_loss(&pred, &target, &weights).unwrap(),
        mse_loss(&pred, &target).unwrap(),
        epsilon = 1e-12
    );
    let g_w = weighted_mse_loss_backward(&pred, &target, &weights).unwrap();
    let g = mse_loss_backward(&pred, &target).unwrap();
    for (a, b) in g_w.data.iter().zip(g.data.iter()) {
        assert_abs_diff_eq!(a.re, b.re, epsilon = 1e-12);
        assert_abs_diff_eq!(a.im, b.im, epsilon = 1e-12);
    }
}

#[test]
fn weighted_mse_backward_matches_finite_differences() {
    let (pred, target) = sample_spectra();
    let weights = vec![0.5, 2.0, 1.0, 3.0];
    let loss =
        |p: &DiffTensor<f64>, t: &DiffTensor<f64>| weighted_mse_loss(p, t, &weights).unwrap();
    let grad = weighted_mse_loss_backward(&pred, &target, &weights).unwrap();
    check_backward_matches_fd(&loss, &grad, &pred, &target);
}

#[test]
fn perceptual_weights_normalize_to_mean_one_and_fall_with_frequency() {
    let freqs = bin_frequencies(9, FS, NFFT);
    assert_eq!(freqs.len(), 9);
    assert_abs_diff_eq!(freqs[0], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(freqs[1], FS / NFFT as f64, epsilon = 1e-9);
    for weights in [bark_weights(&freqs[1..]), erb_weights(&freqs[1..])] {
        let mean: f64 = weights.iter().sum::<f64>() / weights.len() as f64;
        assert_abs_diff_eq!(mean, 1.0, epsilon = 1e-12);
        for w in &weights {
            assert!(*w > 0.0);
        }
        // Inverse-bandwidth weighting: low frequencies get larger weights.
        assert!(weights[0] > weights[weights.len() - 1]);
    }
    // DC / invalid bins get weight 0.
    let edge = bark_weights(&[0.0, -10.0, f64::NAN]);
    assert_eq!(edge, vec![0.0, 0.0, 0.0]);
}

#[test]
fn bark_band_index_assigns_sensible_bands() {
    assert_eq!(bark_band_index(0.0), 0);
    assert_eq!(bark_band_index(-5.0), 0);
    assert_eq!(bark_band_index(f64::NAN), 0);
    let low = bark_band_index(100.0);
    let mid = bark_band_index(1000.0);
    let high = bark_band_index(10_000.0);
    assert!(low < mid && mid < high);
    assert!(high <= 23);
    assert_eq!(bark_band_index(100_000.0), 23);
}

#[test]
fn bark_and_erb_losses_match_weighted_mse_with_their_weights() {
    let (pred, target) = sample_spectra();
    let freqs = vec![100.0, 500.0, 2000.0, 8000.0];
    assert_abs_diff_eq!(
        bark_weighted_loss(&pred, &target, &freqs).unwrap(),
        weighted_mse_loss(&pred, &target, &bark_weights(&freqs)).unwrap(),
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        erb_weighted_loss(&pred, &target, &freqs).unwrap(),
        weighted_mse_loss(&pred, &target, &erb_weights(&freqs)).unwrap(),
        epsilon = 1e-12
    );
    let loss_b =
        |p: &DiffTensor<f64>, t: &DiffTensor<f64>| bark_weighted_loss(p, t, &freqs).unwrap();
    let grad_b = bark_weighted_loss_backward(&pred, &target, &freqs).unwrap();
    check_backward_matches_fd(&loss_b, &grad_b, &pred, &target);
    let loss_e =
        |p: &DiffTensor<f64>, t: &DiffTensor<f64>| erb_weighted_loss(p, t, &freqs).unwrap();
    let grad_e = erb_weighted_loss_backward(&pred, &target, &freqs).unwrap();
    check_backward_matches_fd(&loss_e, &grad_e, &pred, &target);
}

#[test]
fn multi_scale_with_unit_window_matches_magnitude_mse() {
    let (pred, target) = sample_spectra();
    assert_abs_diff_eq!(
        multi_scale_spectral_loss(&pred, &target, &[1]).unwrap(),
        magnitude_mse_loss(&pred, &target).unwrap(),
        epsilon = 1e-12
    );
}

#[test]
fn multi_scale_coarse_scale_smooths_detail() {
    // A zero-mean error pattern cancels under wide pooling but not at scale 1.
    let pred = complex_tensor(&[Complex::new(2.0, 0.0), Complex::new(0.0, 0.0)], &[2]);
    let target = complex_tensor(&[Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)], &[2]);
    let fine = multi_scale_spectral_loss(&pred, &target, &[1]).unwrap();
    assert_abs_diff_eq!(fine, 1.0, epsilon = 1e-12);
    let coarse = multi_scale_spectral_loss(&pred, &target, &[2]).unwrap();
    assert_abs_diff_eq!(coarse, 0.0, epsilon = 1e-12);
    let loss = |p: &DiffTensor<f64>, t: &DiffTensor<f64>| {
        multi_scale_spectral_loss(p, t, &[1, 2, 3]).unwrap()
    };
    let grad = multi_scale_spectral_loss_backward(&pred, &target, &[1, 2, 3]).unwrap();
    check_backward_matches_fd(&loss, &grad, &pred, &target);
}

#[test]
fn spectral_convergence_and_log_magnitude_match_finite_differences() {
    let (pred, target) = sample_spectra();
    let eps = 1e-3;
    let loss_sc =
        |p: &DiffTensor<f64>, t: &DiffTensor<f64>| spectral_convergence_loss(p, t, eps).unwrap();
    let grad_sc = spectral_convergence_loss_backward(&pred, &target, eps).unwrap();
    check_backward_matches_fd(&loss_sc, &grad_sc, &pred, &target);
    let loss_lm = |p: &DiffTensor<f64>, t: &DiffTensor<f64>| log_magnitude_loss(p, t, eps).unwrap();
    let grad_lm = log_magnitude_loss_backward(&pred, &target, eps).unwrap();
    check_backward_matches_fd(&loss_lm, &grad_lm, &pred, &target);
    // Identical spectra give zero loss and zero gradient.
    assert_abs_diff_eq!(
        spectral_convergence_loss(&pred, &pred, eps).unwrap(),
        0.0,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        log_magnitude_loss(&pred, &pred, eps).unwrap(),
        0.0,
        epsilon = 1e-12
    );
}

#[test]
fn loss_error_cases_are_rejected() {
    let (pred, target) = sample_spectra();
    let bad_shape = complex_tensor(
        &[
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ],
        &[3],
    );
    assert!(weighted_mse_loss(&pred, &bad_shape, &[1.0; 4]).is_err());
    assert!(weighted_mse_loss(&pred, &target, &[1.0; 3]).is_err());
    assert!(weighted_mse_loss(&pred, &target, &[1.0, -1.0, 1.0, 1.0]).is_err());
    assert!(weighted_mse_loss(&pred, &target, &[0.0; 4]).is_err());
    assert!(bark_weighted_loss(&pred, &target, &[100.0; 3]).is_err());
    assert!(multi_scale_spectral_loss(&pred, &target, &[]).is_err());
    assert!(multi_scale_spectral_loss(&pred, &target, &[2, 0]).is_err());
    assert!(spectral_convergence_loss(&pred, &target, 0.0).is_err());
    assert!(log_magnitude_loss(&pred, &target, -1.0).is_err());
}

fn peq_param(gains: &[f64], n_channels: usize) -> ArrayD<f64> {
    let n_sections = gains.len();
    let mut param = ArrayD::zeros(IxDyn(&[n_sections, 3, n_channels]));
    for (k, &g) in gains.iter().enumerate() {
        for ch in 0..n_channels {
            param[[k, 2, ch]] = g;
        }
    }
    param
}

#[test]
fn peq_priors_values_and_gradients() {
    let param = peq_param(&[1.0, 3.0, 2.0], 1);
    assert_abs_diff_eq!(
        peq_smoothness_penalty(&param).unwrap(),
        (4.0 + 1.0) / 2.0,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        peq_sparsity_penalty(&param).unwrap(),
        (1.0 + 3.0 + 2.0) / 3.0,
        epsilon = 1e-12
    );
    // Single section: smoothness is zero with zero gradient.
    let single = peq_param(&[2.0], 1);
    assert_abs_diff_eq!(
        peq_smoothness_penalty(&single).unwrap(),
        0.0,
        epsilon = 1e-12
    );
    assert!(
        peq_smoothness_penalty_backward(&single)
            .unwrap()
            .iter()
            .all(|&g| g == 0.0)
    );
    // Finite-difference checks over the gain row.
    let eps = 1e-6;
    for k in 0..3 {
        let mut plus = param.clone();
        let mut minus = param.clone();
        plus[[k, 2, 0]] += eps;
        minus[[k, 2, 0]] -= eps;
        let fd_smooth = (peq_smoothness_penalty(&plus).unwrap()
            - peq_smoothness_penalty(&minus).unwrap())
            / (2.0 * eps);
        assert_abs_diff_eq!(
            peq_smoothness_penalty_backward(&param).unwrap()[[k, 2, 0]],
            fd_smooth,
            epsilon = 1e-6
        );
        let fd_sparse = (peq_sparsity_penalty(&plus).unwrap()
            - peq_sparsity_penalty(&minus).unwrap())
            / (2.0 * eps);
        assert_abs_diff_eq!(
            peq_sparsity_penalty_backward(&param).unwrap()[[k, 2, 0]],
            fd_sparse,
            epsilon = 1e-6
        );
    }
    // fc/Q rows carry no prior gradient; bad shapes are rejected.
    let grad = peq_smoothness_penalty_backward(&param).unwrap();
    assert!(grad[[0, 0, 0]] == 0.0 && grad[[0, 1, 0]] == 0.0);
    assert!(peq_smoothness_penalty(&ArrayD::zeros(IxDyn(&[2, 2, 1]))).is_err());
    assert!(peq_sparsity_penalty(&ArrayD::zeros(IxDyn(&[0, 3, 1]))).is_err());
}

fn ones_spectrum(shape: &[usize]) -> DiffTensor<f64> {
    let n: usize = shape.iter().product();
    complex_tensor(&vec![Complex::new(1.0, 0.0); n], shape)
}

#[test]
fn series_cold_backward_matches_warm_backward() {
    let n_bins = NFFT / 2 + 1;
    let mut gain1 = Gain::new(NFFT, 2, 2).expect("valid gain");
    gain1.param = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.5, -0.5, 2.0]).unwrap();
    let mut gain2 = Gain::new(NFFT, 2, 2).expect("valid gain");
    gain2.param = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.5, 1.5, 1.0, -1.0]).unwrap();
    let input = ones_spectrum(&[1, n_bins, 2]);

    // Warm path: forward then backward.
    let mut warm =
        Series::new(vec![Box::new(gain1.clone()), Box::new(gain2.clone())]).expect("series");
    let output = warm.forward(&input).expect("forward");
    let grad_output = ones_spectrum(output.data.shape());
    let warm_grad = warm
        .backward(&input, &output, &grad_output)
        .expect("warm backward");

    // Cold path: backward with no preceding forward (exercises recompute +
    // re-warm), repeated twice to exercise the re-warmed cache.
    let mut cold = Series::new(vec![Box::new(gain1), Box::new(gain2)]).expect("series");
    let cold_output = cold.modules()[0]
        .forward(&input)
        .and_then(|x1| cold.modules()[1].forward(&x1))
        .expect("manual forward");
    let cold_grad1 = cold
        .backward(&input, &cold_output, &grad_output)
        .expect("cold backward");
    let cold_grad2 = cold
        .backward(&input, &cold_output, &grad_output)
        .expect("re-warmed backward");
    for (w, c) in warm_grad
        .data
        .iter()
        .zip(cold_grad1.data.iter().zip(cold_grad2.data.iter()))
    {
        assert_abs_diff_eq!(w.re, c.0.re, epsilon = 1e-12);
        assert_abs_diff_eq!(w.im, c.0.im, epsilon = 1e-12);
        assert_abs_diff_eq!(w.re, c.1.re, epsilon = 1e-12);
        assert_abs_diff_eq!(w.im, c.1.im, epsilon = 1e-12);
    }
}

#[test]
fn parallel_cold_backward_matches_warm_backward() {
    let n_bins = NFFT / 2 + 1;
    let mut gain_a = Gain::new(NFFT, 2, 2).expect("valid gain");
    gain_a.param = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.5, -0.5, 2.0]).unwrap();
    let mut gain_b = Gain::new(NFFT, 2, 2).expect("valid gain");
    gain_b.param = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.5, 1.5, 1.0, -1.0]).unwrap();
    let input = ones_spectrum(&[1, n_bins, 2]);

    let mut warm =
        Parallel::new(Box::new(gain_a.clone()), Box::new(gain_b.clone())).expect("parallel");
    let output = warm.forward(&input).expect("forward");
    let grad_output = ones_spectrum(output.data.shape());
    let warm_grad = warm
        .backward(&input, &output, &grad_output)
        .expect("warm backward");

    let mut cold = Parallel::new(Box::new(gain_a), Box::new(gain_b)).expect("parallel");
    let (branch_a, branch_b) = cold.branches();
    let out_a = branch_a.forward(&input).expect("branch a");
    let out_b = branch_b.forward(&input).expect("branch b");
    let cold_output = DiffTensor::from_array(&out_a.data + &out_b.data);
    let cold_grad = cold
        .backward(&input, &cold_output, &grad_output)
        .expect("cold backward");
    for (w, c) in warm_grad.data.iter().zip(cold_grad.data.iter()) {
        assert_abs_diff_eq!(w.re, c.re, epsilon = 1e-12);
        assert_abs_diff_eq!(w.im, c.im, epsilon = 1e-12);
    }
}

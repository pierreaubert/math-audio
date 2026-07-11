use approx::{assert_abs_diff_eq, assert_relative_eq};
use math_audio_autodiff::{
    delay::ParallelDelay,
    gain::Gain,
    module::DiffModule,
    recursion::Recursion,
    tensor::DiffTensor,
};
use ndarray::{Array3, ArrayD, IxDyn};
use num_complex::Complex;

const NFFT: usize = 512;

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

    // Forward outputs must match when feedback is zero.
    for (actual, expected) in out_recursion.data.iter().zip(out_standalone.data.iter()) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-12);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-12);
    }

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

#[test]
fn recursion_gradient_matches_finite_difference() {
    // Build a small stable feedback loop and check that the analytical
    // parameter gradients from Recursion::backward match central finite
    // differences of the MSE loss w.r.t. the raw Gain parameters.
    let n = 2;
    let n_bins = NFFT / 2 + 1;

    let mut feedforward = Gain::new(NFFT, n, n).unwrap();
    feedforward.param[[0, 0]] = 0.4;
    feedforward.param[[0, 1]] = -0.15;
    feedforward.param[[1, 0]] = 0.25;
    feedforward.param[[1, 1]] = -0.2;

    let mut feedback = Gain::new(NFFT, n, n).unwrap();
    // Small diagonal feedback keeps (I - H_fb) well-conditioned and stable.
    feedback.param[[0, 0]] = 0.1;
    feedback.param[[1, 1]] = -0.05;

    let input = complex_spectrum(&[1, n_bins, n]);
    let target = complex_spectrum(&[1, n_bins, n]);

    // Analytical gradients.
    let mut recursion =
        Recursion::new(Box::new(feedforward.clone()), Box::new(feedback.clone())).unwrap();
    recursion.zero_grad();
    let output = recursion.forward(&input).unwrap();
    let grad_output_data: Vec<Complex<f64>> = output
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(o, t)| 2.0 * (o - t))
        .collect();
    let grad_output = DiffTensor::from_array(
        ArrayD::from_shape_vec(IxDyn(output.data.shape()), grad_output_data).unwrap(),
    );
    recursion.backward(&input, &output, &grad_output).unwrap();

    let analytical_feedforward = recursion.feedforward.gradients()[0].clone();
    let analytical_feedback = recursion.feedback.gradients()[0].clone();

    // Central finite difference on feedforward parameters.
    let epsilon = 1e-5;
    for i in 0..n {
        for j in 0..n {
            let mut param_plus = feedforward.param.clone();
            param_plus[[i, j]] += epsilon;
            let recursion_plus = Recursion::new(
                Box::new(Gain {
                    nfft: feedforward.nfft,
                    param: param_plus,
                    param_grad: ArrayD::zeros(IxDyn(&[n, n])),
                }),
                Box::new(feedback.clone()),
            )
            .unwrap();
            let out_plus = recursion_plus.forward(&input).unwrap();
            let loss_plus = mse_loss(&out_plus, &target);

            let mut param_minus = feedforward.param.clone();
            param_minus[[i, j]] -= epsilon;
            let recursion_minus = Recursion::new(
                Box::new(Gain {
                    nfft: feedforward.nfft,
                    param: param_minus,
                    param_grad: ArrayD::zeros(IxDyn(&[n, n])),
                }),
                Box::new(feedback.clone()),
            )
            .unwrap();
            let out_minus = recursion_minus.forward(&input).unwrap();
            let loss_minus = mse_loss(&out_minus, &target);

            let finite_diff = (loss_plus - loss_minus) / (2.0 * epsilon);
            let analytical_val = analytical_feedforward[[i, j]];

            let denom = finite_diff.abs().max(1e-8);
            let relative_error = (analytical_val - finite_diff).abs() / denom;
            assert!(
                relative_error < 1e-4 || (analytical_val - finite_diff).abs() < 1e-6,
                "feedforward[{}, {}]: analytical={} finite_diff={} rel_err={}",
                i,
                j,
                analytical_val,
                finite_diff,
                relative_error
            );
        }
    }

    // Central finite difference on feedback parameters.
    for i in 0..n {
        for j in 0..n {
            let mut param_plus = feedback.param.clone();
            param_plus[[i, j]] += epsilon;
            let recursion_plus = Recursion::new(
                Box::new(feedforward.clone()),
                Box::new(Gain {
                    nfft: feedback.nfft,
                    param: param_plus,
                    param_grad: ArrayD::zeros(IxDyn(&[n, n])),
                }),
            )
            .unwrap();
            let out_plus = recursion_plus.forward(&input).unwrap();
            let loss_plus = mse_loss(&out_plus, &target);

            let mut param_minus = feedback.param.clone();
            param_minus[[i, j]] -= epsilon;
            let recursion_minus = Recursion::new(
                Box::new(feedforward.clone()),
                Box::new(Gain {
                    nfft: feedback.nfft,
                    param: param_minus,
                    param_grad: ArrayD::zeros(IxDyn(&[n, n])),
                }),
            )
            .unwrap();
            let out_minus = recursion_minus.forward(&input).unwrap();
            let loss_minus = mse_loss(&out_minus, &target);

            let finite_diff = (loss_plus - loss_minus) / (2.0 * epsilon);
            let analytical_val = analytical_feedback[[i, j]];

            let denom = finite_diff.abs().max(1e-8);
            let relative_error = (analytical_val - finite_diff).abs() / denom;
            assert!(
                relative_error < 1e-4 || (analytical_val - finite_diff).abs() < 1e-6,
                "feedback[{}, {}]: analytical={} finite_diff={} rel_err={}",
                i,
                j,
                analytical_val,
                finite_diff,
                relative_error
            );
        }
    }
}

#[test]
fn recursion_gradient_with_complex_transfer_matches_finite_difference() {
    // Regression guard for the conjugate-transpose bug in Recursion::backward.
    // Gain has a real-valued frequency response, so use ParallelDelay for the
    // feedforward path so that H_ff is genuinely complex.
    let n = 2;
    let n_bins = NFFT / 2 + 1;

    let mut feedforward = ParallelDelay::new(NFFT, n, 0.0).unwrap();
    feedforward.param[[0]] = 3.5;
    feedforward.param[[1]] = 7.25;

    let mut feedback = Gain::new(NFFT, n, n).unwrap();
    // Small diagonal feedback keeps (I - H_fb) well-conditioned and stable.
    feedback.param[[0, 0]] = 0.1;
    feedback.param[[1, 1]] = -0.05;

    let input = complex_spectrum(&[1, n_bins, n]);
    let target = complex_spectrum(&[1, n_bins, n]);

    let mut recursion =
        Recursion::new(Box::new(feedforward.clone()), Box::new(feedback.clone())).unwrap();
    recursion.zero_grad();
    let output = recursion.forward(&input).unwrap();
    let grad_output_data: Vec<Complex<f64>> = output
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(o, t)| 2.0 * (o - t))
        .collect();
    let grad_output = DiffTensor::from_array(
        ArrayD::from_shape_vec(IxDyn(output.data.shape()), grad_output_data).unwrap(),
    );
    recursion.backward(&input, &output, &grad_output).unwrap();

    let analytical_feedforward = recursion.feedforward.gradients()[0].clone();

    // Central finite difference on the raw ParallelDelay parameters.
    let epsilon = 1e-5;
    for ch in 0..n {
        let mut param_plus = feedforward.param.clone();
        param_plus[[ch]] += epsilon;
        let recursion_plus = Recursion::new(
            Box::new(ParallelDelay {
                nfft: feedforward.nfft,
                n_channels: feedforward.n_channels,
                tau_min: feedforward.tau_min,
                param: param_plus,
                param_grad: ArrayD::zeros(IxDyn(&[n])),
            }),
            Box::new(feedback.clone()),
        )
        .unwrap();
        let out_plus = recursion_plus.forward(&input).unwrap();
        let loss_plus = mse_loss(&out_plus, &target);

        let mut param_minus = feedforward.param.clone();
        param_minus[[ch]] -= epsilon;
        let recursion_minus = Recursion::new(
            Box::new(ParallelDelay {
                nfft: feedforward.nfft,
                n_channels: feedforward.n_channels,
                tau_min: feedforward.tau_min,
                param: param_minus,
                param_grad: ArrayD::zeros(IxDyn(&[n])),
            }),
            Box::new(feedback.clone()),
        )
        .unwrap();
        let out_minus = recursion_minus.forward(&input).unwrap();
        let loss_minus = mse_loss(&out_minus, &target);

        let finite_diff = (loss_plus - loss_minus) / (2.0 * epsilon);
        let analytical_val = analytical_feedforward[[ch]];

        let denom = finite_diff.abs().max(1e-8);
        let relative_error = (analytical_val - finite_diff).abs() / denom;
        assert!(
            relative_error < 1e-4 || (analytical_val - finite_diff).abs() < 1e-6,
            "feedforward delay[{}]: analytical={} finite_diff={} rel_err={}",
            ch,
            analytical_val,
            finite_diff,
            relative_error
        );
    }
}

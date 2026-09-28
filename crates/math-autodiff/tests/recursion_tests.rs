use approx::{assert_abs_diff_eq, assert_relative_eq};
use math_audio_autodiff::{
    delay::{Delay, ParallelDelay},
    gain::Gain,
    module::DiffModule,
    recursion::Recursion,
    tensor::DiffTensor,
};
use ndarray::{Array3, ArrayD, IxDyn, s};
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
fn recursion_response_cache_invalidates_after_parameter_change() {
    let mut feedforward = Gain::new(NFFT, 1, 1).unwrap();
    feedforward.param[[0, 0]] = 1.0;
    let mut feedback = Gain::new(NFFT, 1, 1).unwrap();
    feedback.param[[0, 0]] = 0.1;
    let mut recursion = Recursion::new(Box::new(feedforward), Box::new(feedback)).unwrap();
    let input = DiffTensor::from_array(Array3::from_elem(
        (1, NFFT / 2 + 1, 1),
        Complex::new(1.0, 0.0),
    ));

    let before = recursion.forward(&input).unwrap();
    recursion.feedback.parameters_mut()[0][[0, 0]] = 0.2;
    let after = recursion.forward(&input).unwrap();

    assert_abs_diff_eq!(before.data[[0, 0, 0]].re, 1.0 / 0.9, epsilon = 1e-12);
    assert_abs_diff_eq!(after.data[[0, 0, 0]].re, 1.0 / 0.8, epsilon = 1e-12);
}

#[test]
fn recursion_backward_rejects_mismatched_frequency_shape() {
    let feedforward = Gain::new(NFFT, 1, 1).unwrap();
    let feedback = Gain::new(NFFT, 1, 1).unwrap();
    let mut recursion = Recursion::new(Box::new(feedforward), Box::new(feedback)).unwrap();
    let n_bins = NFFT / 2 + 1;
    let input = complex_spectrum(&[1, n_bins, 1]);
    let output = recursion.forward(&input).unwrap();
    let grad = complex_spectrum(&[1, n_bins - 1, 1]);
    assert!(recursion.backward(&input, &output, &grad).is_err());
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
        let mut delay_plus = ParallelDelay::new(
            feedforward.nfft,
            feedforward.n_channels,
            feedforward.tau_min,
        )
        .unwrap();
        delay_plus.param = param_plus;
        let recursion_plus =
            Recursion::new(Box::new(delay_plus), Box::new(feedback.clone())).unwrap();
        let out_plus = recursion_plus.forward(&input).unwrap();
        let loss_plus = mse_loss(&out_plus, &target);

        let mut param_minus = feedforward.param.clone();
        param_minus[[ch]] -= epsilon;
        let mut delay_minus = ParallelDelay::new(
            feedforward.nfft,
            feedforward.n_channels,
            feedforward.tau_min,
        )
        .unwrap();
        delay_minus.param = param_minus;
        let recursion_minus =
            Recursion::new(Box::new(delay_minus), Box::new(feedback.clone())).unwrap();
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

fn small_feedback_gains(n: usize) -> (Gain<f64>, Gain<f64>) {
    let mut feedforward = Gain::new(NFFT, n, n).unwrap();
    feedforward.param[[0, 0]] = 0.4;
    if n > 1 {
        feedforward.param[[0, 1]] = -0.15;
        feedforward.param[[1, 0]] = 0.25;
        feedforward.param[[1, 1]] = -0.2;
    }
    let mut feedback = Gain::new(NFFT, n, n).unwrap();
    feedback.param[[0, 0]] = 0.1;
    if n > 1 {
        feedback.param[[1, 1]] = -0.05;
    }
    (feedforward, feedback)
}

#[test]
fn recursion_backward_wide_channels_is_finite_and_deterministic() {
    // More than MAX_STACK_CHANNELS (16) routes through the buffered 2-D
    // fallback instead of the fused stack-scratch kernel.
    let n = 20;
    let n_bins = NFFT / 2 + 1;
    let (feedforward, feedback) = small_feedback_gains(n);
    let mut recursion = Recursion::new(Box::new(feedforward), Box::new(feedback)).unwrap();

    let input = complex_spectrum(&[1, n_bins, n]);
    let target = complex_spectrum(&[1, n_bins, n]);
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

    recursion.zero_grad();
    let grad_input = recursion.backward(&input, &output, &grad_output).unwrap();
    assert!(grad_input.data.iter().all(|x| x.is_finite()));
    let first_run: Vec<f64> = recursion
        .gradients()
        .iter()
        .flat_map(|g| g.iter().copied())
        .collect();
    assert!(first_run.iter().all(|x| x.is_finite()));

    // A second run from zeroed grads must reproduce the same gradients.
    recursion.zero_grad();
    recursion.backward(&input, &output, &grad_output).unwrap();
    let second_run: Vec<f64> = recursion
        .gradients()
        .iter()
        .flat_map(|g| g.iter().copied())
        .collect();
    assert_eq!(first_run.len(), second_run.len());
    for (a, b) in first_run.iter().zip(second_run.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-12);
    }
}

#[test]
fn recursion_backward_wide_channels_spot_check_finite_difference() {
    // One central finite difference on the buffered fallback path (>16ch).
    let n = 20;
    let n_bins = NFFT / 2 + 1;
    let (feedforward, feedback) = small_feedback_gains(n);
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
    let analytical = recursion.feedforward.gradients()[0][[0, 0]];

    let epsilon = 1e-5;
    let mut param_plus = feedforward.param.clone();
    param_plus[[0, 0]] += epsilon;
    let recursion_plus = Recursion::new(
        Box::new(Gain {
            nfft: feedforward.nfft,
            param: param_plus,
            param_grad: ArrayD::zeros(IxDyn(&[n, n])),
        }),
        Box::new(feedback.clone()),
    )
    .unwrap();
    let loss_plus = mse_loss(&recursion_plus.forward(&input).unwrap(), &target);

    let mut param_minus = feedforward.param.clone();
    param_minus[[0, 0]] -= epsilon;
    let recursion_minus = Recursion::new(
        Box::new(Gain {
            nfft: feedforward.nfft,
            param: param_minus,
            param_grad: ArrayD::zeros(IxDyn(&[n, n])),
        }),
        Box::new(feedback.clone()),
    )
    .unwrap();
    let loss_minus = mse_loss(&recursion_minus.forward(&input).unwrap(), &target);

    let finite_diff = (loss_plus - loss_minus) / (2.0 * epsilon);
    let denom = finite_diff.abs().max(1e-8);
    assert!(
        (analytical - finite_diff).abs() / denom < 1e-4 || (analytical - finite_diff).abs() < 1e-6,
        "wide fallback[0, 0]: analytical={analytical} finite_diff={finite_diff}"
    );
}

#[test]
fn recursion_backward_strided_input_matches_contiguous() {
    // Non-contiguous storage routes through the buffered fallback; the math
    // must agree bit-compatibly with the fused contiguous path.
    let n = 2;
    let n_bins = NFFT / 2 + 1;
    let (feedforward, feedback) = small_feedback_gains(n);

    let wide = ArrayD::from_shape_vec(
        IxDyn(&[1, 2 * n_bins, n]),
        (0..2 * n_bins * n)
            .map(|i| {
                let phase = i as f64 * 0.17;
                Complex::new(0.5 + phase.sin(), phase.cos() - 0.25)
            })
            .collect(),
    )
    .unwrap();
    let strided = wide.slice_move(s![.., ..;2, ..]);
    assert!(strided.as_slice().is_none());
    let strided_input = DiffTensor::from_array(strided);
    let contiguous_input = DiffTensor::from_array(
        ArrayD::from_shape_vec(
            IxDyn(&[1, n_bins, n]),
            strided_input.data.iter().copied().collect(),
        )
        .unwrap(),
    );
    assert!(contiguous_input.data.as_slice().is_some());
    let grad_output = complex_spectrum(&[1, n_bins, n]);

    let mut strided_recursion =
        Recursion::new(Box::new(feedforward.clone()), Box::new(feedback.clone())).unwrap();
    strided_recursion.zero_grad();
    let strided_out = strided_recursion.forward(&strided_input).unwrap();
    let strided_grad = strided_recursion
        .backward(&strided_input, &strided_out, &grad_output)
        .unwrap();

    let mut contiguous_recursion =
        Recursion::new(Box::new(feedforward), Box::new(feedback)).unwrap();
    contiguous_recursion.zero_grad();
    let contiguous_out = contiguous_recursion.forward(&contiguous_input).unwrap();
    let contiguous_grad = contiguous_recursion
        .backward(&contiguous_input, &contiguous_out, &grad_output)
        .unwrap();

    for (a, b) in strided_grad.data.iter().zip(contiguous_grad.data.iter()) {
        assert_abs_diff_eq!(a.re, b.re, epsilon = 1e-12);
        assert_abs_diff_eq!(a.im, b.im, epsilon = 1e-12);
    }
    for (a, b) in strided_recursion
        .gradients()
        .iter()
        .zip(contiguous_recursion.gradients().iter())
    {
        for (x, y) in a.iter().zip(b.iter()) {
            assert_abs_diff_eq!(x, y, epsilon = 1e-12);
        }
    }
}

#[test]
fn gain_backward_params_only_matches_backward() {
    let n = 2;
    let n_bins = NFFT / 2 + 1;
    let mut feedforward = Gain::new(NFFT, n, n).unwrap();
    feedforward.param[[0, 0]] = 0.5;
    feedforward.param[[1, 1]] = -0.3;

    let input = complex_spectrum(&[1, n_bins, n]);
    let output = feedforward.forward(&input).unwrap();
    let grad_output = complex_spectrum(&[1, n_bins, n]);

    let mut full = feedforward.clone();
    full.zero_grad();
    full.backward(&input, &output, &grad_output).unwrap();

    let mut params_only = feedforward.clone();
    params_only.zero_grad();
    params_only
        .backward_params_only(&input, &output, &grad_output)
        .unwrap();

    for (a, b) in full.gradients()[0]
        .iter()
        .zip(params_only.gradients()[0].iter())
    {
        assert_abs_diff_eq!(a, b, epsilon = 1e-12);
    }
}

#[test]
fn gain_backward_params_only_strided_matches_contiguous() {
    // Strided storage exercises the index-view accumulation branch.
    let n = 2;
    let n_bins = NFFT / 2 + 1;
    let gain = Gain::new(NFFT, n, n).unwrap();

    let contiguous_input = complex_spectrum(&[1, n_bins, n]);
    let contiguous_grad = complex_spectrum(&[1, n_bins, n]);
    let output = gain.forward(&contiguous_input).unwrap();

    let mut wide_in = ArrayD::zeros(IxDyn(&[1, 2 * n_bins, n]));
    wide_in
        .slice_mut(s![.., ..;2, ..])
        .assign(&contiguous_input.data);
    let strided_in = DiffTensor::from_array(wide_in.slice_move(s![.., ..;2, ..]));
    assert!(strided_in.data.as_slice().is_none());
    let mut wide_grad = ArrayD::zeros(IxDyn(&[1, 2 * n_bins, n]));
    wide_grad
        .slice_mut(s![.., ..;2, ..])
        .assign(&contiguous_grad.data);
    let strided_grad = DiffTensor::from_array(wide_grad.slice_move(s![.., ..;2, ..]));
    assert!(strided_grad.data.as_slice().is_none());

    let mut contiguous_gain = gain.clone();
    contiguous_gain.zero_grad();
    contiguous_gain
        .backward_params_only(&contiguous_input, &output, &contiguous_grad)
        .unwrap();

    let mut strided_gain = gain.clone();
    strided_gain.zero_grad();
    strided_gain
        .backward_params_only(&strided_in, &output, &strided_grad)
        .unwrap();

    for (a, b) in contiguous_gain.gradients()[0]
        .iter()
        .zip(strided_gain.gradients()[0].iter())
    {
        assert_abs_diff_eq!(a, b, epsilon = 1e-12);
    }
}

#[test]
fn gain_backward_params_only_rejects_mismatched_shapes() {
    let n = 2;
    let n_bins = NFFT / 2 + 1;
    let mut gain = Gain::new(NFFT, n, n).unwrap();
    let input = complex_spectrum(&[1, n_bins, n]);
    let output = gain.forward(&input).unwrap();

    let bad_bins = complex_spectrum(&[1, n_bins - 1, n]);
    assert!(
        gain.backward_params_only(&input, &output, &bad_bins)
            .is_err()
    );
    let bad_channels = complex_spectrum(&[1, n_bins, n + 1]);
    assert!(
        gain.backward_params_only(&input, &output, &bad_channels)
            .is_err()
    );
}

#[test]
fn backward_params_only_default_matches_backward() {
    // Delay has no override, so this exercises the trait default, which must
    // accumulate identical parameter gradients to `backward`.
    let n = 2;
    let n_bins = NFFT / 2 + 1;
    let delay = Delay::new(NFFT, n, n, 0.0).unwrap();
    let input = complex_spectrum(&[1, n_bins, n]);
    let output = delay.forward(&input).unwrap();
    let grad_output = complex_spectrum(&[1, n_bins, n]);

    let mut full = delay.clone();
    full.zero_grad();
    full.backward(&input, &output, &grad_output).unwrap();

    let mut params_only = delay.clone();
    params_only.zero_grad();
    params_only
        .backward_params_only(&input, &output, &grad_output)
        .unwrap();

    assert_eq!(full.gradients().len(), params_only.gradients().len());
    for (a, b) in full.gradients().iter().zip(params_only.gradients().iter()) {
        assert_eq!(a.shape(), b.shape());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_abs_diff_eq!(x, y, epsilon = 1e-12);
        }
    }
}

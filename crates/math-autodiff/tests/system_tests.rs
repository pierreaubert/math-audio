use approx::assert_abs_diff_eq;
use math_audio_autodiff::fft::Fft;
use math_audio_autodiff::gain::{Gain, Magnitude, ParallelGain};
use math_audio_autodiff::iir::biquad::Biquad;
use math_audio_autodiff::module::DiffModule;
use math_audio_autodiff::system::{Parallel, Series, Shell};
use math_audio_autodiff::tensor::DiffTensor;
use math_audio_iir_fir::BiquadFilterType;
use ndarray::{Array1, ArrayD, Axis, IxDyn};
use num_complex::Complex;

const FS: f64 = 48_000.0;
const NFFT: usize = 512;
const ALIAS_DECAY_DB: f64 = 0.0;

fn fc_raw_from_hz(fc: f64, fs: f64) -> f64 {
    let fc_norm = fc / (fs / 2.0);
    (fc_norm / (1.0 - fc_norm)).ln()
}

fn gain_raw_from_db(db_gain: f64) -> f64 {
    10.0_f64.powf(db_gain / 20.0)
}

fn set_biquad_param(biquad: &mut Biquad, fc_hz: f64, db_gain: f64) {
    let fc_raw = fc_raw_from_hz(fc_hz, biquad.fs);
    let gain_raw = gain_raw_from_db(db_gain);
    biquad.param.fill(0.0);
    let shape = biquad.param.shape();
    let n_out = shape[2];
    let n_in = shape[3];
    for s in 0..biquad.n_sections {
        for out in 0..n_out {
            for inp in 0..n_in {
                biquad.param[[s, 0, out, inp]] = fc_raw;
                biquad.param[[s, 1, out, inp]] = gain_raw;
            }
        }
    }
}

fn random_spectrum(shape: &[usize]) -> DiffTensor<f64> {
    let n = shape.iter().product();
    let data: Vec<Complex<f64>> = (0..n)
        .map(|_| Complex::new(rand::random::<f64>() - 0.5, rand::random::<f64>() - 0.5))
        .collect();
    DiffTensor::from_array(ArrayD::from_shape_vec(IxDyn(shape), data).unwrap())
}

fn ones_spectrum(shape: &[usize]) -> DiffTensor<f64> {
    let n = shape.iter().product();
    let data: Vec<Complex<f64>> = std::iter::repeat_n(Complex::new(1.0, 0.0), n).collect();
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
fn gain_forward_and_backward() {
    let n_bins = NFFT / 2 + 1;
    let n_out = 2;
    let n_in = 3;
    let mut gain = Gain::new(NFFT, n_out, n_in).expect("valid gain");
    gain.param =
        ArrayD::from_shape_vec(IxDyn(&[n_out, n_in]), vec![0.5, -0.3, 1.2, -0.7, 0.9, 0.1])
            .unwrap();

    let input = random_spectrum(&[1, n_bins, n_in]);
    let output = gain.forward(&input).expect("forward should succeed");
    assert_eq!(output.data.shape(), &[1, n_bins, n_out]);

    // Verify against explicit matrix multiplication.
    for out_ch in 0..n_out {
        let output_slice = output.data.index_axis(Axis(2), out_ch);
        let mut expected = ArrayD::zeros(IxDyn(output_slice.shape()));
        for in_ch in 0..n_in {
            let h = Complex::new(gain.param[[out_ch, in_ch]], 0.0);
            let input_slice = input.data.index_axis(Axis(2), in_ch);
            expected += &input_slice.mapv(|x| x * h);
        }
        for (&actual, &exp) in output_slice.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual.re, exp.re, epsilon = 1e-12);
            assert_abs_diff_eq!(actual.im, exp.im, epsilon = 1e-12);
        }
    }

    // Target = zero for a simple MSE-like loss.
    let target = DiffTensor::zeros(IxDyn(&[1, n_bins, n_out]));
    let grad_output_data: Vec<Complex<f64>> = output
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(o, t)| 2.0 * (o - t))
        .collect();
    let grad_output = DiffTensor::from_array(
        ArrayD::from_shape_vec(IxDyn(&[1, n_bins, n_out]), grad_output_data).unwrap(),
    );

    gain.zero_grad();
    let _grad_input = gain
        .backward(&input, &output, &grad_output)
        .expect("backward should succeed");
    let analytical = gain.param_grad.clone();

    // Finite difference check on parameters.
    let epsilon = 1e-5;
    for out in 0..n_out {
        for inp in 0..n_in {
            let mut param_plus = gain.param.clone();
            param_plus[[out, inp]] += epsilon;
            let mut gain_plus = gain.clone();
            gain_plus.param = param_plus;
            gain_plus.param_grad.fill(0.0);
            let out_plus = gain_plus.forward(&input).expect("forward should succeed");
            let loss_plus = mse_loss(&out_plus, &target);

            let mut param_minus = gain.param.clone();
            param_minus[[out, inp]] -= epsilon;
            let mut gain_minus = gain.clone();
            gain_minus.param = param_minus;
            gain_minus.param_grad.fill(0.0);
            let out_minus = gain_minus.forward(&input).expect("forward should succeed");
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

#[test]
fn parallel_gain_forward_and_backward() {
    let n_bins = NFFT / 2 + 1;
    let n_channels = 3;
    let mut gain = ParallelGain::new(NFFT, n_channels).expect("valid parallel gain");
    gain.param = ArrayD::from_shape_vec(IxDyn(&[n_channels]), vec![0.5, -0.7, 1.2]).unwrap();

    let input = random_spectrum(&[1, n_bins, n_channels]);
    let output = gain.forward(&input).expect("forward should succeed");
    assert_eq!(output.data.shape(), &[1, n_bins, n_channels]);

    for ch in 0..n_channels {
        let h = Complex::new(gain.param[ch], 0.0);
        let input_slice = input.data.index_axis(Axis(2), ch);
        let output_slice = output.data.index_axis(Axis(2), ch);
        for (&inp, &out) in input_slice.iter().zip(output_slice.iter()) {
            assert_abs_diff_eq!(out.re, (h * inp).re, epsilon = 1e-12);
            assert_abs_diff_eq!(out.im, (h * inp).im, epsilon = 1e-12);
        }
    }

    let target = DiffTensor::zeros(IxDyn(&[1, n_bins, n_channels]));
    let grad_output_data: Vec<Complex<f64>> = output
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(o, t)| 2.0 * (o - t))
        .collect();
    let grad_output = DiffTensor::from_array(
        ArrayD::from_shape_vec(IxDyn(&[1, n_bins, n_channels]), grad_output_data).unwrap(),
    );

    gain.zero_grad();
    let _grad_input = gain
        .backward(&input, &output, &grad_output)
        .expect("backward should succeed");
    let analytical = gain.param_grad.clone();

    let epsilon = 1e-5;
    for ch in 0..n_channels {
        let mut param_plus = gain.param.clone();
        param_plus[ch] += epsilon;
        let mut gain_plus = gain.clone();
        gain_plus.param = param_plus;
        gain_plus.param_grad.fill(0.0);
        let out_plus = gain_plus.forward(&input).expect("forward should succeed");
        let loss_plus = mse_loss(&out_plus, &target);

        let mut param_minus = gain.param.clone();
        param_minus[ch] -= epsilon;
        let mut gain_minus = gain.clone();
        gain_minus.param = param_minus;
        gain_minus.param_grad.fill(0.0);
        let out_minus = gain_minus.forward(&input).expect("forward should succeed");
        let loss_minus = mse_loss(&out_minus, &target);

        let finite_diff = (loss_plus - loss_minus) / (2.0 * epsilon);
        let analytical_val = analytical[ch];

        let denom = finite_diff.abs().max(1e-8);
        let relative_error = (analytical_val - finite_diff).abs() / denom;
        assert!(
            relative_error < 1e-4 || (analytical_val - finite_diff).abs() < 1e-6,
            "ch={}: analytical={} finite_diff={} rel_err={}",
            ch,
            analytical_val,
            finite_diff,
            relative_error
        );
    }
}

#[test]
fn magnitude_backward_ignores_imaginary_upstream_gradient() {
    let n_bins = NFFT / 2 + 1;
    let input = random_spectrum(&[1, n_bins, 1]);
    let mut magnitude = Magnitude::new(NFFT, 1);
    let output = magnitude.forward(&input).expect("forward should succeed");
    let grad_output = DiffTensor::from_array(ArrayD::from_elem(
        IxDyn(&[1, n_bins, 1]),
        Complex::new(2.0, 7.0),
    ));

    let grad_input = magnitude
        .backward(&input, &output, &grad_output)
        .expect("backward should succeed");

    for (actual, sample) in grad_input.data.iter().zip(&input.data) {
        let expected = if sample.norm() > 1e-12 {
            *sample * (2.0 / sample.norm())
        } else {
            Complex::new(0.0, 0.0)
        };
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-12);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-12);
    }
}

#[test]
fn series_two_gains_multiplies() {
    let n_bins = NFFT / 2 + 1;
    let mut gain1 = Gain::new(NFFT, 3, 2).expect("valid gain");
    gain1.param =
        ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let mut gain2 = Gain::new(NFFT, 2, 3).expect("valid gain");
    gain2.param =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0]).unwrap();

    // Expected combined gain matrix: gain2.param * gain1.param.
    let gain1_view = gain1
        .param
        .view()
        .into_shape_with_order((3, 2))
        .expect("gain1 shape");
    let gain2_view = gain2
        .param
        .view()
        .into_shape_with_order((2, 3))
        .expect("gain2 shape");
    let expected = gain2_view.dot(&gain1_view);

    let series = Series::new(vec![Box::new(gain1), Box::new(gain2)]).expect("valid series");
    assert_eq!(series.input_channels(), 2);
    assert_eq!(series.output_channels(), 2);

    let input = ones_spectrum(&[1, n_bins, 2]);
    let output = series.forward(&input).expect("forward should succeed");
    assert_eq!(output.data.shape(), &[1, n_bins, 2]);
    // The combined gain applied to a vector of ones is the row sum of the
    // product matrix.
    let input_vec = Array1::from_vec(vec![1.0, 1.0]);
    let expected_z = expected.dot(&input_vec);
    for out_ch in 0..2 {
        let expected_val = expected_z[out_ch];
        for f in 0..n_bins {
            assert_abs_diff_eq!(
                output.data[[0, f, out_ch]].re,
                expected_val,
                epsilon = 1e-12
            );
        }
    }
}

#[test]
fn series_gradient_matches_manual() {
    let n_bins = NFFT / 2 + 1;
    let mut gain = Gain::new(NFFT, 3, 2).expect("valid gain");
    gain.param =
        ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![0.5, -0.3, 1.2, -0.7, 0.9, 0.1]).unwrap();

    let mut biquad = Biquad::new(NFFT, FS, 1, BiquadFilterType::Lowpass, 3, 3, ALIAS_DECAY_DB)
        .expect("valid biquad");
    set_biquad_param(&mut biquad, 1_000.0, 3.0);

    // Manual chain rule.
    let input = random_spectrum(&[1, n_bins, 2]);
    let x1 = gain.forward(&input).expect("manual gain forward");
    let x2 = biquad.forward(&x1).expect("manual biquad forward");

    let target = DiffTensor::zeros(IxDyn(x2.data.shape()));
    let grad_output_data: Vec<Complex<f64>> = x2
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(o, t)| 2.0 * (o - t))
        .collect();
    let grad_output = DiffTensor::from_array(
        ArrayD::from_shape_vec(IxDyn(x2.data.shape()), grad_output_data).unwrap(),
    );

    let mut gain_manual = gain.clone();
    let mut biquad_manual = biquad.clone();
    gain_manual.zero_grad();
    biquad_manual.zero_grad();
    let grad_x1 = biquad_manual
        .backward(&x1, &x2, &grad_output)
        .expect("manual biquad backward");
    let grad_input_manual = gain_manual
        .backward(&input, &x1, &grad_x1)
        .expect("manual gain backward");

    // Series chain rule.
    let mut series = Series::new(vec![Box::new(gain), Box::new(biquad)]).expect("valid series");
    series.zero_grad();
    let x2_series = series.forward(&input).expect("series forward");
    let grad_input_series = series
        .backward(&input, &x2_series, &grad_output)
        .expect("series backward");

    assert_eq!(
        grad_input_manual.data.shape(),
        grad_input_series.data.shape()
    );
    for (manual, series) in grad_input_manual
        .data
        .iter()
        .zip(grad_input_series.data.iter())
    {
        assert_abs_diff_eq!(manual.re, series.re, epsilon = 1e-9);
        assert_abs_diff_eq!(manual.im, series.im, epsilon = 1e-9);
    }

    // Verify accumulated gradients match.
    let series_gain_grad = series.modules()[0].gradients()[0];
    assert_eq!(gain_manual.param_grad.shape(), series_gain_grad.shape());
    for (manual, series) in gain_manual.param_grad.iter().zip(series_gain_grad.iter()) {
        assert_abs_diff_eq!(manual, series, epsilon = 1e-9);
    }

    let series_biquad_grad = series.modules()[1].gradients()[0];
    assert_eq!(biquad_manual.param_grad.shape(), series_biquad_grad.shape());
    for (manual, series) in biquad_manual
        .param_grad
        .iter()
        .zip(series_biquad_grad.iter())
    {
        assert_abs_diff_eq!(manual, series, epsilon = 1e-9);
    }
}

#[test]
fn shell_parameters_mut_changes_core() {
    let n_bins = NFFT / 2 + 1;
    let input_layer = Magnitude::new(NFFT, 2);
    let mut core = Gain::new(NFFT, 2, 2).expect("valid gain");
    core.param = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let output_layer = Magnitude::new(NFFT, 2);

    let mut shell = Shell::new(
        Box::new(input_layer),
        Box::new(core),
        Box::new(output_layer),
    )
    .expect("valid shell");

    let input = ones_spectrum(&[1, n_bins, 2]);
    let output_before = shell.forward(&input).expect("forward should succeed");

    {
        let mut params_mut = shell.parameters_mut();
        assert_eq!(
            params_mut.len(),
            1,
            "shell should expose exactly the core's parameters"
        );
        let core_param = &mut params_mut[0];
        assert_eq!(core_param.shape(), &[2, 2]);
        // Scale the first output channel by 2.0.
        core_param[[0, 0]] = 2.0;
        core_param[[0, 1]] = 0.0;
        core_param[[1, 0]] = 0.0;
        core_param[[1, 1]] = 1.0;
    }

    let output_after = shell.forward(&input).expect("forward should succeed");

    // Magnitude leaves the all-ones input unchanged; the modified gain matrix
    // should now scale channel 0 by 2.0 and leave channel 1 unchanged.
    for f in 0..n_bins {
        assert_abs_diff_eq!(output_after.data[[0, f, 0]].re, 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(output_after.data[[0, f, 0]].im, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(output_after.data[[0, f, 1]].re, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(output_after.data[[0, f, 1]].im, 0.0, epsilon = 1e-12);
    }

    // Ensure the output actually changed from the pre-modification run.
    assert_ne!(
        output_before.data[[0, 0, 0]].re,
        output_after.data[[0, 0, 0]].re
    );
}

#[test]
fn shell_get_freq_response() {
    let n_bins = NFFT / 2 + 1;
    let fft = Fft::new(NFFT);

    let mut biquad = Biquad::new(NFFT, FS, 1, BiquadFilterType::Lowpass, 1, 1, ALIAS_DECAY_DB)
        .expect("valid biquad");
    set_biquad_param(&mut biquad, 1_000.0, 0.0);

    let magnitude = Magnitude::new(NFFT, 1);

    let shell = Shell::new(Box::new(fft), Box::new(biquad.clone()), Box::new(magnitude))
        .expect("valid shell");

    let response = shell
        .get_freq_response()
        .expect("get_freq_response should succeed");
    assert_eq!(response.data.shape(), &[n_bins, 1, 1]);

    // Compare against the biquad's response to a unit spectrum.
    let input = ones_spectrum(&[1, n_bins, 1]);
    let expected = biquad.forward(&input).expect("biquad forward");
    for f in 0..n_bins {
        assert_abs_diff_eq!(
            response.data[[f, 0, 0]].re,
            expected.data[[0, f, 0]].re,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            response.data[[f, 0, 0]].im,
            expected.data[[0, f, 0]].im,
            epsilon = 1e-9
        );
    }
}

#[test]
fn parallel_two_gains_sums() {
    let n_bins = NFFT / 2 + 1;
    let n_channels = 2;

    let mut gain_a = Gain::new(NFFT, n_channels, n_channels).expect("valid gain");
    gain_a.param =
        ArrayD::from_shape_vec(IxDyn(&[n_channels, n_channels]), vec![1.0, 0.0, 0.0, 2.0]).unwrap();

    let mut gain_b = Gain::new(NFFT, n_channels, n_channels).expect("valid gain");
    gain_b.param =
        ArrayD::from_shape_vec(IxDyn(&[n_channels, n_channels]), vec![0.0, 1.0, 1.0, 0.0]).unwrap();

    let parallel = Parallel::new(Box::new(gain_a), Box::new(gain_b)).expect("valid parallel");
    assert_eq!(parallel.input_channels(), n_channels);
    assert_eq!(parallel.output_channels(), n_channels);

    let input = ones_spectrum(&[1, n_bins, n_channels]);
    let output = parallel.forward(&input).expect("forward should succeed");
    assert_eq!(output.data.shape(), &[1, n_bins, n_channels]);

    // gain_a: out0 = in0, out1 = 2*in1.
    // gain_b: out0 = in1, out1 = in0.
    // With input [1, 1]: out0 = 1 + 1 = 2, out1 = 2 + 1 = 3.
    for f in 0..n_bins {
        assert_abs_diff_eq!(output.data[[0, f, 0]].re, 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(output.data[[0, f, 1]].re, 3.0, epsilon = 1e-12);
    }
}

#[test]
fn parallel_gradient_matches_manual() {
    let n_bins = NFFT / 2 + 1;
    let n_channels = 2;

    let mut gain_a = Gain::new(NFFT, n_channels, n_channels).expect("valid gain");
    gain_a.param =
        ArrayD::from_shape_vec(IxDyn(&[n_channels, n_channels]), vec![0.5, -0.3, 1.2, -0.7])
            .unwrap();

    let mut gain_b = Gain::new(NFFT, n_channels, n_channels).expect("valid gain");
    gain_b.param =
        ArrayD::from_shape_vec(IxDyn(&[n_channels, n_channels]), vec![-0.7, 0.9, 0.1, 0.5])
            .unwrap();

    let input = random_spectrum(&[1, n_bins, n_channels]);
    let target = DiffTensor::zeros(IxDyn(&[1, n_bins, n_channels]));

    let mut parallel =
        Parallel::new(Box::new(gain_a.clone()), Box::new(gain_b.clone())).expect("valid parallel");
    parallel.zero_grad();
    let output = parallel.forward(&input).expect("parallel forward");
    let grad_output_data: Vec<Complex<f64>> = output
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(o, t)| 2.0 * (o - t))
        .collect();
    let grad_output = DiffTensor::from_array(
        ArrayD::from_shape_vec(IxDyn(&[1, n_bins, n_channels]), grad_output_data).unwrap(),
    );
    let grad_input_parallel = parallel
        .backward(&input, &output, &grad_output)
        .expect("parallel backward");

    // Manual sum of branch gradients.
    let mut gain_a_manual = gain_a.clone();
    let mut gain_b_manual = gain_b.clone();
    gain_a_manual.zero_grad();
    gain_b_manual.zero_grad();
    let out_a = gain_a_manual.forward(&input).expect("gain_a forward");
    let out_b = gain_b_manual.forward(&input).expect("gain_b forward");
    let grad_input_a = gain_a_manual
        .backward(&input, &out_a, &grad_output)
        .expect("gain_a backward");
    let grad_input_b = gain_b_manual
        .backward(&input, &out_b, &grad_output)
        .expect("gain_b backward");
    let grad_input_manual = &grad_input_a.data + &grad_input_b.data;

    for (parallel, manual) in grad_input_parallel
        .data
        .iter()
        .zip(grad_input_manual.iter())
    {
        assert_abs_diff_eq!(parallel.re, manual.re, epsilon = 1e-12);
        assert_abs_diff_eq!(parallel.im, manual.im, epsilon = 1e-12);
    }

    // Verify gradients were accumulated in both branches.
    let branch_a_grad = parallel.branches().0.gradients()[0];
    for (manual, parallel) in gain_a_manual.param_grad.iter().zip(branch_a_grad.iter()) {
        assert_abs_diff_eq!(manual, parallel, epsilon = 1e-12);
    }
    let branch_b_grad = parallel.branches().1.gradients()[0];
    for (manual, parallel) in gain_b_manual.param_grad.iter().zip(branch_b_grad.iter()) {
        assert_abs_diff_eq!(manual, parallel, epsilon = 1e-12);
    }
}

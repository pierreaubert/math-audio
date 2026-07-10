use approx::assert_abs_diff_eq;
use math_audio_autodiff::fft::Fft;
use math_audio_autodiff::gain::{Gain, Magnitude, ParallelGain};
use math_audio_autodiff::iir::biquad::Biquad;
use math_audio_autodiff::module::DiffModule;
use math_audio_autodiff::system::{Series, Shell};
use math_audio_autodiff::tensor::DiffTensor;
use math_audio_iir_fir::BiquadFilterType;
use ndarray::{Array1, Array2, ArrayD, Axis, IxDyn};
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
    for s in 0..biquad.n_sections {
        for out in 0..biquad.param.dim().2 {
            for inp in 0..biquad.param.dim().3 {
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
    gain.param = Array2::from_shape_vec(
        (n_out, n_in),
        vec![0.5, -0.3, 1.2, -0.7, 0.9, 0.1],
    )
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
    gain.param = Array1::from_vec(vec![0.5, -0.7, 1.2]);

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
fn series_two_gains_multiplies() {
    let n_bins = NFFT / 2 + 1;
    let mut gain1 = Gain::new(NFFT, 3, 2).expect("valid gain");
    gain1.param = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let mut gain2 = Gain::new(NFFT, 2, 3).expect("valid gain");
    gain2.param = Array2::from_shape_vec((2, 3), vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0]).unwrap();

    // Expected combined gain matrix: gain2.param * gain1.param.
    let expected = gain2.param.dot(&gain1.param);

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
    gain.param = Array2::from_shape_vec(
        (3, 2),
        vec![0.5, -0.3, 1.2, -0.7, 0.9, 0.1],
    )
    .unwrap();

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

    assert_eq!(grad_input_manual.data.shape(), grad_input_series.data.shape());
    for (manual, series) in grad_input_manual
        .data
        .iter()
        .zip(grad_input_series.data.iter())
    {
        assert_abs_diff_eq!(manual.re, series.re, epsilon = 1e-9);
        assert_abs_diff_eq!(manual.im, series.im, epsilon = 1e-9);
    }

    // Verify accumulated gradients match.
    let series_gain = series
        .modules()
        .first()
        .unwrap()
        .as_any()
        .downcast_ref::<Gain>()
        .unwrap();
    assert_eq!(gain_manual.param_grad.shape(), series_gain.param_grad.shape());
    for (manual, series) in gain_manual
        .param_grad
        .iter()
        .zip(series_gain.param_grad.iter())
    {
        assert_abs_diff_eq!(manual, series, epsilon = 1e-9);
    }

    let series_biquad = series
        .modules()
        .get(1)
        .unwrap()
        .as_any()
        .downcast_ref::<Biquad>()
        .unwrap();
    assert_eq!(
        biquad_manual.param_grad.shape(),
        series_biquad.param_grad.shape()
    );
    for (manual, series) in biquad_manual
        .param_grad
        .iter()
        .zip(series_biquad.param_grad.iter())
    {
        assert_abs_diff_eq!(manual, series, epsilon = 1e-9);
    }
}

#[test]
fn shell_get_freq_response() {
    let n_bins = NFFT / 2 + 1;
    let fft = Fft::new(NFFT);

    let mut biquad = Biquad::new(NFFT, FS, 1, BiquadFilterType::Lowpass, 1, 1, ALIAS_DECAY_DB)
        .expect("valid biquad");
    set_biquad_param(&mut biquad, 1_000.0, 0.0);

    let magnitude = Magnitude::new(NFFT, 1);

    let shell = Shell::new(
        Box::new(fft),
        Box::new(biquad.clone()),
        Box::new(magnitude),
    )
    .expect("valid shell");

    let response = shell.get_freq_response().expect("get_freq_response should succeed");
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

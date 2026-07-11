use approx::assert_abs_diff_eq;
use math_audio_autodiff::iir::biquad::{Biquad, ParallelBiquad};
use math_audio_autodiff::module::DiffModule;
use math_audio_autodiff::tensor::DiffTensor;
use math_audio_iir_fir::{Biquad as IirBiquad, BiquadFilterType};
use ndarray::{Array3, ArrayD, IxDyn};
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

fn set_parallel_biquad_param(biquad: &mut ParallelBiquad, fc_hz: f64, db_gain: f64) {
    let fc_raw = fc_raw_from_hz(fc_hz, biquad.fs);
    let gain_raw = gain_raw_from_db(db_gain);
    biquad.param.fill(0.0);
    let n_channels = biquad.param.shape()[2];
    for s in 0..biquad.n_sections {
        for ch in 0..n_channels {
            biquad.param[[s, 0, ch]] = fc_raw;
            biquad.param[[s, 1, ch]] = gain_raw;
        }
    }
}

#[test]
fn biquad_forward_matches_math_iir_fir_response() {
    let fc_hz = 1_000.0;
    let n_bins = NFFT / 2 + 1;

    let mut biquad = Biquad::new(NFFT, FS, 1, BiquadFilterType::Lowpass, 1, 1, ALIAS_DECAY_DB)
        .expect("valid biquad");
    set_biquad_param(&mut biquad, fc_hz, 0.0);

    let input = ones_spectrum(&[1, n_bins, 1]);
    let output = biquad.forward(&input).expect("forward should succeed");

    let reference = IirBiquad::new(BiquadFilterType::Lowpass, fc_hz, FS, 0.0, 0.0);

    for bin in [0, 10, 50, 100, 200, n_bins - 1] {
        let f = bin as f64 * FS / NFFT as f64;
        let expected = reference.complex_response(f).norm();
        let actual = output.data[[0, bin, 0]].norm();
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-6);
    }
}

#[test]
fn biquad_default_gain_is_trainable_unity() {
    let n_bins = NFFT / 2 + 1;
    let biquad = Biquad::new(NFFT, FS, 1, BiquadFilterType::Lowpass, 1, 1, ALIAS_DECAY_DB)
        .expect("valid biquad");
    let input = ones_spectrum(&[1, n_bins, 1]);
    let output = biquad.forward(&input).expect("forward should succeed");
    assert_abs_diff_eq!(output.data[[0, 0, 0]].norm(), 1.0, epsilon = 1e-9);
}

#[test]
fn biquad_zero_linear_gain_has_a_nonzero_gradient() {
    let n_bins = NFFT / 2 + 1;
    let mut biquad =
        Biquad::new(NFFT, FS, 1, BiquadFilterType::Lowpass, 1, 1, ALIAS_DECAY_DB).unwrap();
    biquad.param[[0, 0, 0, 0]] = fc_raw_from_hz(1_000.0, FS);
    biquad.param[[0, 1, 0, 0]] = 0.0;
    let input = ones_spectrum(&[1, n_bins, 1]);
    let output = biquad.forward(&input).unwrap();
    let grad_output = ones_spectrum(&[1, n_bins, 1]);
    biquad.backward(&input, &output, &grad_output).unwrap();
    assert!(biquad.param_grad[[0, 1, 0, 0]].abs() > 1.0);
}

#[test]
fn bandpass_gradient_is_correct_when_cutoffs_are_reversed() {
    let n_bins = NFFT / 2 + 1;
    let mut biquad = Biquad::new(
        NFFT,
        FS,
        1,
        BiquadFilterType::Bandpass,
        1,
        1,
        ALIAS_DECAY_DB,
    )
    .expect("valid bandpass");
    biquad.param[[0, 0, 0, 0]] = fc_raw_from_hz(5_000.0, FS);
    biquad.param[[0, 1, 0, 0]] = fc_raw_from_hz(800.0, FS);
    biquad.param[[0, 2, 0, 0]] = gain_raw_from_db(0.0);

    let input = complex_spectrum(&[1, n_bins, 1]);
    let output = biquad.forward(&input).unwrap();
    let grad_output = DiffTensor::from_array(output.data.clone() * 2.0);
    biquad.zero_grad();
    biquad.backward(&input, &output, &grad_output).unwrap();
    let analytical = biquad.param_grad[[0, 0, 0, 0]];

    let eps = 1e-6;
    let mut plus = biquad.clone();
    plus.param[[0, 0, 0, 0]] += eps;
    let plus_output = plus.forward(&input).unwrap();
    let plus_loss = plus_output.data.iter().map(Complex::norm_sqr).sum::<f64>();
    let mut minus = biquad.clone();
    minus.param[[0, 0, 0, 0]] -= eps;
    let minus_output = minus.forward(&input).unwrap();
    let minus_loss = minus_output.data.iter().map(Complex::norm_sqr).sum::<f64>();
    let numerical = (plus_loss - minus_loss) / (2.0 * eps);

    let scale = numerical.abs().max(1.0);
    assert!(
        (analytical - numerical).abs() / scale < 1e-5,
        "analytical={analytical}, numerical={numerical}"
    );
}

#[test]
fn biquad_gradient_finite_difference() {
    let n_bins = NFFT / 2 + 1;
    let mut biquad = Biquad::new(NFFT, FS, 1, BiquadFilterType::Lowpass, 1, 1, ALIAS_DECAY_DB)
        .expect("valid biquad");
    set_biquad_param(&mut biquad, 1_000.0, 3.0);

    // Target response from a reference filter.
    let target_filter = IirBiquad::new(BiquadFilterType::Lowpass, 1_200.0, FS, 0.0, 0.0);
    let target_values: Vec<Complex<f64>> = (0..n_bins)
        .map(|bin| {
            let f = bin as f64 * FS / NFFT as f64;
            target_filter.complex_response(f)
        })
        .collect();
    let target =
        DiffTensor::from_array(Array3::from_shape_vec((1, n_bins, 1), target_values).unwrap());

    let input = complex_spectrum(&[1, n_bins, 1]);
    let output = biquad.forward(&input).expect("forward should succeed");
    let grad_output_data: Vec<Complex<f64>> = output
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(o, t)| 2.0 * (o - t))
        .collect();
    let grad_output =
        DiffTensor::from_array(Array3::from_shape_vec((1, n_bins, 1), grad_output_data).unwrap());
    biquad.zero_grad();
    let _grad_input = biquad
        .backward(&input, &output, &grad_output)
        .expect("backward should succeed");
    let analytical = biquad.gradients()[0].clone();

    // Finite difference check.
    let epsilon = 1e-5;
    let param_shape = biquad.param.shape();
    let n_params = param_shape[1];
    let n_out = param_shape[2];
    let n_in = param_shape[3];
    for section in 0..biquad.n_sections {
        for p in 0..n_params {
            for out in 0..n_out {
                for inp in 0..n_in {
                    let mut param_plus = biquad.param.clone();
                    param_plus[[section, p, out, inp]] += epsilon;
                    let mut biquad_plus = biquad.clone();
                    biquad_plus.param = param_plus;
                    let out_plus = biquad_plus.forward(&input).expect("forward should succeed");
                    let loss_plus = mse_loss(&out_plus, &target);

                    let mut param_minus = biquad.param.clone();
                    param_minus[[section, p, out, inp]] -= epsilon;
                    let mut biquad_minus = biquad.clone();
                    biquad_minus.param = param_minus;
                    let out_minus = biquad_minus
                        .forward(&input)
                        .expect("forward should succeed");
                    let loss_minus = mse_loss(&out_minus, &target);

                    let finite_diff = (loss_plus - loss_minus) / (2.0 * epsilon);
                    let analytical_val = analytical[[section, p, out, inp]];

                    let denom = finite_diff.abs().max(1e-8);
                    let relative_error = (analytical_val - finite_diff).abs() / denom;
                    assert!(
                        relative_error < 1e-4 || (analytical_val - finite_diff).abs() < 1e-6,
                        "section={} p={} out={} inp={}: analytical={} finite_diff={} rel_err={}",
                        section,
                        p,
                        out,
                        inp,
                        analytical_val,
                        finite_diff,
                        relative_error
                    );
                }
            }
        }
    }
}

#[test]
fn parallel_biquad_forward_matches_math_iir_fir_response() {
    let fc_hz = 1_000.0;
    let n_bins = NFFT / 2 + 1;
    let n_channels = 2;

    let mut biquad = ParallelBiquad::new(
        NFFT,
        FS,
        1,
        BiquadFilterType::Highpass,
        n_channels,
        ALIAS_DECAY_DB,
    )
    .expect("valid parallel biquad");
    set_parallel_biquad_param(&mut biquad, fc_hz, 0.0);

    let input = ones_spectrum(&[1, n_bins, n_channels]);
    let output = biquad.forward(&input).expect("forward should succeed");

    let reference = IirBiquad::new(BiquadFilterType::Highpass, fc_hz, FS, 0.0, 0.0);

    for ch in 0..n_channels {
        for bin in [0, 10, 50, 100, 200, n_bins - 1] {
            let f = bin as f64 * FS / NFFT as f64;
            let expected = reference.complex_response(f).norm();
            let actual = output.data[[0, bin, ch]].norm();
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-6);
        }
    }
}

#[test]
fn parallel_biquad_gradient_finite_difference() {
    let n_bins = NFFT / 2 + 1;
    let n_channels = 2;
    let mut biquad = ParallelBiquad::new(
        NFFT,
        FS,
        1,
        BiquadFilterType::Highpass,
        n_channels,
        ALIAS_DECAY_DB,
    )
    .expect("valid parallel biquad");
    set_parallel_biquad_param(&mut biquad, 1_000.0, 3.0);

    let target_filter = IirBiquad::new(BiquadFilterType::Highpass, 1_200.0, FS, 0.0, 0.0);
    let target_values: Vec<Complex<f64>> = (0..n_bins)
        .flat_map(|bin| {
            let f = bin as f64 * FS / NFFT as f64;
            let h = target_filter.complex_response(f);
            std::iter::repeat_n(h, n_channels)
        })
        .collect();
    let target = DiffTensor::from_array(
        Array3::from_shape_vec((1, n_bins, n_channels), target_values).unwrap(),
    );

    let input = complex_spectrum(&[1, n_bins, n_channels]);
    let output = biquad.forward(&input).expect("forward should succeed");
    let grad_output_data: Vec<Complex<f64>> = output
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(o, t)| 2.0 * (o - t))
        .collect();
    let grad_output = DiffTensor::from_array(
        Array3::from_shape_vec((1, n_bins, n_channels), grad_output_data).unwrap(),
    );
    biquad.zero_grad();
    let _grad_input = biquad
        .backward(&input, &output, &grad_output)
        .expect("backward should succeed");
    let analytical = biquad.gradients()[0].clone();

    let epsilon = 1e-5;
    let param_shape = biquad.param.shape();
    let n_params = param_shape[1];
    for section in 0..biquad.n_sections {
        for p in 0..n_params {
            for ch in 0..n_channels {
                let mut param_plus = biquad.param.clone();
                param_plus[[section, p, ch]] += epsilon;
                let mut biquad_plus = biquad.clone();
                biquad_plus.param = param_plus;
                let out_plus = biquad_plus.forward(&input).expect("forward should succeed");
                let loss_plus = mse_loss(&out_plus, &target);

                let mut param_minus = biquad.param.clone();
                param_minus[[section, p, ch]] -= epsilon;
                let mut biquad_minus = biquad.clone();
                biquad_minus.param = param_minus;
                let out_minus = biquad_minus
                    .forward(&input)
                    .expect("forward should succeed");
                let loss_minus = mse_loss(&out_minus, &target);

                let finite_diff = (loss_plus - loss_minus) / (2.0 * epsilon);
                let analytical_val = analytical[[section, p, ch]];

                let denom = finite_diff.abs().max(1e-8);
                let relative_error = (analytical_val - finite_diff).abs() / denom;
                assert!(
                    relative_error < 1e-4 || (analytical_val - finite_diff).abs() < 1e-6,
                    "section={} p={} ch={}: analytical={} finite_diff={} rel_err={}",
                    section,
                    p,
                    ch,
                    analytical_val,
                    finite_diff,
                    relative_error
                );
            }
        }
    }
}

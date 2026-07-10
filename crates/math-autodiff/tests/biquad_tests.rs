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
    for s in 0..biquad.n_sections {
        for out in 0..biquad.param.dim().2 {
            for inp in 0..biquad.param.dim().3 {
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
    for s in 0..biquad.n_sections {
        for ch in 0..biquad.param.dim().2 {
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
    let target = DiffTensor::from_array(Array3::from_shape_vec((1, n_bins, 1), target_values).unwrap());

    let input = ones_spectrum(&[1, n_bins, 1]);
    let output = biquad.forward(&input).expect("forward should succeed");
    let grad_output_data: Vec<Complex<f64>> = output
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(o, t)| 2.0 * (o - t))
        .collect();
    let grad_output = DiffTensor::from_array(
        Array3::from_shape_vec((1, n_bins, 1), grad_output_data).unwrap(),
    );
    biquad.zero_grad();
    let _grad_input = biquad
        .backward(&input, &output, &grad_output)
        .expect("backward should succeed");
    let analytical = biquad.gradients().clone();

    // Finite difference check.
    let epsilon = 1e-5;
    let (_, n_params, n_out, n_in) = biquad.param.dim();
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
                    let out_minus = biquad_minus.forward(&input).expect("forward should succeed");
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

    let mut biquad =
        ParallelBiquad::new(NFFT, FS, 1, BiquadFilterType::Highpass, n_channels, ALIAS_DECAY_DB)
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
    let mut biquad =
        ParallelBiquad::new(NFFT, FS, 1, BiquadFilterType::Highpass, n_channels, ALIAS_DECAY_DB)
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
    let target =
        DiffTensor::from_array(Array3::from_shape_vec((1, n_bins, n_channels), target_values).unwrap());

    let input = ones_spectrum(&[1, n_bins, n_channels]);
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
    let analytical = biquad.gradients().clone();

    let epsilon = 1e-5;
    let (_, n_params, _) = biquad.param.dim();
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
                let out_minus = biquad_minus.forward(&input).expect("forward should succeed");
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

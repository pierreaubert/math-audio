use math_audio_autodiff::iir::geq::{DEFAULT_Q, GraphicEq};
use math_audio_autodiff::iir::sos_filter::SosFilter;
use math_audio_autodiff::module::DiffModule;
use math_audio_autodiff::tensor::DiffTensor;
use math_audio_iir_fir::{Biquad as IirBiquad, BiquadFilterType};
use ndarray::{Array3, ArrayD, IxDyn};
use num_complex::Complex;

const FS: f64 = 48_000.0;
const NFFT: usize = 512;
const ALIAS_DECAY_DB: f64 = 0.0;

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

#[test]
fn geq_identity_at_unity_gain() {
    let n_bins = NFFT / 2 + 1;
    let n_channels = 2;
    let geq = GraphicEq::new(NFFT, FS, 10, n_channels, ALIAS_DECAY_DB).expect("valid graphic eq");

    let input = complex_spectrum(&[1, n_bins, n_channels]);
    let output = geq.forward(&input).expect("forward should succeed");

    for ch in 0..n_channels {
        for bin in [0, 10, 50, 100, 200, n_bins - 1] {
            let expected = input.data[[0, bin, ch]];
            let actual = output.data[[0, bin, ch]];
            assert!(
                (actual - expected).norm() < 1e-9,
                "identity failed at ch={} bin={}",
                ch,
                bin
            );
        }
    }
}

#[test]
fn geq_forward_matches_hand_built_sos() {
    let n_bins = NFFT / 2 + 1;
    let n_bands = 10;
    let n_channels = 2;

    let mut geq =
        GraphicEq::new(NFFT, FS, n_bands, n_channels, ALIAS_DECAY_DB).expect("valid graphic eq");

    // Set non-unity, per-channel gains.
    for band in 0..n_bands {
        for ch in 0..n_channels {
            geq.param[[band, ch]] = 0.5 + 0.25 * (band as f64) + 0.1 * (ch as f64);
        }
    }

    let input = complex_spectrum(&[1, n_bins, n_channels]);
    let geq_output = geq.forward(&input).expect("forward should succeed");

    // Manually build an equivalent SOS cascade.
    let mut manual = SosFilter::new(NFFT, n_bands, n_channels, n_channels, ALIAS_DECAY_DB)
        .expect("valid manual sos filter");
    manual.param.fill(0.0);
    for band in 0..n_bands {
        for out_ch in 0..n_channels {
            for in_ch in 0..n_channels {
                manual.param[[band, 3, out_ch, in_ch]] = 1.0;
            }
        }
    }
    for (band, &fc) in geq.frequencies.iter().enumerate() {
        let coeffs = IirBiquad::new(BiquadFilterType::Peak, fc, FS, DEFAULT_Q, 0.0).coefficients();
        for ch in 0..n_channels {
            let gain = geq.param[[band, ch]];
            manual.param[[band, 0, ch, ch]] = gain * coeffs.b0;
            manual.param[[band, 1, ch, ch]] = gain * coeffs.b1;
            manual.param[[band, 2, ch, ch]] = gain * coeffs.b2;
            manual.param[[band, 3, ch, ch]] = 1.0;
            manual.param[[band, 4, ch, ch]] = coeffs.a1;
            manual.param[[band, 5, ch, ch]] = coeffs.a2;
        }
    }
    let manual_output = manual
        .forward(&input)
        .expect("manual forward should succeed");

    for ch in 0..n_channels {
        for bin in [0, 10, 50, 100, 200, n_bins - 1] {
            let expected = manual_output.data[[0, bin, ch]];
            let actual = geq_output.data[[0, bin, ch]];
            assert!(
                (actual - expected).norm() < 1e-9,
                "mismatch at ch={} bin={}: actual={:?} expected={:?}",
                ch,
                bin,
                actual,
                expected
            );
        }
    }
}

#[test]
fn geq_gradient_finite_difference() {
    let n_bins = NFFT / 2 + 1;
    let n_bands = 5;
    let n_channels = 2;

    let mut geq =
        GraphicEq::new(NFFT, FS, n_bands, n_channels, ALIAS_DECAY_DB).expect("valid graphic eq");

    // Initialize with non-trivial gains.
    for band in 0..n_bands {
        for ch in 0..n_channels {
            geq.param[[band, ch]] = 0.8 + 0.1 * (band as f64) + 0.05 * (ch as f64);
        }
    }

    let target = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, n_channels), Complex::new(0.3, -0.2))
            .into_dyn(),
    );

    let input = complex_spectrum(&[1, n_bins, n_channels]);
    let output = geq.forward(&input).expect("forward should succeed");
    let grad_output_data: Vec<Complex<f64>> = output
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(o, t)| 2.0 * (o - t))
        .collect();
    let grad_output = DiffTensor::from_array(
        Array3::from_shape_vec((1, n_bins, n_channels), grad_output_data).unwrap(),
    );

    geq.zero_grad();
    let _grad_input = geq
        .backward(&input, &output, &grad_output)
        .expect("backward should succeed");
    let analytical = geq.gradients()[0].clone();

    let epsilon = 1e-5;
    for band in 0..n_bands {
        for ch in 0..n_channels {
            let mut param_plus = geq.param.clone();
            param_plus[[band, ch]] += epsilon;
            let mut geq_plus = geq.clone();
            geq_plus.param = param_plus;
            let out_plus = geq_plus.forward(&input).expect("forward should succeed");
            let loss_plus = mse_loss(&out_plus, &target);

            let mut param_minus = geq.param.clone();
            param_minus[[band, ch]] -= epsilon;
            let mut geq_minus = geq.clone();
            geq_minus.param = param_minus;
            let out_minus = geq_minus.forward(&input).expect("forward should succeed");
            let loss_minus = mse_loss(&out_minus, &target);

            let finite_diff = (loss_plus - loss_minus) / (2.0 * epsilon);
            let analytical_val = analytical[[band, ch]];

            let denom = finite_diff.abs().max(1e-8);
            let relative_error = (analytical_val - finite_diff).abs() / denom;
            assert!(
                relative_error < 1e-4 || (analytical_val - finite_diff).abs() < 1e-6,
                "band={} ch={}: analytical={} finite_diff={} rel_err={}",
                band,
                ch,
                analytical_val,
                finite_diff,
                relative_error
            );
        }
    }
}

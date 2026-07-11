use math_audio_autodiff::iir::peq::{ParametricEq, PeqBandType};
use math_audio_autodiff::iir::sos_filter::SosFilter;
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

fn q_raw_from_q(q: f64) -> f64 {
    q.ln()
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

fn biquad_filter_type(band_type: PeqBandType) -> BiquadFilterType {
    match band_type {
        PeqBandType::Peak => BiquadFilterType::Peak,
        PeqBandType::Lowshelf => BiquadFilterType::Lowshelf,
        PeqBandType::Highshelf => BiquadFilterType::Highshelf,
    }
}

#[test]
fn peq_forward_matches_hand_built_sos() {
    let n_bins = NFFT / 2 + 1;
    let n_sections = 3;
    let n_channels = 2;

    for band_type in [PeqBandType::Peak, PeqBandType::Lowshelf, PeqBandType::Highshelf] {
        let mut peq = ParametricEq::new(NFFT, FS, n_sections, n_channels, band_type, ALIAS_DECAY_DB)
            .expect("valid parametric eq");

        // Set distinct per-section, per-channel parameters.
        for section in 0..n_sections {
            for ch in 0..n_channels {
                let fc_hz = 200.0 + 300.0 * (section as f64) + 50.0 * (ch as f64);
                let q = 0.7 + 0.2 * (section as f64);
                let gain_db = -6.0 + 4.0 * (ch as f64);
                peq.param[[section, 0, ch]] = fc_raw_from_hz(fc_hz, FS);
                peq.param[[section, 1, ch]] = q_raw_from_q(q);
                peq.param[[section, 2, ch]] = gain_db;
            }
        }

        let input = complex_spectrum(&[1, n_bins, n_channels]);
        let peq_output = peq.forward(&input).expect("forward should succeed");

        // Manually build an equivalent SOS cascade.
        let mut manual = SosFilter::new(NFFT, n_sections, n_channels, n_channels, ALIAS_DECAY_DB)
            .expect("valid manual sos filter");
        manual.param.fill(0.0);
        for section in 0..n_sections {
            for out_ch in 0..n_channels {
                for in_ch in 0..n_channels {
                    manual.param[[section, 3, out_ch, in_ch]] = 1.0;
                }
            }
        }
        for section in 0..n_sections {
            for ch in 0..n_channels {
                let fc_hz = 200.0 + 300.0 * (section as f64) + 50.0 * (ch as f64);
                let q = 0.7 + 0.2 * (section as f64);
                let gain_db = -6.0 + 4.0 * (ch as f64);
                let coeffs = IirBiquad::new(biquad_filter_type(band_type), fc_hz, FS, q, gain_db)
                    .coefficients();
                manual.param[[section, 0, ch, ch]] = coeffs.b0;
                manual.param[[section, 1, ch, ch]] = coeffs.b1;
                manual.param[[section, 2, ch, ch]] = coeffs.b2;
                manual.param[[section, 3, ch, ch]] = 1.0;
                manual.param[[section, 4, ch, ch]] = coeffs.a1;
                manual.param[[section, 5, ch, ch]] = coeffs.a2;
            }
        }
        let manual_output = manual.forward(&input).expect("manual forward should succeed");

        for ch in 0..n_channels {
            for bin in [0, 10, 50, 100, 200, n_bins - 1] {
                let expected = manual_output.data[[0, bin, ch]];
                let actual = peq_output.data[[0, bin, ch]];
                assert!(
                    (actual - expected).norm() < 1e-9,
                    "mismatch at band_type={:?} ch={} bin={}: actual={:?} expected={:?}",
                    band_type,
                    ch,
                    bin,
                    actual,
                    expected
                );
            }
        }
    }
}

#[test]
fn peq_identity_at_unity_gain() {
    let n_bins = NFFT / 2 + 1;
    let n_channels = 2;

    for band_type in [PeqBandType::Peak, PeqBandType::Lowshelf, PeqBandType::Highshelf] {
        let peq = ParametricEq::new(NFFT, FS, 4, n_channels, band_type, ALIAS_DECAY_DB)
            .expect("valid parametric eq");

        let input = complex_spectrum(&[1, n_bins, n_channels]);
        let output = peq.forward(&input).expect("forward should succeed");

        for ch in 0..n_channels {
            for bin in [0, 10, 50, 100, 200, n_bins - 1] {
                let expected = input.data[[0, bin, ch]];
                let actual = output.data[[0, bin, ch]];
                assert!(
                    (actual - expected).norm() < 1e-9,
                    "identity failed at band_type={:?} ch={} bin={}",
                    band_type,
                    ch,
                    bin
                );
            }
        }
    }
}

#[test]
fn peq_gradient_finite_difference() {
    let n_bins = NFFT / 2 + 1;
    let n_sections = 3;
    let n_channels = 2;

    for band_type in [PeqBandType::Peak, PeqBandType::Lowshelf, PeqBandType::Highshelf] {
        let mut peq = ParametricEq::new(NFFT, FS, n_sections, n_channels, band_type, ALIAS_DECAY_DB)
            .expect("valid parametric eq");

        // Initialize with non-trivial parameters.
        for section in 0..n_sections {
            for ch in 0..n_channels {
                let fc_hz = 300.0 + 400.0 * (section as f64);
                let q = 1.0 + 0.3 * (ch as f64);
                let gain_db = -3.0 + 2.0 * (ch as f64);
                peq.param[[section, 0, ch]] = fc_raw_from_hz(fc_hz, FS);
                peq.param[[section, 1, ch]] = q_raw_from_q(q);
                peq.param[[section, 2, ch]] = gain_db;
            }
        }

        let target = DiffTensor::from_array(
            Array3::<Complex<f64>>::from_elem((1, n_bins, n_channels), Complex::new(0.3, -0.2))
                .into_dyn(),
        );

        let input = complex_spectrum(&[1, n_bins, n_channels]);
        let output = peq.forward(&input).expect("forward should succeed");
        let grad_output_data: Vec<Complex<f64>> = output
            .data
            .iter()
            .zip(target.data.iter())
            .map(|(o, t)| 2.0 * (o - t))
            .collect();
        let grad_output = DiffTensor::from_array(
            Array3::from_shape_vec((1, n_bins, n_channels), grad_output_data).unwrap(),
        );
        peq.zero_grad();
        let _grad_input = peq
            .backward(&input, &output, &grad_output)
            .expect("backward should succeed");
        let analytical = peq.gradients()[0].clone();

        let epsilon = 1e-5;
        for section in 0..n_sections {
            for p in 0..3 {
                for ch in 0..n_channels {
                    let mut param_plus = peq.param.clone();
                    param_plus[[section, p, ch]] += epsilon;
                    let mut peq_plus = peq.clone();
                    peq_plus.param = param_plus;
                    let out_plus = peq_plus.forward(&input).expect("forward should succeed");
                    let loss_plus = mse_loss(&out_plus, &target);

                    let mut param_minus = peq.param.clone();
                    param_minus[[section, p, ch]] -= epsilon;
                    let mut peq_minus = peq.clone();
                    peq_minus.param = param_minus;
                    let out_minus = peq_minus.forward(&input).expect("forward should succeed");
                    let loss_minus = mse_loss(&out_minus, &target);

                    let finite_diff = (loss_plus - loss_minus) / (2.0 * epsilon);
                    let analytical_val = analytical[[section, p, ch]];

                    let denom = finite_diff.abs().max(1e-8);
                    let relative_error = (analytical_val - finite_diff).abs() / denom;
                    assert!(
                        relative_error < 1e-4 || (analytical_val - finite_diff).abs() < 1e-6,
                        "band_type={:?} section={} p={} ch={}: analytical={} finite_diff={} rel_err={}",
                        band_type,
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
}

#[test]
fn peq_new_rejects_invalid_args() {
    assert!(ParametricEq::new(0, FS, 1, 1, PeqBandType::Peak, ALIAS_DECAY_DB).is_err());
    assert!(ParametricEq::new(NFFT, FS, 0, 1, PeqBandType::Peak, ALIAS_DECAY_DB).is_err());
    assert!(ParametricEq::new(NFFT, FS, 1, 0, PeqBandType::Peak, ALIAS_DECAY_DB).is_err());
    assert!(ParametricEq::new(NFFT, 0.0, 1, 1, PeqBandType::Peak, ALIAS_DECAY_DB).is_err());
}



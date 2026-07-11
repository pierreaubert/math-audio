use approx::assert_relative_eq;
use math_audio_autodiff::{iir::sos_filter::SosFilter, module::DiffModule, tensor::DiffTensor};
use ndarray::{Array3, Array4};
use num_complex::Complex;

const NFFT: usize = 512;

#[test]
fn sos_filter_forward_matches_single_section() {
    let n_bins = NFFT / 2 + 1;
    let mut filter = SosFilter::new(NFFT, 1, 1, 1, 0.0).unwrap();
    // Identity section: b = [1, 0, 0], a = [1, 0, 0]
    filter.param[[0, 0, 0, 0]] = 1.0;
    filter.param[[0, 3, 0, 0]] = 1.0;

    let input = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, 1), Complex::new(1.0, 0.2)).into_dyn(),
    );
    let output = filter.forward(&input).unwrap();
    assert_relative_eq!(output.data[[0, 10, 0]].re, 1.0, epsilon = 1e-9);
    assert_relative_eq!(output.data[[0, 10, 0]].im, 0.2, epsilon = 1e-9);
}

#[test]
fn sos_filter_gradient_matches_finite_difference() {
    let n_bins = NFFT / 2 + 1;
    let mut filter = SosFilter::new(NFFT, 1, 1, 1, 0.0).unwrap();
    filter.param[[0, 0, 0, 0]] = 1.0;
    filter.param[[0, 3, 0, 0]] = 1.0;

    let input = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, 1), Complex::new(1.0, 0.1)).into_dyn(),
    );
    let target = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, 1), Complex::new(0.5, -0.2)).into_dyn(),
    );

    let eps = 1e-6;
    let mut numeric = Array4::<f64>::zeros((1, 6, 1, 1));
    for tap in 0..6 {
        filter.param[[0, tap, 0, 0]] += eps;
        let out_plus = filter.forward(&input).unwrap();
        let loss_plus = (&out_plus.data - &target.data)
            .iter()
            .map(|x| x.norm_sqr())
            .sum::<f64>();
        filter.param[[0, tap, 0, 0]] -= 2.0 * eps;
        let out_minus = filter.forward(&input).unwrap();
        let loss_minus = (&out_minus.data - &target.data)
            .iter()
            .map(|x| x.norm_sqr())
            .sum::<f64>();
        numeric[[0, tap, 0, 0]] = (loss_plus - loss_minus) / (2.0 * eps);
        filter.param[[0, tap, 0, 0]] += eps;
    }

    filter.zero_grad();
    let out = filter.forward(&input).unwrap();
    let diff = &out.data - &target.data;
    let grad = DiffTensor::from_array(diff.into_owned() * 2.0);
    filter.backward(&input, &out, &grad).unwrap();

    for tap in 0..6 {
        assert_relative_eq!(
            filter.param_grad[[0, tap, 0, 0]],
            numeric[[0, tap, 0, 0]],
            epsilon = 1e-4
        );
    }
}

use approx::assert_abs_diff_eq;
use math_audio_autodiff::fft::{Fft, FftAntiAlias, Ifft, IfftAntiAlias};
use math_audio_autodiff::module::DiffModule;
use math_audio_autodiff::tensor::DiffTensor;
use ndarray::Array1;
use num_complex::Complex;

fn impulse(nfft: usize) -> DiffTensor<f64> {
    let signal: Vec<Complex<f64>> = std::iter::once(Complex::new(1.0, 0.0))
        .chain(std::iter::repeat_n(Complex::new(0.0, 0.0), nfft - 1))
        .collect();
    DiffTensor::from_array(Array1::from(signal))
}

fn linear_loss(output: &DiffTensor<f64>, grad_output: &DiffTensor<f64>) -> f64 {
    output
        .data
        .iter()
        .zip(&grad_output.data)
        .map(|(output, grad)| (grad.conj() * output).re)
        .sum()
}

fn assert_backward_matches_finite_difference<M: DiffModule<f64>>(
    module: &mut M,
    input: &DiffTensor<f64>,
    grad_output: &DiffTensor<f64>,
    packed_spectrum_input: bool,
) {
    let output = module.forward(input).expect("forward should succeed");
    let analytical = module
        .backward(input, &output, grad_output)
        .expect("backward should succeed");
    let epsilon = 1e-6;

    for index in 0..input.data.len() {
        for imaginary in [false, true] {
            let is_packed_endpoint = packed_spectrum_input
                && imaginary
                && (index == 0
                    || (output.data.len().is_multiple_of(2)
                        && index == output.data.len() / 2));
            if is_packed_endpoint {
                assert_abs_diff_eq!(analytical.data[index].im, 0.0, epsilon = 1e-12);
                continue;
            }

            let mut plus = input.clone();
            let plus_value = &mut plus
                .data
                .as_slice_memory_order_mut()
                .expect("contiguous test tensor")[index];
            if imaginary {
                plus_value.im += epsilon;
            } else {
                plus_value.re += epsilon;
            }

            let mut minus = input.clone();
            let minus_value = &mut minus
                .data
                .as_slice_memory_order_mut()
                .expect("contiguous test tensor")[index];
            if imaginary {
                minus_value.im -= epsilon;
            } else {
                minus_value.re -= epsilon;
            }

            let loss_plus = linear_loss(
                &module.forward(&plus).expect("plus forward should succeed"),
                grad_output,
            );
            let loss_minus = linear_loss(
                &module.forward(&minus).expect("minus forward should succeed"),
                grad_output,
            );
            let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
            let actual = if imaginary {
                analytical.data[index].im
            } else {
                analytical.data[index].re
            };
            assert_abs_diff_eq!(actual, numerical, epsilon = 1e-7);
        }
    }
}

fn time_signal(nfft: usize) -> DiffTensor<f64> {
    DiffTensor::from_array(Array1::from_iter((0..nfft).map(|i| {
        let phase = i as f64 * 0.37;
        Complex::new(phase.sin() + 0.2 * phase.cos(), 0.1 * phase.cos())
    })))
}

fn packed_spectrum(nfft: usize) -> DiffTensor<f64> {
    let n_bins = nfft / 2 + 1;
    DiffTensor::from_array(Array1::from_iter((0..n_bins).map(|i| {
        let phase = i as f64 * 0.29;
        let imaginary = if i == 0 || (nfft.is_multiple_of(2) && i == nfft / 2) {
            0.0
        } else {
            phase.cos() - 0.3
        };
        Complex::new(phase.sin() + 0.4, imaginary)
    })))
}

#[test]
fn fft_roundtrip_is_identity() {
    let nfft = 512;
    let impulse = impulse(nfft);
    let fft = Fft::new(nfft);
    let spectrum = fft.forward(&impulse).expect("FFT forward should succeed");
    assert_abs_diff_eq!(spectrum.data[0].re, 1.0, epsilon = 1e-9);
}

#[test]
fn ifft_roundtrip_is_identity() {
    let nfft = 512;
    let impulse = impulse(nfft);

    let fft = Fft::new(nfft);
    let spectrum = fft.forward(&impulse).expect("FFT forward should succeed");

    let ifft = Ifft::new(nfft);
    let recovered = ifft.forward(&spectrum).expect("IFFT forward should succeed");

    assert_eq!(recovered.data.shape(), &[nfft]);
    assert_abs_diff_eq!(recovered.data[0].re, 1.0, epsilon = 1e-9);
    for i in 1..nfft {
        assert_abs_diff_eq!(recovered.data[i].re, 0.0, epsilon = 1e-9);
    }
}

#[test]
fn fft_backward_matches_finite_difference() {
    for nfft in [15, 16] {
        let input = time_signal(nfft);
        let grad_output = packed_spectrum(nfft);
        assert_backward_matches_finite_difference(&mut Fft::new(nfft), &input, &grad_output, false);
    }
}

#[test]
fn fft_antialias_impulse_preserves_dc() {
    let nfft = 512;
    let impulse = impulse(nfft);

    let fft = FftAntiAlias::new(nfft, 60.0);
    let spectrum = fft.forward(&impulse).expect("FFT anti-alias forward should succeed");

    assert_eq!(spectrum.data.shape(), &[nfft / 2 + 1]);
    assert_abs_diff_eq!(spectrum.data[0].re, 1.0, epsilon = 1e-9);
}

#[test]
fn ifft_antialias_roundtrip_preserves_first_sample() {
    let nfft = 512;
    let impulse = impulse(nfft);

    let fft = FftAntiAlias::new(nfft, 60.0);
    let spectrum = fft.forward(&impulse).expect("FFT anti-alias forward should succeed");

    let ifft = IfftAntiAlias::new(nfft, 60.0);
    let recovered = ifft.forward(&spectrum).expect("IFFT anti-alias forward should succeed");

    assert_eq!(recovered.data.shape(), &[nfft]);
    assert_abs_diff_eq!(recovered.data[0].re, 1.0, epsilon = 1e-9);
}

#[test]
fn ifft_backward_matches_finite_difference() {
    for nfft in [15, 16] {
        let input = packed_spectrum(nfft);
        let grad_output = time_signal(nfft);
        assert_backward_matches_finite_difference(&mut Ifft::new(nfft), &input, &grad_output, true);
    }
}

#[test]
fn fft_antialias_backward_matches_finite_difference() {
    for nfft in [15, 16] {
        let input = time_signal(nfft);
        let grad_output = packed_spectrum(nfft);
        assert_backward_matches_finite_difference(
            &mut FftAntiAlias::new(nfft, 60.0),
            &input,
            &grad_output,
            false,
        );
    }
}

#[test]
fn ifft_antialias_backward_matches_finite_difference() {
    for nfft in [15, 16] {
        let input = packed_spectrum(nfft);
        let grad_output = time_signal(nfft);
        assert_backward_matches_finite_difference(
            &mut IfftAntiAlias::new(nfft, 60.0),
            &input,
            &grad_output,
            true,
        );
    }
}

#[test]
fn anti_alias_envelope_decays_for_late_samples() {
    let nfft = 512;
    let fft = FftAntiAlias::new(nfft, 60.0);

    assert!(fft.envelope[0] > fft.envelope[nfft - 1]);
    assert_abs_diff_eq!(fft.envelope[0], 1.0, epsilon = 1e-12);
}

fn multi_channel_time_signal(batch: usize, nfft: usize, channels: usize) -> DiffTensor<f64> {
    DiffTensor::from_array(
        ndarray::Array3::from_shape_fn((batch, nfft, channels), |(b, t, c)| {
            let phase = (b * 7 + t * 13 + c * 3) as f64 * 0.11;
            Complex::new(phase.sin() + 0.2 * phase.cos(), 0.0)
        })
        .into_dyn(),
    )
}

fn multi_channel_spectrum(batch: usize, nfft: usize, channels: usize) -> DiffTensor<f64> {
    let n_bins = nfft / 2 + 1;
    DiffTensor::from_array(
        ndarray::Array3::from_shape_fn((batch, n_bins, channels), |(b, bin, c)| {
            let phase = (b * 5 + bin * 17 + c * 11) as f64 * 0.09;
            let imaginary = if bin == 0 || (nfft.is_multiple_of(2) && bin == nfft / 2) {
                0.0
            } else {
                phase.cos() - 0.3
            };
            Complex::new(phase.sin() + 0.4, imaginary)
        })
        .into_dyn(),
    )
}

fn is_packed_endpoint_3d(nfft: usize, bin: usize) -> bool {
    bin == 0 || (nfft.is_multiple_of(2) && bin == nfft / 2)
}

fn assert_backward_matches_finite_difference_3d<M: DiffModule<f64>>(
    module: &mut M,
    input: &DiffTensor<f64>,
    grad_output: &DiffTensor<f64>,
    nfft: usize,
    is_spectrum_input: bool,
) {
    let output = module.forward(input).expect("forward should succeed");
    let analytical = module
        .backward(input, &output, grad_output)
        .expect("backward should succeed");
    let epsilon = 1e-6;

    let shape: Vec<usize> = input.data.shape().to_vec();
    assert_eq!(shape.len(), 3);
    let (batch, mid, channels) = (shape[0], shape[1], shape[2]);
    assert_eq!(mid, if is_spectrum_input { nfft / 2 + 1 } else { nfft });

    for b in 0..batch {
        for k in 0..mid {
            for c in 0..channels {
                for imaginary in [false, true] {
                    let skip_imag = is_spectrum_input && is_packed_endpoint_3d(nfft, k);
                    if imaginary && skip_imag {
                        assert_abs_diff_eq!(analytical.data[[b, k, c]].im, 0.0, epsilon = 1e-12);
                        continue;
                    }

                    let mut plus = input.clone();
                    let plus_value = &mut plus.data[[b, k, c]];
                    if imaginary {
                        plus_value.im += epsilon;
                    } else {
                        plus_value.re += epsilon;
                    }

                    let mut minus = input.clone();
                    let minus_value = &mut minus.data[[b, k, c]];
                    if imaginary {
                        minus_value.im -= epsilon;
                    } else {
                        minus_value.re -= epsilon;
                    }

                    let loss_plus = linear_loss(
                        &module.forward(&plus).expect("plus forward should succeed"),
                        grad_output,
                    );
                    let loss_minus = linear_loss(
                        &module.forward(&minus).expect("minus forward should succeed"),
                        grad_output,
                    );
                    let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
                    let actual = if imaginary {
                        analytical.data[[b, k, c]].im
                    } else {
                        analytical.data[[b, k, c]].re
                    };
                    assert_abs_diff_eq!(actual, numerical, epsilon = 1e-7);
                }
            }
        }
    }
}

#[test]
fn multi_channel_fft_roundtrip_is_identity() {
    let nfft = 512;
    let batch = 2;
    let channels = 3;
    let input = multi_channel_time_signal(batch, nfft, channels);

    let fft = Fft::with_channels(nfft, channels);
    let spectrum = fft.forward(&input).expect("FFT forward should succeed");
    assert_eq!(spectrum.data.shape(), &[batch, nfft / 2 + 1, channels]);

    let ifft = Ifft::with_channels(nfft, channels);
    let recovered = ifft.forward(&spectrum).expect("IFFT forward should succeed");
    assert_eq!(recovered.data.shape(), &[batch, nfft, channels]);

    for b in 0..batch {
        for c in 0..channels {
            assert_abs_diff_eq!(recovered.data[[b, 0, c]].re, input.data[[b, 0, c]].re, epsilon = 1e-9);
            for t in 1..nfft {
                assert_abs_diff_eq!(recovered.data[[b, t, c]].re, input.data[[b, t, c]].re, epsilon = 1e-9);
            }
        }
    }
}

#[test]
fn multi_channel_fft_backward_matches_finite_difference() {
    for nfft in [15, 16] {
        let input = multi_channel_time_signal(2, nfft, 3);
        let grad_output = multi_channel_spectrum(2, nfft, 3);
        assert_backward_matches_finite_difference_3d(
            &mut Fft::with_channels(nfft, 3),
            &input,
            &grad_output,
            nfft,
            false,
        );
    }
}

#[test]
fn multi_channel_ifft_backward_matches_finite_difference() {
    for nfft in [15, 16] {
        let input = multi_channel_spectrum(2, nfft, 3);
        let grad_output = multi_channel_time_signal(2, nfft, 3);
        assert_backward_matches_finite_difference_3d(
            &mut Ifft::with_channels(nfft, 3),
            &input,
            &grad_output,
            nfft,
            true,
        );
    }
}

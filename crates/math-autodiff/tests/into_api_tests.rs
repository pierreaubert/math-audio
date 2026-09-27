use math_audio_autodiff::{delay::Delay, gain::Gain, module::DiffModule, tensor::DiffTensor};
use ndarray::{Array3, IxDyn};
use num_complex::Complex;

fn spectrum_fixture(nfft: usize, channels: usize) -> DiffTensor<f64> {
    DiffTensor::from_array(
        Array3::from_shape_fn((1, nfft / 2 + 1, channels), |(_, f, c)| {
            Complex::new(f as f64 * 0.01 + c as f64, f as f64 * 0.001)
        })
        .into_dyn(),
    )
}

#[test]
fn gain_forward_into_matches_forward_and_reuses_buffer() {
    let gain = Gain::new(64, 2, 2).unwrap();
    let input = spectrum_fixture(64, 2);
    let expected = gain.forward(&input).unwrap();

    let mut out = DiffTensor::zeros(IxDyn(&[0]));
    gain.forward_into(&input, &mut out).unwrap();
    assert_eq!(out.data, expected.data);

    // Second call with a correctly shaped buffer must not reallocate.
    let ptr_before = out.data.as_ptr();
    gain.forward_into(&input, &mut out).unwrap();
    assert_eq!(out.data.shape(), expected.data.shape());
    assert_eq!(out.data.as_ptr(), ptr_before);
    assert_eq!(out.data, expected.data);
}

#[test]
fn gain_backward_into_matches_backward() {
    let mut via_owned = Gain::new(64, 2, 2).unwrap();
    let mut via_into = Gain::new(64, 2, 2).unwrap();
    let input = spectrum_fixture(64, 2);
    let output = via_owned.forward(&input).unwrap();
    let grad_output = DiffTensor::from_array(output.data.clone());

    let expected = via_owned.backward(&input, &output, &grad_output).unwrap();
    let mut grad_input = DiffTensor::zeros(IxDyn(&[0]));
    via_into
        .backward_into(&input, &output, &grad_output, &mut grad_input)
        .unwrap();
    assert_eq!(grad_input.data, expected.data);
    assert_eq!(via_into.gradients()[0], via_owned.gradients()[0]);
}

#[test]
fn delay_forward_into_matches_forward_and_reuses_buffer() {
    let delay = Delay::new(64, 2, 2, 0.0).unwrap();
    let input = spectrum_fixture(64, 2);
    let expected = delay.forward(&input).unwrap();

    let mut out = DiffTensor::zeros(IxDyn(&[0]));
    delay.forward_into(&input, &mut out).unwrap();
    assert_eq!(out.data, expected.data);

    let ptr_before = out.data.as_ptr();
    delay.forward_into(&input, &mut out).unwrap();
    assert_eq!(out.data.shape(), expected.data.shape());
    assert_eq!(out.data.as_ptr(), ptr_before);
    assert_eq!(out.data, expected.data);
}

#[test]
fn delay_backward_into_matches_backward() {
    let mut via_owned = Delay::new(64, 2, 2, 0.0).unwrap();
    let mut via_into = Delay::new(64, 2, 2, 0.0).unwrap();
    let input = spectrum_fixture(64, 2);
    let output = via_owned.forward(&input).unwrap();
    let grad_output = DiffTensor::from_array(output.data.clone());

    let expected = via_owned.backward(&input, &output, &grad_output).unwrap();
    let mut grad_input = DiffTensor::zeros(IxDyn(&[0]));
    via_into
        .backward_into(&input, &output, &grad_output, &mut grad_input)
        .unwrap();
    assert_eq!(grad_input.data, expected.data);
    assert_eq!(via_into.gradients()[0], via_owned.gradients()[0]);
}

#[test]
fn into_apis_resize_wrong_shaped_buffers() {
    let gain = Gain::new(64, 2, 2).unwrap();
    let input = spectrum_fixture(64, 2);
    let expected = gain.forward(&input).unwrap();

    // Deliberately wrong shape: must be resized, not just overwritten.
    let mut out = DiffTensor::from_array(Array3::ones((1, 3, 2)).into_dyn());
    gain.forward_into(&input, &mut out).unwrap();
    assert_eq!(out.data.shape(), expected.data.shape());
    assert_eq!(out.data, expected.data);
}

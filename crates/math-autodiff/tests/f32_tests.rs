use approx::assert_abs_diff_eq;
use math_audio_autodiff::{
    delay::Delay, fft::Fft, gain::Gain, iir::biquad::Biquad, loss::mse_loss, module::DiffModule,
    system::Series, tensor::DiffTensor,
};
use math_audio_iir_fir::BiquadFilterType;
use ndarray::Array3;
use num_complex::Complex;

fn spectrum_fixture_f32(nfft: usize, channels: usize) -> DiffTensor<f32> {
    DiffTensor::from_array(
        Array3::from_shape_fn((1, nfft / 2 + 1, channels), |(_, f, c)| {
            Complex::new(f as f32 * 0.01 + c as f32, f as f32 * 0.001)
        })
        .into_dyn(),
    )
}

fn spectrum_fixture_f64(nfft: usize, channels: usize) -> DiffTensor<f64> {
    DiffTensor::from_array(
        Array3::from_shape_fn((1, nfft / 2 + 1, channels), |(_, f, c)| {
            Complex::new(f as f64 * 0.01 + c as f64, f as f64 * 0.001)
        })
        .into_dyn(),
    )
}

fn assert_spectra_close(actual: &DiffTensor<f32>, expected: &DiffTensor<f64>, eps: f32) {
    assert_eq!(actual.data.shape(), expected.data.shape());
    for (a, e) in actual.data.iter().zip(expected.data.iter()) {
        assert_abs_diff_eq!(a.re, e.re as f32, epsilon = eps);
        assert_abs_diff_eq!(a.im, e.im as f32, epsilon = eps);
    }
}

#[test]
fn gain_f32_matches_f64() {
    let mut gain32 = Gain::<f32>::new(64, 2, 2).unwrap();
    let mut gain64 = Gain::<f64>::new(64, 2, 2).unwrap();
    gain32.parameters_mut()[0].fill(0.5);
    gain64.parameters_mut()[0].fill(0.5);
    let in32 = spectrum_fixture_f32(64, 2);
    let in64 = spectrum_fixture_f64(64, 2);
    let out32 = gain32.forward(&in32).unwrap();
    let out64 = gain64.forward(&in64).unwrap();
    assert_spectra_close(&out32, &out64, 1e-4);

    let grad32 = DiffTensor::from_array(out32.data.clone());
    let grad64 = DiffTensor::from_array(out64.data.clone());
    let back32 = gain32.backward(&in32, &out32, &grad32).unwrap();
    let back64 = gain64.backward(&in64, &out64, &grad64).unwrap();
    assert_spectra_close(&back32, &back64, 1e-3);
}

#[test]
fn delay_f32_matches_f64() {
    let mut delay32 = Delay::<f32>::new(64, 2, 2, 0.0).unwrap();
    let mut delay64 = Delay::<f64>::new(64, 2, 2, 0.0).unwrap();
    delay32.parameters_mut()[0].fill(0.25);
    delay64.parameters_mut()[0].fill(0.25);
    let in32 = spectrum_fixture_f32(64, 2);
    let in64 = spectrum_fixture_f64(64, 2);
    let out32 = delay32.forward(&in32).unwrap();
    let out64 = delay64.forward(&in64).unwrap();
    assert_spectra_close(&out32, &out64, 1e-3);

    let grad32 = DiffTensor::from_array(out32.data.clone());
    let grad64 = DiffTensor::from_array(out64.data.clone());
    let back32 = delay32.backward(&in32, &out32, &grad32).unwrap();
    let back64 = delay64.backward(&in64, &out64, &grad64).unwrap();
    assert_spectra_close(&back32, &back64, 1e-2);
}

#[test]
fn series_f32_end_to_end() {
    let mut gain32 = Gain::<f32>::new(64, 2, 2).unwrap();
    gain32.parameters_mut()[0].fill(0.5);
    let delay32 = Delay::<f32>::new(64, 2, 2, 0.0).unwrap();
    let modules: Vec<Box<dyn DiffModule<f32>>> = vec![Box::new(gain32), Box::new(delay32)];
    let mut series32 = Series::new(modules).unwrap();

    let mut gain64 = Gain::<f64>::new(64, 2, 2).unwrap();
    gain64.parameters_mut()[0].fill(0.5);
    let delay64 = Delay::<f64>::new(64, 2, 2, 0.0).unwrap();
    let modules64: Vec<Box<dyn DiffModule<f64>>> = vec![Box::new(gain64), Box::new(delay64)];
    let mut series64 = Series::new(modules64).unwrap();

    let in32 = spectrum_fixture_f32(64, 2);
    let in64 = spectrum_fixture_f64(64, 2);
    let out32 = series32.forward(&in32).unwrap();
    let out64 = series64.forward(&in64).unwrap();
    assert_spectra_close(&out32, &out64, 1e-3);

    let grad32 = DiffTensor::from_array(out32.data.clone());
    let grad64 = DiffTensor::from_array(out64.data.clone());
    let back32 = series32.backward(&in32, &out32, &grad32).unwrap();
    let back64 = series64.backward(&in64, &out64, &grad64).unwrap();
    assert_spectra_close(&back32, &back64, 1e-2);
}

#[test]
fn fft_f32_roundtrip_matches_f64() {
    let fft32 = Fft::<f32>::with_channels(64, 1);
    let fft64 = Fft::<f64>::with_channels(64, 1);
    let time32 = DiffTensor::from_array(
        Array3::from_shape_fn((1, 64, 1), |(_, t, _)| {
            Complex::new((t as f32 * 0.1).sin(), 0.0)
        })
        .into_dyn(),
    );
    let time64 = DiffTensor::from_array(
        Array3::from_shape_fn((1, 64, 1), |(_, t, _)| {
            Complex::new((t as f64 * 0.1).sin(), 0.0)
        })
        .into_dyn(),
    );
    let spec32 = fft32.forward(&time32).unwrap();
    let spec64 = fft64.forward(&time64).unwrap();
    assert_spectra_close(&spec32, &spec64, 1e-3);
}

#[test]
fn biquad_f32_matches_f64() {
    let biquad32 =
        Biquad::<f32>::new(64, 48_000.0, 1, BiquadFilterType::Lowpass, 1, 1, 60.0).unwrap();
    let biquad64 =
        Biquad::<f64>::new(64, 48_000.0, 1, BiquadFilterType::Lowpass, 1, 1, 60.0).unwrap();
    let in32 = spectrum_fixture_f32(64, 1);
    let in64 = spectrum_fixture_f64(64, 1);
    let out32 = biquad32.forward(&in32).unwrap();
    let out64 = biquad64.forward(&in64).unwrap();
    assert_spectra_close(&out32, &out64, 1e-2);
}

#[test]
fn mse_loss_f32_matches_f64() {
    let pred32 = spectrum_fixture_f32(64, 2);
    let target32 = spectrum_fixture_f32(64, 2);
    let pred64 = spectrum_fixture_f64(64, 2);
    let target64 = spectrum_fixture_f64(64, 2);
    let loss32 = mse_loss(&pred32, &target32).unwrap();
    let loss64 = mse_loss(&pred64, &target64).unwrap();
    assert_abs_diff_eq!(loss32, loss64 as f32, epsilon = 1e-4);
}

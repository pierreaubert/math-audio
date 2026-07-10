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

fn ones_spectrum(nfft: usize) -> DiffTensor<f64> {
    let spectrum: Vec<Complex<f64>> =
        std::iter::repeat_n(Complex::new(1.0, 0.0), nfft / 2 + 1).collect();
    DiffTensor::from_array(Array1::from(spectrum))
}

#[test]
fn fft_roundtrip_is_identity() {
    let nfft = 512;
    let impulse = impulse(nfft);
    let fft = Fft::new(nfft);
    let spectrum = fft.forward(&impulse);
    assert_abs_diff_eq!(spectrum.data[0].re, 1.0, epsilon = 1e-9);
}

#[test]
fn ifft_roundtrip_is_identity() {
    let nfft = 512;
    let impulse = impulse(nfft);

    let fft = Fft::new(nfft);
    let spectrum = fft.forward(&impulse);

    let ifft = Ifft::new(nfft);
    let recovered = ifft.forward(&spectrum);

    assert_eq!(recovered.data.shape(), &[nfft]);
    assert_abs_diff_eq!(recovered.data[0].re, 1.0, epsilon = 1e-9);
    for i in 1..nfft {
        assert_abs_diff_eq!(recovered.data[i].re, 0.0, epsilon = 1e-9);
    }
}

#[test]
fn fft_backward_of_ones_returns_impulse() {
    let nfft = 512;
    let ones = ones_spectrum(nfft);

    let mut fft = Fft::new(nfft);
    let grad = fft.backward(&ones, &ones, &ones);

    assert_eq!(grad.data.shape(), &[nfft]);
    assert_abs_diff_eq!(grad.data[0].re, 1.0, epsilon = 1e-9);
    for i in 1..nfft {
        assert_abs_diff_eq!(grad.data[i].re, 0.0, epsilon = 1e-9);
    }
}

#[test]
fn fft_antialias_impulse_preserves_dc() {
    let nfft = 512;
    let impulse = impulse(nfft);

    let fft = FftAntiAlias::new(nfft, 60.0);
    let spectrum = fft.forward(&impulse);

    assert_eq!(spectrum.data.shape(), &[nfft / 2 + 1]);
    assert_abs_diff_eq!(spectrum.data[0].re, 1.0, epsilon = 1e-9);
}

#[test]
fn ifft_antialias_roundtrip_preserves_first_sample() {
    let nfft = 512;
    let impulse = impulse(nfft);

    let fft = FftAntiAlias::new(nfft, 60.0);
    let spectrum = fft.forward(&impulse);

    let ifft = IfftAntiAlias::new(nfft, 60.0);
    let recovered = ifft.forward(&spectrum);

    assert_eq!(recovered.data.shape(), &[nfft]);
    assert_abs_diff_eq!(recovered.data[0].re, 1.0, epsilon = 1e-9);
}

#[test]
fn ifft_backward_of_impulse_returns_ones() {
    let nfft = 512;
    let impulse = impulse(nfft);

    let mut ifft = Ifft::new(nfft);
    let grad = ifft.backward(&impulse, &impulse, &impulse);

    assert_eq!(grad.data.shape(), &[nfft / 2 + 1]);
    for sample in grad.data.iter() {
        assert_abs_diff_eq!(sample.re, 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(sample.im, 0.0, epsilon = 1e-9);
    }
}

#[test]
fn fft_antialias_backward_of_ones_returns_envelope() {
    let nfft = 512;
    let ones = ones_spectrum(nfft);

    let mut fft = FftAntiAlias::new(nfft, 60.0);
    let grad = fft.backward(&ones, &ones, &ones);

    assert_eq!(grad.data.shape(), &[nfft]);
    assert_abs_diff_eq!(grad.data[0].re, fft.envelope[0], epsilon = 1e-9);
    for i in 1..nfft {
        assert_abs_diff_eq!(grad.data[i].re, 0.0, epsilon = 1e-9);
    }
}

#[test]
fn ifft_antialias_backward_of_impulse_returns_ones() {
    let nfft = 512;
    let impulse = impulse(nfft);

    let mut ifft = IfftAntiAlias::new(nfft, 60.0);
    let grad = ifft.backward(&impulse, &impulse, &impulse);

    assert_eq!(grad.data.shape(), &[nfft / 2 + 1]);
    for sample in grad.data.iter() {
        assert_abs_diff_eq!(sample.re, 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(sample.im, 0.0, epsilon = 1e-9);
    }
}

#[test]
fn anti_alias_envelope_decays_for_late_samples() {
    let nfft = 512;
    let fft = FftAntiAlias::new(nfft, 60.0);

    assert!(fft.envelope[0] > fft.envelope[nfft - 1]);
    assert_abs_diff_eq!(fft.envelope[0], 1.0, epsilon = 1e-12);
}

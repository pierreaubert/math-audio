//! Biquad magnitude-response matching example.
//!
//! Reproduces the FLAMO `e7_biquad.py` workflow: generate a random highpass
//! target response, build a `Shell(FFT -> Biquad -> Magnitude)`, and fit the
//! biquad parameters with MSE loss and SGD.

use math_audio_autodiff::{
    fft::Fft,
    gain::Magnitude,
    iir::biquad::Biquad,
    loss::{mse_loss, mse_loss_backward},
    module::DiffModule,
    optim::Sgd,
    system::Shell,
    tensor::DiffTensor,
};
use math_audio_iir_fir::BiquadFilterType;
use ndarray::{Array3, Array4};
use num_complex::Complex;
use rand::Rng;
use rustfft::FftPlanner;

const FS: f64 = 48_000.0;
const NFFT: usize = 96_000;
const N_SECTIONS: usize = 2;
const IN_CH: usize = 1;
const OUT_CH: usize = 2;
const EPOCHS: usize = 100;
const LEARNING_RATE: f64 = 1e-2;
const ALIAS_DECAY_DB: f64 = 30.0;

/// RBJ highpass coefficients for a single cutoff frequency and gain.
fn highpass_coeffs(fc: f64, gain_db: f64) -> ([f64; 3], [f64; 3]) {
    let omega = 2.0 * std::f64::consts::PI * fc / FS;
    let sn = omega.sin();
    let cs = omega.cos();
    let alpha = sn / 2.0 * std::f64::consts::SQRT_2;

    let mut b = [(1.0 + cs) / 2.0, -(1.0 + cs), (1.0 + cs) / 2.0];
    let a = [1.0 + alpha, -2.0 * cs, 1.0 - alpha];

    let gain_lin = 10.0_f64.powf(gain_db / 20.0);
    for coeff in &mut b {
        *coeff *= gain_lin;
    }

    // Normalize by a0.
    let a0 = a[0];
    (
        [b[0] / a0, b[1] / a0, b[2] / a0],
        [1.0, a[1] / a0, a[2] / a0],
    )
}

/// Compute the complex frequency response of a cascade of highpass sections.
fn target_response<R: Rng>(rng: &mut R) -> Array3<Complex<f64>> {
    let n_bins = NFFT / 2 + 1;
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(NFFT);

    // Per-section coefficients: shape (N_SECTIONS, 3, OUT_CH, IN_CH).
    let mut b_coeffs = Array4::<f64>::zeros((N_SECTIONS, 3, OUT_CH, IN_CH));
    let mut a_coeffs = Array4::<f64>::zeros((N_SECTIONS, 3, OUT_CH, IN_CH));

    for section in 0..N_SECTIONS {
        for out in 0..OUT_CH {
            for inp in 0..IN_CH {
                let fc = rng.random_range(100.0..FS / 2.0 - 100.0);
                let gain_db = rng.random_range(-1.0..1.0);
                let (b, a) = highpass_coeffs(fc, gain_db);
                for tap in 0..3 {
                    b_coeffs[[section, tap, out, inp]] = b[tap];
                    a_coeffs[[section, tap, out, inp]] = a[tap];
                }
            }
        }
    }

    // Compute per-section FFTs.
    let mut b_response = Array4::<Complex<f64>>::zeros((N_SECTIONS, n_bins, OUT_CH, IN_CH));
    let mut a_response = Array4::<Complex<f64>>::zeros((N_SECTIONS, n_bins, OUT_CH, IN_CH));

    for section in 0..N_SECTIONS {
        for out in 0..OUT_CH {
            for inp in 0..IN_CH {
                let mut b_buf = vec![Complex::new(0.0, 0.0); NFFT];
                let mut a_buf = vec![Complex::new(0.0, 0.0); NFFT];
                for tap in 0..3 {
                    b_buf[tap] = Complex::new(b_coeffs[[section, tap, out, inp]], 0.0);
                    a_buf[tap] = Complex::new(a_coeffs[[section, tap, out, inp]], 0.0);
                }
                fft.process(&mut b_buf);
                fft.process(&mut a_buf);
                for bin in 0..n_bins {
                    b_response[[section, bin, out, inp]] = b_buf[bin];
                    a_response[[section, bin, out, inp]] = a_buf[bin];
                }
            }
        }
    }

    // Cascade sections: H = prod_k B_k / A_k.
    let mut h = Array3::<Complex<f64>>::from_elem((n_bins, OUT_CH, IN_CH), Complex::new(1.0, 0.0));
    for section in 0..N_SECTIONS {
        for bin in 0..n_bins {
            for out in 0..OUT_CH {
                for inp in 0..IN_CH {
                    h[[bin, out, inp]] *=
                        b_response[[section, bin, out, inp]] / a_response[[section, bin, out, inp]];
                }
            }
        }
    }

    h
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::rng();

    // Target response.
    let h_target = target_response(&mut rng);
    let n_bins = NFFT / 2 + 1;

    // Input is an impulse in the time domain: shape (batch=1, time=NFFT, channels=1).
    let mut input_time = Array3::<Complex<f64>>::zeros((1, NFFT, 1));
    input_time[[0, 0, 0]] = Complex::new(1.0, 0.0);
    let input = DiffTensor::from_array(input_time.into_dyn());

    // Target magnitude: |H_target * FFT(input)| = |H_target| because input is an impulse.
    let mut target_magnitude = Array3::<Complex<f64>>::zeros((1, n_bins, OUT_CH));
    for bin in 0..n_bins {
        for out in 0..OUT_CH {
            target_magnitude[[0, bin, out]] = Complex::new(h_target[[bin, out, 0]].norm(), 0.0);
        }
    }
    let target = DiffTensor::from_array(target_magnitude.into_dyn());

    // Build Shell(FFT -> Biquad -> Magnitude).
    let fft = Fft::with_channels(NFFT, IN_CH);
    let mut biquad = Biquad::new(
        NFFT,
        FS,
        N_SECTIONS,
        BiquadFilterType::Highpass,
        OUT_CH,
        IN_CH,
        ALIAS_DECAY_DB,
    )?;

    // Initialize to a sensible starting point: fc = fs/4, gain = 0 dB.
    {
        let mut params = biquad.parameters_mut();
        let param = &mut params[0];
        let mut view = param
            .view_mut()
            .into_shape_with_order((N_SECTIONS, 2, OUT_CH, IN_CH))
            .expect("biquad param shape");
        for section in 0..N_SECTIONS {
            for out in 0..OUT_CH {
                for inp in 0..IN_CH {
                    // raw fc = 0.0 -> sigmoid(0) = 0.5 -> fc = fs/4
                    view[[section, 0, out, inp]] = 0.0;
                    // raw gain = 1.0 -> 0 dB
                    view[[section, 1, out, inp]] = 1.0;
                }
            }
        }
    }

    let magnitude = Magnitude::new(NFFT, OUT_CH);
    let mut shell = Shell::new(Box::new(fft), Box::new(biquad), Box::new(magnitude))?;

    let sgd = Sgd::new(LEARNING_RATE);

    let initial_loss = {
        let pred = shell.forward(&input)?;
        mse_loss(&pred, &target)?
    };

    for epoch in 0..EPOCHS {
        shell.zero_grad();
        let pred = shell.forward(&input)?;
        let loss = mse_loss(&pred, &target)?;
        if epoch % 20 == 0 {
            println!("epoch {:3}: loss = {:.6e}", epoch, loss);
        }
        let grad = mse_loss_backward(&pred, &target)?;
        shell.backward(&input, &pred, &grad)?;
        let grads_owned: Vec<ndarray::ArrayD<f64>> =
            shell.gradients().iter().map(|g| (*g).clone()).collect();
        let mut params = shell.parameters_mut();
        let grads: Vec<&ndarray::ArrayD<f64>> = grads_owned.iter().collect();
        sgd.step(&mut params[..], &grads[..])?;
    }

    let final_loss = {
        let pred = shell.forward(&input)?;
        mse_loss(&pred, &target)?
    };

    println!("initial loss = {:.6e}", initial_loss);
    println!("final loss   = {:.6e}", final_loss);

    if final_loss >= initial_loss {
        eprintln!("warning: optimization did not reduce the loss");
    }

    Ok(())
}

//! FDN magnitude-response matching example.

use math_audio_autodiff::{
    delay::ParallelDelay,
    fft::Fft,
    gain::{Gain, Magnitude},
    loss::{mse_loss, mse_loss_backward},
    matrix::{Matrix, MatrixType},
    module::DiffModule,
    optim::Sgd,
    recursion::Recursion,
    system::{Series, Shell},
    tensor::DiffTensor,
};
use ndarray::{Array3, ArrayD};
use num_complex::Complex;

const NFFT: usize = 16_384;
const N: usize = 4;
const EPOCHS: usize = 100;
const LR: f64 = 1e-2;

fn build_fdn_core(
    delay_values: &[f64],
    feedback_gain: f64,
) -> Result<Box<dyn DiffModule<f64>>, Box<dyn std::error::Error>> {
    assert_eq!(delay_values.len(), N);
    let mut input_gain = Gain::new(NFFT, N, 1)?;
    input_gain.param.fill(1.0);
    let mut output_gain = Gain::new(NFFT, 1, N)?;
    output_gain.param.fill(1.0);

    let mut delays = ParallelDelay::new(NFFT, N, 1.0)?;
    for (i, &tau) in delay_values.iter().enumerate() {
        // softplus(raw) + 1.0 ≈ tau for large tau.
        delays.param[[i]] = tau;
    }

    let mut feedback_gain_module = Gain::new(NFFT, N, N)?;
    for i in 0..N {
        feedback_gain_module.param[[i, i]] = feedback_gain;
    }
    let feedback = Series::new(vec![
        Box::new(Matrix::new(NFFT, N, N, MatrixType::Orthogonal)?),
        Box::new(feedback_gain_module),
    ])?;
    let recursion = Recursion::new(Box::new(delays), Box::new(feedback))?;
    let fdn = Series::new(vec![
        Box::new(input_gain),
        Box::new(recursion),
        Box::new(output_gain),
    ])?;
    Ok(Box::new(fdn))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Target FDN: known delays and small feedback gain.
    let target_delays = vec![100.0, 173.0, 251.0, 337.0];
    let target_core = build_fdn_core(&target_delays, 0.3)?;
    let fft = Fft::with_channels(NFFT, 1);
    let magnitude = Magnitude::new(NFFT, 1);
    let target_shell = Shell::new(
        Box::new(fft.clone()),
        target_core,
        Box::new(magnitude.clone()),
    )?;

    let mut input = DiffTensor::from_array(Array3::<Complex<f64>>::zeros((1, NFFT, 1)).into_dyn());
    input.data[[0, 0, 0]] = Complex::new(1.0, 0.0);
    let target = target_shell.forward(&input)?;

    // Optimizable FDN: randomized delays and feedback gain.
    let init_delays = vec![80.0, 150.0, 220.0, 310.0];
    let init_core = build_fdn_core(&init_delays, 0.1)?;
    let mut shell = Shell::new(Box::new(fft), init_core, Box::new(magnitude))?;

    let sgd = Sgd::new(LR);

    let initial_loss = mse_loss(&shell.forward(&input)?, &target)?;
    for epoch in 0..EPOCHS {
        shell.zero_grad();
        let pred = shell.forward(&input)?;
        let loss = mse_loss(&pred, &target)?;
        if epoch % 20 == 0 {
            println!("epoch {:3}: loss = {:.6e}", epoch, loss);
        }
        let grad = mse_loss_backward(&pred, &target)?;
        shell.backward(&input, &pred, &grad)?;
        let grads_owned: Vec<ArrayD<f64>> =
            shell.gradients().iter().map(|g| (*g).clone()).collect();
        let mut params = shell.parameters_mut();
        let grads: Vec<&ArrayD<f64>> = grads_owned.iter().collect();
        sgd.step(&mut params[..], &grads[..])?;
    }

    let final_loss = mse_loss(&shell.forward(&input)?, &target)?;
    println!("initial loss = {:.6e}", initial_loss);
    println!("final loss   = {:.6e}", final_loss);
    if final_loss >= initial_loss {
        eprintln!("warning: optimization did not reduce the loss");
    }
    Ok(())
}

//! FDN with direct-path magnitude-response matching example.
//!
//! Demonstrates the `Parallel` combiner by building a feedback delay network
//! with a separate direct (dry) path. A target FDN + direct gain is matched by
//! an optimizable copy.

use math_audio_autodiff::{
    delay::ParallelDelay,
    fft::Fft,
    gain::{Gain, Magnitude},
    loss::{mse_loss, mse_loss_backward},
    matrix::{Matrix, MatrixType},
    module::DiffModule,
    optim::Sgd,
    recursion::Recursion,
    signals::{SignalType, signal_gallery},
    system::{Parallel, Series, Shell},
};
use ndarray::ArrayD;

const FS: f64 = 48_000.0;
const NFFT: usize = 16_384;
const N: usize = 4;
const EPOCHS: usize = 100;
const LR: f64 = 1e-2;

fn build_fdn_branch(
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

    Ok(Box::new(Series::new(vec![
        Box::new(input_gain),
        Box::new(recursion),
        Box::new(output_gain),
    ])?))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Target FDN: known delays and small feedback gain, plus a direct path.
    let target_delays = vec![100.0, 173.0, 251.0, 337.0];
    let target_core = build_fdn_branch(&target_delays, 0.3)?;

    let mut target_direct = Gain::new(NFFT, 1, 1)?;
    target_direct.param.fill(0.5);

    let target_parallel = Parallel::new(target_core, Box::new(target_direct))?;

    let fft = Fft::with_channels(NFFT, 1);
    let magnitude = Magnitude::new(NFFT, 1);
    let target_shell = Shell::new(
        Box::new(fft.clone()),
        Box::new(target_parallel),
        Box::new(magnitude.clone()),
    )?;

    let input = signal_gallery(SignalType::Impulse, NFFT, 1, FS);
    let target = target_shell.forward(&input)?;

    // Optimizable FDN: randomized delays and feedback gain, plus learnable direct path.
    let init_delays = vec![80.0, 150.0, 220.0, 310.0];
    let init_core = build_fdn_branch(&init_delays, 0.1)?;

    let mut init_direct = Gain::new(NFFT, 1, 1)?;
    init_direct.param.fill(0.1);

    let init_parallel = Parallel::new(init_core, Box::new(init_direct))?;

    let mut shell = Shell::new(Box::new(fft), Box::new(init_parallel), Box::new(magnitude))?;

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

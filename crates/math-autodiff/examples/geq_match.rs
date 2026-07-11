//! Graphic EQ magnitude-response matching example.
//!
//! Builds a target `GraphicEq` with known per-band linear gains, then fits a
//! second `GraphicEq` initialized to flat unity gains by minimizing the MSE
//! between the two magnitude responses.

use math_audio_autodiff::{
    fft::Fft,
    gain::Magnitude,
    iir::geq::GraphicEq,
    loss::{mse_loss, mse_loss_backward},
    module::DiffModule,
    optim::Sgd,
    signals::{SignalType, signal_gallery},
    system::Series,
};
use ndarray::ArrayD;

const FS: f64 = 48_000.0;
const NFFT: usize = 8_192;
const N_BANDS: usize = 8;
const N_CHANNELS: usize = 1;
const EPOCHS: usize = 100;
const LEARNING_RATE: f64 = 1e-2;
const ALIAS_DECAY_DB: f64 = 30.0;

fn set_geq_gains(geq: &mut GraphicEq, gains: &[f64]) {
    assert_eq!(gains.len(), geq.n_bands);
    for (band, &gain) in gains.iter().enumerate().take(geq.n_bands) {
        for ch in 0..geq.n_channels {
            geq.param[[band, ch]] = gain;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Input: a time-domain impulse.
    let input = signal_gallery(SignalType::Impulse, NFFT, N_CHANNELS, FS);

    // Target GEQ: per-band linear gains.
    let target_gains = [0.5, 1.5, 0.8, 2.0, 1.2, 0.6, 1.8, 1.0];
    let mut target_geq = GraphicEq::new(NFFT, FS, N_BANDS, N_CHANNELS, ALIAS_DECAY_DB)?;
    set_geq_gains(&mut target_geq, &target_gains);

    let fft = Fft::with_channels(NFFT, N_CHANNELS);
    let magnitude = Magnitude::new(NFFT, N_CHANNELS);
    let target_chain = Series::new(vec![
        Box::new(fft.clone()),
        Box::new(target_geq),
        Box::new(magnitude.clone()),
    ])?;
    let target = target_chain.forward(&input)?;

    // Optimizable GEQ: initialized to flat unity gains.
    let mut init_geq = GraphicEq::new(NFFT, FS, N_BANDS, N_CHANNELS, ALIAS_DECAY_DB)?;
    set_geq_gains(&mut init_geq, &[1.0; N_BANDS]);

    let mut model = Series::new(vec![Box::new(fft), Box::new(init_geq), Box::new(magnitude)])?;

    let sgd = Sgd::new(LEARNING_RATE);

    let initial_loss = mse_loss(&model.forward(&input)?, &target)?;
    for epoch in 0..EPOCHS {
        model.zero_grad();
        let pred = model.forward(&input)?;
        let loss = mse_loss(&pred, &target)?;
        if epoch % 20 == 0 {
            println!("epoch {:3}: loss = {:.6e}", epoch, loss);
        }
        let grad = mse_loss_backward(&pred, &target)?;
        model.backward(&input, &pred, &grad)?;
        let grads_owned: Vec<ArrayD<f64>> =
            model.gradients().iter().map(|g| (*g).clone()).collect();
        let mut params = model.parameters_mut();
        let grads: Vec<&ArrayD<f64>> = grads_owned.iter().collect();
        sgd.step(&mut params[..], &grads[..])?;
    }

    let final_loss = mse_loss(&model.forward(&input)?, &target)?;
    println!("initial loss = {:.6e}", initial_loss);
    println!("final loss   = {:.6e}", final_loss);
    if final_loss >= initial_loss {
        eprintln!("warning: optimization did not reduce the loss");
    }

    Ok(())
}

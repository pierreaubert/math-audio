//! SVF peak-filter magnitude-response matching example.
//!
//! Builds a target `SvFilter` with known `fc`, `R`, and `gain_db`, then fits a
//! second `SvFilter` initialized to different parameters by minimizing the MSE
//! between the two magnitude responses.

use math_audio_autodiff::{
    fft::Fft,
    gain::Magnitude,
    iir::svf::{SvFilter, SvfType},
    loss::{mse_loss, mse_loss_backward},
    module::DiffModule,
    optim::Sgd,
    signals::{SignalType, signal_gallery},
    system::Series,
};
use ndarray::ArrayD;

const FS: f64 = 48_000.0;
const NFFT: usize = 8_192;
const N_CHANNELS: usize = 1;
const EPOCHS: usize = 300;
const LEARNING_RATE: f64 = 1e2;
const ALIAS_DECAY_DB: f64 = 30.0;

fn set_svf_params(filter: &mut SvFilter, fc_hz: f64, r: f64, gain_db: f64) {
    filter.param.fill(0.0);
    let n_out = filter.n_out;
    let n_in = filter.n_in;
    for out in 0..n_out {
        for inp in 0..n_in {
            filter.param[[0, 0, out, inp]] = fc_hz;
            filter.param[[0, 1, out, inp]] = r;
            filter.param[[0, 2, out, inp]] = gain_db;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Input: a time-domain impulse.
    let input = signal_gallery(SignalType::Impulse, NFFT, N_CHANNELS, FS);

    // Target SVF: known peak filter.
    let mut target_svf = SvFilter::new(
        NFFT,
        FS,
        N_CHANNELS,
        N_CHANNELS,
        SvfType::Peak,
        ALIAS_DECAY_DB,
    )?;
    set_svf_params(&mut target_svf, 1_000.0, 1.0 / 2.0, 6.0);

    let fft = Fft::with_channels(NFFT, N_CHANNELS);
    let magnitude = Magnitude::new(NFFT, N_CHANNELS);
    let target_chain = Series::new(vec![
        Box::new(fft.clone()),
        Box::new(target_svf),
        Box::new(magnitude.clone()),
    ])?;
    let target = target_chain.forward(&input)?;

    // Optimizable SVF: initialized to different parameters.
    let mut init_svf = SvFilter::new(
        NFFT,
        FS,
        N_CHANNELS,
        N_CHANNELS,
        SvfType::Peak,
        ALIAS_DECAY_DB,
    )?;
    set_svf_params(&mut init_svf, 700.0, 1.0, 2.0);

    let mut model = Series::new(vec![Box::new(fft), Box::new(init_svf), Box::new(magnitude)])?;

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

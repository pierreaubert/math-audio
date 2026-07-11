//! Parametric EQ magnitude-response matching example.
//!
//! Builds a target `ParametricEq` with known frequencies, Qs, and gains, then
//! fits a second `ParametricEq` initialized to different parameters by
//! minimizing the MSE between the two magnitude responses.

use math_audio_autodiff::{
    fft::Fft,
    gain::Magnitude,
    iir::peq::{ParametricEq, PeqBandType},
    loss::{mse_loss, mse_loss_backward},
    module::DiffModule,
    optim::Sgd,
    system::Series,
    tensor::DiffTensor,
};
use ndarray::{Array3, ArrayD};
use num_complex::Complex;

const FS: f64 = 48_000.0;
const NFFT: usize = 8_192;
const N_SECTIONS: usize = 3;
const N_CHANNELS: usize = 1;
const EPOCHS: usize = 100;
const LEARNING_RATE: f64 = 10.0;
const ALIAS_DECAY_DB: f64 = 30.0;

/// Convert a physical cutoff frequency to the raw `sigmoid` parameter.
fn fc_to_raw(fc: f64, fs: f64) -> f64 {
    let half_fs = fs / 2.0;
    (fc / (half_fs - fc)).ln()
}

/// Convert a physical quality factor to the raw `exp` parameter.
fn q_to_raw(q: f64) -> f64 {
    q.ln()
}

/// Set the raw parameters of a parametric EQ from physical values.
fn set_peq_params(
    peq: &mut ParametricEq,
    fcs: &[f64],
    qs: &[f64],
    gains_db: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(fcs.len(), N_SECTIONS);
    assert_eq!(qs.len(), N_SECTIONS);
    assert_eq!(gains_db.len(), N_SECTIONS);

    let mut view = peq
        .param
        .view_mut()
        .into_shape_with_order((N_SECTIONS, 3, N_CHANNELS))?;
    for (section, (&fc, (&q, &gain_db))) in
        fcs.iter().zip(qs.iter().zip(gains_db.iter())).enumerate()
    {
        view[[section, 0, 0]] = fc_to_raw(fc, FS);
        view[[section, 1, 0]] = q_to_raw(q);
        view[[section, 2, 0]] = gain_db;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Input: a time-domain impulse (batch=1, time=NFFT, channels=1).
    let mut input_time = Array3::<Complex<f64>>::zeros((1, NFFT, N_CHANNELS));
    input_time[[0, 0, 0]] = Complex::new(1.0, 0.0);
    let input = DiffTensor::from_array(input_time.into_dyn());

    // Target PEQ: known frequencies, Qs, and gains.
    let mut target_peq = ParametricEq::new(
        NFFT,
        FS,
        N_SECTIONS,
        N_CHANNELS,
        PeqBandType::Peak,
        ALIAS_DECAY_DB,
    )?;
    set_peq_params(
        &mut target_peq,
        &[200.0, 1_000.0, 5_000.0],
        &[2.0, 1.5, 3.0],
        &[6.0, -4.0, 3.0],
    )?;

    let fft = Fft::with_channels(NFFT, N_CHANNELS);
    let magnitude = Magnitude::new(NFFT, N_CHANNELS);
    let target_chain = Series::new(vec![
        Box::new(fft.clone()),
        Box::new(target_peq),
        Box::new(magnitude.clone()),
    ])?;
    let target = target_chain.forward(&input)?;

    // Optimizable PEQ: initialized to different parameters.
    let mut init_peq = ParametricEq::new(
        NFFT,
        FS,
        N_SECTIONS,
        N_CHANNELS,
        PeqBandType::Peak,
        ALIAS_DECAY_DB,
    )?;
    set_peq_params(
        &mut init_peq,
        &[500.0, 2_000.0, 8_000.0],
        &[1.0, 1.0, 1.0],
        &[0.0, 0.0, 0.0],
    )?;

    let mut model = Series::new(vec![Box::new(fft), Box::new(init_peq), Box::new(magnitude)])?;

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

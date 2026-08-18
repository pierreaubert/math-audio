//! Loss functions for frequency-domain differentiable DSP.

#![allow(
    clippy::cast_precision_loss,
    reason = "tensor length fits exactly in f64 for practical audio buffer sizes"
)]

use num_complex::Complex;

use crate::error::AutodiffError;
use crate::tensor::DiffTensor;

fn validate_loss_inputs(
    pred: &DiffTensor<f64>,
    target: &DiffTensor<f64>,
) -> Result<(), AutodiffError> {
    if pred.data.shape() != target.data.shape() {
        return Err(AutodiffError::Message(format!(
            "loss: prediction shape {:?} does not match target shape {:?}",
            pred.data.shape(),
            target.data.shape()
        )));
    }
    if pred.data.is_empty() {
        return Err(AutodiffError::Message(
            "loss: tensors must not be empty".to_string(),
        ));
    }
    Ok(())
}

/// Mean squared error between two complex tensors.
///
/// `L = (1/N) * sum_i |pred_i - target_i|^2`.
///
/// # Errors
///
/// Returns an error if the tensors have different shapes or are empty.
pub fn mse_loss(pred: &DiffTensor<f64>, target: &DiffTensor<f64>) -> Result<f64, AutodiffError> {
    validate_loss_inputs(pred, target)?;
    let diff = &pred.data - &target.data;
    Ok(diff.iter().map(Complex::norm_sqr).sum::<f64>() / diff.len() as f64)
}

/// Gradient of [`mse_loss`] with respect to `pred`.
///
/// # Errors
///
/// Returns an error if the tensors have different shapes or are empty.
pub fn mse_loss_backward(
    pred: &DiffTensor<f64>,
    target: &DiffTensor<f64>,
) -> Result<DiffTensor<f64>, AutodiffError> {
    validate_loss_inputs(pred, target)?;
    let scale = 2.0 / pred.data.len() as f64;
    let mut grad = ndarray::ArrayD::<Complex<f64>>::zeros(pred.data.raw_dim());
    ndarray::azip!((g in &mut grad, &p in &pred.data, &t in &target.data) {
        *g = (p - t) * scale;
    });
    Ok(DiffTensor::from_array(grad))
}

/// Mean squared error between magnitude spectra.
///
/// `L = (1/N) * sum_i (|pred_i| - |target_i|)^2`.
///
/// # Errors
///
/// Returns an error if the tensors have different shapes or are empty.
pub fn magnitude_mse_loss(
    pred: &DiffTensor<f64>,
    target: &DiffTensor<f64>,
) -> Result<f64, AutodiffError> {
    validate_loss_inputs(pred, target)?;
    Ok(pred
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(p, t)| {
            let diff = p.norm() - t.norm();
            diff * diff
        })
        .sum::<f64>()
        / pred.data.len() as f64)
}

/// Gradient of [`magnitude_mse_loss`] with respect to `pred`.
///
/// For `|pred_i| > 0` the gradient is `(2/N) * (|pred_i| - |target_i|) * pred_i / |pred_i|`,
/// and zero where `|pred_i|` is zero to avoid division by zero.
///
/// # Errors
///
/// Returns an error if the tensors have different shapes or are empty.
pub fn magnitude_mse_loss_backward(
    pred: &DiffTensor<f64>,
    target: &DiffTensor<f64>,
) -> Result<DiffTensor<f64>, AutodiffError> {
    validate_loss_inputs(pred, target)?;
    let scale = 2.0 / pred.data.len() as f64;
    let mut grad = ndarray::ArrayD::<Complex<f64>>::zeros(pred.data.raw_dim());
    ndarray::azip!((g in &mut grad, p in &pred.data, t in &target.data) {
        let mag = p.norm();
        *g = if mag > 0.0 {
            (scale * (mag - t.norm()) / mag) * *p
        } else {
            Complex::new(0.0, 0.0)
        };
    });
    Ok(DiffTensor::from_array(grad))
}

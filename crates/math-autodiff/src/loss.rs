//! Loss functions for frequency-domain differentiable DSP.

#![allow(
    clippy::cast_precision_loss,
    reason = "tensor length fits exactly in f64 for practical audio buffer sizes"
)]

use math_audio_dsp::psychoacoustics::{BARK_BAND_EDGES, critical_bandwidth};
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

/// Default pooling windows (in bins) for [`multi_scale_spectral_loss`].
///
/// Scale 1 is the full-resolution magnitude MSE; wider windows compare
/// progressively coarser spectral envelopes.
pub const DEFAULT_SPECTRAL_SCALES: &[usize] = &[1, 2, 4, 8];

/// Bin center frequencies in Hz for a real spectrum with `n_bins` bins.
///
/// `f[k] = k * sample_rate / nfft`. Used to build perceptual weightings
/// ([`bark_weights`], [`erb_weights`]) for spectra produced by an `nfft`-point
/// transform.
#[must_use]
pub fn bin_frequencies(n_bins: usize, sample_rate: f64, nfft: usize) -> Vec<f64> {
    let nfft = nfft.max(1) as f64;
    (0..n_bins).map(|k| k as f64 * sample_rate / nfft).collect()
}

/// Assign a frequency in Hz to one of the 24 Bark bands (0-23).
///
/// Uses the standard band edges
/// ([`BARK_BAND_EDGES`](math_audio_dsp::psychoacoustics::BARK_BAND_EDGES))
/// from `math-dsp`, matching its `bark_spectrum` assignment; out-of-range
/// inputs clamp to the nearest band.
#[must_use]
pub fn bark_band_index(f: f64) -> usize {
    BARK_BAND_EDGES
        .partition_point(|&edge| edge <= f)
        .saturating_sub(1)
        .min(23)
}

/// Per-bin perceptual weights inversely proportional to critical bandwidth.
///
/// `w(f) = 1 / critical_bandwidth(f)` with `f` clamped to at least 20 Hz,
/// normalized to mean 1. Narrow (low-frequency) Bark bands span few FFT bins,
/// so without weighting a per-bin loss overemphasizes high frequencies; these
/// weights make each Bark band contribute roughly equally. Non-finite or
/// non-positive frequencies get weight 0.
///
/// The bandwidth comes from
/// [`critical_bandwidth`](math_audio_dsp::psychoacoustics::critical_bandwidth)
/// in `math-dsp`.
#[must_use]
pub fn bark_weights(freqs: &[f64]) -> Vec<f64> {
    let mut weights: Vec<f64> = freqs
        .iter()
        .map(|&f| {
            if !f.is_finite() || f <= 0.0 {
                0.0
            } else {
                1.0 / critical_bandwidth(f.max(20.0))
            }
        })
        .collect();
    normalize_weights_mean_one(&mut weights);
    weights
}

/// Equivalent rectangular bandwidth (Glasberg-Moore) in Hz.
///
/// `ERB(f) = 24.7 * (4.37 * f / 1000 + 1)`. `math-dsp` provides no ERB helper,
/// so the closed form lives here.
fn erb_hz(f: f64) -> f64 {
    24.7 * (4.37 * f / 1000.0 + 1.0)
}

/// Per-bin perceptual weights inversely proportional to ERB.
///
/// Same convention as [`bark_weights`]: `w(f) = 1 / ERB(f)` with `f` clamped
/// to at least 20 Hz, normalized to mean 1, weight 0 for non-finite or
/// non-positive frequencies.
#[must_use]
pub fn erb_weights(freqs: &[f64]) -> Vec<f64> {
    let mut weights: Vec<f64> = freqs
        .iter()
        .map(|&f| {
            if !f.is_finite() || f <= 0.0 {
                0.0
            } else {
                1.0 / erb_hz(f.max(20.0))
            }
        })
        .collect();
    normalize_weights_mean_one(&mut weights);
    weights
}

fn normalize_weights_mean_one(weights: &mut [f64]) {
    if weights.is_empty() {
        return;
    }
    let mean = weights.iter().sum::<f64>() / weights.len() as f64;
    if mean > 0.0 && mean.is_finite() {
        for w in weights.iter_mut() {
            *w /= mean;
        }
    }
}

fn validate_loss_weights(
    pred: &DiffTensor<f64>,
    target: &DiffTensor<f64>,
    weights: &[f64],
) -> Result<(), AutodiffError> {
    validate_loss_inputs(pred, target)?;
    if weights.len() != pred.data.len() {
        return Err(AutodiffError::Message(format!(
            "loss: weights length {} does not match tensor length {}",
            weights.len(),
            pred.data.len()
        )));
    }
    if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return Err(AutodiffError::Message(
            "loss: weights must be finite and non-negative".to_string(),
        ));
    }
    Ok(())
}

/// Weighted mean squared error between two complex tensors.
///
/// `L = sum_i w_i * |pred_i - target_i|^2 / sum_i w_i`.
///
/// # Errors
///
/// Returns an error if the tensors have different shapes, are empty, if
/// `weights` has the wrong length or contains negative/non-finite values, or
/// if all weights are zero.
pub fn weighted_mse_loss(
    pred: &DiffTensor<f64>,
    target: &DiffTensor<f64>,
    weights: &[f64],
) -> Result<f64, AutodiffError> {
    validate_loss_weights(pred, target, weights)?;
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return Err(AutodiffError::Message(
            "loss: weights must not all be zero".to_string(),
        ));
    }
    Ok(pred
        .data
        .iter()
        .zip(target.data.iter())
        .zip(weights.iter())
        .map(|((p, t), &w)| w * (p - t).norm_sqr())
        .sum::<f64>()
        / total)
}

/// Gradient of [`weighted_mse_loss`] with respect to `pred`.
///
/// The gradient is `(2 / sum w) * w_i * (pred_i - target_i)`.
///
/// # Errors
///
/// Same conditions as [`weighted_mse_loss`].
pub fn weighted_mse_loss_backward(
    pred: &DiffTensor<f64>,
    target: &DiffTensor<f64>,
    weights: &[f64],
) -> Result<DiffTensor<f64>, AutodiffError> {
    validate_loss_weights(pred, target, weights)?;
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return Err(AutodiffError::Message(
            "loss: weights must not all be zero".to_string(),
        ));
    }
    let weights_array = ndarray::ArrayD::from_shape_vec(pred.data.raw_dim(), weights.to_vec())
        .map_err(|e| AutodiffError::Message(e.to_string()))?;
    let mut grad = ndarray::ArrayD::<Complex<f64>>::zeros(pred.data.raw_dim());
    ndarray::azip!((g in &mut grad, p in &pred.data, t in &target.data, &w in &weights_array) {
        *g = (p - t) * (2.0 * w / total);
    });
    Ok(DiffTensor::from_array(grad))
}

fn validate_loss_freqs(
    pred: &DiffTensor<f64>,
    target: &DiffTensor<f64>,
    freqs: &[f64],
) -> Result<(), AutodiffError> {
    validate_loss_inputs(pred, target)?;
    if freqs.len() != pred.data.len() {
        return Err(AutodiffError::Message(format!(
            "loss: freqs length {} does not match tensor length {}",
            freqs.len(),
            pred.data.len()
        )));
    }
    Ok(())
}

/// Bark-weighted complex MSE: [`weighted_mse_loss`] with [`bark_weights`].
///
/// # Errors
///
/// Returns an error if the tensors have different shapes, are empty, or if
/// `freqs` has the wrong length.
pub fn bark_weighted_loss(
    pred: &DiffTensor<f64>,
    target: &DiffTensor<f64>,
    freqs: &[f64],
) -> Result<f64, AutodiffError> {
    validate_loss_freqs(pred, target, freqs)?;
    weighted_mse_loss(pred, target, &bark_weights(freqs))
}

/// Gradient of [`bark_weighted_loss`] with respect to `pred`.
///
/// # Errors
///
/// Same conditions as [`bark_weighted_loss`].
pub fn bark_weighted_loss_backward(
    pred: &DiffTensor<f64>,
    target: &DiffTensor<f64>,
    freqs: &[f64],
) -> Result<DiffTensor<f64>, AutodiffError> {
    validate_loss_freqs(pred, target, freqs)?;
    weighted_mse_loss_backward(pred, target, &bark_weights(freqs))
}

/// ERB-weighted complex MSE: [`weighted_mse_loss`] with [`erb_weights`].
///
/// # Errors
///
/// Returns an error if the tensors have different shapes, are empty, or if
/// `freqs` has the wrong length.
pub fn erb_weighted_loss(
    pred: &DiffTensor<f64>,
    target: &DiffTensor<f64>,
    freqs: &[f64],
) -> Result<f64, AutodiffError> {
    validate_loss_freqs(pred, target, freqs)?;
    weighted_mse_loss(pred, target, &erb_weights(freqs))
}

/// Gradient of [`erb_weighted_loss`] with respect to `pred`.
///
/// # Errors
///
/// Same conditions as [`erb_weighted_loss`].
pub fn erb_weighted_loss_backward(
    pred: &DiffTensor<f64>,
    target: &DiffTensor<f64>,
    freqs: &[f64],
) -> Result<DiffTensor<f64>, AutodiffError> {
    validate_loss_freqs(pred, target, freqs)?;
    weighted_mse_loss_backward(pred, target, &erb_weights(freqs))
}

/// Spectral convergence loss between magnitude spectra.
///
/// `L = || |pred| - |target| ||_2 / max(||target||_2, eps)`.
///
/// # Errors
///
/// Returns an error if the tensors have different shapes, are empty, or if
/// `eps` is not positive and finite.
pub fn spectral_convergence_loss(
    pred: &DiffTensor<f64>,
    target: &DiffTensor<f64>,
    eps: f64,
) -> Result<f64, AutodiffError> {
    validate_loss_inputs(pred, target)?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(AutodiffError::Message(
            "loss: eps must be positive and finite".to_string(),
        ));
    }
    let num_sq: f64 = pred
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(p, t)| {
            let diff = p.norm() - t.norm();
            diff * diff
        })
        .sum();
    let den_sq: f64 = target.data.iter().map(Complex::norm_sqr).sum();
    Ok(num_sq.sqrt() / den_sq.sqrt().max(eps))
}

/// Gradient of [`spectral_convergence_loss`] with respect to `pred`.
///
/// Zero where `|pred_i|` is zero or where the prediction already matches the
/// target (`num == 0`).
///
/// # Errors
///
/// Same conditions as [`spectral_convergence_loss`].
pub fn spectral_convergence_loss_backward(
    pred: &DiffTensor<f64>,
    target: &DiffTensor<f64>,
    eps: f64,
) -> Result<DiffTensor<f64>, AutodiffError> {
    validate_loss_inputs(pred, target)?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(AutodiffError::Message(
            "loss: eps must be positive and finite".to_string(),
        ));
    }
    let num_sq: f64 = pred
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(p, t)| {
            let diff = p.norm() - t.norm();
            diff * diff
        })
        .sum();
    let den_sq: f64 = target.data.iter().map(Complex::norm_sqr).sum();
    let num = num_sq.sqrt();
    let den = den_sq.sqrt().max(eps);
    let mut grad = ndarray::ArrayD::<Complex<f64>>::zeros(pred.data.raw_dim());
    if num > 0.0 {
        ndarray::azip!((g in &mut grad, p in &pred.data, t in &target.data) {
            let mag = p.norm();
            *g = if mag > 0.0 {
                *p * ((mag - t.norm()) / (mag * num * den))
            } else {
                Complex::new(0.0, 0.0)
            };
        });
    }
    Ok(DiffTensor::from_array(grad))
}

/// Mean squared error between log magnitude spectra.
///
/// `L = (1/N) * sum_i (log(|pred_i| + eps) - log(|target_i| + eps))^2`.
///
/// # Errors
///
/// Returns an error if the tensors have different shapes, are empty, or if
/// `eps` is not positive and finite.
pub fn log_magnitude_loss(
    pred: &DiffTensor<f64>,
    target: &DiffTensor<f64>,
    eps: f64,
) -> Result<f64, AutodiffError> {
    validate_loss_inputs(pred, target)?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(AutodiffError::Message(
            "loss: eps must be positive and finite".to_string(),
        ));
    }
    Ok(pred
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(p, t)| {
            let diff = (p.norm() + eps).ln() - (t.norm() + eps).ln();
            diff * diff
        })
        .sum::<f64>()
        / pred.data.len() as f64)
}

/// Gradient of [`log_magnitude_loss`] with respect to `pred`.
///
/// Zero where `|pred_i|` is zero to avoid division by zero.
///
/// # Errors
///
/// Same conditions as [`log_magnitude_loss`].
pub fn log_magnitude_loss_backward(
    pred: &DiffTensor<f64>,
    target: &DiffTensor<f64>,
    eps: f64,
) -> Result<DiffTensor<f64>, AutodiffError> {
    validate_loss_inputs(pred, target)?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(AutodiffError::Message(
            "loss: eps must be positive and finite".to_string(),
        ));
    }
    let scale = 2.0 / pred.data.len() as f64;
    let mut grad = ndarray::ArrayD::<Complex<f64>>::zeros(pred.data.raw_dim());
    ndarray::azip!((g in &mut grad, p in &pred.data, t in &target.data) {
        let mag = p.norm();
        *g = if mag > 0.0 {
            let diff = (mag + eps).ln() - (t.norm() + eps).ln();
            *p * (scale * diff / (mag * (mag + eps)))
        } else {
            Complex::new(0.0, 0.0)
        };
    });
    Ok(DiffTensor::from_array(grad))
}

fn validate_spectral_scales(
    pred: &DiffTensor<f64>,
    target: &DiffTensor<f64>,
    scales: &[usize],
) -> Result<(), AutodiffError> {
    validate_loss_inputs(pred, target)?;
    if scales.is_empty() {
        return Err(AutodiffError::Message(
            "loss: scales must not be empty".to_string(),
        ));
    }
    if scales.contains(&0) {
        return Err(AutodiffError::Message(
            "loss: scales must all be at least 1".to_string(),
        ));
    }
    Ok(())
}

/// Multi-scale magnitude-spectrum MSE.
///
/// For each pooling window `s` in `scales`, the magnitude spectra are averaged
/// over non-overlapping pools of `s` bins (the final pool may be shorter) and
/// the MSE of the pooled magnitudes is computed; the loss is the mean over
/// scales. Scale 1 reproduces [`magnitude_mse_loss`]. Coarse scales compare
/// spectral envelopes while fine scales preserve detail, mirroring the intent
/// of multi-resolution STFT losses for fixed-grid frequency-domain tensors.
///
/// # Errors
///
/// Returns an error if the tensors have different shapes, are empty, if
/// `scales` is empty, or if any scale is zero.
pub fn multi_scale_spectral_loss(
    pred: &DiffTensor<f64>,
    target: &DiffTensor<f64>,
    scales: &[usize],
) -> Result<f64, AutodiffError> {
    validate_spectral_scales(pred, target, scales)?;
    let pred_mag: Vec<f64> = pred.data.iter().map(|c| c.norm()).collect();
    let target_mag: Vec<f64> = target.data.iter().map(|c| c.norm()).collect();
    let total: f64 = scales
        .iter()
        .map(|&s| pooled_magnitude_mse(&pred_mag, &target_mag, s))
        .sum();
    Ok(total / scales.len() as f64)
}

fn pooled_magnitude_mse(pred_mag: &[f64], target_mag: &[f64], window: usize) -> f64 {
    let n_pools = pred_mag.len().div_ceil(window);
    let mut sum = 0.0;
    for pool in 0..n_pools {
        let start = pool * window;
        let end = (start + window).min(pred_mag.len());
        let len = (end - start) as f64;
        let mean_pred: f64 = pred_mag[start..end].iter().sum::<f64>() / len;
        let mean_target: f64 = target_mag[start..end].iter().sum::<f64>() / len;
        let diff = mean_pred - mean_target;
        sum += diff * diff;
    }
    sum / n_pools as f64
}

/// Gradient of [`multi_scale_spectral_loss`] with respect to `pred`.
///
/// Each bin's gradient is the mean over scales of its pool's mean-magnitude
/// error, backpropagated through `|pred_i|`; zero where `|pred_i|` is zero.
///
/// # Errors
///
/// Same conditions as [`multi_scale_spectral_loss`].
pub fn multi_scale_spectral_loss_backward(
    pred: &DiffTensor<f64>,
    target: &DiffTensor<f64>,
    scales: &[usize],
) -> Result<DiffTensor<f64>, AutodiffError> {
    validate_spectral_scales(pred, target, scales)?;
    let n = pred.data.len();
    let pred_mag: Vec<f64> = pred.data.iter().map(|c| c.norm()).collect();
    let target_mag: Vec<f64> = target.data.iter().map(|c| c.norm()).collect();
    // Accumulate per-bin dL/d|pred_i| over scales, then chain through the
    // magnitude with one pass over the tensors.
    let mut dmag = vec![0.0_f64; n];
    let n_scales = scales.len() as f64;
    for &window in scales {
        let n_pools = n.div_ceil(window);
        for pool in 0..n_pools {
            let start = pool * window;
            let end = (start + window).min(n);
            let len = (end - start) as f64;
            let mean_pred: f64 = pred_mag[start..end].iter().sum::<f64>() / len;
            let mean_target: f64 = target_mag[start..end].iter().sum::<f64>() / len;
            let coeff = 2.0 * (mean_pred - mean_target) / (n_scales * n_pools as f64 * len);
            for slot in &mut dmag[start..end] {
                *slot += coeff;
            }
        }
    }
    let dmag_array = ndarray::ArrayD::from_shape_vec(pred.data.raw_dim(), dmag)
        .map_err(|e| AutodiffError::Message(e.to_string()))?;
    let mut grad = ndarray::ArrayD::<Complex<f64>>::zeros(pred.data.raw_dim());
    ndarray::azip!((g in &mut grad, p in &pred.data, &d in &dmag_array) {
        let mag = p.norm();
        *g = if mag > 0.0 { *p * (d / mag) } else { Complex::new(0.0, 0.0) };
    });
    Ok(DiffTensor::from_array(grad))
}

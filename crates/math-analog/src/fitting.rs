//! Offline, feature-gated fitting of bounded parallel-Hammerstein models.
//!
//! The fitting path is intentionally separate from the realtime dependency
//! graph.  It allocates FFT workspaces, runs global/local optimization, and
//! emits immutable coefficients with capture provenance.
//!
//! # Capture-fit-validate loop
//!
//! 1. **Capture.** Record one or more stimulus/response pairs at a fixed
//!    sample rate with [`FitCapture::new`]. Keep at least one fully held-out
//!    record (a different level, frequency set, or programme) and assemble
//!    the split with [`FitDataset::from_captures`]; validation records are
//!    never passed to either optimizer. The legacy [`FitDataset::new`]
//!    tail-split constructor is kept for single-recording callers.
//! 2. **Fit (DE global start).** [`fit_hammerstein`] minimizes the spectral
//!    magnitude/phase objective over the training captures with differential
//!    evolution (`math-optimisation`), using gain bounds `[-2, 2]` and cutoff
//!    bounds `[0, sample_rate / 2]` per branch, `FitOptions::de_max_iterations`
//!    / `de_population_size` budget, and `FitOptions::seed` for
//!    reproducibility.
//! 3. **Refine (LM bridge).** The DE optimum becomes `x0` for the bounded
//!    `math-optimisation` Levenberg-Marquardt solver over the *same* residual
//!    vector (`LMConfigBuilder`: `maxiter = lm_max_iterations`, `tol = 1e-10`,
//!    `atol = 1e-14`, finite-difference Jacobian `epsilon = 1e-8`). LM
//!    interpolates Gauss-Newton/gradient-descent steps subject to the same
//!    bounds; its final `fun` is reported as `lm_objective`.
//! 4. **Validate.** [`FitQualityReport`] scores time-domain RMS and spectral
//!    RMS separately on the training captures and on the held-out captures,
//!    plus sample/capture counts and both optimizer objectives.
//!
//! # Numeric acceptance thresholds
//!
//! The synthetic round-trip test
//! (`frozen_coefficients_round_trip_a_synthetic_capture`) pins these gates;
//! use [`FitQualityReport::meets_acceptance_criteria`] for the same check on
//! real captures:
//!
//! - `fit_rms < 0.01` and `held_out_rms < 0.01` (time domain, unity-scale
//!   signals). A held-out RMS far above the training RMS means the model
//!   overfit: add captures, reduce branch orders, or re-tune DE/LM budgets.
//! - `fit_spectral_rms` finite (and, for a healthy fit, of the same order as
//!   the training objective per bin).
//! - Recovered branch gains within `0.15` of the reference and branch
//!   cutoffs within `1000 Hz` (synthetic-identity check only; for measured
//!   hardware, provenance + held-out error replace parameter closeness).
//!
//! A fit that fails these gates must not be checked in as a hardware model:
//! keep the coefficients, provenance, and quality report together and
//! re-capture before claiming the device.

use ndarray::Array1;
use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use math_audio_optimisation::{
    DEConfigBuilder, LMConfigBuilder, differential_evolution, levenberg_marquardt,
};

use crate::{AnalogError, AnalogProcessor, HammersteinBranch, HammersteinModel, ProcessSpec};

const FIT_TOOL_VERSION: &str = "math-analog-fitting-0.2";

#[derive(Debug, Error)]
pub enum FittingError {
    #[error("stimulus and response must have the same non-zero length")]
    InvalidCapture,
    #[error("at least one training capture is required")]
    EmptyTrainingSet,
    #[error("capture contains a non-finite sample")]
    NonFiniteCapture,
    #[error("sample rate must be finite and greater than zero")]
    InvalidSampleRate,
    #[error("at least one and at most five branch orders are required")]
    InvalidBranchOrders,
    #[error("validation fraction must be in [0, 0.5)")]
    InvalidValidationFraction,
    #[error("spectral objective weights must be finite and non-negative")]
    InvalidObjectiveWeights,
    #[error("model construction failed: {0}")]
    Model(#[from] AnalogError),
    #[error("differential evolution failed: {0}")]
    DifferentialEvolution(String),
    #[error("Levenberg–Marquardt failed: {0}")]
    LevenbergMarquardt(String),
}

/// One captured stimulus/response pair.  Captures are kept as separate
/// records so validation can represent a held-out level, frequency, or
/// programme rather than merely the tail of one training recording.
#[derive(Debug, Clone)]
pub struct FitCapture {
    stimulus: Vec<f32>,
    response: Vec<f32>,
}

impl FitCapture {
    pub fn new(stimulus: Vec<f32>, response: Vec<f32>) -> Result<Self, FittingError> {
        if stimulus.is_empty() || stimulus.len() != response.len() {
            return Err(FittingError::InvalidCapture);
        }
        if stimulus
            .iter()
            .chain(&response)
            .any(|sample| !sample.is_finite())
        {
            return Err(FittingError::NonFiniteCapture);
        }
        Ok(Self { stimulus, response })
    }

    pub fn stimulus(&self) -> &[f32] {
        &self.stimulus
    }

    pub fn response(&self) -> &[f32] {
        &self.response
    }

    pub fn len(&self) -> usize {
        self.stimulus.len()
    }
}

/// Captured records split into optimizer-visible training material and fully
/// held-out validation material.  The legacy [`Self::new`] constructor keeps
/// its final-fraction behavior for callers with one continuous capture.
#[derive(Debug, Clone)]
pub struct FitDataset {
    training: Vec<FitCapture>,
    validation: Vec<FitCapture>,
    sample_rate: f32,
}

impl FitDataset {
    pub fn new(
        stimulus: Vec<f32>,
        response: Vec<f32>,
        sample_rate: f32,
        validation_fraction: f32,
    ) -> Result<Self, FittingError> {
        if !(0.0..0.5).contains(&validation_fraction) {
            return Err(FittingError::InvalidValidationFraction);
        }
        let capture = FitCapture::new(stimulus, response)?;
        let held_out = (capture.len() as f32 * validation_fraction).round() as usize;
        let split = capture.len().saturating_sub(held_out).max(1);
        let training = FitCapture::new(
            capture.stimulus[..split].to_vec(),
            capture.response[..split].to_vec(),
        )?;
        let validation = if split < capture.len() {
            vec![FitCapture::new(
                capture.stimulus[split..].to_vec(),
                capture.response[split..].to_vec(),
            )?]
        } else {
            Vec::new()
        };
        Self::from_captures(vec![training], validation, sample_rate)
    }

    /// Construct a dataset with independent training and held-out records.
    /// Validation records are never passed to DE or LM.
    pub fn from_captures(
        training: Vec<FitCapture>,
        validation: Vec<FitCapture>,
        sample_rate: f32,
    ) -> Result<Self, FittingError> {
        if training.is_empty() {
            return Err(FittingError::EmptyTrainingSet);
        }
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(FittingError::InvalidSampleRate);
        }
        if training.iter().chain(&validation).any(|capture| {
            capture.stimulus.len() != capture.response.len()
                || capture.stimulus.is_empty()
                || capture
                    .stimulus
                    .iter()
                    .chain(&capture.response)
                    .any(|sample| !sample.is_finite())
        }) {
            return Err(FittingError::InvalidCapture);
        }
        Ok(Self {
            training,
            validation,
            sample_rate,
        })
    }

    pub fn training_captures(&self) -> &[FitCapture] {
        &self.training
    }

    pub fn validation_captures(&self) -> &[FitCapture] {
        &self.validation
    }

    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    pub fn training_samples(&self) -> usize {
        self.training.iter().map(FitCapture::len).sum()
    }

    pub fn validation_samples(&self) -> usize {
        self.validation.iter().map(FitCapture::len).sum()
    }
}

/// Bounded fitting budget and spectral-objective weights.
#[derive(Debug, Clone)]
pub struct FitOptions {
    pub branch_orders: Vec<usize>,
    pub de_max_iterations: usize,
    pub de_population_size: usize,
    pub lm_max_iterations: usize,
    pub seed: u64,
    pub magnitude_weight: f64,
    pub phase_weight: f64,
    pub target_description: String,
    pub capture_chain: String,
    pub fit_date: String,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            branch_orders: vec![1, 2, 3],
            de_max_iterations: 30,
            de_population_size: 8,
            lm_max_iterations: 40,
            seed: 0xA11A_2026,
            magnitude_weight: 1.0,
            phase_weight: 0.25,
            target_description: "unidentified captured device".to_string(),
            capture_chain: "unspecified capture chain".to_string(),
            fit_date: "unspecified".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FittedHammersteinBranch {
    pub order: usize,
    pub gain: f32,
    pub cutoff_hz: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitProvenance {
    pub target_description: String,
    pub capture_chain: String,
    pub sample_rate: f32,
    pub capture_hash: u64,
    pub fit_date: String,
    pub fit_tool_version: String,
}

/// Immutable, auditable fitted coefficients.  The structure has no optimizer
/// state and can be serialized as a checked-in model record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrozenHammersteinCoefficients {
    branches: Vec<FittedHammersteinBranch>,
    provenance: FitProvenance,
}

impl FrozenHammersteinCoefficients {
    pub fn branches(&self) -> &[FittedHammersteinBranch] {
        &self.branches
    }

    pub fn provenance(&self) -> &FitProvenance {
        &self.provenance
    }

    pub fn to_model(&self) -> Result<HammersteinModel, FittingError> {
        let branches = self
            .branches
            .iter()
            .map(|branch| HammersteinBranch::new(branch.order, branch.gain, branch.cutoff_hz))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(HammersteinModel::with_branches(&branches)?)
    }
}

/// Time-domain RMS acceptance gate for [`FitQualityReport`] (unity-scale signals).
pub const FIT_RMS_ACCEPTANCE: f32 = 0.01;
/// Held-out time-domain RMS acceptance gate; see [`FIT_RMS_ACCEPTANCE`].
pub const HELD_OUT_RMS_ACCEPTANCE: f32 = 0.01;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitQualityReport {
    pub fit_rms: f32,
    pub held_out_rms: f32,
    pub fit_spectral_rms: f32,
    pub held_out_spectral_rms: f32,
    pub fit_samples: usize,
    pub held_out_samples: usize,
    pub fit_captures: usize,
    pub held_out_captures: usize,
    pub de_objective: f64,
    pub lm_objective: f64,
}

impl FitQualityReport {
    /// Numeric acceptance gate documented in the module docs: training and
    /// held-out time-domain RMS below [`FIT_RMS_ACCEPTANCE`] /
    /// [`HELD_OUT_RMS_ACCEPTANCE`] with finite spectral error.
    pub fn meets_acceptance_criteria(&self) -> bool {
        self.fit_rms < FIT_RMS_ACCEPTANCE
            && self.held_out_rms < HELD_OUT_RMS_ACCEPTANCE
            && self.fit_spectral_rms.is_finite()
            && self.held_out_spectral_rms.is_finite()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FitReport {
    pub coefficients: FrozenHammersteinCoefficients,
    pub quality: FitQualityReport,
}

/// Fit branch gains and one-pole cutoffs using a spectral magnitude/phase
/// residual.  DE supplies a global starting point; LM refines the same
/// residual vector.  Only `dataset.training_captures()` enter either solver.
pub fn fit_hammerstein(
    dataset: &FitDataset,
    options: &FitOptions,
) -> Result<FitReport, FittingError> {
    validate_orders(&options.branch_orders)?;
    if !options.magnitude_weight.is_finite()
        || options.magnitude_weight < 0.0
        || !options.phase_weight.is_finite()
        || options.phase_weight < 0.0
        || options.magnitude_weight == 0.0 && options.phase_weight == 0.0
    {
        return Err(FittingError::InvalidObjectiveWeights);
    }
    let bounds = parameter_bounds(dataset.sample_rate, options.branch_orders.len());
    let orders = options.branch_orders.clone();
    let objective = |parameters: &Array1<f64>| {
        spectral_objective(
            parameters.as_slice().unwrap_or(&[]),
            &orders,
            dataset,
            options.magnitude_weight,
            options.phase_weight,
        )
    };
    let de_config = DEConfigBuilder::new()
        .maxiter(options.de_max_iterations)
        .popsize(options.de_population_size.max(4))
        .seed(options.seed)
        .tol(1e-8)
        .atol(1e-12)
        .build()
        .map_err(|error| FittingError::DifferentialEvolution(error.to_string()))?;
    let de = differential_evolution(&objective, &bounds, de_config)
        .map_err(|error| FittingError::DifferentialEvolution(error.to_string()))?;

    let residual = |parameters: &Array1<f64>| {
        spectral_residuals_for_dataset(
            parameters.as_slice().unwrap_or(&[]),
            &orders,
            dataset,
            options.magnitude_weight,
            options.phase_weight,
        )
        .into()
    };
    let lm_config = LMConfigBuilder::new()
        .x0(de.x.clone())
        .maxiter(options.lm_max_iterations)
        .tol(1e-10)
        .atol(1e-14)
        .build();
    let lm = levenberg_marquardt(&residual, &bounds, lm_config)
        .map_err(|error| FittingError::LevenbergMarquardt(error.to_string()))?;
    let final_parameters = lm.x.as_slice().unwrap_or(&[]);
    let coefficients = FrozenHammersteinCoefficients {
        branches: build_branches(final_parameters, &orders),
        provenance: FitProvenance {
            target_description: options.target_description.clone(),
            capture_chain: options.capture_chain.clone(),
            sample_rate: dataset.sample_rate,
            capture_hash: capture_hash(dataset),
            fit_date: options.fit_date.clone(),
            fit_tool_version: FIT_TOOL_VERSION.to_string(),
        },
    };
    let quality = evaluate_quality(
        dataset,
        &coefficients,
        options.magnitude_weight,
        options.phase_weight,
        de.fun,
        lm.fun,
    )?;
    Ok(FitReport {
        coefficients,
        quality,
    })
}

fn validate_orders(orders: &[usize]) -> Result<(), FittingError> {
    if orders.is_empty() || orders.len() > 5 || orders.iter().any(|order| !(1..=5).contains(order))
    {
        Err(FittingError::InvalidBranchOrders)
    } else {
        Ok(())
    }
}

fn parameter_bounds(sample_rate: f32, branches: usize) -> Vec<(f64, f64)> {
    let nyquist = f64::from(sample_rate) * 0.5;
    (0..branches)
        .flat_map(|_| [(-2.0, 2.0), (0.0, nyquist)])
        .collect()
}

fn build_branches(parameters: &[f64], orders: &[usize]) -> Vec<FittedHammersteinBranch> {
    orders
        .iter()
        .enumerate()
        .map(|(index, &order)| FittedHammersteinBranch {
            order,
            gain: parameters.get(index * 2).copied().unwrap_or(0.0) as f32,
            cutoff_hz: parameters.get(index * 2 + 1).copied().unwrap_or(0.0) as f32,
        })
        .collect()
}

fn render_parameters(
    parameters: &[f64],
    orders: &[usize],
    capture: &FitCapture,
    sample_rate: f32,
) -> Result<Vec<f32>, FittingError> {
    let branches = build_branches(parameters, orders)
        .into_iter()
        .map(|branch| HammersteinBranch::new(branch.order, branch.gain, branch.cutoff_hz))
        .collect::<Result<Vec<_>, _>>()?;
    let mut model = HammersteinModel::with_branches(&branches)?;
    model.prepare(ProcessSpec::new(sample_rate, 1, capture.len()))?;
    let mut rendered = capture.stimulus.clone();
    model.process_interleaved(&mut rendered, capture.len())?;
    Ok(rendered)
}

fn render_model(
    template: &HammersteinModel,
    capture: &FitCapture,
    sample_rate: f32,
) -> Result<Vec<f32>, FittingError> {
    let mut model = template.clone();
    model.prepare(ProcessSpec::new(sample_rate, 1, capture.len()))?;
    let mut rendered = capture.stimulus.clone();
    model.process_interleaved(&mut rendered, capture.len())?;
    Ok(rendered)
}

fn spectral_objective(
    parameters: &[f64],
    orders: &[usize],
    dataset: &FitDataset,
    magnitude_weight: f64,
    phase_weight: f64,
) -> f64 {
    spectral_residuals_for_dataset(parameters, orders, dataset, magnitude_weight, phase_weight)
        .iter()
        .map(|residual| residual * residual)
        .sum::<f64>()
}

fn spectral_residuals_for_dataset(
    parameters: &[f64],
    orders: &[usize],
    dataset: &FitDataset,
    magnitude_weight: f64,
    phase_weight: f64,
) -> Vec<f64> {
    let mut residuals = Vec::new();
    for capture in &dataset.training {
        let rendered = match render_parameters(parameters, orders, capture, dataset.sample_rate) {
            Ok(rendered) => rendered,
            Err(_) => return invalid_residuals(dataset),
        };
        residuals.extend(spectral_residuals(
            &rendered,
            &capture.response,
            magnitude_weight,
            phase_weight,
        ));
    }
    if residuals.is_empty() {
        vec![f64::MAX.sqrt()]
    } else {
        residuals
    }
}

fn invalid_residuals(dataset: &FitDataset) -> Vec<f64> {
    let count = dataset
        .training
        .iter()
        .map(|capture| {
            spectral_residuals(&vec![0.0; capture.len()], &capture.response, 1.0, 0.25).len()
        })
        .sum::<usize>();
    vec![1.0e6; count.max(1)]
}

/// Return paired magnitude and phase residuals for the occupied positive
/// frequency bins.  The threshold avoids unstable phase residuals in empty
/// bins while retaining all captured harmonics and programme energy.
fn spectral_residuals(
    rendered: &[f32],
    expected: &[f32],
    magnitude_weight: f64,
    phase_weight: f64,
) -> Vec<f64> {
    if rendered.len() != expected.len() || rendered.is_empty() {
        return vec![f64::MAX.sqrt()];
    }
    let fft_size = rendered.len().next_power_of_two().max(2);
    let mut actual = rendered
        .iter()
        .map(|sample| Complex::new(f64::from(*sample), 0.0))
        .collect::<Vec<_>>();
    let mut target = expected
        .iter()
        .map(|sample| Complex::new(f64::from(*sample), 0.0))
        .collect::<Vec<_>>();
    actual.resize(fft_size, Complex::default());
    target.resize(fft_size, Complex::default());
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(fft_size);
    fft.process(&mut actual);
    fft.process(&mut target);
    let scale = fft_size as f64;
    let target_peak = target
        .iter()
        .take(fft_size / 2 + 1)
        .map(|value| value.norm() / scale)
        .fold(0.0_f64, f64::max);
    let threshold = (target_peak * 1e-7).max(1e-12);
    let mut residuals = Vec::new();
    for bin in 1..=fft_size / 2 {
        let actual_value = actual[bin] / scale;
        let target_value = target[bin] / scale;
        let actual_magnitude = actual_value.norm();
        let target_magnitude = target_value.norm();
        if actual_magnitude <= threshold && target_magnitude <= threshold {
            continue;
        }
        residuals.push((actual_magnitude - target_magnitude) * magnitude_weight);
        let phase_error = if actual_magnitude > threshold && target_magnitude > threshold {
            (actual_value * target_value.conj()).arg()
        } else {
            0.0
        };
        residuals.push(phase_error * target_magnitude.max(actual_magnitude) * phase_weight);
    }
    if residuals.is_empty() {
        vec![f64::MAX.sqrt()]
    } else {
        residuals
    }
}

fn evaluate_quality(
    dataset: &FitDataset,
    coefficients: &FrozenHammersteinCoefficients,
    magnitude_weight: f64,
    phase_weight: f64,
    de_objective: f64,
    lm_objective: f64,
) -> Result<FitQualityReport, FittingError> {
    let model = coefficients.to_model()?;
    let (fit_rms, fit_spectral_rms) = quality_for_captures(
        &model,
        &dataset.training,
        dataset.sample_rate,
        magnitude_weight,
        phase_weight,
    )?;
    let (held_out_rms, held_out_spectral_rms) = quality_for_captures(
        &model,
        &dataset.validation,
        dataset.sample_rate,
        magnitude_weight,
        phase_weight,
    )?;
    Ok(FitQualityReport {
        fit_rms,
        held_out_rms,
        fit_spectral_rms,
        held_out_spectral_rms,
        fit_samples: dataset.training_samples(),
        held_out_samples: dataset.validation_samples(),
        fit_captures: dataset.training.len(),
        held_out_captures: dataset.validation.len(),
        de_objective,
        lm_objective,
    })
}

fn quality_for_captures(
    model: &HammersteinModel,
    captures: &[FitCapture],
    sample_rate: f32,
    magnitude_weight: f64,
    phase_weight: f64,
) -> Result<(f32, f32), FittingError> {
    if captures.is_empty() {
        return Ok((0.0, 0.0));
    }
    let mut squared_error = 0.0_f64;
    let mut sample_count = 0_usize;
    let mut spectral_squared_error = 0.0_f64;
    let mut spectral_count = 0_usize;
    for capture in captures {
        let rendered = render_model(model, capture, sample_rate)?;
        squared_error += rendered
            .iter()
            .zip(&capture.response)
            .map(|(actual, expected)| f64::from(*actual - *expected).powi(2))
            .sum::<f64>();
        sample_count += rendered.len();
        let residuals =
            spectral_residuals(&rendered, &capture.response, magnitude_weight, phase_weight);
        spectral_squared_error += residuals.iter().map(|value| value * value).sum::<f64>();
        spectral_count += residuals.len();
    }
    Ok((
        (squared_error / sample_count.max(1) as f64).sqrt() as f32,
        (spectral_squared_error / spectral_count.max(1) as f64).sqrt() as f32,
    ))
}

fn capture_hash(dataset: &FitDataset) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for capture in dataset.training.iter().chain(&dataset.validation) {
        for sample in capture.stimulus.iter().chain(&capture.response) {
            for byte in sample.to_bits().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x1000_0000_01b3);
            }
        }
        hash ^= capture.len() as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    for byte in dataset.sample_rate.to_bits().to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render_source(
        branches: &[HammersteinBranch],
        stimulus: &[f32],
        sample_rate: f32,
    ) -> Vec<f32> {
        let mut source = HammersteinModel::with_branches(branches).unwrap();
        source
            .prepare(ProcessSpec::new(sample_rate, 1, stimulus.len()))
            .unwrap();
        let mut response = stimulus.to_vec();
        let frames = response.len();
        source.process_interleaved(&mut response, frames).unwrap();
        response
    }

    #[test]
    fn frozen_coefficients_round_trip_a_synthetic_capture() {
        let branches = [
            HammersteinBranch::new(1, 0.85, 0.0).unwrap(),
            HammersteinBranch::new(2, 0.12, 2_000.0).unwrap(),
        ];
        let stimulus: Vec<f32> = (0..1_024)
            .map(|index| {
                let n = index as f32;
                (0.18 * (n * 0.19).sin() + 0.12 * (n * 0.047).sin())
                    * (0.7 + 0.3 * (n / 97.0).sin())
            })
            .collect();
        let response = render_source(&branches, &stimulus, 48_000.0);
        let validation_stimulus: Vec<f32> = (0..512)
            .map(|index| {
                let n = index as f32;
                0.25 * (n * 0.31).sin() + 0.08 * (n * 0.071).sin()
            })
            .collect();
        let validation_response = render_source(&branches, &validation_stimulus, 48_000.0);
        let dataset = FitDataset::from_captures(
            vec![FitCapture::new(stimulus, response).unwrap()],
            vec![FitCapture::new(validation_stimulus, validation_response).unwrap()],
            48_000.0,
        )
        .unwrap();
        let options = FitOptions {
            branch_orders: vec![1, 2],
            de_max_iterations: 24,
            de_population_size: 8,
            lm_max_iterations: 20,
            ..FitOptions::default()
        };
        let report = fit_hammerstein(&dataset, &options).unwrap();
        assert_eq!(report.coefficients.branches().len(), 2);
        assert_ne!(report.coefficients.provenance().capture_hash, 0);
        assert!(report.quality.fit_rms < FIT_RMS_ACCEPTANCE, "{:?}", report.quality);
        assert!(
            report.quality.held_out_rms < HELD_OUT_RMS_ACCEPTANCE,
            "{:?}",
            report.quality
        );
        assert!(report.quality.meets_acceptance_criteria(), "{:?}", report.quality);
        assert!(report.quality.fit_spectral_rms.is_finite());
        assert_eq!(report.quality.held_out_captures, 1);
        assert_eq!(report.quality.held_out_samples, 512);
        let fitted = report.coefficients.branches();
        assert!((fitted[0].gain - 0.85).abs() < 0.15, "{fitted:?}");
        assert!((fitted[1].gain - 0.12).abs() < 0.15, "{fitted:?}");
        assert!(
            (fitted[1].cutoff_hz - 2_000.0).abs() < 1_000.0,
            "{fitted:?}"
        );
    }

    #[test]
    fn validation_material_never_changes_capture_hash_or_objective() {
        let training = FitCapture::new(vec![0.1; 32], vec![0.2; 32]).unwrap();
        let validation_a = FitCapture::new(vec![0.3; 32], vec![0.4; 32]).unwrap();
        let validation_b = FitCapture::new(vec![0.3; 32], vec![0.9; 32]).unwrap();
        let dataset_a =
            FitDataset::from_captures(vec![training.clone()], vec![validation_a], 48_000.0)
                .unwrap();
        let dataset_b =
            FitDataset::from_captures(vec![training], vec![validation_b], 48_000.0).unwrap();
        let parameters = [0.8, 0.0];
        let orders = [1];
        assert_eq!(
            spectral_objective(&parameters, &orders, &dataset_a, 1.0, 0.25),
            spectral_objective(&parameters, &orders, &dataset_b, 1.0, 0.25)
        );
        assert_ne!(capture_hash(&dataset_a), capture_hash(&dataset_b));
    }
}

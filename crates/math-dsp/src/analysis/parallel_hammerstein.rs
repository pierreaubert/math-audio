//! Bounded finite-memory polynomial convolution fitting for offline analysis.
//!
//! This module builds an input-only training design, estimates its numerical
//! rank and condition, and fits a finite parallel-Hammerstein model with LSQR.
//! It does not certify rank, effective physical support, noise bounds, calibration,
//! harmonic availability, or measurement quality. A returned numerical candidate
//! must not be promoted to a measured distortion result without those independent
//! checks. The dense QR step is a bounded increment; long high-rate captures need
//! a scalable factor builder before this API can cover the full recording domain.
//! The preflight estimates visible dense matrices, factor/operator vectors and a
//! fixed workspace reserve. It excludes borrowed input arrays, allocator/runtime
//! overhead and FFT planner internals, so it is not an RSS guarantee. QR and SVD
//! run synchronously until their bounded library calls return.
//!
//! The order scales are norms of the concatenated training input powers. The
//! input-only design is divided by those scales before QR, then fitted taps are
//! mapped back to the original input amplitude. Capture outputs are not read until
//! the input-only factor and its numerical diagnostics have been built.

use super::lsqr::{LeastSquaresOperator, LsqrOptions, LsqrResult, LsqrStop, solve_lsqr};
use super::polynomial_convolution::PolynomialConvolutionOperator;
use super::right_preconditioner::UpperTriangularRightOperator;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::backtrace::Backtrace;
use std::fmt::{Display, Formatter};

/// Highest polynomial order covered by the retained ESS control matrix.
const ORDER_COUNT: usize = 5;
/// Original run05 normalized train and held-out residual gate.
const NORMALIZED_RESIDUAL_LIMIT: f64 = 1.0e-8;
/// Original run05 undamped LSQR tolerances.
const LSQR_TOLERANCE: f64 = 1.0e-13;
/// Original run05 LSQR iteration limit.
const LSQR_MAX_ITERATIONS: usize = 10_000;
/// Original run05 estimated condition limit; this remains a numerical estimate.
const CONDITION_ESTIMATE_LIMIT: f64 = 1.0e8;
/// Original run05 1.5 GiB RSS cap, reused as an estimated-allocation ceiling.
/// This is not a bound on total process RSS or allocations inside external libraries.
pub const PARALLEL_HAMMERSTEIN_MAX_WORKING_BYTES: usize = 1_536 * 1024 * 1024;
/// Fixed extra workspace allowance; it is not an allocator or RSS guarantee.
const FIXED_WORKSPACE_RESERVE_BYTES: usize = 256 * 1024 * 1024;
/// Retained input/record count ceiling from the original A04 resource contract.
const MAX_RECORDS: usize = 16;
/// Retained finite support ceiling from the original A04 resource contract.
const MAX_SUPPORT_SAMPLES: usize = 4_096;
/// Retained combined decoded sample ceiling from the original A04 resource contract.
const MAX_DECODED_SAMPLES: usize = 50_000_000;
/// Hard cap for the transform buffers built by the polynomial convolution operator.
const MAX_FFT_SAMPLES: usize = 1_048_576;

/// Request limits for a finite parallel-Hammerstein fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParallelHammersteinFitOptions {
    /// Independently declared sample rate shared by all reference and capture records.
    pub sample_rate_hz: u32,
    /// Caller-declared finite response horizon, in samples.
    pub support_samples: usize,
    /// Exact trailing guard length after the declared convolution support.
    pub guard_samples: usize,
}

/// Numerical assessment of the input-only design; it is not a rank certificate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum DesignNumericalStatus {
    /// The SVD converged and its estimated rank and condition pass the numeric gates.
    EstimateWithinLimits,
    /// The SVD estimates fewer independent columns than the model requires.
    EstimatedRankDeficient,
    /// The SVD condition estimate exceeds the unchanged run05 limit.
    EstimatedConditionExceeded,
    /// The bounded SVD iteration budget did not converge.
    SvdDidNotConverge,
}

/// SVD diagnostics for the scaled, input-only finite-design factor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignNumericalDiagnostics {
    /// Number of stacked training rows.
    pub rows: usize,
    /// Number of order-major finite-kernel columns.
    pub columns: usize,
    /// SHA-256 over exact ordered reference bytes and finite-design settings.
    /// Signed zero is preserved as a distinct input bit pattern.
    pub input_design_sha256: String,
    /// Numerical rank estimate using max(rows, columns) times machine epsilon.
    pub estimated_rank: Option<usize>,
    /// Estimated largest singular value of the triangular factor.
    pub sigma_max_estimate: Option<f64>,
    /// Estimated smallest singular value of the triangular factor.
    pub sigma_min_estimate: Option<f64>,
    /// Estimated two-norm condition number of the triangular factor.
    pub condition_estimate: Option<f64>,
    /// Relative singular-value threshold used for the numerical rank estimate.
    pub relative_rank_threshold: Option<f64>,
    /// Whether the numeric-only design checks pass.
    pub status: DesignNumericalStatus,
    /// Explicitly states that this implementation has no finite-design certificate.
    pub finite_design_qualification: QualificationStatus,
}

/// Evidence limitation that prevents a numerical fit from being a metrology result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualificationStatus {
    /// This implementation reports numerical estimates, not an outward-rounded certificate.
    NotCertified,
    /// The support length came from the caller; this function cannot verify physical decay.
    CallerDeclaredOnly,
    /// No independent noise bound was provided to this math API.
    NotProvided,
    /// Acquisition quality, calibration and routing were outside this math API.
    NotEvaluated,
    /// This API returns finite kernels, not harmonic transfer or THD metrics.
    NotProduced,
}

/// A borrowed captured output and its independently declared sample rate.
#[derive(Debug, Clone, Copy)]
pub struct CapturedTrainingOutput<'a> {
    /// Recorded samples corresponding to the training references in the same order.
    pub samples: &'a [f64],
    /// Independently declared recording sample rate.
    pub sample_rate_hz: u32,
}

/// One training reference and its independently declared sample rate.
#[derive(Debug, Clone, Copy)]
pub struct TrainingInputReference<'a> {
    /// Exact sample array used as the input for this training capture.
    pub samples: &'a [f64],
    /// Independently declared source sample rate.
    pub sample_rate_hz: u32,
}

/// One held-out reference/capture pair with independent rate declarations.
#[derive(Debug, Clone, Copy)]
pub struct HeldOutRecord<'a> {
    /// Exact input/reference samples for this held-out take.
    pub reference: &'a [f64],
    /// Independently declared reference sample rate.
    pub reference_sample_rate_hz: u32,
    /// Recorded held-out samples.
    pub capture: &'a [f64],
    /// Independently declared capture sample rate.
    pub capture_sample_rate_hz: u32,
}

/// Input-only design prepared before any training capture values are read.
pub struct PreparedParallelHammersteinDesign<'a> {
    references: Vec<&'a [f64]>,
    options: ParallelHammersteinFitOptions,
    input_samples: usize,
    output_samples: usize,
    rows: usize,
    columns: usize,
    order_scales: [f64; ORDER_COUNT],
    factor_row_major: Vec<f64>,
    diagnostics: DesignNumericalDiagnostics,
    preflight: DesignResourceEstimate,
}

impl std::fmt::Debug for PreparedParallelHammersteinDesign<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedParallelHammersteinDesign")
            .field("training_records", &self.references.len())
            .field("sample_rate_hz", &self.options.sample_rate_hz)
            .field("input_samples", &self.input_samples)
            .field("output_samples", &self.output_samples)
            .field("support_samples", &self.options.support_samples)
            .field("guard_samples", &self.options.guard_samples)
            .field("diagnostics", &self.diagnostics)
            .field("preflight", &self.preflight)
            .finish_non_exhaustive()
    }
}

impl PreparedParallelHammersteinDesign<'_> {
    /// Return input-only design dimensions and the bounded factorization preflight.
    #[must_use]
    pub fn resource_estimate(&self) -> &DesignResourceEstimate {
        &self.preflight
    }
}

/// Matrix and factor sizes used by the checked dense-design preflight.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignResourceEstimate {
    /// Dense column-major design matrix bytes.
    pub design_matrix_bytes: usize,
    /// Square triangular factor bytes.
    pub triangular_factor_bytes: usize,
    /// Peak estimate for QR's in-place matrix and extracted triangular factor.
    pub qr_phase_bytes: usize,
    /// Peak estimate for retained factor, SVD matrix and bidiagonal vectors.
    pub svd_phase_bytes: usize,
    /// Fixed extra workspace allowance; this is not an allocator or RSS guarantee.
    pub fixed_workspace_reserve_bytes: usize,
    /// Conservative estimated maximum of QR and SVD phase allocations.
    pub preflight_peak_bytes: usize,
    /// Estimated-allocation ceiling applied by this API.
    pub working_set_limit_bytes: usize,
}

/// Stop or quality gate that prevented a numerical candidate from being returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitUnavailableReason {
    /// Input-only SVD did not converge within its bounded iteration budget.
    DesignSvdDidNotConverge,
    /// Input-only numerical rank estimate is below the required column count.
    DesignRankEstimateInsufficient,
    /// Input-only condition estimate exceeds the unchanged run05 numeric limit.
    DesignConditionEstimateExceeded,
    /// LSQR was cancelled by the caller.
    Cancelled,
    /// LSQR did not stop on either requested residual tolerance.
    SolverDidNotConverge,
    /// One or more training residuals exceed the unchanged run05 limit.
    TrainingResidualExceeded,
    /// One or more held-out residuals exceed the unchanged run05 limit.
    HeldOutResidualExceeded,
}

/// Residual diagnostics for a single training or held-out record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitRecordDiagnostics {
    /// Record position in the corresponding input list.
    pub record_index: usize,
    /// Whole-record RMSE divided by the centered RMS of the captured record.
    pub normalized_rmse: f64,
    /// Ungated trailing-guard RMS divided by the centered captured-record RMS.
    pub guard_rms_normalized: f64,
    /// Whether both unchanged run05 residual gates pass.
    pub residual_gates_passed: bool,
}

/// LSQR termination information; condition is an estimate, not a certificate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SolverDiagnostics {
    /// Solver termination reason.
    pub stop: FitSolverStop,
    /// Number of completed bidiagonal updates.
    pub iterations: usize,
    /// Norm of the actual residual for the preconditioned matrix-free operator.
    pub residual_norm: f64,
    /// Norm of the actual normal residual when it was evaluated.
    pub normal_residual_norm: Option<f64>,
    /// LSQR recurrence condition estimate, not a bound.
    pub condition_estimate: Option<f64>,
}

/// Stable serialized LSQR termination spelling for the polynomial fit report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitSolverStop {
    /// Actual residual met the requested threshold.
    ResidualTolerance,
    /// Actual normal residual met the requested threshold.
    NormalResidualTolerance,
    /// Diagnostic condition estimate reached a configured stop.
    ConditionLimit,
    /// Iteration budget was exhausted.
    IterationLimit,
    /// Caller requested cancellation.
    Cancelled,
    /// Bidiagonalization broke down without meeting a tolerance.
    Breakdown,
}

/// Diagnostics common to a numerical candidate and an unavailable result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolynomialFitDiagnostics {
    /// Numerical input-only design analysis.
    pub design: DesignNumericalDiagnostics,
    /// Solver outcome, absent when fitting was refused before LSQR.
    pub solver: Option<SolverDiagnostics>,
    /// Training-record fit measurements.
    pub training: Vec<FitRecordDiagnostics>,
    /// Held-out-record fit measurements.
    pub held_out: Vec<FitRecordDiagnostics>,
    /// Fixed training and held-out numerical gates.
    pub normalized_residual_limit: f64,
}

/// A finite-kernel fit that passed the numerical-only gates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParallelHammersteinCandidate {
    /// Candidate method revision.
    pub method_version: u32,
    /// Independently declared sample rate.
    pub sample_rate_hz: u32,
    /// Input reference length, excluding convolution tail and guard.
    pub input_samples: usize,
    /// Output length, including the declared convolution tail and guard.
    pub output_samples: usize,
    /// Caller-declared finite response horizon.
    pub support_samples: usize,
    /// Exact trailing guard length.
    pub guard_samples: usize,
    /// Fitted taps, order-major from first through fifth order.
    pub taps_order_major: Vec<f64>,
    /// Fit and input-design diagnostics.
    pub diagnostics: PolynomialFitDiagnostics,
    /// Qualification limits that remain unresolved by this implementation.
    pub qualification: [QualificationStatus; 5],
}

/// Fit diagnostics with a typed reason for refusing a numerical candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitUnavailable {
    /// Specific numerical gate or stop that prevented a candidate.
    pub reason: FitUnavailableReason,
    /// Design, solver and residual diagnostics observed before refusal.
    pub diagnostics: PolynomialFitDiagnostics,
}

/// Result of bounded fitting, kept separate from physical qualification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", content = "result", rename_all = "snake_case")]
pub enum ParallelHammersteinOutcome {
    /// Numerical candidate only; it is not a certified or measured THD result.
    NumericalOnly(ParallelHammersteinCandidate),
    /// A model or residual gate refused the candidate.
    Unavailable(FitUnavailable),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ErrorKind {
    InvalidRequest,
    NonFiniteInput,
    ResourceLimit,
    Allocation,
    Factorization,
    Cancelled,
}

/// Input or resource error from finite polynomial fitting.
#[derive(Debug)]
pub struct PolynomialFitError {
    kind: ErrorKind,
    message: &'static str,
    backtrace: Backtrace,
}

impl PolynomialFitError {
    fn new(kind: ErrorKind, message: &'static str) -> Self {
        Self {
            kind,
            message,
            backtrace: Backtrace::capture(),
        }
    }

    /// Return whether this error came from a declared storage limit.
    #[must_use]
    pub fn is_resource_limit(&self) -> bool {
        self.kind == ErrorKind::ResourceLimit
    }

    /// Return whether preparation was stopped at a documented boundary.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.kind == ErrorKind::Cancelled
    }
}

impl Display for PolynomialFitError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\n{}", self.message, self.backtrace)
    }
}

impl std::error::Error for PolynomialFitError {}

/// Prepare the input-only finite design before the capture outputs are examined.
///
/// The references are the exact sample arrays that generated the captures. Their
/// routing, stimulus law, calibration, and physical support provenance remain the
/// caller's responsibility. Diagnostics include a content identity computed from
/// exact ordered input bytes and the design settings. The factor and SVD are numerical diagnostics;
/// a passing estimate does not certify full rank or a physical response model.
/// QR and SVD are synchronous and cannot be interrupted while their library calls
/// are in progress. Run this bounded offline operation on a worker thread.
///
/// # Errors
/// Rejects invalid rates/shapes, non-finite input,
/// overflow, resource-limit failure,
/// allocation failure, cancellation between major phases, or QR/SVD numerical failure.
pub fn prepare_parallel_hammerstein_design<'a>(
    references: &[TrainingInputReference<'a>],
    options: ParallelHammersteinFitOptions,
    mut cancelled: impl FnMut() -> bool,
) -> Result<PreparedParallelHammersteinDesign<'a>, PolynomialFitError> {
    validate_options(options)?;
    if references.is_empty() || references.len() > MAX_RECORDS {
        return Err(PolynomialFitError::new(
            ErrorKind::InvalidRequest,
            "training reference count is outside the supported range",
        ));
    }
    if cancelled() {
        return Err(PolynomialFitError::new(
            ErrorKind::Cancelled,
            "polynomial design preparation was cancelled",
        ));
    }
    if references.iter().any(|reference| {
        reference.sample_rate_hz == 0 || reference.sample_rate_hz != options.sample_rate_hz
    }) {
        return Err(PolynomialFitError::new(
            ErrorKind::InvalidRequest,
            "training reference sample-rate declarations differ",
        ));
    }
    let input_samples = references[0].samples.len();
    if input_samples == 0
        || references
            .iter()
            .any(|reference| reference.samples.len() != input_samples)
    {
        return Err(PolynomialFitError::new(
            ErrorKind::InvalidRequest,
            "training reference lengths must be equal and nonzero",
        ));
    }
    let total_input_samples = input_samples.checked_mul(references.len()).ok_or_else(|| {
        PolynomialFitError::new(
            ErrorKind::ResourceLimit,
            "training input sample count overflow",
        )
    })?;
    if total_input_samples > MAX_DECODED_SAMPLES {
        return Err(PolynomialFitError::new(
            ErrorKind::ResourceLimit,
            "training input sample count exceeds the A04 decoded-sample limit",
        ));
    }
    if references
        .iter()
        .any(|input| input.samples.iter().any(|sample| !sample.is_finite()))
    {
        return Err(PolynomialFitError::new(
            ErrorKind::NonFiniteInput,
            "training reference contains a non-finite sample",
        ));
    }
    let full_convolution_samples = input_samples
        .checked_add(options.support_samples - 1)
        .ok_or_else(|| {
            PolynomialFitError::new(ErrorKind::ResourceLimit, "convolution length overflow")
        })?;
    let output_samples = full_convolution_samples
        .checked_add(options.guard_samples)
        .ok_or_else(|| {
            PolynomialFitError::new(ErrorKind::ResourceLimit, "guard length overflow")
        })?;
    if full_convolution_samples
        .checked_next_power_of_two()
        .is_none_or(|fft_length| fft_length > MAX_FFT_SAMPLES)
    {
        return Err(PolynomialFitError::new(
            ErrorKind::ResourceLimit,
            "polynomial input FFT exceeds the configured sample limit",
        ));
    }
    let rows = output_samples
        .checked_mul(references.len())
        .ok_or_else(|| {
            PolynomialFitError::new(ErrorKind::ResourceLimit, "design row count overflow")
        })?;
    let columns = ORDER_COUNT
        .checked_mul(options.support_samples)
        .ok_or_else(|| {
            PolynomialFitError::new(ErrorKind::ResourceLimit, "design column count overflow")
        })?;
    if rows < columns {
        return Err(PolynomialFitError::new(
            ErrorKind::InvalidRequest,
            "training design has fewer rows than finite-kernel coefficients",
        ));
    }
    let preflight = estimate_design_resources(rows, columns)?;
    if preflight.preflight_peak_bytes > PARALLEL_HAMMERSTEIN_MAX_WORKING_BYTES {
        return Err(PolynomialFitError::new(
            ErrorKind::ResourceLimit,
            "input-only factorization exceeds the original 1.5 GiB working-set ceiling",
        ));
    }

    let order_scales = compute_order_scales(references)?;
    let matrix_entries = rows.checked_mul(columns).ok_or_else(|| {
        PolynomialFitError::new(ErrorKind::ResourceLimit, "design matrix size overflow")
    })?;
    let mut matrix_values = Vec::new();
    matrix_values
        .try_reserve_exact(matrix_entries)
        .map_err(|_| {
            PolynomialFitError::new(ErrorKind::Allocation, "design matrix allocation failed")
        })?;
    let capacity_bytes = matrix_values
        .capacity()
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| {
            PolynomialFitError::new(ErrorKind::ResourceLimit, "design matrix capacity overflow")
        })?;
    if capacity_bytes
        .checked_add(preflight.preflight_peak_bytes - preflight.design_matrix_bytes)
        .is_none_or(|bytes| bytes > PARALLEL_HAMMERSTEIN_MAX_WORKING_BYTES)
    {
        return Err(PolynomialFitError::new(
            ErrorKind::ResourceLimit,
            "actual design vector capacity exceeds the working-set ceiling",
        ));
    }
    matrix_values.resize(matrix_entries, 0.0);
    fill_design_matrix(
        &mut matrix_values,
        references,
        input_samples,
        output_samples,
        rows,
        columns,
        options.support_samples,
        &order_scales,
    )?;
    if matrix_values.iter().any(|value| !value.is_finite()) {
        return Err(PolynomialFitError::new(
            ErrorKind::NonFiniteInput,
            "scaled input-only design contains a non-finite value",
        ));
    }
    if cancelled() {
        return Err(PolynomialFitError::new(
            ErrorKind::Cancelled,
            "polynomial design preparation was cancelled before QR",
        ));
    }
    let matrix = DMatrix::from_vec(rows, columns, matrix_values);
    let r_matrix = matrix.qr().unpack_r();
    if cancelled() {
        return Err(PolynomialFitError::new(
            ErrorKind::Cancelled,
            "polynomial design preparation was cancelled after QR",
        ));
    }
    let mut factor_row_major = Vec::new();
    let factor_entries = columns.checked_mul(columns).ok_or_else(|| {
        PolynomialFitError::new(ErrorKind::ResourceLimit, "triangular factor size overflow")
    })?;
    factor_row_major
        .try_reserve_exact(factor_entries)
        .map_err(|_| {
            PolynomialFitError::new(ErrorKind::Allocation, "triangular factor allocation failed")
        })?;
    let actual_factor_bytes = bytes_for(factor_row_major.capacity())?;
    let actual_svd_phase_bytes = bytes_for(r_matrix.len())?
        .checked_add(actual_factor_bytes)
        .and_then(|bytes| bytes.checked_add(bytes_for(columns.checked_mul(8)?).ok()?))
        .and_then(|bytes| bytes.checked_add(FIXED_WORKSPACE_RESERVE_BYTES))
        .ok_or_else(|| {
            PolynomialFitError::new(
                ErrorKind::ResourceLimit,
                "actual SVD phase estimate overflow",
            )
        })?;
    if actual_svd_phase_bytes > PARALLEL_HAMMERSTEIN_MAX_WORKING_BYTES {
        return Err(PolynomialFitError::new(
            ErrorKind::ResourceLimit,
            "actual triangular factor capacity exceeds the SVD working-set ceiling",
        ));
    }
    for row in 0..columns {
        for column in 0..columns {
            factor_row_major.push(r_matrix[(row, column)]);
        }
    }
    if factor_row_major.iter().any(|value| !value.is_finite()) {
        return Err(PolynomialFitError::new(
            ErrorKind::Factorization,
            "QR factor contains a non-finite value",
        ));
    }
    let svd_iteration_limit = columns.checked_mul(100).ok_or_else(|| {
        PolynomialFitError::new(ErrorKind::ResourceLimit, "SVD iteration limit overflow")
    })?;
    let svd = nalgebra::linalg::SVD::try_new_unordered(
        r_matrix,
        false,
        false,
        f64::EPSILON * 5.0,
        svd_iteration_limit,
    );
    let input_design_sha256 =
        input_design_sha256(references, options, input_samples, &order_scales)?;
    let diagnostics = svd.map_or_else(
        || DesignNumericalDiagnostics {
            rows,
            columns,
            input_design_sha256: input_design_sha256.clone(),
            estimated_rank: None,
            sigma_max_estimate: None,
            sigma_min_estimate: None,
            condition_estimate: None,
            relative_rank_threshold: None,
            status: DesignNumericalStatus::SvdDidNotConverge,
            finite_design_qualification: QualificationStatus::NotCertified,
        },
        |svd| {
            let sigma_max = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
            let sigma_min = svd
                .singular_values
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let relative_rank_threshold = (rows.max(columns) as f64) * f64::EPSILON;
            let absolute_rank_threshold = relative_rank_threshold * sigma_max;
            let estimated_rank = svd
                .singular_values
                .iter()
                .filter(|value| **value > absolute_rank_threshold)
                .count();
            let condition = if sigma_min > 0.0 {
                sigma_max / sigma_min
            } else {
                f64::INFINITY
            };
            let status = if estimated_rank < columns {
                DesignNumericalStatus::EstimatedRankDeficient
            } else if !condition.is_finite() || condition > CONDITION_ESTIMATE_LIMIT {
                DesignNumericalStatus::EstimatedConditionExceeded
            } else {
                DesignNumericalStatus::EstimateWithinLimits
            };
            DesignNumericalDiagnostics {
                rows,
                columns,
                input_design_sha256: input_design_sha256.clone(),
                estimated_rank: Some(estimated_rank),
                sigma_max_estimate: sigma_max.is_finite().then_some(sigma_max),
                sigma_min_estimate: sigma_min.is_finite().then_some(sigma_min),
                condition_estimate: condition.is_finite().then_some(condition),
                relative_rank_threshold: relative_rank_threshold
                    .is_finite()
                    .then_some(relative_rank_threshold),
                status,
                finite_design_qualification: QualificationStatus::NotCertified,
            }
        },
    );
    if cancelled() {
        return Err(PolynomialFitError::new(
            ErrorKind::Cancelled,
            "polynomial design preparation was cancelled after SVD",
        ));
    }
    Ok(PreparedParallelHammersteinDesign {
        references: references
            .iter()
            .map(|reference| reference.samples)
            .collect(),
        options,
        input_samples,
        output_samples,
        rows,
        columns,
        order_scales,
        factor_row_major,
        diagnostics,
        preflight,
    })
}

/// Fit captures using a previously frozen input-only design.
///
/// Training output values are first read here, after input scaling, QR and SVD
/// diagnostics have completed. The returned candidate uses the fixed run05 LSQR
/// and residual gates. A passing result still lacks certified conditioning,
/// independently bounded physical support and noise, calibrated acquisition
/// provenance, and harmonic metrics.
///
/// # Errors
/// Rejects capture count/rate/shape errors, non-finite samples, sample/resource
/// limit violations, allocation errors, operator errors, or numerical overflow.
pub fn fit_parallel_hammerstein(
    prepared: PreparedParallelHammersteinDesign<'_>,
    training_captures: &[CapturedTrainingOutput<'_>],
    held_out_records: &[HeldOutRecord<'_>],
    mut cancelled: impl FnMut() -> bool,
) -> Result<ParallelHammersteinOutcome, PolynomialFitError> {
    if training_captures.len() != prepared.references.len()
        || held_out_records.is_empty()
        || training_captures
            .len()
            .checked_add(held_out_records.len())
            .is_none_or(|count| count > MAX_RECORDS)
    {
        return Err(PolynomialFitError::new(
            ErrorKind::InvalidRequest,
            "capture counts must match the design and include held-out records",
        ));
    }
    if training_captures.iter().any(|capture| {
        capture.sample_rate_hz == 0 || capture.sample_rate_hz != prepared.options.sample_rate_hz
    }) || held_out_records.iter().any(|record| {
        record.reference_sample_rate_hz == 0
            || record.capture_sample_rate_hz == 0
            || record.reference_sample_rate_hz != prepared.options.sample_rate_hz
            || record.capture_sample_rate_hz != prepared.options.sample_rate_hz
    }) {
        return Err(PolynomialFitError::new(
            ErrorKind::InvalidRequest,
            "reference and capture sample-rate declarations differ",
        ));
    }
    let diagnostics = prepared.diagnostics.clone();
    let unavailable_reason = match diagnostics.status {
        DesignNumericalStatus::EstimateWithinLimits => None,
        DesignNumericalStatus::EstimatedRankDeficient => {
            Some(FitUnavailableReason::DesignRankEstimateInsufficient)
        }
        DesignNumericalStatus::EstimatedConditionExceeded => {
            Some(FitUnavailableReason::DesignConditionEstimateExceeded)
        }
        DesignNumericalStatus::SvdDidNotConverge => {
            Some(FitUnavailableReason::DesignSvdDidNotConverge)
        }
    };
    if let Some(reason) = unavailable_reason {
        return Ok(ParallelHammersteinOutcome::Unavailable(FitUnavailable {
            reason,
            diagnostics: empty_fit_diagnostics(diagnostics),
        }));
    }
    if cancelled() {
        return Ok(ParallelHammersteinOutcome::Unavailable(FitUnavailable {
            reason: FitUnavailableReason::Cancelled,
            diagnostics: empty_fit_diagnostics(diagnostics),
        }));
    }
    validate_captures(&prepared, training_captures, held_out_records)?;

    let total_samples = total_decoded_samples(&prepared, training_captures, held_out_records)?;
    if total_samples > MAX_DECODED_SAMPLES {
        return Err(PolynomialFitError::new(
            ErrorKind::ResourceLimit,
            "combined reference and capture samples exceed the A04 decoded-sample limit",
        ));
    }
    let solver_bytes = lsqr_buffer_bytes(prepared.rows, prepared.columns)?;
    let rhs_bytes = bytes_for(prepared.rows)?;
    let prediction_bytes = rhs_bytes;
    let taps_bytes = bytes_for(prepared.columns)?;
    let factor_bytes = bytes_for(prepared.factor_row_major.capacity())?;
    let mut used_bytes = FIXED_WORKSPACE_RESERVE_BYTES
        .checked_add(factor_bytes)
        .and_then(|bytes| bytes.checked_add(solver_bytes))
        .and_then(|bytes| bytes.checked_add(rhs_bytes))
        .and_then(|bytes| bytes.checked_add(prediction_bytes))
        .and_then(|bytes| bytes.checked_add(taps_bytes))
        .ok_or_else(|| {
            PolynomialFitError::new(
                ErrorKind::ResourceLimit,
                "fit working-set estimate overflow",
            )
        })?;
    if used_bytes > PARALLEL_HAMMERSTEIN_MAX_WORKING_BYTES {
        return Err(PolynomialFitError::new(
            ErrorKind::ResourceLimit,
            "fit vectors exceed the original 1.5 GiB working-set ceiling",
        ));
    }
    let mut operators = Vec::new();
    operators
        .try_reserve_exact(prepared.references.len())
        .map_err(|_| {
            PolynomialFitError::new(ErrorKind::Allocation, "operator list allocation failed")
        })?;
    let fft_length = prepared
        .input_samples
        .checked_add(prepared.options.support_samples - 1)
        .and_then(usize::checked_next_power_of_two)
        .ok_or_else(|| {
            PolynomialFitError::new(ErrorKind::ResourceLimit, "operator FFT length overflow")
        })?;
    let mut operator_bytes = 0usize;
    for reference in &prepared.references {
        let remaining = PARALLEL_HAMMERSTEIN_MAX_WORKING_BYTES
            .saturating_sub(used_bytes)
            .saturating_sub(operator_bytes);
        let operator = PolynomialConvolutionOperator::new(
            reference,
            ORDER_COUNT,
            prepared.options.support_samples,
            prepared.output_samples,
            fft_length.max(1),
            remaining,
        )
        .map_err(|_| {
            PolynomialFitError::new(
                ErrorKind::ResourceLimit,
                "polynomial convolution operator exceeded its remaining storage budget",
            )
        })?;
        operator_bytes = operator_bytes
            .checked_add(operator.buffer_bytes())
            .ok_or_else(|| {
                PolynomialFitError::new(ErrorKind::ResourceLimit, "operator storage overflow")
            })?;
        operators.push(operator);
    }
    let stacked_workspace_bytes = bytes_for(prepared.columns.checked_mul(2).ok_or_else(|| {
        PolynomialFitError::new(
            ErrorKind::ResourceLimit,
            "stacked coefficient workspace overflow",
        )
    })?)?;
    used_bytes = used_bytes
        .checked_add(operator_bytes)
        .and_then(|bytes| bytes.checked_add(stacked_workspace_bytes))
        .ok_or_else(|| {
            PolynomialFitError::new(ErrorKind::ResourceLimit, "operator working-set overflow")
        })?;
    if used_bytes > PARALLEL_HAMMERSTEIN_MAX_WORKING_BYTES {
        return Err(PolynomialFitError::new(
            ErrorKind::ResourceLimit,
            "stacked operator exceeds the original 1.5 GiB working-set ceiling",
        ));
    }
    let base = StackedPolynomialOperator::new(
        operators,
        prepared.output_samples,
        prepared.columns,
        prepared.order_scales,
        stacked_workspace_bytes,
    )?;
    let remaining = PARALLEL_HAMMERSTEIN_MAX_WORKING_BYTES.saturating_sub(used_bytes);
    let remaining_for_map = remaining.saturating_add(factor_bytes);
    let mut mapped =
        UpperTriangularRightOperator::new(base, prepared.factor_row_major, remaining_for_map)
            .map_err(|_| {
                PolynomialFitError::new(
                    ErrorKind::ResourceLimit,
                    "triangular right-map exceeds its remaining storage budget",
                )
            })?;
    let right_map_bytes = mapped.buffer_bytes();
    let additional_map_bytes = right_map_bytes.checked_sub(factor_bytes).ok_or_else(|| {
        PolynomialFitError::new(
            ErrorKind::ResourceLimit,
            "right-map factor accounting underflow",
        )
    })?;
    used_bytes = used_bytes
        .checked_add(additional_map_bytes)
        .ok_or_else(|| {
            PolynomialFitError::new(ErrorKind::ResourceLimit, "right-map storage overflow")
        })?;
    let remaining = PARALLEL_HAMMERSTEIN_MAX_WORKING_BYTES.saturating_sub(used_bytes);
    let mut target = zeroed_vec(prepared.rows, remaining)?;
    copy_training_captures(&mut target, training_captures, prepared.output_samples);
    let mut predictions = zeroed_vec(prepared.rows, remaining.saturating_sub(rhs_bytes))?;
    let solver_result = solve_lsqr(
        &mut mapped,
        &target,
        LsqrOptions {
            atol: LSQR_TOLERANCE,
            btol: LSQR_TOLERANCE,
            condition_limit: 0.0,
            max_iterations: LSQR_MAX_ITERATIONS,
            max_buffer_bytes: remaining
                .saturating_sub(rhs_bytes)
                .saturating_sub(prediction_bytes)
                .saturating_sub(taps_bytes),
        },
        &mut cancelled,
    )
    .map_err(|_| {
        PolynomialFitError::new(
            ErrorKind::Factorization,
            "LSQR rejected the polynomial fit operator or input values",
        )
    })?;
    let solver_diagnostics = solver_diagnostics(&solver_result);
    let mut fit_diagnostics = PolynomialFitDiagnostics {
        design: diagnostics.clone(),
        solver: Some(solver_diagnostics),
        training: Vec::new(),
        held_out: Vec::new(),
        normalized_residual_limit: NORMALIZED_RESIDUAL_LIMIT,
    };
    if solver_result.stop == LsqrStop::Cancelled || cancelled() {
        return Ok(ParallelHammersteinOutcome::Unavailable(FitUnavailable {
            reason: FitUnavailableReason::Cancelled,
            diagnostics: fit_diagnostics,
        }));
    }
    if !matches!(
        solver_result.stop,
        LsqrStop::ResidualTolerance | LsqrStop::NormalResidualTolerance
    ) {
        return Ok(ParallelHammersteinOutcome::Unavailable(FitUnavailable {
            reason: FitUnavailableReason::SolverDidNotConverge,
            diagnostics: fit_diagnostics,
        }));
    }
    let mut taps_order_major = zeroed_vec(
        prepared.columns,
        remaining.saturating_sub(rhs_bytes + prediction_bytes),
    )?;
    mapped
        .map_coefficients(&solver_result.solution, &mut taps_order_major)
        .map_err(|_| {
            PolynomialFitError::new(ErrorKind::Factorization, "triangular fit mapping failed")
        })?;
    for (index, tap) in taps_order_major.iter_mut().enumerate() {
        *tap /= prepared.order_scales[index / prepared.options.support_samples];
    }
    if taps_order_major.iter().any(|tap| !tap.is_finite()) {
        return Err(PolynomialFitError::new(
            ErrorKind::Factorization,
            "mapped finite-kernel taps are non-finite",
        ));
    }
    mapped
        .apply(&solver_result.solution, &mut predictions)
        .map_err(|_| {
            PolynomialFitError::new(ErrorKind::Factorization, "training prediction failed")
        })?;
    fit_diagnostics.training = training_metrics(
        training_captures,
        &predictions,
        prepared.output_samples,
        prepared.input_samples,
        prepared.options.support_samples,
    )?;
    if fit_diagnostics
        .training
        .iter()
        .any(|metrics| !metrics.residual_gates_passed)
    {
        return Ok(ParallelHammersteinOutcome::Unavailable(FitUnavailable {
            reason: FitUnavailableReason::TrainingResidualExceeded,
            diagnostics: fit_diagnostics,
        }));
    }
    drop(mapped);
    drop(target);
    drop(predictions);
    drop(solver_result);
    for (index, record) in held_out_records.iter().enumerate() {
        if cancelled() {
            return Ok(ParallelHammersteinOutcome::Unavailable(FitUnavailable {
                reason: FitUnavailableReason::Cancelled,
                diagnostics: fit_diagnostics,
            }));
        }
        let remaining = PARALLEL_HAMMERSTEIN_MAX_WORKING_BYTES
            .saturating_sub(FIXED_WORKSPACE_RESERVE_BYTES)
            .saturating_sub(taps_bytes);
        let mut operator = PolynomialConvolutionOperator::new(
            record.reference,
            ORDER_COUNT,
            prepared.options.support_samples,
            prepared.output_samples,
            fft_length.max(1),
            remaining,
        )
        .map_err(|_| {
            PolynomialFitError::new(
                ErrorKind::ResourceLimit,
                "held-out operator exceeded its remaining storage budget",
            )
        })?;
        let operator_bytes = operator.buffer_bytes();
        let prediction_cap = remaining.saturating_sub(operator_bytes);
        let mut prediction = zeroed_vec(prepared.output_samples, prediction_cap)?;
        operator
            .apply(&taps_order_major, &mut prediction)
            .map_err(|_| {
                PolynomialFitError::new(ErrorKind::Factorization, "held-out prediction failed")
            })?;
        fit_diagnostics.held_out.push(record_metrics(
            index,
            record.capture,
            &prediction,
            prepared.input_samples,
            prepared.options.support_samples,
        )?);
    }
    if fit_diagnostics
        .held_out
        .iter()
        .any(|metrics| !metrics.residual_gates_passed)
    {
        return Ok(ParallelHammersteinOutcome::Unavailable(FitUnavailable {
            reason: FitUnavailableReason::HeldOutResidualExceeded,
            diagnostics: fit_diagnostics,
        }));
    }
    Ok(ParallelHammersteinOutcome::NumericalOnly(
        ParallelHammersteinCandidate {
            method_version: 1,
            sample_rate_hz: prepared.options.sample_rate_hz,
            input_samples: prepared.input_samples,
            output_samples: prepared.output_samples,
            support_samples: prepared.options.support_samples,
            guard_samples: prepared.options.guard_samples,
            taps_order_major,
            diagnostics: fit_diagnostics,
            qualification: [
                QualificationStatus::NotCertified,
                QualificationStatus::CallerDeclaredOnly,
                QualificationStatus::NotProvided,
                QualificationStatus::NotEvaluated,
                QualificationStatus::NotProduced,
            ],
        },
    ))
}

struct StackedPolynomialOperator {
    operators: Vec<PolynomialConvolutionOperator>,
    output_samples: usize,
    columns: usize,
    order_scales: [f64; ORDER_COUNT],
    physical_coefficients: Vec<f64>,
    coefficient_work: Vec<f64>,
}

impl StackedPolynomialOperator {
    fn new(
        operators: Vec<PolynomialConvolutionOperator>,
        output_samples: usize,
        columns: usize,
        order_scales: [f64; ORDER_COUNT],
        workspace_bytes: usize,
    ) -> Result<Self, PolynomialFitError> {
        let vector_bytes = bytes_for(columns)?;
        let vector_cap = workspace_bytes.saturating_sub(vector_bytes);
        let mut physical_coefficients = zeroed_vec(columns, vector_cap)?;
        let mut coefficient_work = zeroed_vec(columns, vector_cap)?;
        physical_coefficients.fill(0.0);
        coefficient_work.fill(0.0);
        Ok(Self {
            operators,
            output_samples,
            columns,
            order_scales,
            physical_coefficients,
            coefficient_work,
        })
    }
}

impl LeastSquaresOperator for StackedPolynomialOperator {
    fn rows(&self) -> usize {
        self.output_samples * self.operators.len()
    }

    fn columns(&self) -> usize {
        self.columns
    }

    fn apply(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), String> {
        if input.len() != self.columns || output.len() != self.rows() {
            return Err("stacked polynomial forward dimensions mismatch".into());
        }
        let support = self.columns / ORDER_COUNT;
        for (index, (physical, scaled)) in
            self.physical_coefficients.iter_mut().zip(input).enumerate()
        {
            *physical = *scaled / self.order_scales[index / support];
            if !physical.is_finite() {
                return Err("scaled polynomial coefficient conversion is nonfinite".into());
            }
        }
        for (index, operator) in self.operators.iter_mut().enumerate() {
            let begin = index * self.output_samples;
            operator.apply(
                &self.physical_coefficients,
                &mut output[begin..begin + self.output_samples],
            )?;
        }
        Ok(())
    }

    fn apply_adjoint(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), String> {
        if input.len() != self.rows() || output.len() != self.columns {
            return Err("stacked polynomial adjoint dimensions mismatch".into());
        }
        output.fill(0.0);
        for (index, operator) in self.operators.iter_mut().enumerate() {
            let begin = index * self.output_samples;
            operator.apply_adjoint(
                &input[begin..begin + self.output_samples],
                &mut self.coefficient_work,
            )?;
            for (sum, value) in output.iter_mut().zip(&self.coefficient_work) {
                *sum += *value;
            }
        }
        let support = self.columns / ORDER_COUNT;
        for (index, value) in output.iter_mut().enumerate() {
            *value /= self.order_scales[index / support];
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err("stacked polynomial adjoint produced a non-finite value".into());
        }
        Ok(())
    }
}

fn validate_options(options: ParallelHammersteinFitOptions) -> Result<(), PolynomialFitError> {
    if options.sample_rate_hz == 0
        || options.support_samples == 0
        || options.support_samples > MAX_SUPPORT_SAMPLES
        || options.guard_samples == 0
    {
        return Err(PolynomialFitError::new(
            ErrorKind::InvalidRequest,
            "sample rate, support or nonempty output guard is invalid",
        ));
    }
    Ok(())
}

fn estimate_design_resources(
    rows: usize,
    columns: usize,
) -> Result<DesignResourceEstimate, PolynomialFitError> {
    let design_matrix_bytes = bytes_for(rows.checked_mul(columns).ok_or_else(|| {
        PolynomialFitError::new(
            ErrorKind::ResourceLimit,
            "design matrix entry count overflow",
        )
    })?)?;
    let triangular_factor_bytes = bytes_for(columns.checked_mul(columns).ok_or_else(|| {
        PolynomialFitError::new(ErrorKind::ResourceLimit, "factor entry count overflow")
    })?)?;
    let qr_phase_bytes = design_matrix_bytes
        .checked_add(triangular_factor_bytes)
        .and_then(|bytes| bytes.checked_add(bytes_for(columns).ok()?))
        .ok_or_else(|| {
            PolynomialFitError::new(ErrorKind::ResourceLimit, "QR phase estimate overflow")
        })?;
    // nalgebra QR overwrites the dense matrix. unpack_r may temporarily coexist
    // with it; SVD retains the row-major factor while bidiagonalizing one R matrix.
    // Eight n-vectors conservatively cover bidiagonal diagonal/off-diagonal,
    // packed-axis/work arrays, singular values and decomposition bookkeeping.
    let svd_phase_bytes = triangular_factor_bytes
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(bytes_for(columns.checked_mul(8)?).ok()?))
        .ok_or_else(|| {
            PolynomialFitError::new(ErrorKind::ResourceLimit, "SVD phase estimate overflow")
        })?;
    let preflight_peak_bytes = qr_phase_bytes
        .max(svd_phase_bytes)
        .checked_add(FIXED_WORKSPACE_RESERVE_BYTES)
        .ok_or_else(|| {
            PolynomialFitError::new(ErrorKind::ResourceLimit, "factorization estimate overflow")
        })?;
    Ok(DesignResourceEstimate {
        design_matrix_bytes,
        triangular_factor_bytes,
        qr_phase_bytes,
        svd_phase_bytes,
        fixed_workspace_reserve_bytes: FIXED_WORKSPACE_RESERVE_BYTES,
        preflight_peak_bytes,
        working_set_limit_bytes: PARALLEL_HAMMERSTEIN_MAX_WORKING_BYTES,
    })
}

fn input_design_sha256(
    references: &[TrainingInputReference<'_>],
    options: ParallelHammersteinFitOptions,
    input_samples: usize,
    order_scales: &[f64; ORDER_COUNT],
) -> Result<String, PolynomialFitError> {
    let mut digest = Sha256::new();
    digest.update(b"math-dsp/parallel-hammerstein/input-design/v1\0");
    digest.update((ORDER_COUNT as u32).to_le_bytes());
    digest.update(options.sample_rate_hz.to_le_bytes());
    digest.update(to_u64(options.support_samples)?);
    digest.update(to_u64(options.guard_samples)?);
    digest.update(to_u64(references.len())?);
    digest.update(to_u64(input_samples)?);
    for reference in references {
        digest.update(reference.sample_rate_hz.to_le_bytes());
        digest.update(to_u64(reference.samples.len())?);
        for sample in reference.samples {
            // Hash exact IEEE-754 bits: +0 and -0 are distinct design inputs.
            digest.update(sample.to_bits().to_le_bytes());
        }
    }
    for scale in order_scales {
        digest.update(scale.to_bits().to_le_bytes());
    }
    let bytes = digest.finalize();
    let hex_digits = b"0123456789abcdef";
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        hex.push(hex_digits[(byte >> 4) as usize] as char);
        hex.push(hex_digits[(byte & 0x0f) as usize] as char);
    }
    Ok(hex)
}

fn to_u64(value: usize) -> Result<[u8; 8], PolynomialFitError> {
    let value = u64::try_from(value).map_err(|_| {
        PolynomialFitError::new(
            ErrorKind::ResourceLimit,
            "design identity dimension exceeds its canonical encoding",
        )
    })?;
    Ok(value.to_le_bytes())
}

fn compute_order_scales(
    references: &[TrainingInputReference<'_>],
) -> Result<[f64; ORDER_COUNT], PolynomialFitError> {
    let mut scales = [0.0_f64; ORDER_COUNT];
    for input in references {
        for sample in input.samples.iter().copied() {
            let mut power = 1.0;
            for scale in &mut scales {
                power *= sample;
                if !power.is_finite() {
                    return Err(PolynomialFitError::new(
                        ErrorKind::NonFiniteInput,
                        "training reference power is non-finite",
                    ));
                }
                *scale = f64::hypot(*scale, power);
                if !scale.is_finite() {
                    return Err(PolynomialFitError::new(
                        ErrorKind::NonFiniteInput,
                        "training order scale overflowed",
                    ));
                }
            }
        }
    }
    if scales.iter().any(|scale| *scale <= f64::MIN_POSITIVE) {
        return Err(PolynomialFitError::new(
            ErrorKind::InvalidRequest,
            "one or more polynomial input powers have zero numerical scale",
        ));
    }
    Ok(scales)
}

#[expect(
    clippy::too_many_arguments,
    reason = "the explicit dense design dimensions are checked before filling a single allocation"
)]
fn fill_design_matrix(
    matrix: &mut [f64],
    references: &[TrainingInputReference<'_>],
    input_samples: usize,
    output_samples: usize,
    rows: usize,
    columns: usize,
    support: usize,
    scales: &[f64; ORDER_COUNT],
) -> Result<(), PolynomialFitError> {
    let expected_entries = rows.checked_mul(columns).ok_or_else(|| {
        PolynomialFitError::new(ErrorKind::ResourceLimit, "dense design dimensions overflow")
    })?;
    if matrix.len() != expected_entries {
        return Err(PolynomialFitError::new(
            ErrorKind::InvalidRequest,
            "dense design buffer length does not match its dimensions",
        ));
    }
    let mut powers = Vec::new();
    powers.try_reserve_exact(input_samples).map_err(|_| {
        PolynomialFitError::new(
            ErrorKind::Allocation,
            "input power buffer allocation failed",
        )
    })?;
    powers.resize(input_samples, 0.0);
    for (record_index, input) in references.iter().enumerate() {
        powers.copy_from_slice(input.samples);
        for (order, scale) in scales.iter().copied().enumerate() {
            for tap in 0..support {
                let column = order * support + tap;
                let column_begin = column * rows;
                for sample in 0..output_samples {
                    if let Some(source) = sample.checked_sub(tap)
                        && source < input_samples
                    {
                        matrix[column_begin + record_index * output_samples + sample] =
                            powers[source] / scale;
                    }
                }
            }
            if order + 1 < ORDER_COUNT {
                for (power, sample) in powers.iter_mut().zip(input.samples) {
                    *power *= *sample;
                    if !power.is_finite() {
                        return Err(PolynomialFitError::new(
                            ErrorKind::NonFiniteInput,
                            "scaled design input power is non-finite",
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_captures(
    prepared: &PreparedParallelHammersteinDesign<'_>,
    training: &[CapturedTrainingOutput<'_>],
    held_out: &[HeldOutRecord<'_>],
) -> Result<(), PolynomialFitError> {
    for capture in training {
        if capture.samples.len() != prepared.output_samples {
            return Err(PolynomialFitError::new(
                ErrorKind::InvalidRequest,
                "training capture length does not match the exact convolution tail and guard",
            ));
        }
        if capture.samples.iter().any(|sample| !sample.is_finite()) {
            return Err(PolynomialFitError::new(
                ErrorKind::NonFiniteInput,
                "training capture contains a non-finite sample",
            ));
        }
    }
    for record in held_out {
        if record.reference.len() != prepared.input_samples
            || record.capture.len() != prepared.output_samples
        {
            return Err(PolynomialFitError::new(
                ErrorKind::InvalidRequest,
                "held-out record length differs from the prepared design",
            ));
        }
        if record
            .reference
            .iter()
            .chain(record.capture)
            .any(|sample| !sample.is_finite())
        {
            return Err(PolynomialFitError::new(
                ErrorKind::NonFiniteInput,
                "held-out record contains a non-finite sample",
            ));
        }
    }
    Ok(())
}

fn total_decoded_samples(
    prepared: &PreparedParallelHammersteinDesign<'_>,
    training: &[CapturedTrainingOutput<'_>],
    held_out: &[HeldOutRecord<'_>],
) -> Result<usize, PolynomialFitError> {
    let mut total = prepared
        .references
        .iter()
        .try_fold(0usize, |sum, input| sum.checked_add(input.len()))
        .ok_or_else(|| {
            PolynomialFitError::new(ErrorKind::ResourceLimit, "decoded sample count overflow")
        })?;
    for capture in training {
        total = total.checked_add(capture.samples.len()).ok_or_else(|| {
            PolynomialFitError::new(ErrorKind::ResourceLimit, "decoded sample count overflow")
        })?;
    }
    for record in held_out {
        total = total
            .checked_add(record.reference.len())
            .and_then(|sum| sum.checked_add(record.capture.len()))
            .ok_or_else(|| {
                PolynomialFitError::new(ErrorKind::ResourceLimit, "decoded sample count overflow")
            })?;
    }
    Ok(total)
}

fn copy_training_captures(
    target: &mut [f64],
    training: &[CapturedTrainingOutput<'_>],
    output_samples: usize,
) {
    for (index, capture) in training.iter().enumerate() {
        let begin = index * output_samples;
        target[begin..begin + output_samples].copy_from_slice(capture.samples);
    }
}

fn training_metrics(
    captures: &[CapturedTrainingOutput<'_>],
    predictions: &[f64],
    output_samples: usize,
    input_samples: usize,
    support_samples: usize,
) -> Result<Vec<FitRecordDiagnostics>, PolynomialFitError> {
    let mut metrics = Vec::new();
    metrics.try_reserve_exact(captures.len()).map_err(|_| {
        PolynomialFitError::new(
            ErrorKind::Allocation,
            "training diagnostics allocation failed",
        )
    })?;
    for (index, capture) in captures.iter().enumerate() {
        let begin = index * output_samples;
        metrics.push(record_metrics(
            index,
            capture.samples,
            &predictions[begin..begin + output_samples],
            input_samples,
            support_samples,
        )?);
    }
    Ok(metrics)
}

fn record_metrics(
    record_index: usize,
    actual: &[f64],
    predicted: &[f64],
    input_samples: usize,
    support_samples: usize,
) -> Result<FitRecordDiagnostics, PolynomialFitError> {
    if actual.len() != predicted.len() {
        return Err(PolynomialFitError::new(
            ErrorKind::InvalidRequest,
            "fit prediction and captured output lengths differ",
        ));
    }
    let scale = centered_rms(actual)?;
    if scale <= f64::MIN_POSITIVE {
        return Err(PolynomialFitError::new(
            ErrorKind::InvalidRequest,
            "captured output has no finite nonzero centered RMS",
        ));
    }
    let mut error_norm = 0.0_f64;
    for (sample, prediction) in actual.iter().zip(predicted) {
        let error = sample - prediction;
        if !error.is_finite() {
            return Err(PolynomialFitError::new(
                ErrorKind::Factorization,
                "fit residual overflowed",
            ));
        }
        error_norm = error_norm.hypot(error);
    }
    let normalized_rmse = error_norm / (actual.len() as f64).sqrt() / scale;
    let guard_start = input_samples
        .checked_add(support_samples - 1)
        .ok_or_else(|| PolynomialFitError::new(ErrorKind::ResourceLimit, "guard start overflow"))?;
    let guard = actual.get(guard_start..).ok_or_else(|| {
        PolynomialFitError::new(ErrorKind::InvalidRequest, "captured guard is missing")
    })?;
    if guard.is_empty() {
        return Err(PolynomialFitError::new(
            ErrorKind::InvalidRequest,
            "captured guard must contain at least one sample",
        ));
    }
    let guard_norm = guard.iter().copied().fold(0.0_f64, f64::hypot);
    let guard_rms_normalized = guard_norm / (guard.len() as f64).sqrt() / scale;
    if !normalized_rmse.is_finite() || !guard_rms_normalized.is_finite() {
        return Err(PolynomialFitError::new(
            ErrorKind::Factorization,
            "fit residual metric is non-finite",
        ));
    }
    Ok(FitRecordDiagnostics {
        record_index,
        normalized_rmse,
        guard_rms_normalized,
        residual_gates_passed: normalized_rmse <= NORMALIZED_RESIDUAL_LIMIT
            && guard_rms_normalized <= NORMALIZED_RESIDUAL_LIMIT,
    })
}

fn centered_rms(values: &[f64]) -> Result<f64, PolynomialFitError> {
    if values.is_empty() {
        return Err(PolynomialFitError::new(
            ErrorKind::InvalidRequest,
            "cannot normalize an empty capture",
        ));
    }
    let mut mean = 0.0_f64;
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(PolynomialFitError::new(
                ErrorKind::NonFiniteInput,
                "captured output contains a non-finite sample",
            ));
        }
        let count = (index + 1) as f64;
        mean = mean * ((count - 1.0) / count) + value / count;
    }
    let norm = values
        .iter()
        .map(|value| value - mean)
        .try_fold(0.0_f64, |norm, value| {
            if value.is_finite() {
                Some(norm.hypot(value))
            } else {
                None
            }
        })
        .ok_or_else(|| {
            PolynomialFitError::new(ErrorKind::NonFiniteInput, "capture centering overflowed")
        })?;
    let result = norm / (values.len() as f64).sqrt();
    if !result.is_finite() {
        return Err(PolynomialFitError::new(
            ErrorKind::NonFiniteInput,
            "capture RMS is non-finite",
        ));
    }
    Ok(result)
}

fn solver_diagnostics(result: &LsqrResult) -> SolverDiagnostics {
    SolverDiagnostics {
        stop: match result.stop {
            LsqrStop::ResidualTolerance => FitSolverStop::ResidualTolerance,
            LsqrStop::NormalResidualTolerance => FitSolverStop::NormalResidualTolerance,
            LsqrStop::ConditionLimit => FitSolverStop::ConditionLimit,
            LsqrStop::IterationLimit => FitSolverStop::IterationLimit,
            LsqrStop::Cancelled => FitSolverStop::Cancelled,
            LsqrStop::Breakdown => FitSolverStop::Breakdown,
        },
        iterations: result.iterations,
        residual_norm: result.residual_norm,
        normal_residual_norm: result
            .normal_residual_norm
            .filter(|value| value.is_finite()),
        condition_estimate: result
            .condition_estimate
            .is_finite()
            .then_some(result.condition_estimate),
    }
}

fn empty_fit_diagnostics(design: DesignNumericalDiagnostics) -> PolynomialFitDiagnostics {
    PolynomialFitDiagnostics {
        design,
        solver: None,
        training: Vec::new(),
        held_out: Vec::new(),
        normalized_residual_limit: NORMALIZED_RESIDUAL_LIMIT,
    }
}

fn lsqr_buffer_bytes(rows: usize, columns: usize) -> Result<usize, PolynomialFitError> {
    columns
        .checked_mul(4)
        .and_then(|count| {
            rows.checked_mul(2)
                .and_then(|row_count| count.checked_add(row_count))
        })
        .and_then(|count| count.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(|| {
            PolynomialFitError::new(ErrorKind::ResourceLimit, "LSQR vector estimate overflow")
        })
}

fn bytes_for(elements: usize) -> Result<usize, PolynomialFitError> {
    elements
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| {
            PolynomialFitError::new(ErrorKind::ResourceLimit, "working byte count overflow")
        })
}

fn zeroed_vec(length: usize, max_bytes: usize) -> Result<Vec<f64>, PolynomialFitError> {
    let bytes = bytes_for(length)?;
    if bytes > max_bytes || bytes > isize::MAX as usize {
        return Err(PolynomialFitError::new(
            ErrorKind::ResourceLimit,
            "vector exceeds its remaining storage budget",
        ));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| PolynomialFitError::new(ErrorKind::Allocation, "vector allocation failed"))?;
    if bytes_for(values.capacity())? > max_bytes {
        return Err(PolynomialFitError::new(
            ErrorKind::ResourceLimit,
            "actual vector capacity exceeds its remaining storage budget",
        ));
    }
    values.resize(length, 0.0);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    const INPUT_SAMPLES: usize = 96;
    const SUPPORT: usize = 6;
    const GUARD: usize = 16;

    fn reference(amplitude: f64, phase: f64) -> Vec<f64> {
        (0..INPUT_SAMPLES)
            .map(|index| {
                let time = index as f64;
                amplitude
                    * (0.53 * (0.17 * time + phase).sin()
                        + 0.31 * (0.43 * time - phase * 0.7).cos()
                        + 0.16 * (0.071 * time + phase * 1.3).sin())
            })
            .collect()
    }

    fn known_taps() -> Vec<f64> {
        let mut taps = Vec::with_capacity(ORDER_COUNT * SUPPORT);
        for order in 0..ORDER_COUNT {
            for tap in 0..SUPPORT {
                let sign = if (order + tap) % 2 == 0 { 1.0 } else { -1.0 };
                taps.push(sign * (0.11 / (order + 1) as f64) / (tap + 1) as f64);
            }
        }
        taps
    }

    fn direct_output(input: &[f64], taps: &[f64], guard: usize) -> Vec<f64> {
        let output_len = input.len() + SUPPORT - 1 + guard;
        let mut output = vec![0.0; output_len];
        for order in 0..ORDER_COUNT {
            for (sample_index, sample) in input.iter().copied().enumerate() {
                let mut power = sample;
                for _ in 0..order {
                    power *= sample;
                }
                for tap in 0..SUPPORT {
                    output[sample_index + tap] += taps[order * SUPPORT + tap] * power;
                }
            }
        }
        output
    }

    fn fit_inputs() -> Vec<Vec<f64>> {
        (0..5)
            .map(|index| reference(0.22 + index as f64 * 0.045, index as f64 * 0.37))
            .collect()
    }

    fn training_references(inputs: &[Vec<f64>]) -> Vec<TrainingInputReference<'_>> {
        inputs
            .iter()
            .map(|samples| TrainingInputReference {
                samples,
                sample_rate_hz: 4_000,
            })
            .collect()
    }

    fn options() -> ParallelHammersteinFitOptions {
        ParallelHammersteinFitOptions {
            sample_rate_hz: 4_000,
            support_samples: SUPPORT,
            guard_samples: GUARD,
        }
    }

    #[test]
    fn unequal_order_scales_preserve_adjoint_taps_and_held_out_convolution() {
        let inputs = fit_inputs();
        let references = training_references(&inputs);
        let taps = known_taps();
        let prepared =
            prepare_parallel_hammerstein_design(&references, options(), || false).unwrap();
        assert_eq!(prepared.rows, references.len() * prepared.output_samples);
        assert_eq!(prepared.columns, ORDER_COUNT * SUPPORT);
        assert_eq!(
            prepared.diagnostics.status,
            DesignNumericalStatus::EstimateWithinLimits
        );
        assert_eq!(
            prepared.diagnostics.finite_design_qualification,
            QualificationStatus::NotCertified
        );
        assert_eq!(prepared.diagnostics.input_design_sha256.len(), 64);
        assert!(
            prepared
                .order_scales
                .windows(2)
                .any(|pair| (pair[0] - pair[1]).abs() > 1.0e-6)
        );
        assert!(
            prepared.resource_estimate().preflight_peak_bytes
                <= PARALLEL_HAMMERSTEIN_MAX_WORKING_BYTES
        );
        let fft_len = (INPUT_SAMPLES + SUPPORT - 1).next_power_of_two();
        let operators = references
            .iter()
            .map(|reference| {
                PolynomialConvolutionOperator::new(
                    reference.samples,
                    ORDER_COUNT,
                    SUPPORT,
                    prepared.output_samples,
                    fft_len,
                    PARALLEL_HAMMERSTEIN_MAX_WORKING_BYTES,
                )
                .unwrap()
            })
            .collect();
        let coefficient_workspace = bytes_for(prepared.columns * 2).unwrap();
        let mut scaled_operator = StackedPolynomialOperator::new(
            operators,
            prepared.output_samples,
            prepared.columns,
            prepared.order_scales,
            coefficient_workspace,
        )
        .unwrap();
        let coefficients: Vec<f64> = taps
            .iter()
            .enumerate()
            .map(|(index, tap)| tap * prepared.order_scales[index / SUPPORT])
            .collect();
        let dual: Vec<f64> = (0..prepared.rows)
            .map(|index| (index as f64 * 0.07).cos())
            .collect();
        let mut forward = vec![0.0; prepared.rows];
        let mut adjoint = vec![0.0; prepared.columns];
        scaled_operator.apply(&coefficients, &mut forward).unwrap();
        scaled_operator.apply_adjoint(&dual, &mut adjoint).unwrap();
        for (record_index, input) in inputs.iter().enumerate() {
            let expected = direct_output(input, &taps, GUARD);
            let actual = &forward[record_index * prepared.output_samples
                ..(record_index + 1) * prepared.output_samples];
            for (actual, expected) in actual.iter().zip(expected) {
                assert_abs_diff_eq!(*actual, expected, epsilon = 1.0e-12);
            }
        }
        let left: f64 = forward.iter().zip(&dual).map(|(a, b)| a * b).sum();
        let right: f64 = coefficients.iter().zip(&adjoint).map(|(a, b)| a * b).sum();
        assert!((left - right).abs() <= 1.0e-10 * left.abs().max(right.abs()).max(1.0));
        let training: Vec<Vec<f64>> = inputs
            .iter()
            .map(|input| direct_output(input, &taps, GUARD))
            .collect();
        let training_records: Vec<CapturedTrainingOutput<'_>> = training
            .iter()
            .map(|samples| CapturedTrainingOutput {
                samples,
                sample_rate_hz: 4_000,
            })
            .collect();
        let held_input = reference(0.36, 0.91);
        let held_output = direct_output(&held_input, &taps, GUARD);
        let fft_len = (INPUT_SAMPLES + SUPPORT - 1).next_power_of_two();
        let mut held_operator = PolynomialConvolutionOperator::new(
            &held_input,
            ORDER_COUNT,
            SUPPORT,
            prepared.output_samples,
            fft_len,
            PARALLEL_HAMMERSTEIN_MAX_WORKING_BYTES,
        )
        .unwrap();
        let mut operator_output = vec![0.0; prepared.output_samples];
        held_operator.apply(&taps, &mut operator_output).unwrap();
        for (actual, expected) in held_output.iter().zip(&operator_output) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1.0e-12);
        }
        let held = [HeldOutRecord {
            reference: &held_input,
            reference_sample_rate_hz: 4_000,
            capture: &held_output,
            capture_sample_rate_hz: 4_000,
        }];
        let output_samples = prepared.output_samples;
        let outcome =
            fit_parallel_hammerstein(prepared, &training_records, &held, || false).unwrap();
        let ParallelHammersteinOutcome::NumericalOnly(candidate) = outcome else {
            panic!("expected numerical candidate, got {outcome:#?}");
        };
        assert_eq!(candidate.taps_order_major.len(), taps.len());
        for (actual, expected) in candidate.taps_order_major.iter().zip(&taps) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1.0e-8);
        }
        assert!(
            candidate
                .diagnostics
                .training
                .iter()
                .all(|row| row.residual_gates_passed)
        );
        assert!(
            candidate
                .diagnostics
                .held_out
                .iter()
                .all(|row| row.residual_gates_passed)
        );
        assert!(candidate.diagnostics.held_out[0].normalized_rmse < 1.0e-8);
        let mut fitted_heldout_operator = PolynomialConvolutionOperator::new(
            &held_input,
            ORDER_COUNT,
            SUPPORT,
            output_samples,
            fft_len,
            PARALLEL_HAMMERSTEIN_MAX_WORKING_BYTES,
        )
        .unwrap();
        let mut fitted_heldout = vec![0.0; output_samples];
        fitted_heldout_operator
            .apply(&candidate.taps_order_major, &mut fitted_heldout)
            .unwrap();
        for (actual, expected) in fitted_heldout.iter().zip(&held_output) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1.0e-8);
        }
        assert_eq!(
            candidate.qualification,
            [
                QualificationStatus::NotCertified,
                QualificationStatus::CallerDeclaredOnly,
                QualificationStatus::NotProvided,
                QualificationStatus::NotEvaluated,
                QualificationStatus::NotProduced,
            ]
        );
        let encoded = serde_json::to_vec(&candidate).unwrap();
        let decoded: ParallelHammersteinCandidate = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(candidate.method_version, decoded.method_version);
        assert_eq!(candidate.sample_rate_hz, decoded.sample_rate_hz);
        assert_eq!(candidate.input_samples, decoded.input_samples);
        assert_eq!(candidate.output_samples, decoded.output_samples);
        assert_eq!(candidate.support_samples, decoded.support_samples);
        assert_eq!(candidate.guard_samples, decoded.guard_samples);
        assert_eq!(
            candidate.diagnostics.design.input_design_sha256,
            decoded.diagnostics.design.input_design_sha256
        );
        assert_eq!(
            candidate.diagnostics.design.status,
            decoded.diagnostics.design.status
        );
        assert_eq!(candidate.qualification, decoded.qualification);
        for (actual, expected) in candidate
            .taps_order_major
            .iter()
            .zip(&decoded.taps_order_major)
        {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1.0e-14);
        }
        assert_eq!(
            candidate.diagnostics.training.len(),
            decoded.diagnostics.training.len()
        );
        for (actual, expected) in candidate
            .diagnostics
            .training
            .iter()
            .chain(&candidate.diagnostics.held_out)
            .zip(
                decoded
                    .diagnostics
                    .training
                    .iter()
                    .chain(&decoded.diagnostics.held_out),
            )
        {
            assert_eq!(actual.record_index, expected.record_index);
            assert_abs_diff_eq!(
                actual.normalized_rmse,
                expected.normalized_rmse,
                epsilon = 1.0e-14
            );
            assert_abs_diff_eq!(
                actual.guard_rms_normalized,
                expected.guard_rms_normalized,
                epsilon = 1.0e-14
            );
            assert_eq!(actual.residual_gates_passed, expected.residual_gates_passed);
        }
    }

    #[test]
    fn repeated_training_reference_is_refused_by_unchanged_condition_estimate_gate() {
        // Constant input makes each normalized polynomial-order block identical,
        // so the repeated-reference design is exactly rank deficient.
        let input = vec![0.31; INPUT_SAMPLES];
        let references = [TrainingInputReference {
            samples: &input,
            sample_rate_hz: 4_000,
        }; 5];
        let prepared =
            prepare_parallel_hammerstein_design(&references, options(), || false).unwrap();
        assert!(matches!(
            prepared.diagnostics.status,
            DesignNumericalStatus::EstimatedRankDeficient
                | DesignNumericalStatus::EstimatedConditionExceeded
        ));
        let bad_capture = vec![f64::NAN; prepared.output_samples];
        let training = [CapturedTrainingOutput {
            samples: &bad_capture,
            sample_rate_hz: 4_000,
        }; 5];
        let held = [HeldOutRecord {
            reference: &input,
            reference_sample_rate_hz: 4_000,
            capture: &bad_capture,
            capture_sample_rate_hz: 4_000,
        }];
        let result = fit_parallel_hammerstein(prepared, &training, &held, || false).unwrap();
        let ParallelHammersteinOutcome::Unavailable(unavailable) = result else {
            panic!("repeated reference must be refused");
        };
        assert!(matches!(
            unavailable.reason,
            FitUnavailableReason::DesignRankEstimateInsufficient
                | FitUnavailableReason::DesignConditionEstimateExceeded
        ));
        assert!(unavailable.diagnostics.training.is_empty());
    }

    #[test]
    fn mismatched_declared_capture_rate_is_rejected() {
        let inputs = fit_inputs();
        let references = training_references(&inputs);
        let prepared =
            prepare_parallel_hammerstein_design(&references, options(), || false).unwrap();
        let captures = vec![0.0; prepared.output_samples];
        let training: Vec<_> = (0..references.len())
            .map(|_| CapturedTrainingOutput {
                samples: &captures,
                sample_rate_hz: 48_000,
            })
            .collect();
        let held_input = reference(0.3, 0.1);
        let held_output = vec![0.0; prepared.output_samples];
        let held = [HeldOutRecord {
            reference: &held_input,
            reference_sample_rate_hz: 4_000,
            capture: &held_output,
            capture_sample_rate_hz: 4_000,
        }];
        let error = fit_parallel_hammerstein(prepared, &training, &held, || false).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("sample-rate declarations differ")
        );
    }

    #[test]
    fn training_rate_mismatch_is_rejected_before_factorization() {
        let inputs = fit_inputs();
        let mut references = training_references(&inputs);
        references[2].sample_rate_hz = 48_000;
        let mut cancellation_polls = 0;
        let mismatched = prepare_parallel_hammerstein_design(&references, options(), || {
            cancellation_polls += 1;
            false
        })
        .unwrap_err();
        assert_eq!(cancellation_polls, 1);
        assert!(
            mismatched
                .to_string()
                .contains("training reference sample-rate declarations differ")
        );
    }

    #[test]
    fn cancellation_after_svd_refuses_prepared_design() {
        let inputs = fit_inputs();
        let references = training_references(&inputs);
        let mut cancellation_polls = 0;
        let error = prepare_parallel_hammerstein_design(&references, options(), || {
            cancellation_polls += 1;
            cancellation_polls == 4
        })
        .unwrap_err();
        assert!(error.is_cancelled());
        assert_eq!(cancellation_polls, 4);
        assert!(error.to_string().contains("cancelled after SVD"));
    }

    #[test]
    fn input_design_identity_binds_exact_ordered_samples_and_settings() {
        let positive_zero = [0.0_f64, 0.2, -0.1, 0.3];
        let negative_zero = [-0.0_f64, 0.2, -0.1, 0.3];
        let other = [0.1_f64, -0.3, 0.2, 0.4];
        let base = [
            TrainingInputReference {
                samples: &positive_zero,
                sample_rate_hz: 4_000,
            },
            TrainingInputReference {
                samples: &other,
                sample_rate_hz: 4_000,
            },
        ];
        let base_scales = compute_order_scales(&base).unwrap();
        let base_hash =
            input_design_sha256(&base, options(), positive_zero.len(), &base_scales).unwrap();
        assert_eq!(
            base_hash,
            input_design_sha256(&base, options(), positive_zero.len(), &base_scales).unwrap()
        );
        let swapped = [base[1], base[0]];
        let swapped_scales = compute_order_scales(&swapped).unwrap();
        assert_ne!(
            base_hash,
            input_design_sha256(&swapped, options(), positive_zero.len(), &swapped_scales).unwrap()
        );
        let mut signed_zero = base;
        signed_zero[0].samples = &negative_zero;
        let signed_zero_scales = compute_order_scales(&signed_zero).unwrap();
        assert_ne!(
            base_hash,
            input_design_sha256(
                &signed_zero,
                options(),
                positive_zero.len(),
                &signed_zero_scales
            )
            .unwrap()
        );
        let changed_settings = ParallelHammersteinFitOptions {
            guard_samples: GUARD + 1,
            ..options()
        };
        assert_ne!(
            base_hash,
            input_design_sha256(&base, changed_settings, positive_zero.len(), &base_scales)
                .unwrap()
        );
    }

    #[test]
    fn oversized_dense_design_refuses_before_matrix_allocation() {
        let input = vec![0.1; 1_000_000];
        let references = [TrainingInputReference {
            samples: &input,
            sample_rate_hz: 48_000,
        }; 5];
        let error = prepare_parallel_hammerstein_design(
            &references,
            ParallelHammersteinFitOptions {
                sample_rate_hz: 48_000,
                support_samples: MAX_SUPPORT_SAMPLES,
                guard_samples: 640,
            },
            || false,
        )
        .unwrap_err();
        assert!(error.is_resource_limit());
    }

    #[test]
    fn pre_cancelled_design_allocates_no_factorization() {
        let inputs = fit_inputs();
        let references = training_references(&inputs);
        let error =
            prepare_parallel_hammerstein_design(&references, options(), || true).unwrap_err();
        assert!(error.is_cancelled());
    }
}

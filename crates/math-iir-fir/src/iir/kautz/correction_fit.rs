//! Checked fitting for the dry-plus-bank Kautz correction.
//!
//! [`fit_correction`] drives the realized complex transfer's magnitude
//! `|H(f)| = |1 + Σ g_k·B_k(f)|` toward a declared dB target with a
//! log-frequency-weighted squared-error objective, solved by Gauss–Newton
//! steps with analytic Jacobians through the regularized QR solver. Every
//! trial step is checked against the composite guard grid before acceptance,
//! and gains are committed only as complete, validated vectors.

use super::correction::{KautzCorrection, KautzError, KautzResult};
use super::misc::qr_least_squares;
use num_complex::Complex;
use std::f64::consts::LN_10;

/// Floor for `|H|` inside dB evaluation (keeps the objective finite when a
/// trial step drives the transfer through zero).
const MAG_FLOOR: f64 = 1e-12;
/// dB conversion factor for magnitude derivatives.
const DB_FACTOR: f64 = 20.0 / LN_10;
/// Backtracking tries per Gauss–Newton step before declaring bounds-active.
const MAX_BACKTRACKS: usize = 10;

/// How measurement points are weighted in the fit objective.
#[derive(Debug, Clone)]
pub enum FitWeights {
    /// `w ∝ 1/f`: each octave contributes equally (default).
    LogEqual,
    /// Caller-supplied non-negative weights, one per grid point.
    Custom(Vec<f64>),
}

/// Normalization policy for the fit target.
///
/// Reported errors always include the absolute (unnormalized) RMS beside
/// the policy objective, so a success claim can never hide behind a
/// renormalization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalizationPolicy {
    /// Fit the declared target as-is.
    Absolute,
    /// Subtract the weighted mean target over the anchor band before
    /// fitting, and report the shift. The absolute error is still reported
    /// against the unshifted target.
    MeanAnchored {
        /// Lower anchor edge in Hz.
        band_lo_hz: f64,
        /// Upper anchor edge in Hz.
        band_hi_hz: f64,
    },
}

/// Frequency-dependent composite bound: inside `[lo_hz, hi_hz]` the
/// correction magnitude must stay within `[-max_cut_db, +max_boost_db]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandLimit {
    /// Lower band edge in Hz (inclusive).
    pub lo_hz: f64,
    /// Upper band edge in Hz (inclusive).
    pub hi_hz: f64,
    /// Largest allowed boost in dB (≥ 0).
    pub max_boost_db: f64,
    /// Largest allowed cut in dB, as a positive magnitude (≥ 0).
    pub max_cut_db: f64,
}

/// Configuration for [`fit_correction`]. All bounds apply to the composite
/// realized transfer, never to individual coefficients.
#[derive(Debug, Clone)]
pub struct FitConfig {
    /// Optional correction band: outside it `|H − 1|` must stay within
    /// [`neutrality_db`](Self::neutrality_db).
    pub correction_band: Option<(f64, f64)>,
    /// Outside-band neutrality limit in dB (≥ 0).
    pub neutrality_db: f64,
    /// Global composite boost ceiling in dB (≥ 0).
    pub max_boost_db: f64,
    /// Global composite cut floor in dB, as a positive magnitude (≥ 0).
    pub max_cut_db: f64,
    /// Extra frequency-dependent composite bounds.
    pub extra_constraints: Vec<BandLimit>,
    /// Target normalization policy.
    pub normalization: NormalizationPolicy,
    /// Point weighting in the objective.
    pub weights: FitWeights,
    /// Gauss–Newton iteration budget (must be nonzero).
    pub max_iterations: usize,
    /// Convergence threshold on objective improvement (must be finite, > 0).
    pub tolerance: f64,
    /// Gauss–Newton Tikhonov factor (must be finite, ≥ 0).
    pub regularization: f64,
    /// Guard points per side around every pole (≥ 1).
    pub pole_guard_points: usize,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            correction_band: None,
            neutrality_db: 0.5,
            max_boost_db: 6.0,
            max_cut_db: 20.0,
            extra_constraints: Vec::new(),
            normalization: NormalizationPolicy::Absolute,
            weights: FitWeights::LogEqual,
            max_iterations: 200,
            tolerance: 1e-9,
            regularization: 1e-6,
            pole_guard_points: 4,
        }
    }
}

/// How the fit loop terminated. Only [`Converged`](Self::Converged) claims
/// an optimum; every other status names exactly what stopped the search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitStatus {
    /// Objective improvement fell below tolerance with guards satisfied.
    Converged,
    /// No improving, guard-satisfying step exists from the start point;
    /// gains are unchanged from entry.
    Stalled,
    /// Iteration budget exhausted without converging; last good gains kept.
    IterationExhausted,
    /// Every improving direction violates composite bounds; last good gains kept.
    BoundsActive,
}

/// Identity and composition of the guard grid used for a fit or check.
///
/// Guards are sampled, not continuous: passing them certifies the sampled
/// points only, never a continuous-frequency extremum. Pole neighborhoods
/// are placed from each pole's 3 dB bandwidth (`−sr·ln(r)/π`).
#[derive(Debug, Clone)]
pub struct GuardGrid {
    /// Sorted unique guard frequencies in Hz (measurement grid first in
    /// composition, then DC, Nyquist, pole neighborhoods).
    pub freqs: Vec<f64>,
    /// Measurement-grid point count.
    pub measurement_count: usize,
    /// Guard points placed per side of every pole.
    pub pole_points_per_side: usize,
    /// Whether DC (0 Hz) is included (always true).
    pub includes_dc: bool,
    /// Whether Nyquist is included (always true).
    pub includes_nyquist: bool,
    /// Correction band the grid was built against, if any.
    pub band: Option<(f64, f64)>,
}

/// Composite-bound check over a guard grid.
#[derive(Debug, Clone)]
pub struct GuardReport {
    /// Largest correction magnitude in dB and where it occurs.
    pub worst_boost_db: f64,
    /// Frequency of the worst boost in Hz.
    pub worst_boost_hz: f64,
    /// Deepest correction magnitude in dB (≤ 0 side) and where it occurs.
    pub worst_cut_db: f64,
    /// Frequency of the worst cut in Hz.
    pub worst_cut_hz: f64,
    /// Largest outside-band `|dB|` and where it occurs (0 when no band set).
    pub neutrality_violation_db: f64,
    /// Frequency of the worst neutrality violation in Hz.
    pub neutrality_violation_hz: f64,
    /// True when every configured bound holds on the grid.
    pub passed: bool,
    /// Grid identity behind this report.
    pub grid: GuardGrid,
}

/// Fit result: both the policy objective and the absolute error, so a
/// success claim can be audited without trusting any renormalization.
#[derive(Debug, Clone)]
pub struct FitDiagnostics {
    /// How the search terminated.
    pub status: FitStatus,
    /// Gauss–Newton iterations used.
    pub iterations_used: usize,
    /// Weighted mean squared dB error before the fit (policy objective).
    pub initial_objective: f64,
    /// Weighted mean squared dB error after the fit (policy objective).
    pub final_objective: f64,
    /// Unweighted RMS dB error against the declared (unshifted) target,
    /// before the fit.
    pub initial_abs_rms_db: f64,
    /// Unweighted RMS dB error against the declared (unshifted) target,
    /// after the fit.
    pub final_abs_rms_db: f64,
    /// Unweighted RMS dB error against the normalized target, after the
    /// fit (equals the absolute error under [`Absolute`](NormalizationPolicy::Absolute) policy).
    pub final_normalized_rms_db: f64,
    /// Target shift applied under anchoring (0 under absolute policy).
    pub normalization_shift_db: f64,
    /// Worst composite boost in dB on the guard grid, after the fit.
    pub worst_boost_db: f64,
    /// Its frequency in Hz.
    pub worst_boost_hz: f64,
    /// Worst composite cut in dB on the guard grid, after the fit.
    pub worst_cut_db: f64,
    /// Its frequency in Hz.
    pub worst_cut_hz: f64,
    /// Guard grid identity behind the bound checks.
    pub guard_grid: GuardGrid,
}

fn check_grid(freqs: &[f64], target_db: &[f64], weights: &FitWeights) -> KautzResult<()> {
    if freqs.len() != target_db.len() {
        return Err(KautzError::LengthMismatch {
            what: "frequency grid vs target",
            expected: target_db.len(),
            got: freqs.len(),
        });
    }
    if freqs.is_empty() {
        return Err(KautzError::LengthMismatch {
            what: "nonempty frequency grid",
            expected: 1,
            got: 0,
        });
    }
    for (index, &f) in freqs.iter().enumerate() {
        if !f.is_finite() {
            return Err(KautzError::NonFinite {
                what: "grid frequency",
                index,
                value: f,
            });
        }
        if index > 0 && f <= freqs[index - 1] {
            return Err(KautzError::UnorderedGrid { index });
        }
    }
    for (index, &t) in target_db.iter().enumerate() {
        if !t.is_finite() {
            return Err(KautzError::NonFinite {
                what: "target dB",
                index,
                value: t,
            });
        }
    }
    if let FitWeights::Custom(w) = weights {
        if w.len() != freqs.len() {
            return Err(KautzError::LengthMismatch {
                what: "custom weights vs grid",
                expected: freqs.len(),
                got: w.len(),
            });
        }
        for (index, &weight) in w.iter().enumerate() {
            if !weight.is_finite() {
                return Err(KautzError::NonFinite {
                    what: "weight",
                    index,
                    value: weight,
                });
            }
            if weight < 0.0 {
                return Err(KautzError::IncompatibleBounds {
                    reason: format!("negative weight {weight} at index {index}"),
                });
            }
        }
    }
    Ok(())
}

fn check_config(config: &FitConfig) -> KautzResult<()> {
    if config.max_iterations == 0 {
        return Err(KautzError::ZeroBudget {
            what: "fit iterations",
        });
    }
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return Err(KautzError::IncompatibleBounds {
            reason: format!(
                "convergence tolerance must be finite and > 0 (got {})",
                config.tolerance
            ),
        });
    }
    if !(config.regularization.is_finite() && config.regularization >= 0.0) {
        return Err(KautzError::IncompatibleBounds {
            reason: format!(
                "regularization must be finite and >= 0 (got {})",
                config.regularization
            ),
        });
    }
    for (name, value) in [
        ("neutrality_db", config.neutrality_db),
        ("max_boost_db", config.max_boost_db),
        ("max_cut_db", config.max_cut_db),
    ] {
        if !(value.is_finite() && value >= 0.0) {
            return Err(KautzError::IncompatibleBounds {
                reason: format!("{name} must be finite and >= 0 (got {value})"),
            });
        }
    }
    if let Some((lo, hi)) = config.correction_band
        && !(lo.is_finite() && hi.is_finite() && lo < hi && lo >= 0.0)
    {
        return Err(KautzError::IncompatibleBounds {
            reason: format!("correction band must satisfy 0 <= lo < hi (got {lo}, {hi})"),
        });
    }
    for (index, band) in config.extra_constraints.iter().enumerate() {
        if !(band.lo_hz.is_finite()
            && band.hi_hz.is_finite()
            && band.lo_hz < band.hi_hz
            && band.lo_hz >= 0.0)
        {
            return Err(KautzError::IncompatibleBounds {
                reason: format!("constraint band {index} must satisfy 0 <= lo < hi"),
            });
        }
        if !(band.max_boost_db.is_finite()
            && band.max_boost_db >= 0.0
            && band.max_cut_db.is_finite()
            && band.max_cut_db >= 0.0)
        {
            return Err(KautzError::IncompatibleBounds {
                reason: format!("constraint band {index} ceilings must be finite and >= 0"),
            });
        }
    }
    if config.pole_guard_points == 0 {
        return Err(KautzError::IncompatibleBounds {
            reason: "pole_guard_points must be >= 1 (pole neighborhoods are required guards)"
                .to_string(),
        });
    }
    if let NormalizationPolicy::MeanAnchored {
        band_lo_hz,
        band_hi_hz,
    } = config.normalization
        && !(band_lo_hz.is_finite() && band_hi_hz.is_finite() && band_lo_hz < band_hi_hz)
    {
        return Err(KautzError::IncompatibleBounds {
            reason: "anchor band must satisfy lo < hi with finite edges".to_string(),
        });
    }
    Ok(())
}

/// Measurement weights: log-equal `1/f`, or caller weights, normalized to
/// unit mean so the objective is grid-size independent.
fn measurement_weights(freqs: &[f64], weights: &FitWeights) -> KautzResult<(Vec<f64>, Vec<f64>)> {
    let raw: Vec<f64> = match weights {
        FitWeights::LogEqual => freqs.iter().map(|&f| 1.0 / f.max(1e-9)).collect(),
        FitWeights::Custom(w) => w.clone(),
    };
    let mean = raw.iter().sum::<f64>() / raw.len() as f64;
    if !(mean.is_finite() && mean > 0.0) {
        return Err(KautzError::IncompatibleBounds {
            reason: "weights are all zero (no measured point would count)".to_string(),
        });
    }
    let sqrtw: Vec<f64> = raw.iter().map(|&w| (w / mean).sqrt()).collect();
    Ok((raw.iter().map(|&w| w / mean).collect(), sqrtw))
}

/// Ordered basis matrix: `basis[i][k] = B_k(freqs[i])` with the allpass
/// chain accumulated in section order, matching sample processing.
fn basis_matrix(modes: &[(f64, f64)], srate: f64, freqs: &[f64]) -> Vec<Vec<Complex<f64>>> {
    let bank = modes
        .iter()
        .map(|&(freq, q)| {
            let theta = 2.0 * std::f64::consts::PI * freq / srate;
            let radius = (-std::f64::consts::PI * freq / (q * srate)).exp();
            let r2 = radius * radius;
            (r2, -2.0 * radius * theta.cos())
        })
        .collect::<Vec<_>>();
    freqs
        .iter()
        .map(|&freq| {
            let omega = 2.0 * std::f64::consts::PI * freq / srate;
            let z_inv = Complex::from_polar(1.0, -omega);
            let z_inv2 = z_inv * z_inv;
            let mut chain = Complex::new(1.0, 0.0);
            let mut row = Vec::with_capacity(bank.len());
            for &(r2, a1) in &bank {
                let den = Complex::new(1.0, 0.0) + z_inv * a1 + z_inv2 * r2;
                let num = Complex::new((1.0 - r2).sqrt() * (1.0 - r2), 0.0);
                row.push(num / den * chain);
                let ap_num = Complex::new(r2, 0.0) + z_inv * a1 + z_inv2;
                chain *= ap_num / den;
            }
            row
        })
        .collect()
}

/// Build the guard grid: measurement points plus DC, Nyquist and pole
/// neighborhoods placed from each pole's 3 dB bandwidth.
fn guard_grid(
    bank: &KautzCorrection<f64>,
    freqs: &[f64],
    per_side: usize,
    band: Option<(f64, f64)>,
) -> GuardGrid {
    let srate = bank.sample_rate();
    let nyquist = srate / 2.0;
    let mut grid: Vec<f64> = freqs.to_vec();
    grid.push(0.0);
    grid.push(nyquist);
    for &(freq, q) in bank.modes() {
        let radius = (-std::f64::consts::PI * freq / (q * srate))
            .exp()
            .min(0.999_999);
        let bw = -srate * radius.ln() / std::f64::consts::PI;
        grid.push(freq);
        for k in 1..=per_side {
            let step = bw * k as f64 / per_side as f64;
            if freq - step > 0.0 {
                grid.push(freq - step);
            }
            if freq + step < nyquist {
                grid.push(freq + step);
            }
        }
    }
    grid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    grid.dedup();
    GuardGrid {
        freqs: grid,
        measurement_count: freqs.len(),
        pole_points_per_side: per_side,
        includes_dc: true,
        includes_nyquist: true,
        band,
    }
}

fn correction_db_at(basis_row: &[Complex<f64>], gains: &[f64]) -> f64 {
    let mut h = Complex::new(1.0, 0.0);
    for (basis, &g) in basis_row.iter().zip(gains.iter()) {
        h += *basis * g;
    }
    20.0 * h.norm().max(MAG_FLOOR).log10()
}

/// Composite-bound check of a gain vector on the guard grid. Returns the
/// report; `passed == false` names a bound violation, never a fit failure.
fn check_gains(
    guard_basis: &[Vec<Complex<f64>>],
    guard_freqs: &[f64],
    gains: &[f64],
    config: &FitConfig,
) -> (bool, f64, f64, f64, f64, f64, f64) {
    let mut worst_boost = f64::NEG_INFINITY;
    let mut worst_boost_hz = 0.0;
    let mut worst_cut = f64::INFINITY;
    let mut worst_cut_hz = 0.0;
    let mut worst_neut = 0.0;
    let mut worst_neut_hz = 0.0;
    let mut passed = true;
    for (row, &freq) in guard_basis.iter().zip(guard_freqs.iter()) {
        let db = correction_db_at(row, gains);
        if db > worst_boost {
            worst_boost = db;
            worst_boost_hz = freq;
        }
        if db < worst_cut {
            worst_cut = db;
            worst_cut_hz = freq;
        }
        if db > config.max_boost_db || db < -config.max_cut_db {
            passed = false;
        }
        if let Some((lo, hi)) = config.correction_band {
            if (freq < lo || freq > hi) && db.abs() > worst_neut {
                worst_neut = db.abs();
                worst_neut_hz = freq;
            }
            if (freq < lo || freq > hi) && db.abs() > config.neutrality_db {
                passed = false;
            }
        }
        for band in &config.extra_constraints {
            if freq >= band.lo_hz
                && freq <= band.hi_hz
                && (db > band.max_boost_db || db < -band.max_cut_db)
            {
                passed = false;
            }
        }
    }
    (
        passed,
        worst_boost,
        worst_boost_hz,
        worst_cut,
        worst_cut_hz,
        worst_neut,
        worst_neut_hz,
    )
}

/// Check a live bank against composite bounds without fitting.
///
/// Builds the guard grid from the bank's modes and rate, evaluates the
/// realized transfer on it and reports worst values with frequencies.
/// Invalid configuration is refused; bound violations read `passed == false`.
pub fn check_bounds(bank: &KautzCorrection<f64>, config: &FitConfig) -> KautzResult<GuardReport> {
    check_config(config)?;
    let grid = guard_grid(bank, &[], config.pole_guard_points, config.correction_band);
    let guard_basis = basis_matrix(bank.modes(), bank.sample_rate(), &grid.freqs);
    let gains = bank.gains();
    let gains_scaled: Vec<f64> = gains.iter().map(|&g| bank.strength() * g).collect();
    let (passed, worst_boost, worst_boost_hz, worst_cut, worst_cut_hz, worst_neut, worst_neut_hz) =
        check_gains(&guard_basis, &grid.freqs, &gains_scaled, config);
    Ok(GuardReport {
        worst_boost_db: worst_boost,
        worst_boost_hz,
        worst_cut_db: worst_cut,
        worst_cut_hz,
        neutrality_violation_db: worst_neut,
        neutrality_violation_hz: worst_neut_hz,
        passed,
        grid,
    })
}

fn objective(
    meas_basis: &[Vec<Complex<f64>>],
    gains: &[f64],
    target_shifted: &[f64],
    weights: &[f64],
) -> f64 {
    let mut sum = 0.0;
    for ((row, &t), &w) in meas_basis
        .iter()
        .zip(target_shifted.iter())
        .zip(weights.iter())
    {
        let err = correction_db_at(row, gains) - t;
        sum += w * err * err;
    }
    sum
}

fn abs_rms(meas_basis: &[Vec<Complex<f64>>], gains: &[f64], target: &[f64]) -> f64 {
    let sum: f64 = meas_basis
        .iter()
        .zip(target.iter())
        .map(|(row, &t)| {
            let err = correction_db_at(row, gains) - t;
            err * err
        })
        .sum();
    (sum / target.len() as f64).sqrt()
}

/// Fit the bank's realized magnitude to a declared dB correction target.
///
/// `target_db[i]` is the desired `20·log10|H(freqs[i])|` (positive boosts).
/// Gains start from the bank's current values and are committed only as a
/// complete validated vector; invalid input returns `Err` with the bank
/// untouched. Termination is reported honestly: only
/// [`Converged`](FitStatus::Converged) claims an optimum.
pub fn fit_correction(
    bank: &mut KautzCorrection<f64>,
    freqs: &[f64],
    target_db: &[f64],
    config: &FitConfig,
) -> KautzResult<FitDiagnostics> {
    check_grid(freqs, target_db, &config.weights)?;
    check_config(config)?;
    let (weights, sqrtw) = measurement_weights(freqs, &config.weights)?;

    // Normalization: shift the working target, remember the shift, and
    // always report the absolute error beside the policy objective.
    let mut shift = 0.0;
    if let NormalizationPolicy::MeanAnchored {
        band_lo_hz,
        band_hi_hz,
    } = config.normalization
    {
        let mut num = 0.0;
        let mut den = 0.0;
        for ((&f, &t), &w) in freqs.iter().zip(target_db.iter()).zip(weights.iter()) {
            if f >= band_lo_hz && f <= band_hi_hz {
                num += w * t;
                den += w;
            }
        }
        if den <= 0.0 {
            return Err(KautzError::IncompatibleBounds {
                reason: "anchor band covers no measurement grid point".to_string(),
            });
        }
        shift = num / den;
    }
    let target_shifted: Vec<f64> = target_db.iter().map(|&t| t - shift).collect();

    let modes = bank.modes().to_vec();
    let srate = bank.sample_rate();
    let meas_basis = basis_matrix(&modes, srate, freqs);
    let grid = guard_grid(
        bank,
        freqs,
        config.pole_guard_points,
        config.correction_band,
    );
    let guard_basis = basis_matrix(&modes, srate, &grid.freqs);
    let m = modes.len();

    // Gains enter the search strength-scaled; the bank stores raw gains.
    let strength = bank.strength();
    let mut best: Vec<f64> = bank.gains().iter().map(|&g| strength * g).collect();
    let initial_objective = objective(&meas_basis, &best, &target_shifted, &weights);
    let initial_abs = abs_rms(&meas_basis, &best, target_db);
    let mut current_objective = initial_objective;

    let mut status = FitStatus::Stalled;
    let mut iterations_used = 0;
    let mut improved_once = false;
    // A start point that already violates bounds refuses to search from an
    // illegal state: report bounds-active with gains unchanged.
    let (start_ok, ..) = check_gains(&guard_basis, &grid.freqs, &best, config);
    if !start_ok {
        status = FitStatus::BoundsActive;
    } else {
        for iter in 0..config.max_iterations {
            // Analytic Jacobian of the weighted dB residual.
            let mut jac = vec![0.0f64; freqs.len() * m];
            let mut neg_res = vec![0.0f64; freqs.len()];
            for (i, row) in meas_basis.iter().enumerate() {
                let mut h = Complex::new(1.0, 0.0);
                for (basis, &g) in row.iter().zip(best.iter()) {
                    h += *basis * g;
                }
                let mag_sq = (h.norm_sqr()).max(MAG_FLOOR * MAG_FLOOR);
                let residual = 20.0 * mag_sq.sqrt().log10() - target_shifted[i];
                neg_res[i] = -sqrtw[i] * residual;
                let conj_h = h.conj();
                for (k, basis) in row.iter().enumerate() {
                    jac[i * m + k] = sqrtw[i] * DB_FACTOR * (basis * conj_h).re / mag_sq;
                }
            }
            let step = match qr_least_squares(freqs.len(), m, &jac, &neg_res, config.regularization)
            {
                Some(step) => step,
                None => {
                    status = FitStatus::IterationExhausted;
                    break;
                }
            };
            // Trial with backtracking on guard violations and non-finites.
            let mut trial: Vec<f64> = best.iter().zip(step.iter()).map(|(g, d)| g + d).collect();
            let mut accepted = false;
            let mut scale = 1.0;
            for _ in 0..=MAX_BACKTRACKS {
                let finite = trial.iter().all(|v| v.is_finite());
                let (trial_ok, ..) = if finite {
                    check_gains(&guard_basis, &grid.freqs, &trial, config)
                } else {
                    (false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                };
                if trial_ok {
                    let trial_obj = objective(&meas_basis, &trial, &target_shifted, &weights);
                    if trial_obj <= current_objective {
                        let improvement = current_objective - trial_obj;
                        current_objective = trial_obj;
                        best = trial;
                        iterations_used = iter + 1;
                        if improvement > config.tolerance * (1.0 + current_objective) {
                            improved_once = true;
                        } else if improved_once {
                            status = FitStatus::Converged;
                        }
                        accepted = true;
                        break;
                    }
                }
                scale *= 0.5;
                trial = best
                    .iter()
                    .zip(step.iter())
                    .map(|(g, d)| g + scale * d)
                    .collect();
            }
            if !accepted {
                status = FitStatus::BoundsActive;
                break;
            }
            if scale < 1.0 / 256.0 {
                // The step had to shrink 256x to respect the guards: the
                // bounds, not the objective, are driving from here.
                status = FitStatus::BoundsActive;
                break;
            }
            if status == FitStatus::Converged {
                break;
            }
            if iter + 1 == config.max_iterations {
                status = FitStatus::IterationExhausted;
            }
        }
        if status == FitStatus::Stalled && improved_once {
            status = FitStatus::Converged;
        }
    }

    // Commit the complete validated vector (strength divided back out).
    bank.set_gains(&best_scaled_back(strength, &best))?;
    let final_abs = abs_rms(&meas_basis, &best, target_db);
    let final_norm = abs_rms(&meas_basis, &best, &target_shifted);
    let (_, worst_boost, worst_boost_hz, worst_cut, worst_cut_hz, _, _) =
        check_gains(&guard_basis, &grid.freqs, &best, config);
    Ok(FitDiagnostics {
        status,
        iterations_used,
        initial_objective,
        final_objective: current_objective,
        initial_abs_rms_db: initial_abs,
        final_abs_rms_db: final_abs,
        final_normalized_rms_db: final_norm,
        normalization_shift_db: shift,
        worst_boost_db: worst_boost,
        worst_boost_hz,
        worst_cut_db: worst_cut,
        worst_cut_hz,
        guard_grid: grid,
    })
}

/// Divide strength back out of searched (effective) gains for storage.
fn best_scaled_back(strength: f64, best: &[f64]) -> Vec<f64> {
    if strength == 0.0 {
        return vec![0.0; best.len()];
    }
    best.iter().map(|&g| g / strength).collect()
}

//! Trust-region constrained Bayesian optimisation (SCBO-lite).
//!
//! A single trust region (TuRBO-style) combined with constrained expected
//! improvement: one Gaussian process models the objective and one more
//! models each constraint, all on normalised coordinates. Candidates are
//! drawn inside the trust region and scored by `EI * P(feasible)` once a
//! feasible point is known, or by feasibility probability alone while
//! searching for one. The region expands on success streaks, shrinks on
//! failure streaks, and restarts from a global acquisition scan when it
//! collapses.

use super::bayes_opt_config::{BayesOptConfig, fit_gp, initial_design};
use super::derive::{derive_candidate_pool_size, derive_initial_samples, derive_lengthscales};
use super::expected::expected_improvement;
use super::gaussian_process::GaussianProcess;
use super::misc::{denormalize, is_new_point, make_rng, normalize, squared_distance};
use super::normal::normal_cdf;
use crate::CallbackAction;
use crate::error::{DEError, Result};
use crate::init_sobol::init_halton;
use crate::parallel_eval::ParallelConfig;
use ndarray::Array1;
use rand::RngExt;
use rand::rngs::StdRng;
use std::sync::Arc;

/// Erased inequality-constraint closure: feasible when `<= 0`.
pub type BayesOptConstraintFn = Arc<dyn Fn(&Array1<f64>) -> f64 + Send + Sync>;

/// A single inequality constraint `fun(x) <= 0` for
/// [`constrained_bayesian_optimization`].
#[derive(Clone)]
pub struct BayesOptConstraint {
    /// Constraint function. Feasible when `<= 0`.
    pub fun: BayesOptConstraintFn,
}

/// Per-iteration callback payload for [`constrained_bayesian_optimization`].
pub struct ConstrainedBayesOptIntermediate {
    /// Current best parameter vector (best feasible, or least-violating).
    pub x: Array1<f64>,
    /// Objective value at [`Self::x`].
    pub fun: f64,
    /// Maximum constraint violation at [`Self::x`] (0 when feasible).
    pub max_violation: f64,
    /// Whether [`Self::x`] is feasible.
    pub feasible: bool,
    /// BO iterations completed after the initial design.
    pub iter: usize,
    /// Objective evaluations consumed so far.
    pub nfev: usize,
}

/// Callback type used by [`ConstrainedBayesOptConfig`].
pub type ConstrainedBayesOptCallback =
    Box<dyn FnMut(&ConstrainedBayesOptIntermediate) -> CallbackAction + Send>;

/// Configuration for [`constrained_bayesian_optimization`].
pub struct ConstrainedBayesOptConfig {
    /// `(lower, upper)` bounds per parameter.
    pub bounds: Vec<(f64, f64)>,
    /// Optional initial point. Values outside bounds are clipped.
    pub x0: Option<Array1<f64>>,
    /// Number of initial design samples. `0` uses a dimension default.
    pub initial_samples: usize,
    /// Candidates evaluated per BO iteration. `0` uses `1`.
    pub batch_size: usize,
    /// Maximum objective evaluations.
    pub maxeval: usize,
    /// Acquisition candidates drawn per iteration. `0` uses a default.
    pub candidate_pool_size: usize,
    /// Optional ARD lengthscales in normalised coordinates.
    pub lengthscales: Option<Vec<f64>>,
    /// Signal variance for the Matérn-5/2 kernel.
    pub kernel_variance: f64,
    /// Diagonal observation-noise/jitter floor.
    pub noise: f64,
    /// EI exploration offset. Larger values prefer uncertain candidates.
    pub exploration: f64,
    /// Stop once a feasible point reaches this objective value.
    pub target_f: f64,
    /// Initial trust-region side length (normalised units). `0.0` uses 0.8.
    pub tr_length_init: f64,
    /// Minimum trust-region length before a restart. `0.0` uses 2^-7.
    pub tr_length_min: f64,
    /// Maximum trust-region length. `0.0` uses 1.6.
    pub tr_length_max: f64,
    /// Consecutive failures before halving the region. `0` uses `dim`.
    pub tr_fail_tol: usize,
    /// Consecutive successes before doubling the region. `0` uses 3.
    pub tr_succ_tol: usize,
    /// Fraction of `maxeval` reserved for a final COBYLA polish on the true
    /// functions from the best point. `0.0` disables it. Default 0.1.
    pub polish_fraction: f64,
    /// Parallel batch-evaluation configuration.
    pub parallel: ParallelConfig,
    /// Optional RNG seed for deterministic runs.
    pub seed: Option<u64>,
    /// Optional per-iteration callback.
    pub callback: Option<ConstrainedBayesOptCallback>,
}

impl Default for ConstrainedBayesOptConfig {
    fn default() -> Self {
        Self {
            bounds: Vec::new(),
            x0: None,
            initial_samples: 0,
            batch_size: 1,
            maxeval: 300,
            candidate_pool_size: 0,
            lengthscales: None,
            kernel_variance: 1.0,
            noise: 1e-8,
            exploration: 0.01,
            target_f: f64::NEG_INFINITY,
            tr_length_init: 0.0,
            tr_length_min: 0.0,
            tr_length_max: 0.0,
            tr_fail_tol: 0,
            tr_succ_tol: 0,
            polish_fraction: 0.1,
            parallel: ParallelConfig::default(),
            seed: None,
            callback: None,
        }
    }
}

/// Result of a [`constrained_bayesian_optimization`] run.
#[derive(Clone)]
pub struct ConstrainedBayesOptReport {
    /// Best parameter vector found (best feasible, or least-violating).
    pub x: Array1<f64>,
    /// Objective value at [`Self::x`].
    pub fun: f64,
    /// Whether [`Self::x`] is feasible.
    pub feasible: bool,
    /// Maximum constraint violation at [`Self::x`] (0 when feasible).
    pub max_violation: f64,
    /// Whether the run met a target/callback stop condition before
    /// exhausting the evaluation budget.
    pub success: bool,
    /// Human-readable termination message.
    pub message: String,
    /// Objective evaluations consumed.
    pub nfev: usize,
    /// BO iterations completed after the initial design.
    pub nit: usize,
}

impl std::fmt::Debug for ConstrainedBayesOptReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConstrainedBayesOptReport")
            .field("x_len", &self.x.len())
            .field("fun", &self.fun)
            .field("feasible", &self.feasible)
            .field("max_violation", &self.max_violation)
            .field("success", &self.success)
            .field("message", &self.message)
            .field("nfev", &self.nfev)
            .field("nit", &self.nit)
            .finish()
    }
}

/// Minimise `f` subject to `constraints` with trust-region constrained BO.
///
/// Each entry of `constraints` must be `<= 0` at feasible points. One GP
/// models the objective and one more models each constraint; candidates are
/// scored by constrained expected improvement inside a single adaptive
/// trust region.
pub fn constrained_bayesian_optimization<F>(
    f: &F,
    constraints: &[BayesOptConstraint],
    mut config: ConstrainedBayesOptConfig,
) -> Result<ConstrainedBayesOptReport>
where
    F: Fn(&Array1<f64>) -> f64 + Sync,
{
    validate_constrained_config(&config)?;
    let n = config.bounds.len();
    let batch_size = config.batch_size.max(1);
    // Scratch base config so the shared BO helpers (initial design,
    // GP fitting, derived defaults) can be reused directly.
    let base = BayesOptConfig {
        bounds: config.bounds.clone(),
        x0: config.x0.clone(),
        initial_samples: config.initial_samples,
        candidate_pool_size: config.candidate_pool_size,
        lengthscales: config.lengthscales.clone(),
        kernel_variance: config.kernel_variance,
        noise: config.noise,
        ..Default::default()
    };
    let initial_samples = derive_initial_samples(n, &base).min(config.maxeval);
    let pool_size = derive_candidate_pool_size(n, &base).clamp(64, 2048);
    let lengthscales_0 = derive_lengthscales(n, &base)?;
    let fixed_lengthscales = config.lengthscales.is_some();
    let mut rng = make_rng(config.seed);

    let tr_init = if config.tr_length_init > 0.0 {
        config.tr_length_init
    } else {
        0.8
    };
    let tr_min = if config.tr_length_min > 0.0 {
        config.tr_length_min
    } else {
        0.5f64.powi(7)
    };
    let tr_max = if config.tr_length_max > 0.0 {
        config.tr_length_max
    } else {
        1.6
    };
    let fail_tol = if config.tr_fail_tol == 0 {
        n.max(1)
    } else {
        config.tr_fail_tol
    };
    let succ_tol = if config.tr_succ_tol == 0 {
        3
    } else {
        config.tr_succ_tol
    };

    // Initial design and evaluation.
    let mut design = initial_design(&base, initial_samples);
    let mut xs_norm: Vec<Array1<f64>> = Vec::with_capacity(config.maxeval);
    let mut ys: Vec<f64> = Vec::with_capacity(config.maxeval);
    let mut cons: Vec<Vec<f64>> = Vec::with_capacity(config.maxeval);
    evaluate_constrained(
        f,
        constraints,
        &mut design,
        &config,
        &mut xs_norm,
        &mut ys,
        &mut cons,
    );

    let mut best = best_indices(&ys, &cons);
    let mut center_norm = best_center(&xs_norm, best);
    let mut tr_length = tr_init;
    let mut succ = 0usize;
    let mut fail = 0usize;
    let mut nit = 0usize;
    let mut nfev = ys.len();
    let mut success = false;
    let mut message = String::from("evaluation budget reached");
    if nfev >= config.maxeval {
        message = String::from("initial design consumed evaluation budget");
    }
    // Evaluations reserved for the final polish (plus one for the final
    // true evaluation at the polished point).
    let reserve = if config.polish_fraction > 0.0 {
        ((config.maxeval as f64 * config.polish_fraction.clamp(0.0, 0.5)) as usize + 1)
            .min(config.maxeval)
    } else {
        0
    };

    while nfev + reserve < config.maxeval {
        let gp_obj = fit_gp(&xs_norm, &ys, &lengthscales_0, fixed_lengthscales, &base)?;
        // Constraint GPs share the objective's learned lengthscales: joint
        // per-model lengthscale learning would multiply the O(n^3) fits.
        let obj_ls = gp_obj.lengthscales.clone();
        let kernel_variance = config.kernel_variance.max(1e-12);
        let mut gp_cons = Vec::with_capacity(constraints.len());
        for c in 0..constraints.len() {
            let yc: Vec<f64> = cons.iter().map(|v| v[c]).collect();
            gp_cons.push(GaussianProcess::fit(
                &xs_norm,
                &yc,
                &obj_ls,
                kernel_variance,
                config.noise,
            )?);
        }

        let half_widths = trust_half_widths(&obj_ls, tr_length);
        let candidates = trust_pool(
            &center_norm,
            &half_widths,
            &config,
            pool_size,
            &xs_norm,
            &mut rng,
        );
        if candidates.is_empty() {
            message = String::from("candidate pool exhausted");
            break;
        }
        let feasible_best = best.0.map(|i| ys[i]).unwrap_or(f64::INFINITY);
        let remaining = config.maxeval - nfev - reserve;
        let mut selected = select_constrained(
            &gp_obj,
            &gp_cons,
            &candidates,
            feasible_best,
            batch_size.min(remaining),
            config.exploration,
            &config.bounds,
        );
        let before = best_point(&ys, &cons, best);
        evaluate_constrained(
            f,
            constraints,
            &mut selected,
            &config,
            &mut xs_norm,
            &mut ys,
            &mut cons,
        );
        nit += 1;
        nfev += selected.len();
        best = best_indices(&ys, &cons);
        let after = best_point(&ys, &cons, best);

        // Trust-region update on feasibility-first improvement.
        let improved = if after.3 {
            !before.3 || after.1 < before.1
        } else {
            !before.3 && after.2 < before.2
        };
        if improved {
            succ += 1;
            fail = 0;
        } else {
            succ = 0;
            fail += 1;
        }
        if succ >= succ_tol {
            tr_length = (2.0 * tr_length).min(tr_max);
            succ = 0;
        } else if fail >= fail_tol {
            tr_length /= 2.0;
            fail = 0;
        }
        center_norm = best_center(&xs_norm, best);
        if tr_length < tr_min {
            // Restart the region from a global acquisition scan.
            tr_length = tr_init;
            succ = 0;
            fail = 0;
            if let Some(global) = global_recenter(
                &gp_obj,
                &gp_cons,
                &config,
                pool_size,
                feasible_best,
                config.exploration,
                &xs_norm,
                &mut rng,
            ) {
                center_norm = global;
            }
        }

        let (cb_x, cb_fun, cb_viol, cb_feasible) =
            best_point_owned(&xs_norm, &ys, &cons, best, &config.bounds);
        if let Some(callback) = config.callback.as_mut() {
            let payload = ConstrainedBayesOptIntermediate {
                x: cb_x,
                fun: cb_fun,
                max_violation: cb_viol,
                feasible: cb_feasible,
                iter: nit,
                nfev,
            };
            if matches!(callback(&payload), CallbackAction::Stop) {
                success = true;
                message = String::from("stopped by callback");
                break;
            }
        }
        let (_, bfun, _, feasible) = best_point(&ys, &cons, best);
        if feasible && bfun <= config.target_f {
            success = true;
            message = format!("target_f reached: {bfun:.6e}");
            break;
        }
    }

    if !success {
        // Final polish: COBYLA on the true functions from the best point.
        // COBYLA reports its configured budget as consumed, so one
        // evaluation is held back to score the polished point exactly.
        let remaining = config.maxeval - nfev;
        let (bx, _, _, _) = best_point_owned(&xs_norm, &ys, &cons, best, &config.bounds);
        if reserve > 0 && remaining >= 2 {
            let min_width = config
                .bounds
                .iter()
                .map(|(lo, hi)| hi - lo)
                .fold(f64::INFINITY, f64::min);
            let model_cons: Vec<crate::cobyla::CobylaConstraint> = constraints
                .iter()
                .map(|c| crate::cobyla::CobylaConstraint { fun: c.fun.clone() })
                .collect();
            let cfg = crate::cobyla::CobylaConfig {
                x0: bx,
                bounds: config.bounds.clone(),
                rho_begin: crate::cobyla::CobylaRhoBegin::All(0.1 * min_width),
                maxeval: remaining - 1,
                ..Default::default()
            };
            let polished = match crate::cobyla::cobyla(f, &model_cons, cfg) {
                Ok(report) => report.x,
                Err(_) => random_original(&config.bounds, &mut rng),
            };
            let mut endpoint = vec![polished];
            evaluate_constrained(
                f,
                constraints,
                &mut endpoint,
                &config,
                &mut xs_norm,
                &mut ys,
                &mut cons,
            );
            nfev = config.maxeval;
            best = best_indices(&ys, &cons);
        } else {
            while nfev < config.maxeval {
                let mut filler = vec![random_original(&config.bounds, &mut rng)];
                evaluate_constrained(
                    f,
                    constraints,
                    &mut filler,
                    &config,
                    &mut xs_norm,
                    &mut ys,
                    &mut cons,
                );
                nfev += 1;
            }
            best = best_indices(&ys, &cons);
        }
    }

    let (bx, bfun, bviol, feasible) = best_point_owned(&xs_norm, &ys, &cons, best, &config.bounds);
    Ok(ConstrainedBayesOptReport {
        x: bx,
        fun: bfun,
        feasible,
        max_violation: bviol,
        success,
        message,
        nfev,
        nit,
    })
}

/// Uniform random point in the original bounded coordinates.
fn random_original(bounds: &[(f64, f64)], rng: &mut StdRng) -> Array1<f64> {
    denormalize(
        &Array1::from_vec(
            (0..bounds.len())
                .map(|_| rng.random_range(0.0..=1.0))
                .collect(),
        ),
        bounds,
    )
}

fn validate_constrained_config(config: &ConstrainedBayesOptConfig) -> Result<()> {
    if config.bounds.is_empty() {
        return Err(DEError::BoundsMismatch {
            lower_len: 0,
            upper_len: 0,
        });
    }
    for (i, (lo, hi)) in config.bounds.iter().enumerate() {
        if lo > hi {
            return Err(DEError::InvalidBounds {
                index: i,
                lower: *lo,
                upper: *hi,
            });
        }
    }
    if let Some(ref x0) = config.x0
        && x0.len() != config.bounds.len()
    {
        return Err(DEError::X0DimensionMismatch {
            expected: config.bounds.len(),
            got: x0.len(),
        });
    }
    if config.maxeval == 0 {
        return Err(DEError::InvalidConfig {
            message: "maxeval must be greater than zero".into(),
        });
    }
    Ok(())
}

fn evaluate_constrained<F>(
    f: &F,
    constraints: &[BayesOptConstraint],
    candidates: &mut [Array1<f64>],
    config: &ConstrainedBayesOptConfig,
    xs_norm: &mut Vec<Array1<f64>>,
    ys: &mut Vec<f64>,
    cons: &mut Vec<Vec<f64>>,
) where
    F: Fn(&Array1<f64>) -> f64 + Sync,
{
    for x in candidates.iter() {
        let y = f(x);
        xs_norm.push(normalize(x, &config.bounds));
        ys.push(if y.is_finite() { y } else { f64::INFINITY });
        cons.push(constraints.iter().map(|c| (c.fun)(x)).collect());
    }
}

/// `(best_feasible, least_violating)` data indices.
fn best_indices(ys: &[f64], cons: &[Vec<f64>]) -> (Option<usize>, usize) {
    let mut feasible: Option<usize> = None;
    let mut least_violating = 0usize;
    for (i, (y, c)) in ys.iter().zip(cons.iter()).enumerate() {
        if max_violation(c) < max_violation(&cons[least_violating]) {
            least_violating = i;
        }
        if max_violation(c) <= 0.0 {
            let better = feasible.map(|b| *y < ys[b]).unwrap_or(true);
            if better {
                feasible = Some(i);
            }
        }
    }
    (feasible, least_violating)
}

fn max_violation(values: &[f64]) -> f64 {
    let mut worst = 0.0f64;
    for &v in values {
        let v = if v.is_nan() { f64::INFINITY } else { v };
        worst = worst.max(v);
    }
    worst
}

/// `(index, fun-or-viol-value, violation, feasible)` of the current best.
fn best_point(
    ys: &[f64],
    cons: &[Vec<f64>],
    best: (Option<usize>, usize),
) -> (usize, f64, f64, bool) {
    if let Some(i) = best.0 {
        (i, ys[i], 0.0, true)
    } else {
        (best.1, ys[best.1], max_violation(&cons[best.1]), false)
    }
}

fn best_point_owned(
    xs_norm: &[Array1<f64>],
    ys: &[f64],
    cons: &[Vec<f64>],
    best: (Option<usize>, usize),
    bounds: &[(f64, f64)],
) -> (Array1<f64>, f64, f64, bool) {
    let (i, fun, viol, feasible) = best_point(ys, cons, best);
    (denormalize(&xs_norm[i], bounds), fun, viol, feasible)
}

fn best_center(xs_norm: &[Array1<f64>], best: (Option<usize>, usize)) -> Array1<f64> {
    xs_norm[best.0.unwrap_or(best.1)].clone()
}

/// Per-dimension trust-region half-widths, scaled by learned lengthscales
/// (TuRBO-style): `h_i = L/2 * l_i / gmean(l)`.
fn trust_half_widths(lengthscales: &[f64], length: f64) -> Vec<f64> {
    let log_mean =
        lengthscales.iter().map(|l| l.max(1e-12).ln()).sum::<f64>() / lengthscales.len() as f64;
    let geometric_mean = log_mean.exp();
    lengthscales
        .iter()
        .map(|l| 0.5 * length * l.max(1e-12) / geometric_mean)
        .collect()
}

/// Uniform candidate pool inside the trust box (normalised coordinates
/// clipped to the unit cube), returned in original coordinates.
fn trust_pool(
    center_norm: &Array1<f64>,
    half_widths: &[f64],
    config: &ConstrainedBayesOptConfig,
    size: usize,
    existing_norm: &[Array1<f64>],
    rng: &mut StdRng,
) -> Vec<Array1<f64>> {
    let mut candidates = Vec::with_capacity(size);
    let mut attempts = 0usize;
    while candidates.len() < size && attempts < size.saturating_mul(50).max(2000) {
        attempts += 1;
        let x_norm = Array1::from_vec(
            center_norm
                .iter()
                .zip(half_widths.iter())
                .map(|(&c, &h)| rng.random_range((c - h).max(0.0)..=(c + h).min(1.0)))
                .collect(),
        );
        let x = denormalize(&x_norm, &config.bounds);
        if is_new_point(&x_norm, existing_norm, &candidates, &config.bounds) {
            candidates.push(x);
        }
    }
    candidates
}

/// Greedy top-`q` diverse selection by constrained EI (or feasibility
/// probability while nothing feasible is known). Candidates arrive in
/// original coordinates; scoring and diversity use normalised ones.
fn select_constrained(
    gp_obj: &GaussianProcess,
    gp_cons: &[GaussianProcess],
    candidates: &[Array1<f64>],
    feasible_best: f64,
    q: usize,
    exploration: f64,
    bounds: &[(f64, f64)],
) -> Vec<Array1<f64>> {
    let mut scored: Vec<(f64, Array1<f64>, Array1<f64>)> = candidates
        .iter()
        .map(|x| {
            let x_norm = normalize(x, bounds);
            (
                constrained_score(gp_obj, gp_cons, &x_norm, feasible_best, exploration),
                x.clone(),
                x_norm,
            )
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut selected = Vec::with_capacity(q);
    let mut selected_norm = Vec::with_capacity(q);
    for (_, x, x_norm) in scored {
        let diverse = selected_norm
            .iter()
            .all(|s| squared_distance(&x_norm, s) > 1e-8);
        if diverse {
            selected.push(x);
            selected_norm.push(x_norm);
        }
        if selected.len() == q {
            break;
        }
    }
    selected
}

/// `EI * P(feasible)` once feasible (candidates are in normalised
/// coordinates); pure feasibility probability otherwise.
fn constrained_score(
    gp_obj: &GaussianProcess,
    gp_cons: &[GaussianProcess],
    x_norm: &Array1<f64>,
    feasible_best: f64,
    exploration: f64,
) -> f64 {
    let mut p_feasible = 1.0;
    for gp in gp_cons {
        let (mean, std) = gp.predict(x_norm);
        p_feasible *= normal_cdf(-mean / std.max(1e-12));
    }
    if !feasible_best.is_finite() {
        return p_feasible;
    }
    let (mean, std) = gp_obj.predict(x_norm);
    expected_improvement(feasible_best, mean, std, exploration) * p_feasible
}

/// Global acquisition scan for trust-region restarts: best constrained
/// score over a space-filling pool, returned in normalised coordinates.
#[allow(clippy::too_many_arguments)]
fn global_recenter(
    gp_obj: &GaussianProcess,
    gp_cons: &[GaussianProcess],
    config: &ConstrainedBayesOptConfig,
    size: usize,
    feasible_best: f64,
    exploration: f64,
    existing_norm: &[Array1<f64>],
    rng: &mut StdRng,
) -> Option<Array1<f64>> {
    let mut best: Option<(f64, Array1<f64>)> = None;
    let mut consider = |x_norm: Array1<f64>| {
        if !is_new_point(&x_norm, existing_norm, &[], &config.bounds) {
            return;
        }
        let score = constrained_score(gp_obj, gp_cons, &x_norm, feasible_best, exploration);
        let better = best.as_ref().map(|(s, _)| score > *s).unwrap_or(true);
        if better {
            best = Some((score, x_norm));
        }
    };
    for s in init_halton(config.bounds.len(), size / 2, &config.bounds) {
        consider(normalize(&Array1::from(s), &config.bounds));
    }
    for _ in 0..size.div_ceil(2) {
        let x_norm = Array1::from_vec(
            (0..config.bounds.len())
                .map(|_| rng.random_range(0.0..=1.0))
                .collect(),
        );
        consider(x_norm);
    }
    best.map(|(_, x)| x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget(release: usize, debug: usize) -> usize {
        if cfg!(debug_assertions) {
            debug
        } else {
            release
        }
    }

    fn box_constraint() -> BayesOptConstraint {
        BayesOptConstraint {
            fun: Arc::new(|x: &Array1<f64>| 1.0 - x[0]),
        }
    }

    #[test]
    fn constrained_bo_respects_inequality_constraint() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let report = constrained_bayesian_optimization(
            &sphere,
            &[box_constraint()],
            ConstrainedBayesOptConfig {
                bounds: vec![(-5.0, 5.0); 2],
                maxeval: budget(120, 40),
                seed: Some(11),
                ..Default::default()
            },
        )
        .expect("constrained BO should run");
        assert!(report.feasible, "should find a feasible point");
        assert!(
            (report.fun - 1.0).abs() < 0.2,
            "constrained minimum is 1, got {}",
            report.fun
        );
    }

    #[test]
    fn constrained_bo_solves_g24() {
        use math_audio_test_functions::{g24_constraint1, g24_constraint2, g24_objective};
        let constraints = [
            g24_constraint1 as fn(&Array1<f64>) -> f64,
            g24_constraint2 as fn(&Array1<f64>) -> f64,
        ]
        .map(|g| BayesOptConstraint {
            fun: Arc::new(move |x: &Array1<f64>| g(x)),
        });
        let report = constrained_bayesian_optimization(
            &g24_objective,
            &constraints,
            ConstrainedBayesOptConfig {
                bounds: vec![(0.0, 3.0), (0.0, 4.0)],
                maxeval: budget(200, 60),
                seed: Some(12),
                ..Default::default()
            },
        )
        .expect("constrained BO should run");
        assert!(report.feasible, "should find a feasible point");
        assert!(
            report.fun <= -5.3,
            "should approach -5.508, got {}",
            report.fun
        );
    }

    #[test]
    fn constrained_bo_reports_infeasible_gracefully() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let impossible = BayesOptConstraint {
            fun: Arc::new(|x: &Array1<f64>| 100.0 - x[0]),
        };
        let report = constrained_bayesian_optimization(
            &sphere,
            &[impossible],
            ConstrainedBayesOptConfig {
                bounds: vec![(-5.0, 5.0); 2],
                maxeval: budget(60, 20),
                seed: Some(13),
                ..Default::default()
            },
        )
        .expect("constrained BO should run");
        assert!(!report.feasible);
        assert!(report.max_violation > 0.0);
        assert!(report.fun.is_finite());
    }

    #[test]
    fn constrained_bo_is_deterministic() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let run = || {
            constrained_bayesian_optimization(
                &sphere,
                &[box_constraint()],
                ConstrainedBayesOptConfig {
                    bounds: vec![(-5.0, 5.0); 2],
                    maxeval: budget(60, 20),
                    seed: Some(14),
                    ..Default::default()
                },
            )
            .expect("constrained BO should run")
        };
        let first = run();
        let second = run();
        assert_eq!(first.fun, second.fun);
        assert_eq!(first.x, second.x);
        assert_eq!(first.nfev, second.nfev);
    }

    #[test]
    fn constrained_bo_rejects_bad_config() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        assert!(
            constrained_bayesian_optimization(
                &sphere,
                &[],
                ConstrainedBayesOptConfig {
                    bounds: vec![],
                    ..Default::default()
                },
            )
            .is_err()
        );
        assert!(
            constrained_bayesian_optimization(
                &sphere,
                &[],
                ConstrainedBayesOptConfig {
                    bounds: vec![(-5.0, 5.0); 2],
                    maxeval: 0,
                    ..Default::default()
                },
            )
            .is_err()
        );
    }

    #[test]
    fn trust_half_widths_follow_lengthscales() {
        let widths = trust_half_widths(&[0.5, 2.0], 0.8);
        // Geometric mean is 1, so half-widths are L/2 * l_i.
        assert!((widths[0] - 0.2).abs() < 1e-12);
        assert!((widths[1] - 0.8).abs() < 1e-12);
    }
}

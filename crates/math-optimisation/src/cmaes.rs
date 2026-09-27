//! Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
//!
//! This is a pure-Rust, bounded, full-covariance CMA-ES implementation for
//! continuous black-box minimisation. Internally the search runs in a
//! normalised `[0, 1]^n` box so parameter scales such as log-frequency, Q, and
//! gain can share one covariance matrix without manual preconditioning.

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::Arc;

use crate::CallbackAction;
use crate::error::{DEError, Result};
use crate::parallel_eval::ParallelConfig;

thread_local! {
    static CMA_X_SCRATCH: RefCell<Array1<f64>> = RefCell::new(Array1::zeros(0));
}

/// Per-generation callback payload for [`cma_es`].
pub struct CmaEsIntermediate {
    /// Current best parameter vector in the original bounded coordinates.
    pub x: Array1<f64>,
    /// Current best objective value: best feasible value, or the objective
    /// at the least-violating point while nothing feasible is known.
    pub fun: f64,
    /// Current generation index.
    pub iter: usize,
    /// Number of objective evaluations consumed so far.
    pub nfev: usize,
    /// Current global step size in normalised coordinates.
    pub sigma: f64,
}

/// Callback type used by [`CmaEsConfig`].
pub type CmaEsCallback = Box<dyn FnMut(&CmaEsIntermediate) -> CallbackAction + Send>;

/// Erased inequality-constraint closure: feasible when `<= 0`.
pub type CmaEsConstraintFn = Arc<dyn Fn(&Array1<f64>) -> f64 + Send + Sync>;

/// A single inequality constraint `fun(x) <= 0` for [`cma_es`].
#[derive(Clone)]
pub struct CmaEsConstraint {
    /// Constraint function. Feasible when `<= 0`; positive values count as
    /// the violation magnitude used in stochastic ranking.
    pub fun: CmaEsConstraintFn,
}

/// Covariance model used by [`cma_es`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CmaCovariance {
    /// Full covariance matrix (default). Learns parameter couplings at
    /// O(n^3) eigendecomposition cost per refresh.
    #[default]
    Full,
    /// Diagonal-only (separable) covariance. Ignores couplings but adapts
    /// per-axis scales at O(n) cost — faster per generation and fewer
    /// evaluations to adapt on high-dimensional near-separable landscapes.
    Diagonal,
}

/// Configuration for [`cma_es`].
pub struct CmaEsConfig {
    /// `(lower, upper)` bounds per parameter.
    pub bounds: Vec<(f64, f64)>,
    /// Optional initial mean. Values outside bounds are clipped.
    pub x0: Option<Array1<f64>>,
    /// Initial step size in normalised `[0, 1]` coordinates.
    ///
    /// `None` uses `0.3`, the standard broad-search default for bounded
    /// CMA-ES. Smaller values are appropriate for local refinement.
    pub sigma0: Option<f64>,
    /// Offspring population size. `0` uses `4 + floor(3 ln(n))`.
    pub lambda: usize,
    /// Parent count. `0` uses `lambda / 2`.
    pub mu: usize,
    /// Maximum objective evaluations.
    pub maxeval: usize,
    /// Optional RNG seed for deterministic runs.
    pub seed: Option<u64>,
    /// Stop after this many generations with improvement below [`Self::f_tol`].
    pub stagnation_window: usize,
    /// Objective-improvement tolerance for stagnation detection.
    pub f_tol: f64,
    /// Stop once the best objective is at or below this value.
    pub target_f: f64,
    /// Maximum IPOP restarts after stagnation or step-size collapse.
    /// `0` (default) runs once; each restart grows the offspring population
    /// and resets the distribution while keeping the best point.
    pub max_restarts: usize,
    /// Offspring-population growth factor per restart. Must be `>= 1.0`
    /// when `max_restarts > 0`; `2.0` is the standard IPOP doubling.
    pub restart_lambda_growth: f64,
    /// Restart the mean at the best point so far (`true`, default) or at a
    /// fresh uniform-random point in the bounds (`false`).
    pub restart_from_best: bool,
    /// Optional per-generation callback. Returning [`CallbackAction::Stop`]
    /// terminates the run early and returns the best point seen so far.
    pub callback: Option<CmaEsCallback>,
    /// Parallel evaluation configuration for offspring fitness calls.
    pub parallel: ParallelConfig,
    /// Covariance model: full or diagonal-only (separable).
    pub covariance: CmaCovariance,
    /// Inequality constraints `g_i(x) <= 0`. Empty (default) means
    /// unconstrained. Offspring are selected by an adaptive-penalty merit
    /// (`f + w * violation^2` with a self-tuning weight) and the reported
    /// best is the best feasible point (or the least-violating point when
    /// nothing feasible was found).
    pub constraints: Vec<CmaEsConstraint>,
}

impl Default for CmaEsConfig {
    fn default() -> Self {
        Self {
            bounds: Vec::new(),
            x0: None,
            sigma0: None,
            lambda: 0,
            mu: 0,
            maxeval: 10_000,
            seed: None,
            stagnation_window: 80,
            f_tol: 1e-10,
            target_f: f64::NEG_INFINITY,
            max_restarts: 0,
            restart_lambda_growth: 2.0,
            restart_from_best: true,
            callback: None,
            parallel: ParallelConfig::default(),
            covariance: CmaCovariance::Full,
            constraints: Vec::new(),
        }
    }
}

/// Internal restartable termination reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CmaStopReason {
    Stagnation,
    SigmaCollapse,
}

/// Result of a [`cma_es`] run.
#[derive(Clone)]
pub struct CmaEsReport {
    /// Best parameter vector found.
    pub x: Array1<f64>,
    /// Objective value at [`Self::x`].
    pub fun: f64,
    /// Whether the run met a convergence/target/callback stop condition before
    /// exhausting the evaluation budget.
    pub success: bool,
    /// Human-readable termination message.
    pub message: String,
    /// Objective evaluations consumed.
    pub nfev: usize,
    /// Generations completed.
    pub nit: usize,
    /// Final global step size in normalised coordinates.
    pub sigma: f64,
    /// IPOP restarts performed during the run.
    pub restarts: usize,
    /// Whether the reported point satisfies all constraints. Always true
    /// for unconstrained runs.
    pub feasible: bool,
    /// Maximum constraint violation at the reported point (0 when feasible).
    pub max_violation: f64,
}

impl std::fmt::Debug for CmaEsReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CmaEsReport")
            .field("x_len", &self.x.len())
            .field("fun", &self.fun)
            .field("success", &self.success)
            .field("message", &self.message)
            .field("nfev", &self.nfev)
            .field("nit", &self.nit)
            .field("sigma", &self.sigma)
            .field("restarts", &self.restarts)
            .field("feasible", &self.feasible)
            .field("max_violation", &self.max_violation)
            .finish()
    }
}

/// Minimise `f` with bounded CMA-ES.
///
/// The objective receives parameters in the original coordinate system. Bounds
/// are handled by clipping sampled normalised points before evaluation.
/// When [`CmaEsConfig::constraints`] is non-empty, offspring are selected by
/// adaptive-penalty merit and the reported best is the best feasible point
/// (or the least-violating point when nothing feasible was found); constraint
/// values do not consume extra evaluations.
///
/// Note — clipping bound bias: probability mass sampled outside the box piles
/// up exactly on the boundary, so the adapted mean and the reported best point
/// can hug a bound even when the unconstrained optimum lies beyond it. Treat
/// boundary-hugging results as a hint to widen the bounds rather than as proof
/// that the optimum is on the bound.
pub fn cma_es<F>(f: &F, mut config: CmaEsConfig) -> Result<CmaEsReport>
where
    F: Fn(&Array1<f64>) -> f64 + Sync,
{
    let n = config.bounds.len();
    if n == 0 {
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
        && x0.len() != n
    {
        return Err(DEError::X0DimensionMismatch {
            expected: n,
            got: x0.len(),
        });
    }
    if config.max_restarts > 0
        && (config.restart_lambda_growth.is_nan() || config.restart_lambda_growth < 1.0)
    {
        return Err(DEError::InvalidConfig {
            message: format!(
                "restart_lambda_growth must be >= 1.0, got {}",
                config.restart_lambda_growth
            ),
        });
    }
    let constrained = !config.constraints.is_empty();

    let mut coeffs = population_coeffs(n, config.lambda, config.mu)?;
    let n_f = n as f64;
    let chi_n = n_f.sqrt() * (1.0 - 1.0 / (4.0 * n_f) + 1.0 / (21.0 * n_f * n_f));

    let mut mean = initial_mean(&config);
    let mut old_mean = DVector::<f64>::zeros(n);
    let sigma0_init = config.sigma0.unwrap_or(0.3).clamp(1e-12, 2.0);
    let mut sigma = sigma0_init;
    let mut covariance = DMatrix::<f64>::identity(n, n);
    let mut b = DMatrix::<f64>::identity(n, n);
    let mut d = DVector::<f64>::from_element(n, 1.0);
    let mut invsqrt_c = DMatrix::<f64>::identity(n, n);
    // Scratch buffer reused for the eigendecomposition so `covariance` is
    // never cloned per generation (only `copy_from` into existing storage).
    let mut eig_work = DMatrix::<f64>::zeros(n, n);
    // Lazy eigendecomposition schedule: the O(n^3) factorisation is refreshed
    // every `eig_gap` generations and the cached (B, D, C^{-1/2}) factors are
    // reused in between. Small problems refresh every generation.
    let eig_gap = (n / 4).clamp(1, 10);
    let mut gens_since_eig = eig_gap;
    let mut pc = DVector::<f64>::zeros(n);
    let mut ps = DVector::<f64>::zeros(n);

    // Reusable per-generation scratch buffers.
    let mut z_buf = DVector::<f64>::zeros(n);
    let mut scaled_z = DVector::<f64>::zeros(n);
    let mut step_buf = DVector::<f64>::zeros(n);
    let mut y_w = DVector::<f64>::zeros(n);
    let mut tmp_n = DVector::<f64>::zeros(n);
    let mut rank_mu = DMatrix::<f64>::zeros(n, n);

    // Pre-allocated population storage so offspring `DVector`s are reused across
    // generations instead of re-allocated each iteration.
    let mut y_pool: Vec<DVector<f64>> = Vec::with_capacity(coeffs.lambda);
    for _ in 0..coeffs.lambda {
        y_pool.push(DVector::<f64>::zeros(n));
    }
    let mut funs: Vec<f64> = vec![0.0; coeffs.lambda];
    let mut viols: Vec<f64> = vec![0.0; coeffs.lambda];
    let mut merits: Vec<f64> = vec![0.0; coeffs.lambda];
    let mut order: Vec<usize> = (0..coeffs.lambda).collect();
    // Adaptive-penalty state for constrained selection.
    let mut penalty_w = 1.0;
    let mut penalty_seeded = false;

    let mut rng: StdRng = match config.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut thread_rng = rand::rng();
            StdRng::from_rng(&mut thread_rng)
        }
    };

    let initial_x = denormalise(&mean, &config.bounds);
    let initial_fun = finite_or_infinity(f(&initial_x));
    let initial_viol = max_violation_at(&initial_x, &config.constraints);
    // Best feasible point (`best_fun` stays infinite until one is found) plus
    // the least-violating point as a fallback for infeasible runs.
    let mut found_feasible = initial_viol <= 0.0;
    let mut best_x = initial_x.clone();
    let mut best_fun = if found_feasible {
        initial_fun
    } else {
        f64::INFINITY
    };
    let mut min_viol_x = initial_x;
    let mut min_viol_fun = initial_fun;
    let mut min_viol = initial_viol;
    let mut nfev = 1usize;
    let mut nit = 0usize;
    let mut last_improvement_fun = best_fun;
    let mut last_improvement_viol = min_viol;
    let mut stagnation_counter = 0usize;
    let mut message = String::from("maximum evaluations reached");
    let mut success = false;
    let mut restarts_used = 0usize;

    while nfev < config.maxeval {
        std::mem::swap(&mut mean, &mut old_mean);
        let eval_budget = (config.maxeval - nfev).min(coeffs.lambda);

        // Generate offspring in normalised coordinates.  Instead of forming the
        // full `B * diag(D)` transform matrix, sample `z ~ N(0, I)` and apply
        // the equivalent cheaper operation `B * (D .* z)`.
        for y_i in y_pool.iter_mut().take(eval_budget) {
            fill_standard_normal(&mut z_buf, &mut rng);
            for j in 0..n {
                scaled_z[j] = d[j] * z_buf[j];
            }
            gemv_inplace(&b, &scaled_z, &mut step_buf);
            for j in 0..n {
                y_i[j] = (old_mean[j] + step_buf[j] * sigma).clamp(0.0, 1.0);
            }
        }

        // Evaluate offspring. Constraint values ride along with the
        // objective evaluation and do not consume extra `nfev`.
        if config.parallel.enabled && eval_budget >= 4 {
            funs[..eval_budget]
                .par_iter_mut()
                .zip(viols[..eval_budget].par_iter_mut())
                .enumerate()
                .with_min_len(16)
                .for_each(|(i, (fun, viol))| {
                    CMA_X_SCRATCH.with(|slot| {
                        let mut x_array = slot.borrow_mut();
                        if x_array.len() != n {
                            *x_array = Array1::zeros(n);
                        }
                        denormalise_into(&y_pool[i], &config.bounds, &mut x_array);
                        *fun = finite_or_infinity(f(&x_array));
                        *viol = max_violation_at(&x_array, &config.constraints);
                    });
                });
        } else {
            let mut x_array = Array1::<f64>::zeros(n);
            for i in 0..eval_budget {
                denormalise_into(&y_pool[i], &config.bounds, &mut x_array);
                funs[i] = finite_or_infinity(f(&x_array));
                viols[i] = max_violation_at(&x_array, &config.constraints);
            }
        }
        nfev += eval_budget;

        // Track the best feasible point and the least-violating fallback.
        for i in 0..eval_budget {
            if viols[i] < min_viol {
                min_viol = viols[i];
                min_viol_fun = funs[i];
                denormalise_into(&y_pool[i], &config.bounds, &mut min_viol_x);
            }
            if viols[i] <= 0.0 {
                found_feasible = true;
                if funs[i] < best_fun {
                    best_fun = funs[i];
                    denormalise_into(&y_pool[i], &config.bounds, &mut best_x);
                }
            }
        }

        if eval_budget == 0 {
            break;
        }

        // Rank offspring by fitness, or by adaptive-penalty merit when
        // constraints are present.
        if constrained {
            if !penalty_seeded {
                penalty_seeded = true;
                penalty_w = initial_penalty_weight(&funs[..eval_budget], &viols[..eval_budget]);
            } else if eval_budget > 0 {
                let feasible_count = viols[..eval_budget].iter().filter(|&&v| v <= 0.0).count();
                penalty_w =
                    adapt_penalty_weight(penalty_w, feasible_count as f64 / eval_budget as f64);
            }
            for i in 0..eval_budget {
                merits[i] = funs[i] + penalty_w * viols[i] * viols[i];
            }
            order[..eval_budget].sort_by(|&a, &b| merits[a].total_cmp(&merits[b]));
        } else {
            order[..eval_budget].sort_by(|&a, &b| funs[a].total_cmp(&funs[b]));
        }
        let mu_used = coeffs.mu.min(eval_budget);

        // Recombine parents into the new mean.
        mean.fill(0.0);
        for (&idx, &w) in order.iter().zip(coeffs.weights.iter()).take(mu_used) {
            let y_i = &y_pool[idx];
            for j in 0..n {
                mean[j] += w * y_i[j];
            }
        }
        clamp_unit_vector_inplace(&mut mean);

        // Evolution paths.
        for j in 0..n {
            y_w[j] = (mean[j] - old_mean[j]) / sigma.max(1e-30);
        }

        gemv_inplace(&invsqrt_c, &y_w, &mut tmp_n);
        let ps_factor = (coeffs.cs * (2.0 - coeffs.cs) * coeffs.mueff).sqrt();
        for j in 0..n {
            ps[j] = ps[j] * (1.0 - coeffs.cs) + tmp_n[j] * ps_factor;
        }
        let norm_ps = ps.norm();

        let hsig_den = (1.0 - (1.0 - coeffs.cs).powi(2 * (nit as i32 + 1))).sqrt() * chi_n;
        let hsig = if hsig_den > 0.0 {
            norm_ps / hsig_den < 1.4 + 2.0 / (n_f + 1.0)
        } else {
            true
        };

        let pc_factor = (coeffs.cc * (2.0 - coeffs.cc) * coeffs.mueff).sqrt();
        for j in 0..n {
            pc[j] *= 1.0 - coeffs.cc;
            if hsig {
                pc[j] += y_w[j] * pc_factor;
            }
        }

        // Rank-mu update matrix.
        rank_mu.fill(0.0);
        let inv_sigma = 1.0 / sigma.max(1e-30);
        for (&idx, &w) in order.iter().zip(coeffs.weights.iter()).take(mu_used) {
            let y_i = &y_pool[idx];
            for j in 0..n {
                let diff_j = (y_i[j] - old_mean[j]) * inv_sigma;
                for k in 0..n {
                    let diff_k = (y_i[k] - old_mean[k]) * inv_sigma;
                    rank_mu[(j, k)] += w * diff_j * diff_k;
                }
            }
        }

        // Covariance matrix update.
        let hsig_correction = if hsig {
            0.0
        } else {
            coeffs.c1 * coeffs.cc * (2.0 - coeffs.cc)
        };
        let cov_scale = 1.0 - coeffs.c1 - coeffs.cmu + hsig_correction;
        for j in 0..n {
            for k in 0..n {
                covariance[(j, k)] = covariance[(j, k)] * cov_scale
                    + coeffs.c1 * pc[j] * pc[k]
                    + coeffs.cmu * rank_mu[(j, k)];
            }
        }
        symmetrise_and_regularise(&mut covariance);
        if config.covariance == CmaCovariance::Diagonal {
            zero_off_diagonal(&mut covariance);
        }

        // Step-size update.
        sigma *= ((coeffs.cs / coeffs.damps) * (norm_ps / chi_n - 1.0)).exp();
        sigma = sigma.clamp(1e-14, 10.0);

        // Lazily refresh the eigen-decomposition of the updated covariance
        // matrix, reusing the cached (B, D, C^{-1/2}) factors on the
        // generations in between. The factorisation moves out of the reused
        // `eig_work` scratch buffer (refilled via `copy_from`, no per-generation
        // clone of `covariance`); the placeholder left behind is correctly
        // sized so the next refresh copies without reallocating.
        gens_since_eig += 1;
        if gens_since_eig >= eig_gap {
            gens_since_eig = 0;
            if config.covariance == CmaCovariance::Diagonal {
                // Diagonal covariance: B = I, D = sqrt(diag(C)) — no O(n^3)
                // eigendecomposition needed.
                b.fill(0.0);
                invsqrt_c.fill(0.0);
                for j in 0..n {
                    b[(j, j)] = 1.0;
                    let dj = covariance[(j, j)].max(1e-30).sqrt();
                    d[j] = dj;
                    invsqrt_c[(j, j)] = 1.0 / dj.max(1e-30);
                }
            } else {
                eig_work.copy_from(&covariance);
                let eig =
                    SymmetricEigen::new(std::mem::replace(&mut eig_work, DMatrix::zeros(n, n)));
                b = eig.eigenvectors;
                d = eig.eigenvalues.map(|v| v.max(1e-30).sqrt());

                // Recompute C^{-1/2} = B * diag(1/d) * B^T without materialising the
                // intermediate diagonal matrix.
                for j in 0..n {
                    for k in 0..n {
                        let mut sum = 0.0;
                        for l in 0..n {
                            sum += b[(j, l)] * b[(k, l)] / d[l].max(1e-30);
                        }
                        invsqrt_c[(j, k)] = sum;
                    }
                }
            }
        }

        nit += 1;
        // Progress means a better feasible value, or — while nothing
        // feasible is known — a smaller constraint violation.
        let improved = if constrained && !found_feasible {
            (last_improvement_viol - min_viol) > config.f_tol
        } else {
            (last_improvement_fun - best_fun).abs() > config.f_tol
        };
        if improved {
            stagnation_counter = 0;
            last_improvement_fun = best_fun;
            last_improvement_viol = min_viol;
        } else {
            stagnation_counter += 1;
        }

        if let Some(ref mut callback) = config.callback {
            let (callback_x, callback_fun) = if found_feasible {
                (&best_x, best_fun)
            } else {
                (&min_viol_x, min_viol_fun)
            };
            let intermediate = CmaEsIntermediate {
                x: callback_x.clone(),
                fun: callback_fun,
                iter: nit,
                nfev,
                sigma,
            };
            if matches!(callback(&intermediate), CallbackAction::Stop) {
                success = true;
                message = String::from("stopped by callback");
                break;
            }
        }

        if best_fun <= config.target_f {
            success = true;
            message = format!("target_f reached: {:.6e}", best_fun);
            break;
        }
        // Stagnation and step-size collapse are restartable: with IPOP
        // restarts enabled the run continues with a larger population and a
        // reset distribution instead of terminating.
        let mut stop_reason: Option<CmaStopReason> = None;
        if config.stagnation_window > 0 && stagnation_counter >= config.stagnation_window {
            stop_reason = Some(CmaStopReason::Stagnation);
        } else if sigma < 1e-12 {
            stop_reason = Some(CmaStopReason::SigmaCollapse);
        }
        if let Some(reason) = stop_reason {
            let grown = ((coeffs.lambda as f64 * config.restart_lambda_growth).ceil() as usize)
                .max(coeffs.lambda + 1);
            let budget_for_restart = nfev.saturating_add(grown).saturating_add(1) <= config.maxeval;
            if restarts_used < config.max_restarts && budget_for_restart {
                restarts_used += 1;
                coeffs = population_coeffs(n, grown, config.mu)?;
                y_pool.resize_with(coeffs.lambda, || DVector::<f64>::zeros(n));
                funs.resize(coeffs.lambda, 0.0);
                viols.resize(coeffs.lambda, 0.0);
                merits.resize(coeffs.lambda, 0.0);
                order = (0..coeffs.lambda).collect();
                mean = if config.restart_from_best {
                    normalise_point(&best_x, &config.bounds)
                } else {
                    DVector::<f64>::from_iterator(n, (0..n).map(|_| rng.random::<f64>()))
                };
                covariance = DMatrix::<f64>::identity(n, n);
                b = DMatrix::<f64>::identity(n, n);
                d = DVector::<f64>::from_element(n, 1.0);
                invsqrt_c = DMatrix::<f64>::identity(n, n);
                pc.fill(0.0);
                ps.fill(0.0);
                sigma = sigma0_init;
                stagnation_counter = 0;
                last_improvement_fun = best_fun;
                last_improvement_viol = min_viol;
                gens_since_eig = eig_gap;
                continue;
            }
            success = true;
            message = match reason {
                CmaStopReason::Stagnation => format!(
                    "stagnated for {} generations below f_tol={:.3e}",
                    config.stagnation_window, config.f_tol
                ),
                CmaStopReason::SigmaCollapse => String::from("step size collapsed"),
            };
            break;
        }
    }

    if restarts_used > 0 {
        message = format!("{message} (after {restarts_used} restarts)");
    }

    // Report the best feasible point, or the least-violating point when
    // nothing feasible was found.
    let (final_x, final_fun, final_viol) = if found_feasible {
        (best_x, best_fun, 0.0)
    } else {
        (min_viol_x, min_viol_fun, min_viol)
    };

    Ok(CmaEsReport {
        x: final_x,
        fun: final_fun,
        success,
        message,
        nfev,
        nit,
        sigma,
        restarts: restarts_used,
        feasible: found_feasible,
        max_violation: final_viol,
    })
}

/// Population-dependent strategy coefficients, recomputed on every IPOP
/// restart because the offspring population grows.
struct CmaCoeffs {
    lambda: usize,
    mu: usize,
    weights: Vec<f64>,
    mueff: f64,
    cc: f64,
    cs: f64,
    c1: f64,
    cmu: f64,
    damps: f64,
}

fn population_coeffs(n: usize, lambda_cfg: usize, mu_cfg: usize) -> Result<CmaCoeffs> {
    let lambda = if lambda_cfg == 0 {
        (4.0 + (3.0 * (n as f64).ln()).floor()).max(4.0) as usize
    } else {
        lambda_cfg
    };
    if lambda < 2 {
        return Err(DEError::PopulationTooSmall { pop_size: lambda });
    }
    let mu = if mu_cfg == 0 {
        lambda / 2
    } else {
        mu_cfg.min(lambda)
    }
    .max(1);

    let weights = recombination_weights(mu);
    let mueff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
    let n_f = n as f64;

    let cc = (4.0 + mueff / n_f) / (n_f + 4.0 + 2.0 * mueff / n_f);
    let cs = (mueff + 2.0) / (n_f + mueff + 5.0);
    let c1 = 2.0 / ((n_f + 1.3).powi(2) + mueff);
    let cmu = (1.0 - c1).min(2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n_f + 2.0).powi(2) + mueff));
    let damps = 1.0 + 2.0 * ((mueff - 1.0) / (n_f + 1.0)).sqrt().max(1.0) - 2.0 + cs;

    Ok(CmaCoeffs {
        lambda,
        mu,
        weights,
        mueff,
        cc,
        cs,
        c1,
        cmu,
        damps,
    })
}

fn recombination_weights(mu: usize) -> Vec<f64> {
    let mu_f = mu as f64;
    let mut weights: Vec<f64> = (1..=mu)
        .map(|i| (mu_f + 0.5).ln() - (i as f64).ln())
        .collect();
    let sum = weights.iter().sum::<f64>();
    for w in &mut weights {
        *w /= sum;
    }
    weights
}

fn initial_mean(config: &CmaEsConfig) -> DVector<f64> {
    if let Some(ref x0) = config.x0 {
        normalise_point(x0, &config.bounds)
    } else {
        DVector::<f64>::from_element(config.bounds.len(), 0.5)
    }
}

fn normalise_point(x: &Array1<f64>, bounds: &[(f64, f64)]) -> DVector<f64> {
    let mut y = DVector::<f64>::zeros(bounds.len());
    for (i, (lo, hi)) in bounds.iter().enumerate() {
        let span = hi - lo;
        y[i] = if span > 0.0 {
            ((x[i].clamp(*lo, *hi) - lo) / span).clamp(0.0, 1.0)
        } else {
            0.5
        };
    }
    y
}

fn denormalise(y: &DVector<f64>, bounds: &[(f64, f64)]) -> Array1<f64> {
    let mut x = Vec::with_capacity(bounds.len());
    for (i, (lo, hi)) in bounds.iter().enumerate() {
        x.push(lo + y[i].clamp(0.0, 1.0) * (hi - lo));
    }
    Array1::from(x)
}

fn denormalise_into(y: &DVector<f64>, bounds: &[(f64, f64)], out: &mut Array1<f64>) {
    for (i, (lo, hi)) in bounds.iter().enumerate() {
        out[i] = lo + y[i].clamp(0.0, 1.0) * (hi - lo);
    }
}

fn clamp_unit_vector_inplace(y: &mut DVector<f64>) {
    for v in y.iter_mut() {
        *v = v.clamp(0.0, 1.0);
    }
}

fn fill_standard_normal<R: Rng + ?Sized>(out: &mut DVector<f64>, rng: &mut R) {
    let n = out.len();
    let mut i = 0usize;
    while i < n {
        let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        let u2 = rng.random::<f64>();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out[i] = radius * theta.cos();
        if i + 1 < n {
            out[i + 1] = radius * theta.sin();
        }
        i += 2;
    }
}

fn finite_or_infinity(v: f64) -> f64 {
    if v.is_finite() { v } else { f64::INFINITY }
}

/// Maximum constraint violation at `x` (0 when feasible). NaN constraint
/// values fail closed as infinite violation.
fn max_violation_at(x: &Array1<f64>, constraints: &[CmaEsConstraint]) -> f64 {
    let mut worst = 0.0f64;
    for c in constraints {
        let v = (c.fun)(x);
        let v = if v.is_nan() { f64::INFINITY } else { v };
        if v > worst {
            worst = v;
        }
    }
    worst
}

/// Stochastic ranking (Runarsson & Yao 2005): bubble-sort `order` by
/// interleaving objective and violation comparisons. With probability `pf`
/// two adjacent individuals compare by objective, otherwise by violation.
/// Initial adaptive-penalty weight from first-generation statistics:
/// the penalty at the largest violation roughly matches the observed
/// objective spread, so neither term dominates blindly.
fn initial_penalty_weight(funs: &[f64], viols: &[f64]) -> f64 {
    let mut finite: Vec<f64> = funs.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.len() < 2 {
        return 1.0;
    }
    finite.sort_by(f64::total_cmp);
    let spread = finite[finite.len() - 1] - finite[0];
    let worst = viols.iter().fold(0.0f64, |a, &b| a.max(b));
    if spread > 0.0 && worst > 0.0 {
        (spread / (worst * worst)).clamp(1e-6, 1e12)
    } else {
        1.0
    }
}

/// Adapt the penalty weight toward a target feasible fraction: push inside
/// when nearly everything is infeasible, relax when nearly everything is
/// feasible so the search can ride the boundary.
fn adapt_penalty_weight(weight: f64, feasible_fraction: f64) -> f64 {
    if feasible_fraction < 0.1 {
        (weight * 5.0).min(1e14)
    } else if feasible_fraction < 0.3 {
        (weight * 1.5).min(1e14)
    } else if feasible_fraction > 0.7 {
        (weight * 0.7).max(1e-10)
    } else {
        weight
    }
}

/// In-place dense matrix-vector multiply `y = A * x` without allocating an
/// intermediate result vector.
fn gemv_inplace(a: &DMatrix<f64>, x: &DVector<f64>, y: &mut DVector<f64>) {
    let n = a.nrows();
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += a[(i, j)] * x[j];
        }
        y[i] = sum;
    }
}

fn zero_off_diagonal(c: &mut DMatrix<f64>) {
    let n = c.nrows();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                c[(i, j)] = 0.0;
            }
        }
    }
}

fn symmetrise_and_regularise(c: &mut DMatrix<f64>) {
    let n = c.nrows();
    for i in 0..n {
        for j in 0..i {
            let v = 0.5 * (c[(i, j)] + c[(j, i)]);
            c[(i, j)] = v;
            c[(j, i)] = v;
        }
        c[(i, i)] = c[(i, i)].max(1e-30);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn cma_es_converges_on_sphere() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let report = cma_es(
            &sphere,
            CmaEsConfig {
                bounds: vec![(-5.0, 5.0); 4],
                maxeval: 5_000,
                seed: Some(42),
                target_f: 1e-10,
                ..Default::default()
            },
        )
        .expect("CMA-ES should run");

        assert!(
            report.fun < 1e-6,
            "CMA-ES should converge near origin, got {}",
            report.fun
        );
    }

    #[test]
    fn cma_es_handles_coupled_rotated_quadratic() {
        let rotated = |x: &Array1<f64>| {
            let u = (x[0] + x[1]) / 2.0_f64.sqrt();
            let v = (x[0] - x[1]) / 2.0_f64.sqrt();
            1_000.0 * u * u + v * v
        };
        let report = cma_es(
            &rotated,
            CmaEsConfig {
                bounds: vec![(-3.0, 3.0), (-3.0, 3.0)],
                maxeval: 4_000,
                seed: Some(7),
                target_f: 1e-9,
                ..Default::default()
            },
        )
        .expect("CMA-ES should run");

        assert!(
            report.fun < 1e-5,
            "CMA-ES should solve rotated ill-conditioned quadratic, got {}",
            report.fun
        );
    }

    #[test]
    fn cma_es_rejects_empty_bounds() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let result = cma_es(
            &sphere,
            CmaEsConfig {
                bounds: vec![],
                ..Default::default()
            },
        );
        assert!(result.is_err(), "Empty bounds should error");
    }

    #[test]
    fn cma_es_rejects_inverted_bounds() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let result = cma_es(
            &sphere,
            CmaEsConfig {
                bounds: vec![(1.0, -1.0)],
                ..Default::default()
            },
        );
        assert!(result.is_err(), "Inverted bounds should error");
    }

    #[test]
    fn cma_es_rejects_mismatched_x0() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let result = cma_es(
            &sphere,
            CmaEsConfig {
                bounds: vec![(-5.0, 5.0); 3],
                x0: Some(array![0.0, 0.0]),
                ..Default::default()
            },
        );
        assert!(result.is_err(), "Mismatched x0 dimension should error");
    }

    #[test]
    fn cma_es_callback_stop() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();
        let report = cma_es(
            &sphere,
            CmaEsConfig {
                bounds: vec![(-5.0, 5.0); 2],
                maxeval: 10_000,
                seed: Some(42),
                callback: Some(Box::new(move |_| {
                    let c = call_count_clone.fetch_add(1, Ordering::SeqCst);
                    if c + 1 >= 3 {
                        crate::CallbackAction::Stop
                    } else {
                        crate::CallbackAction::Continue
                    }
                })),
                ..Default::default()
            },
        )
        .expect("CMA-ES should run");
        assert!(report.success, "Should stop successfully by callback");
        assert_eq!(report.message, "stopped by callback");
    }

    #[test]
    fn cma_es_target_f_reached() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let report = cma_es(
            &sphere,
            CmaEsConfig {
                bounds: vec![(-5.0, 5.0); 2],
                maxeval: 10_000,
                seed: Some(42),
                target_f: 1.0,
                ..Default::default()
            },
        )
        .expect("CMA-ES should run");
        assert!(
            report.fun <= 1.0,
            "Should stop when target_f reached: f={}",
            report.fun
        );
        assert!(report.success);
    }

    #[test]
    fn cma_es_single_dimension() {
        let sphere = |x: &Array1<f64>| x[0] * x[0];
        let report = cma_es(
            &sphere,
            CmaEsConfig {
                bounds: vec![(-5.0, 5.0)],
                maxeval: 2_000,
                seed: Some(42),
                target_f: 1e-8,
                ..Default::default()
            },
        )
        .expect("CMA-ES should run");
        assert!(
            report.fun < 1e-6,
            "1D CMA-ES should converge: f={}",
            report.fun
        );
    }

    #[test]
    fn cma_es_custom_lambda_mu() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let report = cma_es(
            &sphere,
            CmaEsConfig {
                bounds: vec![(-5.0, 5.0); 2],
                lambda: 20,
                mu: 8,
                maxeval: 3_000,
                seed: Some(42),
                target_f: 1e-8,
                ..Default::default()
            },
        )
        .expect("CMA-ES should run");
        assert!(
            report.fun < 1e-6,
            "Custom lambda/mu should converge: f={}",
            report.fun
        );
    }

    #[test]
    fn cma_es_sigma0_override() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let report = cma_es(
            &sphere,
            CmaEsConfig {
                bounds: vec![(-5.0, 5.0); 2],
                sigma0: Some(0.1),
                maxeval: 3_000,
                seed: Some(42),
                target_f: 1e-8,
                ..Default::default()
            },
        )
        .expect("CMA-ES should run");
        assert!(
            report.fun < 1e-6,
            "Small sigma0 should still converge: f={}",
            report.fun
        );
    }

    #[test]
    fn cma_es_diagonal_converges_on_separable_sphere() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let report = cma_es(
            &sphere,
            CmaEsConfig {
                bounds: vec![(-5.0, 5.0); 4],
                covariance: CmaCovariance::Diagonal,
                maxeval: 5_000,
                seed: Some(42),
                target_f: 1e-10,
                ..Default::default()
            },
        )
        .expect("diagonal CMA-ES should run");

        assert!(
            report.fun < 1e-6,
            "diagonal CMA-ES should converge on sphere, got {}",
            report.fun
        );
    }

    #[test]
    fn cma_es_diagonal_improves_on_ellipsoid() {
        // Separable but ill-conditioned: diagonal model should adapt
        // per-axis scales without any coupling information.
        let ellipsoid = |x: &Array1<f64>| {
            x.iter()
                .enumerate()
                .map(|(i, &xi)| 1000_f64.powi(i as i32) * xi * xi)
                .sum::<f64>()
        };
        let report = cma_es(
            &ellipsoid,
            CmaEsConfig {
                bounds: vec![(-5.0, 5.0); 3],
                covariance: CmaCovariance::Diagonal,
                x0: Some(array![4.0, 4.0, 4.0]),
                maxeval: 4_000,
                seed: Some(11),
                target_f: 1e-8,
                ..Default::default()
            },
        )
        .expect("diagonal CMA-ES should run");

        assert!(
            report.fun < 1e-4,
            "diagonal CMA-ES should solve ellipsoid, got {}",
            report.fun
        );
    }

    #[test]
    fn cma_es_rejects_sub_unit_restart_growth() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let result = cma_es(
            &sphere,
            CmaEsConfig {
                bounds: vec![(-5.0, 5.0); 2],
                max_restarts: 1,
                restart_lambda_growth: 0.5,
                ..Default::default()
            },
        );
        assert!(result.is_err(), "sub-unit growth should error");
    }

    #[test]
    fn cma_es_restart_triggers_on_forced_stagnation() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let report = cma_es(
            &sphere,
            CmaEsConfig {
                bounds: vec![(-5.0, 5.0); 2],
                maxeval: 3_000,
                seed: Some(42),
                // Absurd tolerance: every generation "stagnates".
                f_tol: 1e300,
                stagnation_window: 2,
                max_restarts: 2,
                ..Default::default()
            },
        )
        .expect("CMA-ES should run");

        assert_eq!(report.restarts, 2, "both restarts should be consumed");
        assert!(report.success);
        assert!(
            report.message.contains("restarts"),
            "message should note restarts: {}",
            report.message
        );
        assert!(report.nfev <= 3_000, "nfev {} over budget", report.nfev);
    }

    #[test]
    fn cma_es_random_restarts_escape_rastrigin_basins() {
        let rastrigin = |x: &Array1<f64>| {
            20.0 + x
                .iter()
                .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                .sum::<f64>()
        };
        let report = cma_es(
            &rastrigin,
            CmaEsConfig {
                bounds: vec![(-5.12, 5.12); 2],
                maxeval: 8_000,
                seed: Some(5),
                max_restarts: 4,
                restart_from_best: false,
                ..Default::default()
            },
        )
        .expect("CMA-ES should run");

        assert!(
            report.restarts >= 1,
            "expected at least one restart, got {}",
            report.restarts
        );
        // Local minima sit at f >= ~1; below that proves the global basin.
        assert!(
            report.fun < 1.0,
            "restarts should reach the global basin, got {}",
            report.fun
        );
    }

    #[test]
    fn cma_es_respects_inequality_constraint() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let x0_at_least_one = CmaEsConstraint {
            fun: Arc::new(|x: &Array1<f64>| 1.0 - x[0]),
        };
        let report = cma_es(
            &sphere,
            CmaEsConfig {
                bounds: vec![(-5.0, 5.0); 2],
                maxeval: 6_000,
                seed: Some(42),
                constraints: vec![x0_at_least_one],
                ..Default::default()
            },
        )
        .expect("constrained CMA-ES should run");

        assert!(report.feasible, "should find a feasible point");
        assert_eq!(report.max_violation, 0.0);
        // Constrained minimum is (1, 0) with f = 1.
        assert!(
            (report.fun - 1.0).abs() < 0.05,
            "should converge near (1, 0), got f={} at {:?}",
            report.fun,
            report.x
        );
        assert!(report.x[0] >= 1.0 - 1e-2);
    }

    #[test]
    fn cma_es_reports_least_violating_point_when_infeasible() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        // Unsatisfiable within [-5, 5]: x0 >= 100.
        let impossible = CmaEsConstraint {
            fun: Arc::new(|x: &Array1<f64>| 100.0 - x[0]),
        };
        let report = cma_es(
            &sphere,
            CmaEsConfig {
                bounds: vec![(-5.0, 5.0); 2],
                maxeval: 500,
                seed: Some(42),
                constraints: vec![impossible],
                ..Default::default()
            },
        )
        .expect("constrained CMA-ES should run");

        assert!(!report.feasible);
        assert!(
            report.max_violation > 0.0,
            "infeasible run must report positive violation"
        );
        assert!(report.fun.is_finite());
    }

    #[test]
    fn cma_es_adaptive_penalty_rides_disk_boundary() {
        // Minimum of x0 + x1 over the disk x0^2 + x1^2 <= 2 sits exactly on
        // the boundary at (-1, -1) with f = -2.
        let linear = |x: &Array1<f64>| x[0] + x[1];
        let disk = CmaEsConstraint {
            fun: Arc::new(|x: &Array1<f64>| x[0].powi(2) + x[1].powi(2) - 2.0),
        };
        let report = cma_es(
            &linear,
            CmaEsConfig {
                bounds: vec![(-2.0, 2.0); 2],
                maxeval: 4_000,
                seed: Some(3),
                constraints: vec![disk],
                ..Default::default()
            },
        )
        .expect("constrained CMA-ES should run");

        assert!(report.feasible, "should find a feasible point");
        assert!(
            report.fun < -1.9,
            "should ride the boundary to (-1, -1), got f={} at {:?}",
            report.fun,
            report.x
        );
    }

    #[test]
    fn adaptive_penalty_weight_helpers_behave() {
        let w = initial_penalty_weight(&[0.0, 10.0], &[0.0, 2.0]);
        assert!((w - 2.5).abs() < 1e-12, "spread/viol^2 = 10/4, got {w}");
        assert_eq!(initial_penalty_weight(&[1.0, 1.0], &[0.0, 1.0]), 1.0);
        assert_eq!(initial_penalty_weight(&[0.0, 10.0], &[0.0, 0.0]), 1.0);

        assert_eq!(adapt_penalty_weight(1.0, 0.0), 5.0);
        assert_eq!(adapt_penalty_weight(1.0, 0.2), 1.5);
        assert_eq!(adapt_penalty_weight(1.0, 0.5), 1.0);
        assert!((adapt_penalty_weight(1.0, 0.9) - 0.7).abs() < 1e-12);
        assert_eq!(adapt_penalty_weight(1e14, 0.0), 1e14);
        assert_eq!(adapt_penalty_weight(1e-10, 1.0), 1e-10);
    }

    #[test]
    fn cma_es_uses_the_caller_pool_and_keeps_fixed_seed_results() {
        let run = |threads: usize| {
            let worker_ids = Arc::new(Mutex::new(std::collections::BTreeSet::new()));
            let worker_names = Arc::new(Mutex::new(std::collections::BTreeSet::new()));
            let scratch_by_worker = Arc::new(Mutex::new(std::collections::BTreeMap::new()));
            let calls = Arc::new(AtomicUsize::new(0));
            let observed = worker_ids.clone();
            let observed_names = worker_names.clone();
            let observed_scratch = scratch_by_worker.clone();
            let observed_calls = calls.clone();
            let objective = move |x: &Array1<f64>| {
                let worker = rayon::current_thread_index().expect("inside caller pool");
                observed.lock().expect("worker set lock").insert(worker);
                observed_names.lock().expect("worker name set lock").insert(
                    std::thread::current()
                        .name()
                        .expect("named caller-pool worker")
                        .to_owned(),
                );
                if observed_calls.fetch_add(1, Ordering::Relaxed) > 0 {
                    let mut scratch = observed_scratch.lock().expect("scratch map lock");
                    let pointer = x.as_ptr() as usize;
                    assert_eq!(
                        *scratch.entry(worker).or_insert(pointer),
                        pointer,
                        "worker received a newly allocated CMA offspring buffer"
                    );
                }
                let mut guard = 0.0;
                for _ in 0..2_000 {
                    guard = std::hint::black_box(guard + x[0] * f64::EPSILON);
                }
                x.iter().map(|value| value * value).sum::<f64>() + guard * 0.0
            };
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(move |index| format!("cma-es-caller-{threads}-{index}"))
                .build()
                .expect("caller-owned pool");
            let report = pool
                .install(|| {
                    cma_es(
                        &objective,
                        CmaEsConfig {
                            bounds: vec![(-5.0, 5.0); 4],
                            lambda: 64,
                            maxeval: 257,
                            seed: Some(1234),
                            parallel: ParallelConfig {
                                enabled: true,
                                num_threads: Some(threads),
                            },
                            ..Default::default()
                        },
                    )
                })
                .expect("CMA-ES run");
            let used = worker_ids.lock().expect("worker set lock").len();
            let names = worker_names.lock().expect("worker name set lock");
            assert!(
                !names.is_empty() && names.iter().all(|name| name.starts_with("cma-es-caller-")),
                "CMA-ES should evaluate on the caller-owned Rayon pool, got {names:?}"
            );
            (report, used)
        };

        let (single, single_used) = run(1);
        let (parallel, parallel_used) = run(2);
        assert_eq!(single_used, 1);
        assert!(parallel_used <= 2);
        assert_eq!(single.fun, parallel.fun);
        assert_eq!(single.x, parallel.x);
    }
}

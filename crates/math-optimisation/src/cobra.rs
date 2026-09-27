//! Surrogate-assisted constrained optimisation in the COBRA style.
//!
//! COBRA (Constrained Optimisation by Radial-basis-function Approximation)
//! fits cheap RBF surrogates to the objective and each constraint, optimises
//! an infill point on the models, evaluates the true functions there, and
//! repeats — one true evaluation per iteration after the initial design. This
//! makes it suited to expensive objectives where evolution strategies need
//! too many evaluations.
//!
//! The loop has two phases. Phase I (no feasible point known yet) drives
//! into the feasible region by minimising predicted violation over a random
//! candidate pool. Phase II optimises the predicted objective under the
//! predicted constraints with multi-start COBYLA on the models. A
//! distance-requirement cycle (DRC) keeps infill points spread away from
//! evaluated points, cycling the required distance geometrically.

use crate::CallbackAction;
use crate::cobyla::{CobylaConfig, CobylaConstraint, CobylaRhoBegin, cobyla};
use crate::error::{DEError, Result};
use crate::init_sobol::init_halton;
use crate::surrogate::{RbfKind, RbfSurrogate};
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use std::sync::Arc;

/// Erased inequality-constraint closure: feasible when `<= 0`.
pub type CobraConstraintFn = Arc<dyn Fn(&Array1<f64>) -> f64 + Send + Sync>;

/// A single inequality constraint `fun(x) <= 0` for [`cobra`].
#[derive(Clone)]
pub struct CobraConstraint {
    /// Constraint function. Feasible when `<= 0`.
    pub fun: CobraConstraintFn,
}

/// Per-iteration callback payload for [`cobra`].
pub struct CobraIntermediate {
    /// Current best parameter vector (best feasible, or least-violating).
    pub x: Array1<f64>,
    /// Objective value at [`Self::x`].
    pub fun: f64,
    /// Maximum constraint violation at [`Self::x`] (0 when feasible).
    pub max_violation: f64,
    /// Whether [`Self::x`] is feasible.
    pub feasible: bool,
    /// Surrogate (infill) iterations completed.
    pub iter: usize,
    /// True evaluations consumed so far.
    pub nfev: usize,
}

/// Callback type used by [`CobraConfig`].
pub type CobraCallback = Box<dyn FnMut(&CobraIntermediate) -> CallbackAction + Send>;

/// Configuration for [`cobra`].
pub struct CobraConfig {
    /// `(lower, upper)` bounds per parameter.
    pub bounds: Vec<(f64, f64)>,
    /// Maximum true evaluations (initial design included).
    pub maxeval: usize,
    /// Optional RNG seed for deterministic runs.
    pub seed: Option<u64>,
    /// Stop once a feasible point reaches this objective value.
    pub target_f: f64,
    /// Initial design size. `0` uses `3 * dim + 1`.
    pub initial_samples: usize,
    /// Internal COBYLA starts per Phase II iteration. `0` uses 8.
    pub n_starts: usize,
    /// Surrogate training-set cap. `0` uses 400.
    pub max_train: usize,
    /// Re-run kernel selection every `k` surrogate fits. `0` selects once
    /// and reuses the kernel families afterwards. Default 10.
    pub reselect_every: usize,
    /// Fraction of `maxeval` reserved for a final COBYLA polish on the true
    /// functions from the best point. `0.0` disables the polish phase and
    /// spends the whole budget on surrogate infill. Default 0.1.
    pub polish_fraction: f64,
    /// Probe a random point after this many iterations without progress
    /// (SACOBRA-style exploration kick). `0` disables the kicks. Default 15.
    pub stall_kick_every: usize,
    /// Probe a global uniform point every this many iterations regardless
    /// of progress, so refining the wrong basin cannot trap the search
    /// forever. `0` disables scheduled exploration. Default 25.
    pub explore_every: usize,
    /// Fall back to a random candidate pool when every COBYLA endpoint
    /// clusters near evaluated points during a stall. Otherwise (and when
    /// `false`) the best endpoint is used regardless of distance, so
    /// converging runs keep refining. Default `true`.
    pub pool_fallback: bool,
    /// Optional per-iteration callback. Returning [`CallbackAction::Stop`]
    /// terminates the run early with the best point seen so far.
    pub callback: Option<CobraCallback>,
}

impl Default for CobraConfig {
    fn default() -> Self {
        Self {
            bounds: Vec::new(),
            maxeval: 500,
            seed: None,
            target_f: f64::NEG_INFINITY,
            initial_samples: 0,
            n_starts: 0,
            max_train: 0,
            reselect_every: 10,
            polish_fraction: 0.1,
            stall_kick_every: 15,
            explore_every: 25,
            pool_fallback: true,
            callback: None,
        }
    }
}

/// Result of a [`cobra`] run.
#[derive(Clone)]
pub struct CobraReport {
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
    /// True evaluations consumed.
    pub nfev: usize,
    /// Surrogate (infill) iterations completed.
    pub nit: usize,
}

impl std::fmt::Debug for CobraReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CobraReport")
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

/// DRC radii in normalised coordinates, cycled per iteration.
const DRC_RADII: [f64; 4] = [0.2, 0.1, 0.05, 0.02];

/// One evaluated point: location, objective, constraint values.
struct ArchiveEntry {
    x: Array1<f64>,
    fun: f64,
    cons: Vec<f64>,
}

impl ArchiveEntry {
    fn max_violation(&self) -> f64 {
        let mut worst = 0.0f64;
        for &v in &self.cons {
            let v = if v.is_nan() { f64::INFINITY } else { v };
            worst = worst.max(v);
        }
        worst
    }
}

/// Minimise `f` subject to `constraints` with RBF-surrogate assistance.
///
/// The objective and constraints receive parameters in the original
/// coordinate system. Surrogates are fitted on normalised `[0, 1]^n`
/// coordinates. Each iteration consumes exactly one true evaluation.
pub fn cobra<F>(
    f: &F,
    constraints: &[CobraConstraint],
    mut config: CobraConfig,
) -> Result<CobraReport>
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
    if config.maxeval == 0 {
        return Err(DEError::InvalidConfig {
            message: "maxeval must be greater than zero".to_string(),
        });
    }

    let mut rng = match config.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_rng(&mut rand::rng()),
    };
    let n_starts = config.n_starts.max(1);
    let max_train = if config.max_train == 0 {
        400
    } else {
        config.max_train
    };
    let default_init = 3 * n + 1;
    let n_init = if config.initial_samples == 0 {
        default_init
    } else {
        config.initial_samples
    }
    .clamp(1, config.maxeval);

    // Initial design: Halton quasi-random coverage, seed-shuffled so
    // different seeds explore different orders.
    let mut design = init_halton(n, n_init, &config.bounds);
    design.shuffle(&mut rng);
    let mut archive: Vec<ArchiveEntry> = Vec::with_capacity(config.maxeval);
    for point in &design {
        let x = Array1::from(point.clone());
        archive.push(evaluate_point(f, constraints, &x));
    }
    let mut nfev = n_init;
    let mut best = best_indices(&archive);

    // Degenerate budgets cannot fit the linear-tail surrogates; return the
    // best design point honestly.
    if n_init < n + 1 {
        return Ok(finish_report(
            &archive,
            best,
            false,
            String::from("budget too small for surrogates; returning best initial point"),
            nfev,
            0,
        ));
    }

    // Evaluations reserved for the final polish (plus one for the final
    // true evaluation at the polished point).
    let reserve = if config.polish_fraction > 0.0 {
        ((config.maxeval as f64 * config.polish_fraction.clamp(0.0, 0.5)) as usize + 1)
            .min(config.maxeval)
    } else {
        0
    };

    let mut nit = 0usize;
    let mut success = false;
    let mut message = String::from("evaluation budget reached");
    // Cached kernel families (objective first, then constraints); refreshed
    // by `fit_auto` every `reselect_every` fits.
    let mut kinds: Vec<RbfKind> = Vec::new();
    // Stagnation-triggered exploration kicks: alternating global (uniform)
    // and local (shrinking box around the best point) probes.
    let mut stall_counter = 0usize;
    let mut kick_count = 0usize;
    let mut last_merit = merit_of(&archive, best);

    while nfev + reserve < config.maxeval {
        nit += 1;
        // Probe instead of trusting the models when stalled, or on the
        // scheduled exploration cadence: stall kicks alternate global
        // uniform probes (basin escape) with local probes in a shrinking
        // box around the best point (basin refinement), while scheduled
        // probes are always global.
        let stalled_kick = config.stall_kick_every > 0 && stall_counter >= config.stall_kick_every;
        let scheduled = config.explore_every > 0 && nit.is_multiple_of(config.explore_every);
        let x = if stalled_kick || scheduled {
            stall_counter = 0;
            if stalled_kick {
                let (bx, _, _, _) = best_point(&archive, best);
                let point = kick_point(bx, kick_count, &config.bounds, &mut rng);
                kick_count += 1;
                point
            } else {
                random_point(&config.bounds, &mut rng)
            }
        } else {
            let reselect = kinds.is_empty()
                || (config.reselect_every > 0 && (nit - 1).is_multiple_of(config.reselect_every));
            let models = fit_models(&archive, &config.bounds, max_train, reselect, &mut kinds);
            let stalled =
                config.stall_kick_every > 0 && stall_counter >= config.stall_kick_every / 2;
            let candidate = match models {
                Ok(models) => {
                    search_infill(&models, &archive, &config, n_starts, nit, stalled, &mut rng)
                }
                Err(_) => None,
            };
            // Surrogate or search failure degrades to a random point rather
            // than aborting the optimisation.
            candidate.unwrap_or_else(|| random_point(&config.bounds, &mut rng))
        };
        archive.push(evaluate_point(f, constraints, &x));
        nfev += 1;
        best = best_indices(&archive);
        let merit = merit_of(&archive, best);
        let improved =
            (!last_merit.0 && merit.0) || (merit.0 == last_merit.0 && merit.1 < last_merit.1);
        if improved {
            stall_counter = 0;
            kick_count = 0;
            last_merit = merit;
        } else {
            stall_counter += 1;
        }

        if let Some(ref mut callback) = config.callback {
            let (bx, bfun, bviol, feasible) = best_point(&archive, best);
            let intermediate = CobraIntermediate {
                x: bx.clone(),
                fun: bfun,
                max_violation: bviol,
                feasible,
                iter: nit,
                nfev,
            };
            if matches!(callback(&intermediate), CallbackAction::Stop) {
                success = true;
                message = String::from("stopped by callback");
                break;
            }
        }
        let (_, bfun, _, feasible) = best_point(&archive, best);
        if feasible && bfun <= config.target_f {
            success = true;
            message = format!("target_f reached: {bfun:.6e}");
            break;
        }
    }

    if !success {
        // Final polish: COBYLA on the true functions from the best point.
        // COBYLA reports its configured budget as consumed (exact counts
        // are unavailable), so one evaluation is held back to score the
        // polished point exactly.
        let remaining = config.maxeval - nfev;
        let (bx, _, _, _) = best_point(&archive, best);
        if reserve > 0 && remaining >= 2 {
            let min_width = config
                .bounds
                .iter()
                .map(|(lo, hi)| hi - lo)
                .fold(f64::INFINITY, f64::min);
            let model_cons: Vec<CobylaConstraint> = constraints
                .iter()
                .map(|c| CobylaConstraint { fun: c.fun.clone() })
                .collect();
            let cfg = CobylaConfig {
                x0: bx.clone(),
                bounds: config.bounds.clone(),
                rho_begin: CobylaRhoBegin::All(0.1 * min_width),
                maxeval: remaining - 1,
                ..Default::default()
            };
            match cobyla(f, &model_cons, cfg) {
                Ok(report) => archive.push(evaluate_point(f, constraints, &report.x)),
                Err(_) => {
                    let x = random_point(&config.bounds, &mut rng);
                    archive.push(evaluate_point(f, constraints, &x));
                }
            }
            nfev = config.maxeval;
            best = best_indices(&archive);
        } else {
            while nfev < config.maxeval {
                let x = random_point(&config.bounds, &mut rng);
                archive.push(evaluate_point(f, constraints, &x));
                nfev += 1;
            }
            best = best_indices(&archive);
        }
    }

    Ok(finish_report(&archive, best, success, message, nfev, nit))
}

fn evaluate_point<F>(f: &F, constraints: &[CobraConstraint], x: &Array1<f64>) -> ArchiveEntry
where
    F: Fn(&Array1<f64>) -> f64 + Sync,
{
    let fun = f(x);
    let fun = if fun.is_finite() { fun } else { f64::INFINITY };
    let cons = constraints.iter().map(|c| (c.fun)(x)).collect();
    ArchiveEntry {
        x: x.clone(),
        fun,
        cons,
    }
}

/// `(best_feasible, least_violating)` archive indices.
fn best_indices(archive: &[ArchiveEntry]) -> (Option<usize>, usize) {
    let mut feasible: Option<usize> = None;
    let mut least_violating = 0usize;
    for (i, entry) in archive.iter().enumerate() {
        if entry.max_violation() < archive[least_violating].max_violation() {
            least_violating = i;
        }
        if entry.max_violation() <= 0.0 {
            let better = feasible.map(|b| entry.fun < archive[b].fun).unwrap_or(true);
            if better {
                feasible = Some(i);
            }
        }
    }
    (feasible, least_violating)
}

fn best_point(
    archive: &[ArchiveEntry],
    best: (Option<usize>, usize),
) -> (&Array1<f64>, f64, f64, bool) {
    if let Some(i) = best.0 {
        (&archive[i].x, archive[i].fun, 0.0, true)
    } else {
        let entry = &archive[best.1];
        (&entry.x, entry.fun, entry.max_violation(), false)
    }
}

/// Search merit: feasibility first, then objective (feasible) or
/// violation (infeasible).
fn merit_of(archive: &[ArchiveEntry], best: (Option<usize>, usize)) -> (bool, f64) {
    if let Some(i) = best.0 {
        (true, archive[i].fun)
    } else {
        (false, archive[best.1].max_violation())
    }
}

fn finish_report(
    archive: &[ArchiveEntry],
    best: (Option<usize>, usize),
    success: bool,
    message: String,
    nfev: usize,
    nit: usize,
) -> CobraReport {
    let (x, fun, max_violation, feasible) = best_point(archive, best);
    CobraReport {
        x: x.clone(),
        fun,
        feasible,
        max_violation,
        success,
        message,
        nfev,
        nit,
    }
}

/// Fitted models: objective surrogate plus one surrogate per constraint.
struct FittedModels {
    objective: RbfSurrogate,
    constraints: Vec<RbfSurrogate>,
}

/// Sign-preserving log transform for modelled outputs.
///
/// Compresses large output ranges (e.g. objectives spanning 1e2..1e7) into
/// a range RBF interpolation handles well. Monotonic with `plog(0) = 0`,
/// so optima orderings and constraint feasibility boundaries are preserved
/// and model predictions never need transforming back for ranking.
fn plog(y: f64) -> f64 {
    y.signum() * (1.0 + y.abs()).ln()
}

/// Whether the log transform helps a training target: only when the range
/// is both large and wide relative to the typical magnitude. Compressing
/// ordinary ranges would flatten the very valleys the infill search needs.
fn use_plog(ys: &[f64]) -> bool {
    let mut sorted: Vec<f64> = ys.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.len() < 2 {
        return false;
    }
    sorted.sort_by(f64::total_cmp);
    let range = sorted[sorted.len() - 1] - sorted[0];
    let median = sorted[sorted.len() / 2].abs();
    range > 1e3 && range > 20.0 * median.max(1.0)
}

fn maybe_plog(ys: &[f64]) -> Vec<f64> {
    if use_plog(ys) {
        ys.iter().map(|&y| plog(y)).collect()
    } else {
        ys.to_vec()
    }
}

/// Fit (or refit) all surrogates on normalised coordinates and
/// log-transformed outputs.
fn fit_models(
    archive: &[ArchiveEntry],
    bounds: &[(f64, f64)],
    max_train: usize,
    reselect: bool,
    kinds: &mut Vec<RbfKind>,
) -> Result<FittedModels> {
    let idx = select_training_indices(archive, max_train);
    let train_norm: Vec<Array1<f64>> = idx
        .iter()
        .map(|&i| normalize_point(&archive[i].x, bounds))
        .collect();
    let n_models = archive[0].cons.len() + 1;
    let raw_obj: Vec<f64> = idx.iter().map(|&i| archive[i].fun).collect();
    let obj_ys = maybe_plog(&raw_obj);
    if reselect || kinds.len() != n_models {
        kinds.clear();
        kinds.push(RbfSurrogate::fit_auto(&train_norm, &obj_ys)?.kind());
        for c in 0..archive[0].cons.len() {
            let raw: Vec<f64> = idx.iter().map(|&i| archive[i].cons[c]).collect();
            kinds.push(RbfSurrogate::fit_auto(&train_norm, &maybe_plog(&raw))?.kind());
        }
    }
    let objective = RbfSurrogate::fit(&train_norm, &obj_ys, kinds[0])?;
    let mut constraints = Vec::with_capacity(archive[0].cons.len());
    for (c, &kind) in kinds.iter().skip(1).enumerate() {
        let raw: Vec<f64> = idx.iter().map(|&i| archive[i].cons[c]).collect();
        constraints.push(RbfSurrogate::fit(&train_norm, &maybe_plog(&raw), kind)?);
    }
    Ok(FittedModels {
        objective,
        constraints,
    })
}

/// Training subset: the full archive while small, otherwise the
/// best-predicted half (feasible-first) plus the most recent half.
fn select_training_indices(archive: &[ArchiveEntry], max_train: usize) -> Vec<usize> {
    let n = archive.len();
    if n <= max_train {
        return (0..n).collect();
    }
    let mut best_first: Vec<usize> = (0..n).collect();
    best_first.sort_by(|&a, &b| {
        let (va, vb) = (archive[a].max_violation(), archive[b].max_violation());
        match (va <= 0.0, vb <= 0.0) {
            (true, true) => archive[a].fun.total_cmp(&archive[b].fun),
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => va.total_cmp(&vb),
        }
    });
    let half = max_train / 2;
    let mut selected: Vec<usize> = best_first.into_iter().take(half.max(1)).collect();
    for i in (n.saturating_sub(max_train - selected.len())..n).rev() {
        if selected.len() >= max_train {
            break;
        }
        if !selected.contains(&i) {
            selected.push(i);
        }
    }
    selected
}

/// Search one infill point on the models: Phase I (feasibility drive over
/// a random pool) or Phase II (multi-start COBYLA under predicted
/// constraints), both filtered by the distance-requirement cycle. When the
/// search has stalled, clustered endpoints trigger pool exploration;
/// otherwise the best endpoint is used for continued refinement.
fn search_infill(
    models: &FittedModels,
    archive: &[ArchiveEntry],
    config: &CobraConfig,
    n_starts: usize,
    nit: usize,
    stalled: bool,
    rng: &mut StdRng,
) -> Option<Array1<f64>> {
    let feasible_known = archive.iter().any(|e| e.max_violation() <= 0.0);
    let bounds = &config.bounds;
    let predict_obj = |x: &Array1<f64>| models.objective.predict(&normalize_point(x, bounds));
    let predict_viol = |x: &Array1<f64>| {
        let xn = normalize_point(x, bounds);
        let mut worst = 0.0f64;
        for m in &models.constraints {
            worst = worst.max(m.predict(&xn));
        }
        worst
    };

    if !feasible_known && !models.constraints.is_empty() {
        // Phase I: best predicted violation over a random pool.
        let pool = (1024 * bounds.len()).clamp(2048, 8192);
        let mut candidates: Vec<(Array1<f64>, f64, f64)> = Vec::with_capacity(pool);
        for _ in 0..pool {
            let x = random_point(bounds, rng);
            candidates.push((x.clone(), predict_obj(&x), predict_viol(&x)));
        }
        sort_by_predicted_merit(&mut candidates);
        pick_with_drc(candidates, archive, bounds, nit)
    } else {
        // Phase II: multi-start COBYLA on the models (possibly with no
        // model constraints for unconstrained problems). The local
        // model-optima are the primary infill source; a random pool only
        // serves as fallback when every endpoint clusters near evaluated
        // points (a large pool ranked ahead would chase model artifacts).
        let inner_budget = (100 * bounds.len()).max(50);
        let min_width = bounds
            .iter()
            .map(|(lo, hi)| hi - lo)
            .fold(f64::INFINITY, f64::min);
        let mut endpoints: Vec<(Array1<f64>, f64, f64)> = Vec::new();
        for _ in 0..n_starts.max(1) {
            let x0 = random_point(bounds, rng);
            let model_cons: Vec<CobylaConstraint> = models
                .constraints
                .iter()
                .map(|m| {
                    let model = m.clone();
                    let owned_bounds = bounds.clone();
                    CobylaConstraint {
                        fun: Arc::new(move |x: &Array1<f64>| {
                            model.predict(&normalize_point(x, &owned_bounds))
                        }),
                    }
                })
                .collect();
            let cfg = CobylaConfig {
                x0,
                bounds: bounds.clone(),
                rho_begin: CobylaRhoBegin::All(0.25 * min_width),
                maxeval: inner_budget,
                ..Default::default()
            };
            if let Ok(report) = cobyla(&predict_obj, &model_cons, cfg) {
                let x = report.x;
                endpoints.push((x.clone(), predict_obj(&x), predict_viol(&x)));
            }
        }
        sort_by_predicted_merit(&mut endpoints);
        if let Some(x) = pick_strict(&endpoints, archive, bounds, nit) {
            return Some(x);
        }
        if !config.pool_fallback || !stalled {
            // Refine: use the best endpoint regardless of distance. The
            // pool fallback only engages on genuine stalls, so converging
            // runs keep drilling down instead of being diverted.
            return endpoints.into_iter().next().map(|(x, _, _)| x);
        }
        // Stalled with every endpoint clustering: fall back to a random
        // pool for exploration.
        let pool = (512 * bounds.len()).clamp(1024, 4096);
        let mut candidates: Vec<(Array1<f64>, f64, f64)> = Vec::with_capacity(pool);
        for _ in 0..pool {
            let x = random_point(bounds, rng);
            candidates.push((x.clone(), predict_obj(&x), predict_viol(&x)));
        }
        sort_by_predicted_merit(&mut candidates);
        pick_with_drc(candidates, archive, bounds, nit)
    }
}

/// Sort by predicted merit: predicted-feasible by objective, then the rest
/// by predicted violation.
fn sort_by_predicted_merit(candidates: &mut [(Array1<f64>, f64, f64)]) {
    candidates.sort_by(|a, b| match (a.2 <= 0.0, b.2 <= 0.0) {
        (true, true) => a.1.total_cmp(&b.1),
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => a.2.total_cmp(&b.2),
    });
}

/// Pick the best candidate at least `δ` (normalised) away from the archive,
/// cycling `δ`; return `None` when every candidate clusters.
fn pick_strict(
    candidates: &[(Array1<f64>, f64, f64)],
    archive: &[ArchiveEntry],
    bounds: &[(f64, f64)],
    nit: usize,
) -> Option<Array1<f64>> {
    let delta = DRC_RADII[(nit - 1) % DRC_RADII.len()];
    candidates
        .iter()
        .find(|(x, _, _)| min_archive_distance(x, archive, bounds) >= delta)
        .map(|(x, _, _)| x.clone())
}

/// Pick the best candidate at least `δ` (normalised) away from the archive,
/// cycling `δ`; fall back to the best candidate when none qualifies.
fn pick_with_drc(
    candidates: Vec<(Array1<f64>, f64, f64)>,
    archive: &[ArchiveEntry],
    bounds: &[(f64, f64)],
    nit: usize,
) -> Option<Array1<f64>> {
    let delta = DRC_RADII[(nit - 1) % DRC_RADII.len()];
    for (x, _, _) in &candidates {
        if min_archive_distance(x, archive, bounds) >= delta {
            return Some(x.clone());
        }
    }
    candidates.into_iter().next().map(|(x, _, _)| x)
}

fn min_archive_distance(x: &Array1<f64>, archive: &[ArchiveEntry], bounds: &[(f64, f64)]) -> f64 {
    let xn = normalize_point(x, bounds);
    archive
        .iter()
        .map(|e| {
            let en = normalize_point(&e.x, bounds);
            xn.iter()
                .zip(en.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

fn normalize_point(x: &Array1<f64>, bounds: &[(f64, f64)]) -> Array1<f64> {
    Array1::from_vec(
        x.iter()
            .zip(bounds.iter())
            .map(|(&v, &(lo, hi))| {
                let span = hi - lo;
                if span > 0.0 {
                    ((v - lo) / span).clamp(0.0, 1.0)
                } else {
                    0.5
                }
            })
            .collect(),
    )
}

/// Stagnation kick: even kicks probe uniformly (basin escape), odd kicks
/// probe a box around the best point with geometrically cycling radius
/// (basin refinement).
fn kick_point(
    best_x: &Array1<f64>,
    kick_count: usize,
    bounds: &[(f64, f64)],
    rng: &mut StdRng,
) -> Array1<f64> {
    if kick_count.is_multiple_of(2) {
        return random_point(bounds, rng);
    }
    let radius = DRC_RADII[(kick_count / 2) % DRC_RADII.len()];
    Array1::from_vec(
        best_x
            .iter()
            .zip(bounds.iter())
            .map(|(&b, &(lo, hi))| {
                let half = radius * (hi - lo);
                rng.random_range((b - half).max(lo)..=(b + half).min(hi))
            })
            .collect(),
    )
}

fn random_point(bounds: &[(f64, f64)], rng: &mut StdRng) -> Array1<f64> {
    Array1::from_vec(
        bounds
            .iter()
            .map(|(lo, hi)| rng.random_range(*lo..=*hi))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_constraint() -> CobraConstraint {
        CobraConstraint {
            fun: Arc::new(|x: &Array1<f64>| 1.0 - x[0]),
        }
    }

    /// Smaller budgets in debug builds (surrogate refits are slow
    /// unoptimized); full budgets in release. Thresholds below hold for
    /// both modes.
    fn budget(release: usize, debug: usize) -> usize {
        if cfg!(debug_assertions) {
            debug
        } else {
            release
        }
    }

    #[test]
    fn cobra_solves_sphere_unconstrained() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let report = cobra(
            &sphere,
            &[],
            CobraConfig {
                bounds: vec![(-5.0, 5.0); 2],
                maxeval: budget(120, 50),
                seed: Some(1),
                ..Default::default()
            },
        )
        .expect("cobra should run");
        assert!(report.feasible);
        let tol = if cfg!(debug_assertions) { 2e-2 } else { 1e-3 };
        assert!(
            report.fun < tol,
            "should converge on sphere, got {}",
            report.fun
        );
        assert_eq!(report.nfev, budget(120, 50));
    }

    #[test]
    fn cobra_respects_inequality_constraint() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let report = cobra(
            &sphere,
            &[box_constraint()],
            CobraConfig {
                bounds: vec![(-5.0, 5.0); 2],
                maxeval: budget(150, 60),
                seed: Some(2),
                ..Default::default()
            },
        )
        .expect("cobra should run");
        assert!(report.feasible, "should find a feasible point");
        assert!(
            (report.fun - 1.0).abs() < 0.1,
            "constrained minimum is 1, got {}",
            report.fun
        );
    }

    #[test]
    fn cobra_rides_disk_boundary() {
        let linear = |x: &Array1<f64>| x[0] + x[1];
        let disk = CobraConstraint {
            fun: Arc::new(|x: &Array1<f64>| x[0].powi(2) + x[1].powi(2) - 2.0),
        };
        let report = cobra(
            &linear,
            &[disk],
            CobraConfig {
                bounds: vec![(-2.0, 2.0); 2],
                maxeval: budget(200, 60),
                seed: Some(3),
                ..Default::default()
            },
        )
        .expect("cobra should run");
        assert!(report.feasible);
        assert!(
            report.fun < -1.9,
            "should ride the boundary to -2, got {}",
            report.fun
        );
    }

    #[test]
    fn cobra_solves_g24() {
        use math_audio_test_functions::{g24_constraint1, g24_constraint2, g24_objective};
        let constraints = [
            g24_constraint1 as fn(&Array1<f64>) -> f64,
            g24_constraint2 as fn(&Array1<f64>) -> f64,
        ]
        .map(|g| CobraConstraint {
            fun: Arc::new(move |x: &Array1<f64>| g(x)),
        });
        let report = cobra(
            &g24_objective,
            &constraints,
            CobraConfig {
                bounds: vec![(0.0, 3.0), (0.0, 4.0)],
                maxeval: budget(250, 80),
                seed: Some(2),
                ..Default::default()
            },
        )
        .expect("cobra should run");
        assert!(report.feasible, "should find a feasible point");
        let bar = if cfg!(debug_assertions) { -5.3 } else { -5.4 };
        assert!(
            report.fun <= bar,
            "should approach -5.508, got {}",
            report.fun
        );
    }

    #[test]
    fn cobra_reports_infeasible_gracefully() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let impossible = CobraConstraint {
            fun: Arc::new(|x: &Array1<f64>| 100.0 - x[0]),
        };
        let report = cobra(
            &sphere,
            &[impossible],
            CobraConfig {
                bounds: vec![(-5.0, 5.0); 2],
                maxeval: budget(60, 30),
                seed: Some(4),
                ..Default::default()
            },
        )
        .expect("cobra should run");
        assert!(!report.feasible);
        assert!(report.max_violation > 0.0);
        assert!(report.fun.is_finite());
        assert_eq!(report.nfev, budget(60, 30));
    }

    #[test]
    fn cobra_is_deterministic() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let run = || {
            cobra(
                &sphere,
                &[box_constraint()],
                CobraConfig {
                    bounds: vec![(-5.0, 5.0); 2],
                    maxeval: budget(80, 40),
                    seed: Some(9),
                    ..Default::default()
                },
            )
            .expect("cobra should run")
        };
        let first = run();
        let second = run();
        assert_eq!(first.fun, second.fun);
        assert_eq!(first.x, second.x);
        assert_eq!(first.nfev, second.nfev);
    }

    #[test]
    fn cobra_handles_tiny_budget() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let report = cobra(
            &sphere,
            &[],
            CobraConfig {
                bounds: vec![(-5.0, 5.0); 2],
                maxeval: 2,
                seed: Some(5),
                ..Default::default()
            },
        )
        .expect("cobra should run");
        assert_eq!(report.nfev, 2);
        assert!(report.fun.is_finite());
    }

    #[test]
    fn cobra_rejects_bad_config() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        assert!(
            cobra(
                &sphere,
                &[],
                CobraConfig {
                    bounds: vec![],
                    ..Default::default()
                },
            )
            .is_err()
        );
        assert!(
            cobra(
                &sphere,
                &[],
                CobraConfig {
                    bounds: vec![(-5.0, 5.0); 2],
                    maxeval: 0,
                    ..Default::default()
                },
            )
            .is_err()
        );
    }

    #[test]
    fn plog_is_monotonic_and_sign_preserving() {
        assert_eq!(plog(0.0), 0.0);
        assert!((plog(1.0) - 2.0f64.ln()).abs() < 1e-12);
        assert!(plog(1e7) > plog(1e2));
        assert!(plog(-1e2) > plog(-1e7));
        assert!(plog(-5.0) < 0.0 && plog(5.0) > 0.0);
    }

    #[test]
    fn plog_applies_only_to_wide_ranges() {
        // G09-like: huge range relative to magnitude -> transform.
        assert!(use_plog(&[1e2, 1e4, 1e6, 5e4, 1e3]));
        // G04-like: narrow relative range -> raw.
        assert!(!use_plog(&[-30665.0, -25000.0, -20000.0, -28000.0]));
        // Small absolute range -> raw.
        assert!(!use_plog(&[-5.5, -3.0, 0.0, -7.0]));
        assert!(!use_plog(&[42.0]));
    }

    #[test]
    fn kick_point_alternates_global_and_local() {
        let bounds = vec![(0.0, 10.0); 2];
        let best = Array1::from(vec![5.0, 5.0]);
        let mut rng = StdRng::seed_from_u64(0);
        // Even kicks are global: just check bounds.
        for _ in 0..5 {
            let x = kick_point(&best, 0, &bounds, &mut rng);
            assert!(x.iter().all(|&v| (0.0..=10.0).contains(&v)));
        }
        // Odd kicks stay in the radius box: radius 0.2 * width 10 = 2.
        for _ in 0..20 {
            let x = kick_point(&best, 1, &bounds, &mut rng);
            assert!(
                x.iter().all(|&v| (3.0..=7.0).contains(&v)),
                "local kick escaped its box: {x:?}"
            );
        }
        // Later kicks shrink: count 5 -> radius index 2 -> 0.05 * 10 = 0.5.
        for _ in 0..20 {
            let x = kick_point(&best, 5, &bounds, &mut rng);
            assert!(
                x.iter().all(|&v| (4.5..=5.5).contains(&v)),
                "kick should shrink: {x:?}"
            );
        }
    }

    #[test]
    fn training_subset_prefers_best_and_recent() {
        let mut archive = Vec::new();
        for i in 0..10 {
            archive.push(ArchiveEntry {
                x: Array1::from(vec![i as f64]),
                fun: i as f64,
                cons: vec![],
            });
        }
        let idx = select_training_indices(&archive, 6);
        assert_eq!(idx.len(), 6);
        // Best-first half: lowest objective values.
        assert!(idx.contains(&0));
        assert!(idx.contains(&1));
        assert!(idx.contains(&2));
        // Recent half: highest archive indices.
        assert!(idx.contains(&9));
        assert!(idx.contains(&8));
        assert!(idx.contains(&7));
    }
}

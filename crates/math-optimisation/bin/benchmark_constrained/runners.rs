//! Solver runners for the constrained benchmark suite.
//!
//! Every runner maps its solver onto one nominal true-evaluation `budget` and
//! reports a [`SolverOutcome`] with the *unpenalized* objective value and the
//! true constraint violation at the returned point, so penalty-based and
//! native-constraint solvers are judged identically.

use super::problems::{ConstrainedProblem, FEASIBILITY_TOL};
use math_audio_optimisation::cobyla::{CobylaConstraint, CobylaRhoBegin, cobyla};
use math_audio_optimisation::{
    BayesOptConfig, BayesOptConstraint, CmaEsConfig, CmaEsConstraint, CobraConfig, CobraConstraint,
    CobylaConfig, ConstrainedBayesOptConfig, DEConfigBuilder, IsresConfig, IsresConstraint,
    LShadeConfig, Strategy, bayesian_optimization, cma_es, cobra,
    constrained_bayesian_optimization, differential_evolution, isres,
};
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Solvers compared by the suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SolverKind {
    /// CMA-ES on the shared penalized objective (current best).
    CmaEsPenalty,
    /// L-SHADE with per-constraint quadratic penalties.
    LShadePenalty,
    /// ISRES with native stochastic-ranking constraints.
    Isres,
    /// Multi-start COBYLA with native constraints.
    CobylaMultistart,
    /// GP Bayesian optimisation on the penalized objective (small budgets).
    BayesPenalty,
    /// CMA-ES with native adaptive-penalty constraints and IPOP restarts.
    CmaesNative,
    /// COBRA-style RBF-surrogate constrained optimisation.
    Cobra,
    /// Trust-region constrained Bayesian optimisation (SCBO-lite).
    Scbo,
}

impl SolverKind {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::CmaEsPenalty => "cmaes+pen",
            Self::LShadePenalty => "lshade+pen",
            Self::Isres => "isres",
            Self::CobylaMultistart => "cobyla-ms",
            Self::BayesPenalty => "bo+pen",
            Self::CmaesNative => "cmaes+nat",
            Self::Cobra => "cobra",
            Self::Scbo => "scbo",
        }
    }

    pub(super) fn all() -> Vec<SolverKind> {
        vec![
            Self::CmaEsPenalty,
            Self::LShadePenalty,
            Self::Isres,
            Self::CobylaMultistart,
            Self::BayesPenalty,
            Self::CmaesNative,
            Self::Cobra,
            Self::Scbo,
        ]
    }
}

/// Uniform outcome of one solver run.
pub(super) struct SolverOutcome {
    /// True (unpenalized) objective value at the returned point.
    pub(super) fun: f64,
    pub(super) max_violation: f64,
    pub(super) nfev: usize,
    pub(super) success: bool,
    pub(super) duration: Duration,
    /// True when the solver skipped this budget (e.g. BO over its cap).
    pub(super) skipped: bool,
    pub(super) error: Option<String>,
}

impl SolverOutcome {
    fn skipped(_dim: usize) -> Self {
        Self {
            fun: f64::INFINITY,
            max_violation: f64::INFINITY,
            nfev: 0,
            success: false,
            duration: Duration::from_secs(0),
            skipped: true,
            error: None,
        }
    }

    fn failed(_dim: usize, duration: Duration, error: String) -> Self {
        Self {
            fun: f64::INFINITY,
            max_violation: f64::INFINITY,
            nfev: 0,
            success: false,
            duration,
            skipped: false,
            error: Some(error),
        }
    }

    fn scored(
        problem: &ConstrainedProblem,
        x: Array1<f64>,
        nfev: usize,
        duration: Duration,
    ) -> Self {
        let fun = (problem.objective)(&x);
        let max_violation = problem.max_violation(&x);
        let success = max_violation <= FEASIBILITY_TOL && fun <= problem.target();
        Self {
            fun,
            max_violation,
            nfev,
            success,
            duration,
            skipped: false,
            error: None,
        }
    }
}

/// Run one solver on one problem with a nominal true-evaluation budget.
pub(super) fn run_solver(
    kind: SolverKind,
    problem: &ConstrainedProblem,
    budget: usize,
    seed: u64,
) -> SolverOutcome {
    match kind {
        SolverKind::CmaEsPenalty => run_cmaes_penalty(problem, budget, seed),
        SolverKind::LShadePenalty => run_lshade_penalty(problem, budget, seed),
        SolverKind::Isres => run_isres(problem, budget, seed),
        SolverKind::CobylaMultistart => run_cobyla_multistart(problem, budget, seed),
        SolverKind::BayesPenalty => run_bayes_penalty(problem, budget, seed),
        SolverKind::CmaesNative => run_cmaes_native(problem, budget, seed),
        SolverKind::Cobra => run_cobra(problem, budget, seed),
        SolverKind::Scbo => run_scbo(problem, budget, seed),
    }
}

fn run_cmaes_penalty(problem: &ConstrainedProblem, budget: usize, seed: u64) -> SolverOutcome {
    let start = Instant::now();
    let penalized = |x: &Array1<f64>| problem.penalized(x);
    let config = CmaEsConfig {
        bounds: problem.bounds.clone(),
        maxeval: budget.max(1),
        seed: Some(seed),
        ..Default::default()
    };
    match cma_es(&penalized, config) {
        Ok(report) => SolverOutcome::scored(problem, report.x, report.nfev, start.elapsed()),
        Err(e) => SolverOutcome::failed(problem.dim(), start.elapsed(), format!("{e}")),
    }
}

fn run_cmaes_native(problem: &ConstrainedProblem, budget: usize, seed: u64) -> SolverOutcome {
    let start = Instant::now();
    let constraints: Vec<CmaEsConstraint> = problem
        .constraints
        .iter()
        .map(|g| {
            let gf = *g;
            CmaEsConstraint {
                fun: Arc::new(move |x: &Array1<f64>| gf(x)),
            }
        })
        .collect();
    let objective = |x: &Array1<f64>| (problem.objective)(x);
    let config = CmaEsConfig {
        bounds: problem.bounds.clone(),
        maxeval: budget.max(1),
        seed: Some(seed),
        constraints,
        max_restarts: 3,
        ..Default::default()
    };
    match cma_es(&objective, config) {
        Ok(report) => SolverOutcome::scored(problem, report.x, report.nfev, start.elapsed()),
        Err(e) => SolverOutcome::failed(problem.dim(), start.elapsed(), format!("{e}")),
    }
}

fn run_cobra(problem: &ConstrainedProblem, budget: usize, seed: u64) -> SolverOutcome {
    let start = Instant::now();
    let constraints: Vec<CobraConstraint> = problem
        .constraints
        .iter()
        .map(|g| {
            let gf = *g;
            CobraConstraint {
                fun: Arc::new(move |x: &Array1<f64>| gf(x)),
            }
        })
        .collect();
    let objective = |x: &Array1<f64>| (problem.objective)(x);
    let config = CobraConfig {
        bounds: problem.bounds.clone(),
        maxeval: budget.max(1),
        seed: Some(seed),
        ..Default::default()
    };
    match cobra(&objective, &constraints, config) {
        Ok(report) => SolverOutcome::scored(problem, report.x, report.nfev, start.elapsed()),
        Err(e) => SolverOutcome::failed(problem.dim(), start.elapsed(), format!("{e}")),
    }
}

/// SCBO-lite fits m+1 GPs per iteration, so like GP-BO it only runs
/// small-budget rungs.
const SCBO_MAX_BUDGET: usize = 250;

fn run_scbo(problem: &ConstrainedProblem, budget: usize, seed: u64) -> SolverOutcome {
    if budget > SCBO_MAX_BUDGET {
        return SolverOutcome::skipped(problem.dim());
    }
    let start = Instant::now();
    let constraints: Vec<BayesOptConstraint> = problem
        .constraints
        .iter()
        .map(|g| {
            let gf = *g;
            BayesOptConstraint {
                fun: Arc::new(move |x: &Array1<f64>| gf(x)),
            }
        })
        .collect();
    let objective = |x: &Array1<f64>| (problem.objective)(x);
    let config = ConstrainedBayesOptConfig {
        bounds: problem.bounds.clone(),
        maxeval: budget.max(1),
        seed: Some(seed),
        ..Default::default()
    };
    match constrained_bayesian_optimization(&objective, &constraints, config) {
        Ok(report) => SolverOutcome::scored(problem, report.x, report.nfev, start.elapsed()),
        Err(e) => SolverOutcome::failed(problem.dim(), start.elapsed(), format!("{e}")),
    }
}

fn run_lshade_penalty(problem: &ConstrainedProblem, budget: usize, seed: u64) -> SolverOutcome {
    let start = Instant::now();
    let dim = problem.dim();
    // L-SHADE shrinks NP from 18*dim to 4, so average population is ~9*dim + 2.
    let avg_pop = 9usize.saturating_mul(dim).saturating_add(2).max(1);
    let maxiter = budget.div_ceil(avg_pop).max(1);
    let mut builder = DEConfigBuilder::new()
        .seed(seed)
        .maxiter(maxiter)
        .popsize(18usize.saturating_mul(dim).max(4))
        .strategy(Strategy::LShadeBin)
        .lshade(LShadeConfig::default());
    for g in &problem.constraints {
        let gf = *g;
        let weight = problem.penalty_weight;
        builder = builder.add_penalty_ineq(move |x: &Array1<f64>| gf(x), weight);
    }
    let penalized = |x: &Array1<f64>| (problem.objective)(x);
    match builder.build() {
        Ok(config) => match differential_evolution(&penalized, &problem.bounds, config) {
            Ok(report) => SolverOutcome::scored(problem, report.x, report.nfev, start.elapsed()),
            Err(e) => SolverOutcome::failed(dim, start.elapsed(), format!("{e}")),
        },
        Err(e) => SolverOutcome::failed(dim, start.elapsed(), format!("config: {e}")),
    }
}

fn run_isres(problem: &ConstrainedProblem, budget: usize, seed: u64) -> SolverOutcome {
    let start = Instant::now();
    let constraints: Vec<IsresConstraint> = problem
        .constraints
        .iter()
        .map(|g| {
            let gf = *g;
            IsresConstraint {
                fun: Arc::new(move |x: &Array1<f64>| gf(x)),
            }
        })
        .collect();
    // Scale the population down for small budgets: one generation costs
    // mu + lambda = 8*mu evaluations, so mu ~= budget/40 allows about
    // five generations per run.
    let mu = (budget / 40).clamp(3, 20);
    let config = IsresConfig {
        bounds: problem.bounds.clone(),
        maxeval: budget.max(1),
        seed: Some(seed),
        mu,
        lambda: 0, // 0 means 7*mu
        ..Default::default()
    };
    let objective = |x: &Array1<f64>| (problem.objective)(x);
    match isres(&objective, &constraints, config) {
        Ok(report) => SolverOutcome::scored(problem, report.x, report.nfev, start.elapsed()),
        Err(e) => SolverOutcome::failed(problem.dim(), start.elapsed(), format!("{e}")),
    }
}

fn run_cobyla_multistart(problem: &ConstrainedProblem, budget: usize, seed: u64) -> SolverOutcome {
    let start = Instant::now();
    let dim = problem.dim();
    let constraints: Vec<CobylaConstraint> = problem
        .constraints
        .iter()
        .map(|g| {
            let gf = *g;
            CobylaConstraint {
                fun: Arc::new(move |x: &Array1<f64>| gf(x)),
            }
        })
        .collect();
    let objective = |x: &Array1<f64>| (problem.objective)(x);
    let starts = (budget / (50usize.saturating_mul(dim).max(1))).clamp(2, 8);
    let per_start = (budget / starts).max(1);
    let min_width = problem
        .bounds
        .iter()
        .map(|(lo, hi)| hi - lo)
        .fold(f64::INFINITY, f64::min);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut best_x: Option<Array1<f64>> = None;
    let mut best_fun = f64::INFINITY;
    let mut consumed = 0usize;
    for _ in 0..starts {
        let x0 = Array1::from_vec(
            problem
                .bounds
                .iter()
                .map(|(lo, hi)| rng.random_range(*lo..=*hi))
                .collect(),
        );
        let config = CobylaConfig {
            x0,
            bounds: problem.bounds.clone(),
            rho_begin: CobylaRhoBegin::All(0.25 * min_width),
            maxeval: per_start,
            ..Default::default()
        };
        match cobyla(&objective, &constraints, config) {
            Ok(report) => {
                // The COBYLA wrapper reports the configured budget as nfev
                // (exact counts are unavailable), so account per-start budgets.
                consumed = consumed.saturating_add(report.nfev);
                let fun = (problem.objective)(&report.x);
                let violation = problem.max_violation(&report.x);
                let feasible = violation <= FEASIBILITY_TOL;
                let best_violation = best_x
                    .as_ref()
                    .map(|x| problem.max_violation(x))
                    .unwrap_or(f64::INFINITY);
                let improves = match (feasible, best_violation <= FEASIBILITY_TOL) {
                    (true, true) | (false, false) => fun < best_fun,
                    (true, false) => true,
                    (false, true) => false,
                };
                if improves {
                    best_fun = fun;
                    best_x = Some(report.x.clone());
                }
            }
            Err(e) => {
                return SolverOutcome::failed(dim, start.elapsed(), format!("{e}"));
            }
        }
    }
    match best_x {
        Some(x) => SolverOutcome::scored(problem, x, consumed, start.elapsed()),
        None => SolverOutcome::failed(dim, start.elapsed(), "no start produced a result".into()),
    }
}

/// GP-BO is only competitive on small budgets (Cholesky cost grows with the
/// cube of the evaluation count), so it sits out larger rungs.
const BAYES_MAX_BUDGET: usize = 200;

fn run_bayes_penalty(problem: &ConstrainedProblem, budget: usize, seed: u64) -> SolverOutcome {
    if budget > BAYES_MAX_BUDGET {
        return SolverOutcome::skipped(problem.dim());
    }
    let start = Instant::now();
    let penalized = |x: &Array1<f64>| problem.penalized(x);
    let config = BayesOptConfig {
        bounds: problem.bounds.clone(),
        maxeval: budget.max(1),
        seed: Some(seed),
        ..Default::default()
    };
    match bayesian_optimization(&penalized, config) {
        Ok(report) => SolverOutcome::scored(problem, report.x, report.nfev, start.elapsed()),
        Err(e) => SolverOutcome::failed(problem.dim(), start.elapsed(), format!("{e}")),
    }
}

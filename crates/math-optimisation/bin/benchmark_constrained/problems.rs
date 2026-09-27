//! Constrained optimisation benchmark suite.
//!
//! Compares solvers on genuinely constrained non-convex problems with a
//! budget-normalised metric: best feasible objective value versus number of
//! true function evaluations (`nfev`). Each evaluation of the objective counts
//! as one `nfev`; constraint values ride along for free, matching the
//! expensive-simulation model where one run yields `f` and all `g` together.
//!
//! Problems come from the CEC 2006 G-function family (see
//! `math-test-functions`) plus a Rosenbrock-disk sanity case. Penalty-based
//! solvers share one quadratic penalty (`w * sum(max(0, g)^2)`) with a
//! per-problem weight, so differences reflect solver behaviour rather than
//! penalty tuning.

use ndarray::Array1;

/// Budget rungs, expressed as multiples of the problem dimension.
pub(super) const RUNGS_PER_DIM: [usize; 5] = [10, 25, 50, 100, 200];

/// Seeds used for every (solver, rung) cell.
pub(super) const SEEDS: [u64; 3] = [42, 43, 44];

/// Maximum constraint violation still counted as feasible.
pub(super) const FEASIBILITY_TOL: f64 = 1e-4;

/// A constrained benchmark problem: minimise `objective` subject to
/// `constraints[i](x) <= 0` within `bounds`.
pub(super) struct ConstrainedProblem {
    pub(super) name: &'static str,
    pub(super) bounds: Vec<(f64, f64)>,
    pub(super) objective: fn(&Array1<f64>) -> f64,
    pub(super) constraints: Vec<fn(&Array1<f64>) -> f64>,
    /// Best known feasible objective value (literature).
    pub(super) best_known: f64,
    /// Success threshold is `best_known + success_tol`.
    pub(super) success_tol: f64,
    /// Shared quadratic-penalty weight for penalty-based solvers.
    pub(super) penalty_weight: f64,
    /// Literature minimiser, used to validate the problem definition.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) reference_point: Vec<f64>,
}

impl ConstrainedProblem {
    pub(super) fn dim(&self) -> usize {
        self.bounds.len()
    }

    pub(super) fn target(&self) -> f64 {
        self.best_known + self.success_tol
    }

    pub(super) fn n_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Maximum constraint violation at `x` (0 when feasible).
    pub(super) fn max_violation(&self, x: &Array1<f64>) -> f64 {
        let mut worst = 0.0f64;
        for g in &self.constraints {
            let v = g(x);
            if v > worst {
                worst = v;
            }
        }
        worst
    }

    /// Objective plus shared quadratic penalty for penalty-based solvers.
    pub(super) fn penalized(&self, x: &Array1<f64>) -> f64 {
        let base = (self.objective)(x);
        let mut p = 0.0;
        for g in &self.constraints {
            let v = g(x).max(0.0);
            p += v * v;
        }
        base + self.penalty_weight * p
    }

    /// Nominal evaluation budgets for this problem (rungs scaled by dimension).
    pub(super) fn budgets(&self) -> Vec<usize> {
        RUNGS_PER_DIM
            .iter()
            .map(|r| r.saturating_mul(self.dim()).max(1))
            .collect()
    }
}

/// All constrained benchmark problems.
pub(super) fn all_problems() -> Vec<ConstrainedProblem> {
    use math_audio_test_functions as tf;

    vec![
        ConstrainedProblem {
            name: "g06",
            bounds: vec![(13.0, 100.0), (0.0, 100.0)],
            objective: tf::g06_objective,
            constraints: vec![tf::g06_constraint1, tf::g06_constraint2],
            best_known: -6961.81387558,
            success_tol: 1.0,
            penalty_weight: 7.0e6,
            reference_point: vec![14.095, 0.84296],
        },
        ConstrainedProblem {
            name: "g08",
            bounds: vec![(0.0, 10.0), (0.0, 10.0)],
            objective: tf::g08_objective,
            constraints: vec![tf::g08_constraint1, tf::g08_constraint2],
            best_known: -0.095825,
            success_tol: 1e-3,
            penalty_weight: 1.0e3,
            reference_point: vec![1.2279713, 4.2453733],
        },
        ConstrainedProblem {
            name: "g24",
            bounds: vec![(0.0, 3.0), (0.0, 4.0)],
            objective: tf::g24_objective,
            constraints: vec![tf::g24_constraint1, tf::g24_constraint2],
            best_known: -5.50801327,
            success_tol: 1e-3,
            penalty_weight: 6.0e3,
            reference_point: vec![2.32952019, 3.17849307],
        },
        ConstrainedProblem {
            name: "g04",
            bounds: vec![
                (78.0, 102.0),
                (33.0, 45.0),
                (27.0, 45.0),
                (27.0, 45.0),
                (27.0, 45.0),
            ],
            objective: tf::g04_objective,
            constraints: vec![
                tf::g04_constraint1,
                tf::g04_constraint2,
                tf::g04_constraint3,
                tf::g04_constraint4,
                tf::g04_constraint5,
                tf::g04_constraint6,
            ],
            best_known: -30665.53867178,
            success_tol: 1.0,
            penalty_weight: 3.0e7,
            reference_point: vec![78.0, 33.0, 29.99525602, 45.0, 36.77581290],
        },
        ConstrainedProblem {
            name: "g09",
            bounds: vec![(-10.0, 10.0); 7],
            objective: tf::g09_objective,
            constraints: vec![
                tf::g09_constraint1,
                tf::g09_constraint2,
                tf::g09_constraint3,
                tf::g09_constraint4,
            ],
            best_known: 680.63005737,
            success_tol: 0.5,
            penalty_weight: 7.0e5,
            reference_point: vec![
                2.33049935,
                1.95137236,
                -0.47754139,
                4.36572624,
                -0.62448747,
                1.03813099,
                1.59422667,
            ],
        },
        ConstrainedProblem {
            name: "rosenbrock_disk",
            bounds: vec![(-2.048, 2.048), (-2.048, 2.048)],
            objective: tf::rosenbrock_objective,
            constraints: vec![tf::rosenbrock_disk_constraint],
            best_known: 0.0,
            success_tol: 1e-2,
            penalty_weight: 1.0e3,
            reference_point: vec![1.0, 1.0],
        },
    ]
}

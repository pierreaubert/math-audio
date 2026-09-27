//! Smoke tests for the constrained benchmark suite.

use super::problems::{FEASIBILITY_TOL, all_problems};
use super::runners::{SolverKind, run_solver};
use ndarray::Array1;

#[test]
fn reference_points_are_feasible_and_match_best_known() {
    for problem in all_problems() {
        let x = Array1::from(problem.reference_point.clone());
        let violation = problem.max_violation(&x);
        assert!(
            violation <= 1e-2,
            "{} reference violates constraints by {violation:.3e}",
            problem.name
        );
        let fun = (problem.objective)(&x);
        assert!(
            (fun - problem.best_known).abs() <= problem.success_tol,
            "{} reference fun {fun:.6e} differs from best_known {:.6e}",
            problem.name,
            problem.best_known
        );
        // The shared penalty must be (near-)zero at the reference point.
        let penalized = problem.penalized(&x);
        assert!(
            (penalized - fun).abs() <= problem.success_tol,
            "{} penalized {penalized:.6e} differs from fun {fun:.6e}",
            problem.name
        );
    }
}

#[test]
fn budgets_scale_with_dimension() {
    for problem in all_problems() {
        let budgets = problem.budgets();
        assert_eq!(budgets.len(), 5);
        assert!(
            budgets.windows(2).all(|w| w[0] < w[1]),
            "{} budgets not increasing: {budgets:?}",
            problem.name
        );
        assert_eq!(budgets[0], 10 * problem.dim());
        assert_eq!(budgets[4], 200 * problem.dim());
    }
}

#[test]
fn every_non_bo_solver_runs_on_g24() {
    let problems = all_problems();
    let problem = problems.iter().find(|p| p.name == "g24").unwrap();
    for solver in [
        SolverKind::CmaEsPenalty,
        SolverKind::LShadePenalty,
        SolverKind::Isres,
        SolverKind::CobylaMultistart,
        SolverKind::CmaesNative,
    ] {
        let outcome = run_solver(solver, problem, 200, 7);
        assert!(
            !outcome.skipped,
            "{:?} unexpectedly skipped a 200-eval budget",
            solver
        );
        assert!(
            outcome.error.is_none(),
            "{:?} failed: {:?}",
            solver,
            outcome.error
        );
        assert!(
            outcome.fun.is_finite(),
            "{:?} returned non-finite fun",
            solver
        );
        assert!(outcome.nfev > 0, "{:?} consumed no evaluations", solver);
    }
}

#[test]
fn cobra_runs_on_small_budget() {
    // Kept small: each iteration refits several RBF surrogates, which is
    // slow in debug builds. Convergence quality is the release-mode
    // suite's job.
    let problems = all_problems();
    let problem = problems.iter().find(|p| p.name == "g24").unwrap();
    let outcome = run_solver(SolverKind::Cobra, problem, 60, 7);
    assert!(!outcome.skipped);
    assert!(outcome.error.is_none());
    assert!(outcome.fun.is_finite());
    assert!(outcome.nfev > 0);
}

#[test]
fn scbo_runs_on_small_budget() {
    // Same rationale as the BO smoke test: GP refits are slow in debug.
    let problems = all_problems();
    let problem = problems.iter().find(|p| p.name == "g24").unwrap();
    let budget = if cfg!(debug_assertions) { 25 } else { 100 };
    let outcome = run_solver(SolverKind::Scbo, problem, budget, 7);
    assert!(!outcome.skipped);
    assert!(outcome.error.is_none());
    assert!(outcome.fun.is_finite());
    assert!(outcome.nfev > 0);
}

#[test]
fn bayes_penalty_runs_on_small_budget() {
    // Kept small: each BO iteration refits the GP, which is slow in debug
    // builds. Convergence quality is the release-mode suite's job.
    let problems = all_problems();
    let problem = problems.iter().find(|p| p.name == "g24").unwrap();
    let budget = if cfg!(debug_assertions) { 12 } else { 100 };
    let outcome = run_solver(SolverKind::BayesPenalty, problem, budget, 7);
    assert!(!outcome.skipped);
    assert!(outcome.error.is_none());
    assert!(outcome.fun.is_finite());
    assert!(outcome.nfev > 0);
}

#[test]
fn budgeted_solvers_respect_small_budgets() {
    let problems = all_problems();
    let problem = problems.iter().find(|p| p.name == "g24").unwrap();
    for solver in [SolverKind::CmaEsPenalty, SolverKind::Isres] {
        let outcome = run_solver(solver, problem, 100, 7);
        assert!(
            outcome.nfev <= 100,
            "{:?} consumed {} evals over a 100 budget",
            solver,
            outcome.nfev
        );
    }
}

#[test]
fn feasibility_tolerance_is_positive_and_tight() {
    const {
        assert!(FEASIBILITY_TOL > 0.0 && FEASIBILITY_TOL <= 1e-3);
    }
}

//! Tabular reporting for the constrained benchmark suite.

use super::problems::{ConstrainedProblem, FEASIBILITY_TOL};
use super::runners::{SolverKind, SolverOutcome};

/// One recorded run.
pub(super) struct Record<'a> {
    pub(super) problem: &'a ConstrainedProblem,
    pub(super) solver: SolverKind,
    pub(super) budget: usize,
    pub(super) seed: u64,
    pub(super) outcome: SolverOutcome,
}

/// Median of a non-empty list.
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        0.5 * (values[mid - 1] + values[mid])
    }
}

/// Compact cell: median feasible value with success count, `infeas` when no
/// seed found a feasible point, or `---` when the solver sat the rung out.
fn cell_text(outcomes: &[&SolverOutcome]) -> String {
    if outcomes.iter().all(|o| o.skipped) {
        return "---".to_string();
    }
    let feasible: Vec<f64> = outcomes
        .iter()
        .filter(|o| !o.skipped && o.error.is_none() && o.max_violation <= FEASIBILITY_TOL)
        .map(|o| o.fun)
        .collect();
    let successes = outcomes.iter().filter(|o| o.success).count();
    if feasible.is_empty() {
        return format!("infeas(0/{})", outcomes.len());
    }
    format!("{:.4e}({}/{})", median(feasible), successes, outcomes.len())
}

/// Print the per-rung comparison table for one problem.
pub(super) fn print_problem_table(problem: &ConstrainedProblem, records: &[&Record<'_>]) {
    println!(
        "\n=== {} (dim={}, constraints={}, target={:.6e}) ===",
        problem.name,
        problem.dim(),
        problem.n_constraints(),
        problem.target()
    );
    let mut budgets: Vec<usize> = records.iter().map(|r| r.budget).collect();
    budgets.sort_unstable();
    budgets.dedup();
    print!("{:12}", "solver");
    for b in &budgets {
        print!(" {:>22}", format!("n={b}"));
    }
    println!();
    for solver in SolverKind::all() {
        print!("{:12}", solver.label());
        for b in &budgets {
            let outcomes: Vec<&SolverOutcome> = records
                .iter()
                .filter(|r| r.solver == solver && r.budget == *b)
                .map(|r| &r.outcome)
                .collect();
            print!(" {:>22}", cell_text(&outcomes));
        }
        println!();
    }
}

/// Print per-seed detail rows for one problem.
pub(super) fn print_problem_verbose(problem: &ConstrainedProblem, records: &[&Record<'_>]) {
    println!(
        "\n--- {} detail (target={:.6e}) ---",
        problem.name,
        problem.target()
    );
    for solver in SolverKind::all() {
        for r in records.iter().filter(|r| r.solver == solver) {
            let o = &r.outcome;
            if o.skipped {
                println!(
                    "  {:12} budget={:5} seed={} skipped",
                    solver.label(),
                    r.budget,
                    r.seed
                );
            } else if let Some(err) = &o.error {
                println!(
                    "  {:12} budget={:5} seed={} ERROR {err}",
                    solver.label(),
                    r.budget,
                    r.seed
                );
            } else {
                println!(
                    "  {:12} budget={:5} seed={} fun={:.6e} viol={:.2e} nfev={:5} {} {:.1?}",
                    solver.label(),
                    r.budget,
                    r.seed,
                    o.fun,
                    o.max_violation,
                    o.nfev,
                    if o.success { "OK " } else { "miss" },
                    o.duration,
                );
            }
        }
    }
}

/// Print the cross-problem summary: per solver, problems solved at the final
/// rung (majority of seeds) and total rung-level successes.
pub(super) fn print_summary(records: &[&Record<'_>]) {
    println!("\n=== SUMMARY (final rung, majority of seeds) ===");
    let mut problems: Vec<&str> = records.iter().map(|r| r.problem.name).collect();
    problems.sort_unstable();
    problems.dedup();
    for solver in SolverKind::all() {
        let mut solved = 0usize;
        let mut total_successes = 0usize;
        let mut total_runs = 0usize;
        for name in &problems {
            let solver_records: Vec<&&Record<'_>> = records
                .iter()
                .filter(|r| r.problem.name == *name && r.solver == solver)
                .collect();
            // The final rung is the largest budget the solver actually ran;
            // skipped rungs (small-budget-only solvers) must not count.
            let Some(final_budget) = solver_records
                .iter()
                .filter(|r| !r.outcome.skipped)
                .map(|r| r.budget)
                .max()
            else {
                continue;
            };
            let final_runs: Vec<&&&Record<'_>> = solver_records
                .iter()
                .filter(|r| r.budget == final_budget && !r.outcome.skipped)
                .collect();
            if !final_runs.is_empty() {
                let ok = final_runs.iter().filter(|r| r.outcome.success).count();
                if ok * 2 > final_runs.len() {
                    solved += 1;
                }
            }
            for r in solver_records {
                if !r.outcome.skipped {
                    total_runs += 1;
                    if r.outcome.success {
                        total_successes += 1;
                    }
                }
            }
        }
        println!(
            "  {:12} problems solved: {solved}/{}   rung successes: {total_successes}/{total_runs}",
            solver.label(),
            problems.len()
        );
    }
}

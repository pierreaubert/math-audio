use clap::{Arg, Command};
use std::time::Instant;

#[path = "benchmark_constrained/print.rs"]
mod print;
#[path = "benchmark_constrained/problems.rs"]
mod problems;
#[path = "benchmark_constrained/runners.rs"]
mod runners;
#[cfg(test)]
#[path = "benchmark_constrained/tests.rs"]
mod tests;

use print::{Record, print_problem_table, print_problem_verbose, print_summary};
use problems::{SEEDS, all_problems};
use runners::{SolverKind, run_solver};

fn main() {
    let matches = Command::new("benchmark_constrained")
        .version("0.1.0")
        .about("Constrained optimisation benchmarks: best feasible value vs evaluation budget")
        .arg(
            Arg::new("filter")
                .short('f')
                .long("filter")
                .value_name("PATTERN")
                .help("Only run problems matching this pattern")
                .num_args(1),
        )
        .arg(
            Arg::new("list")
                .short('l')
                .long("list")
                .help("List available problems")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Show per-seed detail rows")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("quick")
                .short('q')
                .long("quick")
                .help("Single seed and middle three budget rungs")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    let problems = all_problems();

    if matches.get_flag("list") {
        println!("Available problems:");
        for p in &problems {
            println!(
                "  {} (dim={}, constraints={}, target={:.6e})",
                p.name,
                p.dim(),
                p.n_constraints(),
                p.target()
            );
        }
        return;
    }

    let filter = matches.get_one::<String>("filter");
    let verbose = matches.get_flag("verbose");
    let quick = matches.get_flag("quick");
    let seeds: Vec<u64> = if quick {
        vec![SEEDS[0]]
    } else {
        SEEDS.to_vec()
    };

    let selected: Vec<&problems::ConstrainedProblem> = problems
        .iter()
        .filter(|p| filter.map(|f| p.name.contains(f)).unwrap_or(true))
        .collect();
    if selected.is_empty() {
        eprintln!("No problems match the filter criteria");
        std::process::exit(1);
    }

    println!("Running {} problem(s)...", selected.len());
    let total_start = Instant::now();
    let mut records: Vec<Record<'_>> = Vec::new();

    for problem in selected {
        let mut budgets = problem.budgets();
        if quick {
            budgets = budgets.into_iter().skip(1).take(3).collect();
        }
        for solver in SolverKind::all() {
            for &budget in &budgets {
                for &seed in &seeds {
                    println!(
                        "Running {} {} budget={} seed={}...",
                        problem.name,
                        solver.label(),
                        budget,
                        seed
                    );
                    let outcome = run_solver(solver, problem, budget, seed);
                    records.push(Record {
                        problem,
                        solver,
                        budget,
                        seed,
                        outcome,
                    });
                }
            }
        }
        let problem_records: Vec<&Record<'_>> = records
            .iter()
            .filter(|r| r.problem.name == problem.name)
            .collect();
        print_problem_table(problem, &problem_records);
        if verbose {
            print_problem_verbose(problem, &problem_records);
        }
    }

    let all_refs: Vec<&Record<'_>> = records.iter().collect();
    print_summary(&all_refs);

    println!("\nTotal wall time: {:.1?}", total_start.elapsed());
}

use ndarray::{Array1, Array2};
use std::fmt;

/// Why a DE run stopped. Typed counterpart of the human-readable
/// `message`; `success` alone conflates convergence with callback stops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DETermination {
    /// Population spread met the convergence threshold.
    Converged,
    /// Population spread stayed within tolerance for the configured
    /// plateau window of consecutive generations.
    Plateau { window_generations: usize },
    /// The generation limit was reached.
    MaxIterations,
    /// The callback stopped the run.
    CallbackStop,
    /// The callback stopped the run, accepting the incumbent.
    CallbackStopAcceptBest,
    /// Every variable was fixed by bounds; nothing to search.
    AllFixed,
}

/// Result/report of a DE optimization run.
///
/// Contains the optimal solution, convergence status, and statistics.
#[derive(Clone)]
pub struct DEReport {
    /// The optimal solution vector.
    pub x: Array1<f64>,
    /// The objective function value at the optimal solution.
    pub fun: f64,
    /// Whether the optimization converged successfully.
    pub success: bool,
    /// Human-readable status message.
    pub message: String,
    /// Number of iterations (generations) performed.
    pub nit: usize,
    /// Number of function evaluations performed.
    pub nfev: usize,
    /// Final population matrix (NP x n).
    pub population: Array2<f64>,
    /// Fitness values for each population member.
    pub population_energies: Array1<f64>,
    /// Typed stop reason.
    pub termination: DETermination,
    /// Seed candidates actually placed in the initial population
    /// (after clamping, integrality, dedup, and truncation).
    pub initial_seed_count: usize,
}

impl fmt::Debug for DEReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DEReport")
            .field("x", &format!("len={}", self.x.len()))
            .field("fun", &self.fun)
            .field("success", &self.success)
            .field("message", &self.message)
            .field("nit", &self.nit)
            .field("nfev", &self.nfev)
            .field(
                "population",
                &format!("{}x{}", self.population.nrows(), self.population.ncols()),
            )
            .field(
                "population_energies",
                &format!("len={}", self.population_energies.len()),
            )
            .field("termination", &self.termination)
            .field("initial_seed_count", &self.initial_seed_count)
            .finish()
    }
}

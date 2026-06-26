//! NSGA-II and NSGA-III Pareto multi-objective optimizers.
//!
//! The implementation minimises every objective. It uses Deb's fast
//! non-dominated sorting, simulated binary crossover, polynomial mutation,
//! crowding-distance survival for NSGA-II, and reference-direction niching for
//! NSGA-III.

mod assign;
mod compare;
mod individual;
mod misc;
mod nsga_config;
mod reference;
#[cfg(test)]
mod tests;
mod types;

pub use individual::*;
pub use nsga_config::*;
pub use types::*;

//! Levenberg-Marquardt bounded nonlinear least-squares solver.
//!
//! Solves `min_x  ||r(x)||^2_W` subject to `lb <= x <= ub`, where `r(x)` is a
//! vector-valued residual function and `W` is a diagonal weight matrix.
//!
//! The solver interpolates between Gauss-Newton (fast near the minimum) and
//! gradient descent (robust far from it) via an adaptive damping parameter `lambda`.
//! Bounds are enforced by projecting the trial step.

mod error;
mod jtw;
mod lmconfig_builder;
mod misc;
#[cfg(test)]
mod tests;
mod types;

pub use error::*;
pub use lmconfig_builder::*;
pub use types::*;

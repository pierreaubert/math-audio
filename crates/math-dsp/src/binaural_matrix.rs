//! Binaural transfer-matrix utilities.
//!
//! The routines here deliberately avoid roomEQ-specific concepts such as
//! speaker roles, head-position names, or artifact formats. Callers provide
//! frequency-bin matrices and receive regularized inverse filters.

mod direct;
mod misc;
mod solve;
#[cfg(test)]
mod tests;
mod transfer_matrix_bin;
mod types;

pub use direct::*;
pub use misc::*;
pub use solve::*;
pub use transfer_matrix_bin::*;
pub use types::*;

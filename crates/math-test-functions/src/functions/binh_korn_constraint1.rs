//! Binh Korn Constraint1 test function

use ndarray::Array1;

/// Binh-Korn constraint 1: x1^2 + x2^2 <= 25
pub fn binh_korn_constraint1(x: &Array1<f64>) -> f64 {
    assert!(
        x.len() == 2,
        "binh_korn_constraint1 requires 2 dimensions, got {}",
        x.len()
    );
    x[0].powi(2) + x[1].powi(2) - 25.0
}

//! Mishras Bird Objective test function

use ndarray::Array1;

/// Mishra's Bird objective function
pub fn mishras_bird_objective(x: &Array1<f64>) -> f64 {
    assert!(
        x.len() == 2,
        "mishras_bird_objective requires 2 dimensions, got {}",
        x.len()
    );
    let x1 = x[0];
    let x2 = x[1];
    let sin_term = ((x1 * x2).exp().cos() - (x1.powi(2) + x2.powi(2)).cos()).sin();
    sin_term.powi(2) + 0.01 * (x1 + x2)
}

//! G08 objective (CEC 2006 constrained benchmark).

use ndarray::Array1;
use std::f64::consts::PI;

/// G08 objective: `f(x) = -sin^3(2*pi*x1) * sin(2*pi*x2) / (x1^3 * (x1 + x2))`.
///
/// Bounds: `x in [0, 10]^2`. Known constrained minimum `f* = -0.095825`
/// at `(1.2279713, 4.2453733)`.
pub fn g08_objective(x: &Array1<f64>) -> f64 {
    assert!(
        x.len() == 2,
        "g08_objective requires 2 dimensions, got {}",
        x.len()
    );
    let numerator = -(2.0 * PI * x[0]).sin().powi(3) * (2.0 * PI * x[1]).sin();
    let denominator = x[0].powi(3) * (x[0] + x[1]);
    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "requires 2 dimensions")]
    fn rejects_wrong_dimension() {
        let x = Array1::from(vec![0.0]);
        let _ = g08_objective(&x);
    }

    #[test]
    fn known_optimum_matches_literature_value() {
        let x = Array1::from(vec![1.2279713, 4.2453733]);
        let value = g08_objective(&x);
        assert!((value - -0.095825).abs() < 1e-5, "got {value}");
    }

    #[test]
    fn finite_at_interior_point() {
        let x = Array1::from(vec![2.0, 3.0]);
        assert!(g08_objective(&x).is_finite());
    }
}

//! G04 objective (CEC 2006 constrained benchmark).

use ndarray::Array1;

/// G04 objective: `f(x) = 5.3578547*x3^2 + 0.8356891*x1*x5 + 37.293239*x1 - 40792.141`.
///
/// Bounds: `x1 in [78, 102]`, `x2 in [33, 45]`, `x3, x4, x5 in [27, 45]`.
/// Known constrained minimum `f* = -30665.53867178` at
/// `(78, 33, 29.99525602, 45, 36.77581290)`.
pub fn g04_objective(x: &Array1<f64>) -> f64 {
    assert!(
        x.len() == 5,
        "g04_objective requires 5 dimensions, got {}",
        x.len()
    );
    5.3578547 * x[2].powi(2) + 0.8356891 * x[0] * x[4] + 37.293239 * x[0] - 40792.141
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "requires 5 dimensions")]
    fn rejects_wrong_dimension() {
        let x = Array1::from(vec![0.0]);
        let _ = g04_objective(&x);
    }

    #[test]
    fn known_optimum_matches_literature_value() {
        let x = Array1::from(vec![78.0, 33.0, 29.99525602, 45.0, 36.77581290]);
        let value = g04_objective(&x);
        assert!((value - -30665.53867178).abs() < 1e-3, "got {value}");
    }

    #[test]
    fn finite_at_lower_bounds() {
        let x = Array1::from(vec![78.0, 33.0, 27.0, 27.0, 27.0]);
        assert!(g04_objective(&x).is_finite());
    }
}

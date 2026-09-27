//! G06 objective (CEC 2006 constrained benchmark).

use ndarray::Array1;

/// G06 objective: `f(x) = (x1 - 10)^3 + (x2 - 20)^3`.
///
/// Bounds: `x1 in [13, 100]`, `x2 in [0, 100]`. Known constrained
/// minimum `f* = -6961.81387558` at `(14.095, 0.84296)`.
pub fn g06_objective(x: &Array1<f64>) -> f64 {
    assert!(
        x.len() == 2,
        "g06_objective requires 2 dimensions, got {}",
        x.len()
    );
    (x[0] - 10.0).powi(3) + (x[1] - 20.0).powi(3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "requires 2 dimensions")]
    fn rejects_wrong_dimension() {
        let x = Array1::from(vec![0.0]);
        let _ = g06_objective(&x);
    }

    #[test]
    fn known_optimum_matches_literature_value() {
        let x = Array1::from(vec![14.095, 0.84296]);
        let value = g06_objective(&x);
        // Tolerance accounts for the rounded literature location (steep cubic).
        assert!((value - -6961.81387558).abs() < 2e-3, "got {value}");
    }

    #[test]
    fn finite_at_bounds_corners() {
        for x in [
            Array1::from(vec![13.0, 0.0]),
            Array1::from(vec![100.0, 100.0]),
        ] {
            assert!(g06_objective(&x).is_finite());
        }
    }
}

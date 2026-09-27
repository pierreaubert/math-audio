//! G06 first constraint (CEC 2006 constrained benchmark).

use ndarray::Array1;

/// G06 constraint 1: `g1(x) = -(x1 - 5)^2 - (x2 - 5)^2 + 100 <= 0`.
///
/// Returns the violation amount (feasible when `<= 0`).
pub fn g06_constraint1(x: &Array1<f64>) -> f64 {
    assert!(
        x.len() == 2,
        "g06_constraint1 requires 2 dimensions, got {}",
        x.len()
    );
    -(x[0] - 5.0).powi(2) - (x[1] - 5.0).powi(2) + 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "requires 2 dimensions")]
    fn rejects_wrong_dimension() {
        let x = Array1::from(vec![0.0]);
        let _ = g06_constraint1(&x);
    }

    #[test]
    fn active_at_known_optimum() {
        let x = Array1::from(vec![14.095, 0.84296]);
        let value = g06_constraint1(&x);
        assert!(value.abs() < 1e-3, "got {value}");
    }

    #[test]
    fn violated_inside_forbidden_disk() {
        let x = Array1::from(vec![5.0, 5.0]);
        assert_eq!(g06_constraint1(&x), 100.0);
    }
}

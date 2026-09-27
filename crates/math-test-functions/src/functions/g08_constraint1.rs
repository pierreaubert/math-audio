//! G08 first constraint (CEC 2006 constrained benchmark).

use ndarray::Array1;

/// G08 constraint 1: `g1(x) = x1^2 - x2 + 1 <= 0`.
///
/// Returns the violation amount (feasible when `<= 0`).
pub fn g08_constraint1(x: &Array1<f64>) -> f64 {
    assert!(
        x.len() == 2,
        "g08_constraint1 requires 2 dimensions, got {}",
        x.len()
    );
    x[0].powi(2) - x[1] + 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "requires 2 dimensions")]
    fn rejects_wrong_dimension() {
        let x = Array1::from(vec![0.0]);
        let _ = g08_constraint1(&x);
    }

    #[test]
    fn feasible_at_known_optimum() {
        let x = Array1::from(vec![1.2279713, 4.2453733]);
        assert!(g08_constraint1(&x) <= 0.0);
    }

    #[test]
    fn violated_for_large_x1_small_x2() {
        let x = Array1::from(vec![10.0, 0.0]);
        assert!(g08_constraint1(&x) > 0.0);
    }
}

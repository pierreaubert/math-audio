//! G24 first constraint (CEC 2006 constrained benchmark).

use ndarray::Array1;

/// G24 constraint 1: `g1(x) = -2*x1^4 + 8*x1^3 - 8*x1^2 + x2 - 2 <= 0`.
///
/// Returns the violation amount (feasible when `<= 0`).
pub fn g24_constraint1(x: &Array1<f64>) -> f64 {
    assert!(
        x.len() == 2,
        "g24_constraint1 requires 2 dimensions, got {}",
        x.len()
    );
    -2.0 * x[0].powi(4) + 8.0 * x[0].powi(3) - 8.0 * x[0].powi(2) + x[1] - 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "requires 2 dimensions")]
    fn rejects_wrong_dimension() {
        let x = Array1::from(vec![0.0]);
        let _ = g24_constraint1(&x);
    }

    #[test]
    fn active_at_known_optimum() {
        let x = Array1::from(vec![2.32952019, 3.17849307]);
        let value = g24_constraint1(&x);
        assert!(value.abs() < 1e-4, "got {value}");
    }

    #[test]
    fn violated_at_origin_top_corner() {
        let x = Array1::from(vec![0.0, 4.0]);
        assert_eq!(g24_constraint1(&x), 2.0);
    }
}

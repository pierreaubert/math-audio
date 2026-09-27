//! G24 second constraint (CEC 2006 constrained benchmark).

use ndarray::Array1;

/// G24 constraint 2: `g2(x) = -4*x1^4 + 32*x1^3 - 88*x1^2 + 96*x1 + x2 - 36 <= 0`.
///
/// Returns the violation amount (feasible when `<= 0`).
pub fn g24_constraint2(x: &Array1<f64>) -> f64 {
    assert!(
        x.len() == 2,
        "g24_constraint2 requires 2 dimensions, got {}",
        x.len()
    );
    -4.0 * x[0].powi(4) + 32.0 * x[0].powi(3) - 88.0 * x[0].powi(2) + 96.0 * x[0] + x[1] - 36.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "requires 2 dimensions")]
    fn rejects_wrong_dimension() {
        let x = Array1::from(vec![0.0]);
        let _ = g24_constraint2(&x);
    }

    #[test]
    fn feasible_at_known_optimum() {
        let x = Array1::from(vec![2.32952019, 3.17849307]);
        assert!(g24_constraint2(&x) <= 1e-6);
    }

    #[test]
    fn violated_at_midpoint_top_edge() {
        let x = Array1::from(vec![1.5, 4.0]);
        assert!(g24_constraint2(&x) > 0.0);
    }
}

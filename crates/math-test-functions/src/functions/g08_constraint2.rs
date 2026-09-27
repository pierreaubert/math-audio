//! G08 second constraint (CEC 2006 constrained benchmark).

use ndarray::Array1;

/// G08 constraint 2: `g2(x) = 1 - x1 + (x2 - 4)^2 <= 0`.
///
/// Returns the violation amount (feasible when `<= 0`).
pub fn g08_constraint2(x: &Array1<f64>) -> f64 {
    assert!(
        x.len() == 2,
        "g08_constraint2 requires 2 dimensions, got {}",
        x.len()
    );
    1.0 - x[0] + (x[1] - 4.0).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "requires 2 dimensions")]
    fn rejects_wrong_dimension() {
        let x = Array1::from(vec![0.0]);
        let _ = g08_constraint2(&x);
    }

    #[test]
    fn feasible_at_known_optimum() {
        let x = Array1::from(vec![1.2279713, 4.2453733]);
        assert!(g08_constraint2(&x) <= 0.0);
    }

    #[test]
    fn violated_for_small_x1_large_x2() {
        let x = Array1::from(vec![0.0, 10.0]);
        assert!(g08_constraint2(&x) > 0.0);
    }
}

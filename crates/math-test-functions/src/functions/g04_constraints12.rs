//! G04 constraints 1-2 (CEC 2006 constrained benchmark).

use ndarray::Array1;

/// Shared term of G04 constraints 1 and 2:
/// `A(x) = 85.334407 + 0.0056858*x2*x5 + 0.0006262*x1*x4 - 0.0022053*x3*x5`.
fn g04_first_term(x: &Array1<f64>) -> f64 {
    85.334407 + 0.0056858 * x[1] * x[4] + 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4]
}

/// G04 constraint 1: `g1(x) = A(x) - 92 <= 0` (see [`g04_first_term`]).
///
/// Returns the violation amount (feasible when `<= 0`).
pub fn g04_constraint1(x: &Array1<f64>) -> f64 {
    assert!(
        x.len() == 5,
        "g04_constraint1 requires 5 dimensions, got {}",
        x.len()
    );
    g04_first_term(x) - 92.0
}

/// G04 constraint 2: `g2(x) = -A(x) <= 0` (see [`g04_first_term`]).
///
/// Returns the violation amount (feasible when `<= 0`).
pub fn g04_constraint2(x: &Array1<f64>) -> f64 {
    assert!(
        x.len() == 5,
        "g04_constraint2 requires 5 dimensions, got {}",
        x.len()
    );
    -g04_first_term(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "requires 5 dimensions")]
    fn first_rejects_wrong_dimension() {
        let x = Array1::from(vec![0.0]);
        let _ = g04_constraint1(&x);
    }

    #[test]
    #[should_panic(expected = "requires 5 dimensions")]
    fn second_rejects_wrong_dimension() {
        let x = Array1::from(vec![0.0]);
        let _ = g04_constraint2(&x);
    }

    #[test]
    fn first_constraint_active_at_known_optimum() {
        let x = Array1::from(vec![78.0, 33.0, 29.99525602, 45.0, 36.77581290]);
        let value = g04_constraint1(&x);
        assert!(value.abs() < 1e-3, "got {value}");
    }

    #[test]
    fn second_constraint_feasible_at_known_optimum() {
        let x = Array1::from(vec![78.0, 33.0, 29.99525602, 45.0, 36.77581290]);
        assert!(g04_constraint2(&x) <= 0.0);
    }

    #[test]
    fn both_finite_at_lower_bounds() {
        let x = Array1::from(vec![78.0, 33.0, 27.0, 27.0, 27.0]);
        assert!(g04_constraint1(&x).is_finite());
        assert!(g04_constraint2(&x).is_finite());
    }
}

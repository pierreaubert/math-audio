//! G04 constraints 5-6 (CEC 2006 constrained benchmark).

use ndarray::Array1;

/// Shared term of G04 constraints 5 and 6:
/// `C(x) = 9.300961 + 0.0047026*x3*x5 + 0.0012547*x1*x3 + 0.0019085*x3*x4`.
fn g04_third_term(x: &Array1<f64>) -> f64 {
    9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3]
}

/// G04 constraint 5: `g5(x) = C(x) - 25 <= 0` (see [`g04_third_term`]).
///
/// Returns the violation amount (feasible when `<= 0`).
pub fn g04_constraint5(x: &Array1<f64>) -> f64 {
    assert!(
        x.len() == 5,
        "g04_constraint5 requires 5 dimensions, got {}",
        x.len()
    );
    g04_third_term(x) - 25.0
}

/// G04 constraint 6: `g6(x) = -C(x) + 20 <= 0` (see [`g04_third_term`]).
///
/// Returns the violation amount (feasible when `<= 0`).
pub fn g04_constraint6(x: &Array1<f64>) -> f64 {
    assert!(
        x.len() == 5,
        "g04_constraint6 requires 5 dimensions, got {}",
        x.len()
    );
    -g04_third_term(x) + 20.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "requires 5 dimensions")]
    fn fifth_rejects_wrong_dimension() {
        let x = Array1::from(vec![0.0]);
        let _ = g04_constraint5(&x);
    }

    #[test]
    #[should_panic(expected = "requires 5 dimensions")]
    fn sixth_rejects_wrong_dimension() {
        let x = Array1::from(vec![0.0]);
        let _ = g04_constraint6(&x);
    }

    #[test]
    fn sixth_constraint_active_at_known_optimum() {
        let x = Array1::from(vec![78.0, 33.0, 29.99525602, 45.0, 36.77581290]);
        let value = g04_constraint6(&x);
        assert!(value.abs() < 1e-3, "got {value}");
    }

    #[test]
    fn fifth_constraint_feasible_at_known_optimum() {
        let x = Array1::from(vec![78.0, 33.0, 29.99525602, 45.0, 36.77581290]);
        assert!(g04_constraint5(&x) <= 0.0);
    }

    #[test]
    fn both_finite_at_lower_bounds() {
        let x = Array1::from(vec![78.0, 33.0, 27.0, 27.0, 27.0]);
        assert!(g04_constraint5(&x).is_finite());
        assert!(g04_constraint6(&x).is_finite());
    }
}

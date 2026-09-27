//! G04 constraints 3-4 (CEC 2006 constrained benchmark).

use ndarray::Array1;

/// Shared term of G04 constraints 3 and 4:
/// `B(x) = 80.51249 + 0.0071317*x2*x5 + 0.0029955*x1*x2 + 0.0021813*x3^2`.
fn g04_second_term(x: &Array1<f64>) -> f64 {
    80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2].powi(2)
}

/// G04 constraint 3: `g3(x) = B(x) - 110 <= 0` (see [`g04_second_term`]).
///
/// Returns the violation amount (feasible when `<= 0`).
pub fn g04_constraint3(x: &Array1<f64>) -> f64 {
    assert!(
        x.len() == 5,
        "g04_constraint3 requires 5 dimensions, got {}",
        x.len()
    );
    g04_second_term(x) - 110.0
}

/// G04 constraint 4: `g4(x) = -B(x) + 90 <= 0` (see [`g04_second_term`]).
///
/// Returns the violation amount (feasible when `<= 0`).
pub fn g04_constraint4(x: &Array1<f64>) -> f64 {
    assert!(
        x.len() == 5,
        "g04_constraint4 requires 5 dimensions, got {}",
        x.len()
    );
    -g04_second_term(x) + 90.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "requires 5 dimensions")]
    fn third_rejects_wrong_dimension() {
        let x = Array1::from(vec![0.0]);
        let _ = g04_constraint3(&x);
    }

    #[test]
    #[should_panic(expected = "requires 5 dimensions")]
    fn fourth_rejects_wrong_dimension() {
        let x = Array1::from(vec![0.0]);
        let _ = g04_constraint4(&x);
    }

    #[test]
    fn both_feasible_at_known_optimum() {
        let x = Array1::from(vec![78.0, 33.0, 29.99525602, 45.0, 36.77581290]);
        assert!(g04_constraint3(&x) <= 1e-6);
        assert!(g04_constraint4(&x) <= 1e-6);
    }

    #[test]
    fn both_finite_at_lower_bounds() {
        let x = Array1::from(vec![78.0, 33.0, 27.0, 27.0, 27.0]);
        assert!(g04_constraint3(&x).is_finite());
        assert!(g04_constraint4(&x).is_finite());
    }
}

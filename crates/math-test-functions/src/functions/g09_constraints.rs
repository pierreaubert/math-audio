//! G09 constraints (CEC 2006 constrained benchmark).

use ndarray::Array1;

/// Known constrained minimiser of G09 (see [`super::g09_objective`]).
pub fn g09_optimum() -> Array1<f64> {
    Array1::from(vec![
        2.33049935,
        1.95137236,
        -0.47754139,
        4.36572624,
        -0.62448747,
        1.03813099,
        1.59422667,
    ])
}

/// G09 constraint 1: `g1(x) = -127 + 2*x1^2 + 3*x2^4 + x3 + 4*x4^2 + 5*x5 <= 0`.
///
/// Returns the violation amount (feasible when `<= 0`).
pub fn g09_constraint1(x: &Array1<f64>) -> f64 {
    assert!(
        x.len() == 7,
        "g09_constraint1 requires 7 dimensions, got {}",
        x.len()
    );
    -127.0 + 2.0 * x[0].powi(2) + 3.0 * x[1].powi(4) + x[2] + 4.0 * x[3].powi(2) + 5.0 * x[4]
}

/// G09 constraint 2: `g2(x) = -282 + 7*x1 + 3*x2 + 10*x3^2 + x4 - x5 <= 0`.
///
/// Returns the violation amount (feasible when `<= 0`).
pub fn g09_constraint2(x: &Array1<f64>) -> f64 {
    assert!(
        x.len() == 7,
        "g09_constraint2 requires 7 dimensions, got {}",
        x.len()
    );
    -282.0 + 7.0 * x[0] + 3.0 * x[1] + 10.0 * x[2].powi(2) + x[3] - x[4]
}

/// G09 constraint 3: `g3(x) = -196 + 23*x1 + x2^2 + 6*x6^2 - 8*x7 <= 0`.
///
/// Returns the violation amount (feasible when `<= 0`).
pub fn g09_constraint3(x: &Array1<f64>) -> f64 {
    assert!(
        x.len() == 7,
        "g09_constraint3 requires 7 dimensions, got {}",
        x.len()
    );
    -196.0 + 23.0 * x[0] + x[1].powi(2) + 6.0 * x[5].powi(2) - 8.0 * x[6]
}

/// G09 constraint 4: `g4(x) = 4*x1^2 + x2^2 - 3*x1*x2 + 2*x3^2 + 5*x6 - 11*x7 <= 0`.
///
/// Returns the violation amount (feasible when `<= 0`).
pub fn g09_constraint4(x: &Array1<f64>) -> f64 {
    assert!(
        x.len() == 7,
        "g09_constraint4 requires 7 dimensions, got {}",
        x.len()
    );
    4.0 * x[0].powi(2) + x[1].powi(2) - 3.0 * x[0] * x[1] + 2.0 * x[2].powi(2) + 5.0 * x[5]
        - 11.0 * x[6]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "requires 7 dimensions")]
    fn first_rejects_wrong_dimension() {
        let x = Array1::from(vec![0.0]);
        let _ = g09_constraint1(&x);
    }

    #[test]
    #[should_panic(expected = "requires 7 dimensions")]
    fn second_rejects_wrong_dimension() {
        let x = Array1::from(vec![0.0]);
        let _ = g09_constraint2(&x);
    }

    #[test]
    #[should_panic(expected = "requires 7 dimensions")]
    fn third_rejects_wrong_dimension() {
        let x = Array1::from(vec![0.0]);
        let _ = g09_constraint3(&x);
    }

    #[test]
    #[should_panic(expected = "requires 7 dimensions")]
    fn fourth_rejects_wrong_dimension() {
        let x = Array1::from(vec![0.0]);
        let _ = g09_constraint4(&x);
    }

    #[test]
    fn all_feasible_at_known_optimum() {
        let x = g09_optimum();
        assert!(g09_constraint1(&x) <= 1e-6);
        assert!(g09_constraint2(&x) <= 1e-6);
        assert!(g09_constraint3(&x) <= 1e-6);
        assert!(g09_constraint4(&x) <= 1e-6);
    }

    #[test]
    fn first_constraint_active_at_known_optimum() {
        let x = g09_optimum();
        assert!(g09_constraint1(&x).abs() < 1e-3);
    }

    #[test]
    fn first_constraint_violated_at_lower_corner() {
        let x = Array1::from(vec![-10.0; 7]);
        assert!(g09_constraint1(&x) > 0.0);
    }
}

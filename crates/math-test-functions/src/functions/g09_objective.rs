//! G09 objective (CEC 2006 constrained benchmark).

use ndarray::Array1;

/// G09 objective: `f(x) = (x1-10)^2 + 5*(x2-12)^2 + x3^4 + 3*(x4-11)^2
/// + 10*x5^6 + 7*x6^2 + x7^4 - 4*x6*x7 - 10*x6 - 8*x7`.
///
/// Bounds: `x in [-10, 10]^7`. Known constrained minimum `f* = 680.63005737`
/// at `(2.33049935, 1.95137236, -0.47754139, 4.36572624, -0.62448747,
/// 1.03813099, 1.59422667)`.
pub fn g09_objective(x: &Array1<f64>) -> f64 {
    assert!(
        x.len() == 7,
        "g09_objective requires 7 dimensions, got {}",
        x.len()
    );
    (x[0] - 10.0).powi(2)
        + 5.0 * (x[1] - 12.0).powi(2)
        + x[2].powi(4)
        + 3.0 * (x[3] - 11.0).powi(2)
        + 10.0 * x[4].powi(6)
        + 7.0 * x[5].powi(2)
        + x[6].powi(4)
        - 4.0 * x[5] * x[6]
        - 10.0 * x[5]
        - 8.0 * x[6]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "requires 7 dimensions")]
    fn rejects_wrong_dimension() {
        let x = Array1::from(vec![0.0]);
        let _ = g09_objective(&x);
    }

    #[test]
    fn known_optimum_matches_literature_value() {
        let x = Array1::from(vec![
            2.33049935,
            1.95137236,
            -0.47754139,
            4.36572624,
            -0.62448747,
            1.03813099,
            1.59422667,
        ]);
        let value = g09_objective(&x);
        assert!((value - 680.63005737).abs() < 1e-3, "got {value}");
    }

    #[test]
    fn finite_at_origin() {
        let x = Array1::from(vec![0.0; 7]);
        assert!(g09_objective(&x).is_finite());
    }
}

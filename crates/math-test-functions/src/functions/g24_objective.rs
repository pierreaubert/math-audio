//! G24 objective (CEC 2006 constrained benchmark).

use ndarray::Array1;

/// G24 objective: `f(x) = -x1 - x2`.
///
/// Bounds: `x1 in [0, 3]`, `x2 in [0, 4]`. Known constrained minimum
/// `f* = -5.50801327` at `(2.32952019, 3.17849307)`.
pub fn g24_objective(x: &Array1<f64>) -> f64 {
    assert!(
        x.len() == 2,
        "g24_objective requires 2 dimensions, got {}",
        x.len()
    );
    -x[0] - x[1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "requires 2 dimensions")]
    fn rejects_wrong_dimension() {
        let x = Array1::from(vec![0.0]);
        let _ = g24_objective(&x);
    }

    #[test]
    fn known_optimum_matches_literature_value() {
        let x = Array1::from(vec![2.32952019, 3.17849307]);
        let value = g24_objective(&x);
        assert!((value - -5.50801327).abs() < 1e-6, "got {value}");
    }
}

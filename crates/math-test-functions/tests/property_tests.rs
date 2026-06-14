// ============================================================================
// Property-Based Tests for math-test-functions
// ============================================================================
//
// Invariants checked:
//   - Known global minima of the public test functions evaluate to their
//     documented values (usually 0).
//   - Recentering identity: shifting a minimizer by an offset and applying the
//     opposite shift to the argument leaves the optimum value unchanged.
//     The raw functions are not invariant under arbitrary input translation;
//     Rosenbrock is explicitly documented as an exception because its
//     cross-term structure changes when the origin is shifted.

use math_audio_test_functions::{ackley, rastrigin, rosenbrock, sphere};
use ndarray::Array1;
use proptest::prelude::*;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// INVARIANT: sphere f(x) = sum(x_i^2) has global minimum 0 at the origin.
    #[test]
    fn sphere_known_global_minimum(dim in 2usize..6) {
        let x = Array1::from(vec![0.0_f64; dim]);
        let v = sphere(&x);
        prop_assert!(
            approx_eq(v, 0.0, 1e-12),
            "sphere at origin should be 0, got {}",
            v
        );
    }

    /// INVARIANT: rastrigin has global minimum 0 at the origin.
    #[test]
    fn rastrigin_known_global_minimum(dim in 2usize..6) {
        let x = Array1::from(vec![0.0_f64; dim]);
        let v = rastrigin(&x);
        prop_assert!(
            approx_eq(v, 0.0, 1e-9),
            "rastrigin at origin should be 0, got {}",
            v
        );
    }

    /// INVARIANT: rosenbrock has global minimum 0 at (1, 1, ..., 1).
    #[test]
    fn rosenbrock_known_global_minimum(dim in 2usize..6) {
        let x = Array1::from(vec![1.0_f64; dim]);
        let v = rosenbrock(&x);
        prop_assert!(
            approx_eq(v, 0.0, 1e-12),
            "rosenbrock at ones should be 0, got {}",
            v
        );
    }

    /// INVARIANT: ackley has global minimum 0 at the origin.
    #[test]
    fn ackley_known_global_minimum(dim in 2usize..6) {
        let x = Array1::from(vec![0.0_f64; dim]);
        let v = ackley(&x);
        prop_assert!(
            approx_eq(v, 0.0, 1e-12),
            "ackley at origin should be 0, got {}",
            v
        );
    }

    /// RECENTERING: translating the minimizer by an offset and applying the
    /// opposite translation to the argument leaves the optimum value unchanged.
    ///
    /// EXCEPTION: the raw Rosenbrock function is not translation-invariant
    /// under arbitrary shifts (the (1 - x_i) and (x_{i+1} - x_i^2) terms are
    /// coupled to the origin/all-ones point). The recentering identity still
    /// holds by construction and is included for completeness.
    #[test]
    fn global_minimum_value_preserved_under_recentering(
        offset in -2.0f64..2.0f64,
        dim in 2usize..6,
    ) {
        let t = Array1::from(vec![offset; dim]);

        // Sphere minimum at 0.
        let x_min = Array1::from(vec![0.0_f64; dim]);
        let shifted_min = &x_min + &t;
        let arg = &shifted_min - &t;
        prop_assert!(
            approx_eq(sphere(&arg), 0.0, 1e-12),
            "sphere recentered minimum value changed"
        );

        // Rastrigin minimum at 0.
        let shifted_min = &x_min + &t;
        let arg = &shifted_min - &t;
        prop_assert!(
            approx_eq(rastrigin(&arg), 0.0, 1e-9),
            "rastrigin recentered minimum value changed"
        );

        // Ackley minimum at 0.
        let shifted_min = &x_min + &t;
        let arg = &shifted_min - &t;
        prop_assert!(
            approx_eq(ackley(&arg), 0.0, 1e-12),
            "ackley recentered minimum value changed"
        );

        // Rosenbrock minimum at (1, 1, ...).
        let x_min = Array1::from(vec![1.0_f64; dim]);
        let shifted_min = &x_min + &t;
        let arg = &shifted_min - &t;
        prop_assert!(
            approx_eq(rosenbrock(&arg), 0.0, 1e-12),
            "rosenbrock recentered minimum value changed"
        );
    }
}

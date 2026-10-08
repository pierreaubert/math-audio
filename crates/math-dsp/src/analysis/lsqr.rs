//! Bounded matrix-free least squares for offline identification.
//!
//! LSQR uses Golub-Kahan bidiagonalization and orthogonal rotations, following
//! Paige and Saunders (1982), https://web.stanford.edu/group/SOL/software/lsqr/.
//! This implementation solves undamped least squares from a zero initial vector.
//! Actual residual and normal-residual vectors are evaluated at every iterate,
//! rather than accepting convergence from recurrence estimates alone. The operator
//! may include a caller-validated right preconditioner, whose inverse mapping and
//! conditioning certificate remain the caller's responsibility. Neither convergence
//! nor the diagnostic condition estimate certifies rank or physical identifiability.
//! Run on a worker thread; this is not an audio callback solver.

// Rust guideline compliant 2026-02-21

/// Linear map and its transpose with fixed dimensions.
///
/// Implementations must replace every output component, retain the same map
/// throughout a solve, and supply its mathematical transpose. The solver checks
/// dimensions and finite outputs, but cannot prove these semantic requirements.
pub trait LeastSquaresOperator {
    /// Return the number of output samples.
    fn rows(&self) -> usize;
    /// Return the number of unknown coefficients.
    fn columns(&self) -> usize;
    /// Replace `output` with the map applied to `input`.
    ///
    /// # Errors
    /// Return an error if the map cannot be evaluated.
    fn apply(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), String>;
    /// Replace `output` with the transpose applied to `input`.
    ///
    /// # Errors
    /// Return an error if the transpose cannot be evaluated.
    fn apply_adjoint(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), String>;
}

impl LeastSquaresOperator for super::polynomial_convolution::PolynomialConvolutionOperator {
    fn rows(&self) -> usize {
        self.output_len()
    }
    fn columns(&self) -> usize {
        self.coefficient_len()
    }
    fn apply(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), String> {
        self.apply(input, output)
    }
    fn apply_adjoint(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), String> {
        self.apply_adjoint(input, output)
    }
}

/// Explicit termination and allocation limits for LSQR.
#[derive(Debug, Clone, Copy)]
pub struct LsqrOptions {
    /// Backward-error tolerance for the map and normal residual, between zero and one.
    pub atol: f64,
    /// Relative right-hand-side tolerance, between zero and one.
    pub btol: f64,
    /// Diagnostic condition limit; zero disables this stop, without certifying conditioning.
    pub condition_limit: f64,
    /// Maximum number of completed bidiagonal iterations.
    pub max_iterations: usize,
    /// Cap on solver vectors; excludes caller buffers, operator storage and allocator overhead.
    pub max_buffer_bytes: usize,
}

/// Why a solve stopped; resource and cancellation stops are distinct from convergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LsqrStop {
    /// The actual residual satisfies the requested backward-error threshold.
    ResidualTolerance,
    /// The actual normal residual satisfies the requested stationarity threshold.
    NormalResidualTolerance,
    /// The recurrence's diagnostic condition estimate reached the requested limit.
    ConditionLimit,
    /// The requested iteration budget was exhausted.
    IterationLimit,
    /// The caller requested cancellation at an operator boundary.
    Cancelled,
    /// Bidiagonalization could not advance, without satisfying either tolerance.
    Breakdown,
}

/// Finite iterate, actual residual diagnostics, and explicit termination reason.
#[derive(Debug)]
pub struct LsqrResult {
    /// Solution in the supplied operator's coordinates, before inverse preconditioner mapping.
    pub solution: Vec<f64>,
    /// Termination reason; limits and breakdown do not mean a converged fit.
    pub stop: LsqrStop,
    /// Number of completed solution updates.
    pub iterations: usize,
    /// Euclidean norm of the actual `b - A*x` residual.
    pub residual_norm: f64,
    /// Norm of the actual `A^T*(b - A*x)`; absent if cancelled before its evaluation.
    pub normal_residual_norm: Option<f64>,
    /// Recurrence estimate of the map's Frobenius norm, not a certified bound.
    pub operator_norm_estimate: f64,
    /// Recurrence condition estimate, not a certified bound.
    pub condition_estimate: f64,
    /// Vector storage allocated by this solve, including the returned solution.
    pub buffer_bytes: usize,
}

fn norm(values: &[f64]) -> Result<f64, String> {
    let mut result = 0.0_f64;
    for value in values {
        if !value.is_finite() {
            return Err("LSQR vector is non-finite".into());
        }
        result = result.hypot(*value);
    }
    if !result.is_finite() {
        return Err("LSQR vector norm overflow".into());
    }
    Ok(result)
}

fn normalize(values: &mut [f64], scale: f64) {
    if scale != 0.0 {
        for value in values {
            *value /= scale;
        }
    }
}

/// Solve undamped least squares with bounded vectors and actual residual checks.
///
/// `cancelled` is polled before initialization and each bidiagonal operator call,
/// after the initial transpose, and after each complete residual pair.
/// Each solution update and its two residual evaluations form one complete step.
/// Operator calls must provide their own duration bounds; this function cannot
/// interrupt them. The
/// caller must inspect `stop` before accepting a fit. For a preconditioned map,
/// tolerances refer to that map; validate the mapped-back solution separately.
///
/// # Errors
/// Rejects invalid dimensions/options, non-finite inputs, buffer arithmetic
/// overflow, storage exceeding the cap, operator errors, or numerical overflow.
pub fn solve_lsqr(
    operator: &mut impl LeastSquaresOperator,
    rhs: &[f64],
    options: LsqrOptions,
    mut cancelled: impl FnMut() -> bool,
) -> Result<LsqrResult, String> {
    let rows = operator.rows();
    let columns = operator.columns();
    if rows == 0 || columns == 0 || rhs.len() != rows {
        return Err("LSQR dimensions mismatch or empty operator".into());
    }
    if !options.atol.is_finite()
        || !(0.0..=1.0).contains(&options.atol)
        || !options.btol.is_finite()
        || !(0.0..=1.0).contains(&options.btol)
        || !options.condition_limit.is_finite()
        || options.condition_limit < 0.0
        || (options.condition_limit > 0.0 && options.condition_limit < 1.0)
    {
        return Err("LSQR tolerances or condition limit are invalid".into());
    }
    let buffer_bytes = columns
        .checked_mul(4)
        .and_then(|n| rows.checked_mul(2).and_then(|m| n.checked_add(m)))
        .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
        .ok_or("LSQR vector byte count overflow")?;
    if buffer_bytes > options.max_buffer_bytes || buffer_bytes > isize::MAX as usize {
        return Err("LSQR vector byte limit exceeded".into());
    }
    let rhs_norm = norm(rhs)?;
    let mut u = rhs.to_vec();
    let mut row_work = vec![0.0; rows];
    let mut v = vec![0.0; columns];
    let mut column_work = vec![0.0; columns];
    let mut w = vec![0.0; columns];
    let mut result = LsqrResult {
        solution: vec![0.0; columns],
        stop: LsqrStop::IterationLimit,
        iterations: 0,
        residual_norm: rhs_norm,
        normal_residual_norm: None,
        operator_norm_estimate: 0.0,
        condition_estimate: 0.0,
        buffer_bytes,
    };
    if cancelled() {
        result.stop = LsqrStop::Cancelled;
        return Ok(result);
    }
    if rhs_norm == 0.0 {
        result.stop = LsqrStop::ResidualTolerance;
        result.normal_residual_norm = Some(0.0);
        return Ok(result);
    }
    normalize(&mut u, rhs_norm);
    operator.apply_adjoint(&u, &mut v)?;
    let mut alpha = norm(&v)?;
    let initial_normal = alpha * rhs_norm;
    if !initial_normal.is_finite() {
        return Err("LSQR initial normal norm overflow".into());
    }
    result.normal_residual_norm = Some(initial_normal);
    result.operator_norm_estimate = alpha;
    if cancelled() {
        result.stop = LsqrStop::Cancelled;
        return Ok(result);
    }
    if alpha == 0.0 {
        result.stop = LsqrStop::NormalResidualTolerance;
        return Ok(result);
    }
    normalize(&mut v, alpha);
    w.copy_from_slice(&v);
    let mut rho_bar = alpha;
    let mut phi_bar = rhs_norm;
    let mut inverse_norm_estimate = 0.0_f64;
    for iteration in 1..=options.max_iterations {
        if cancelled() {
            result.stop = LsqrStop::Cancelled;
            return Ok(result);
        }
        operator.apply(&v, &mut row_work)?;
        for (next, current) in row_work.iter_mut().zip(&u) {
            *next -= alpha * current;
        }
        let beta = norm(&row_work)?;
        std::mem::swap(&mut u, &mut row_work);
        normalize(&mut u, beta);
        if cancelled() {
            result.stop = LsqrStop::Cancelled;
            return Ok(result);
        }
        operator.apply_adjoint(&u, &mut column_work)?;
        for (next, current) in column_work.iter_mut().zip(&v) {
            *next -= beta * current;
        }
        let next_alpha = norm(&column_work)?;
        std::mem::swap(&mut v, &mut column_work);
        normalize(&mut v, next_alpha);
        result.operator_norm_estimate = result.operator_norm_estimate.hypot(beta).hypot(next_alpha);
        let rho = rho_bar.hypot(beta);
        if rho == 0.0 {
            result.stop = LsqrStop::Breakdown;
            return Ok(result);
        }
        let cosine = rho_bar / rho;
        let sine = beta / rho;
        let theta = sine * next_alpha;
        rho_bar = -cosine * next_alpha;
        let phi = cosine * phi_bar;
        phi_bar *= sine;
        inverse_norm_estimate = inverse_norm_estimate.hypot(norm(&w)? / rho);
        for (solution, direction) in result.solution.iter_mut().zip(&w) {
            *solution += (phi / rho) * direction;
        }
        for (direction, next) in w.iter_mut().zip(&v) {
            *direction = next - (theta / rho) * *direction;
        }
        let solution_norm = norm(&result.solution)?;
        norm(&w)?;
        result.condition_estimate = result.operator_norm_estimate * inverse_norm_estimate;
        if !result.operator_norm_estimate.is_finite() || !result.condition_estimate.is_finite() {
            return Err("LSQR recurrence diagnostic overflow".into());
        }
        result.iterations = iteration;
        // Complete both actual residuals before the next cancellation boundary.
        // Operator failures never expose this partially checked iterate.
        operator.apply(&result.solution, &mut row_work)?;
        for (residual, b) in row_work.iter_mut().zip(rhs) {
            *residual = b - *residual;
        }
        result.residual_norm = norm(&row_work)?;
        operator.apply_adjoint(&row_work, &mut column_work)?;
        let normal = norm(&column_work)?;
        result.normal_residual_norm = Some(normal);
        if cancelled() {
            result.stop = LsqrStop::Cancelled;
            return Ok(result);
        }
        let residual_threshold =
            options.atol * result.operator_norm_estimate * solution_norm + options.btol * rhs_norm;
        let normal_threshold = options.atol * result.operator_norm_estimate * result.residual_norm;
        if !residual_threshold.is_finite() || !normal_threshold.is_finite() {
            return Err("LSQR stopping threshold overflow".into());
        }
        if result.residual_norm <= residual_threshold {
            result.stop = LsqrStop::ResidualTolerance;
            return Ok(result);
        }
        if normal <= normal_threshold {
            result.stop = LsqrStop::NormalResidualTolerance;
            return Ok(result);
        }
        if options.condition_limit > 0.0 && result.condition_estimate >= options.condition_limit {
            result.stop = LsqrStop::ConditionLimit;
            return Ok(result);
        }
        if beta == 0.0 || next_alpha == 0.0 {
            result.stop = LsqrStop::Breakdown;
            return Ok(result);
        }
        alpha = next_alpha;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Dense {
        rows: usize,
        columns: usize,
        values: Vec<f64>,
        calls: usize,
    }
    impl LeastSquaresOperator for Dense {
        fn rows(&self) -> usize {
            self.rows
        }
        fn columns(&self) -> usize {
            self.columns
        }
        fn apply(&mut self, x: &[f64], y: &mut [f64]) -> Result<(), String> {
            self.calls += 1;
            for (row, output) in self.values.chunks_exact(self.columns).zip(y) {
                *output = row.iter().zip(x).map(|(a, b)| a * b).sum();
            }
            Ok(())
        }
        fn apply_adjoint(&mut self, y: &[f64], x: &mut [f64]) -> Result<(), String> {
            self.calls += 1;
            x.fill(0.0);
            for (row, input) in self.values.chunks_exact(self.columns).zip(y) {
                for (a, output) in row.iter().zip(x.iter_mut()) {
                    *output += a * input;
                }
            }
            Ok(())
        }
    }
    fn options() -> LsqrOptions {
        LsqrOptions {
            atol: 1e-13,
            btol: 1e-13,
            condition_limit: 1e8,
            max_iterations: 100,
            max_buffer_bytes: 1 << 20,
        }
    }
    fn dense(rows: usize, columns: usize, values: &[f64]) -> Dense {
        Dense {
            rows,
            columns,
            values: values.to_vec(),
            calls: 0,
        }
    }
    fn assert_solution(result: &LsqrResult, expected: &[f64]) {
        assert!(
            matches!(
                result.stop,
                LsqrStop::ResidualTolerance | LsqrStop::NormalResidualTolerance
            ),
            "{result:?}"
        );
        for (a, b) in result.solution.iter().zip(expected) {
            assert!((a - b).abs() < 1e-11, "{result:?}, expected {expected:?}");
        }
    }
    #[test]
    fn independently_known_full_rank_inconsistent_rank_deficient_and_underdetermined_solutions() {
        let mut a = dense(3, 2, &[1., 0., 1., 1., 0., 1.]);
        let exact = solve_lsqr(&mut a, &[1., 0., -1.], options(), || false).unwrap();
        assert_solution(&exact, &[1., -1.]);
        let inconsistent = solve_lsqr(&mut a, &[1., 0.01, -1.], options(), || false).unwrap();
        assert_solution(&inconsistent, &[1. + 0.01 / 3., -1. + 0.01 / 3.]);
        assert!((inconsistent.residual_norm - 0.01 / 3.0_f64.sqrt()).abs() < 1e-13);
        assert!(inconsistent.normal_residual_norm.unwrap() < 1e-13);
        let rank_deficient = solve_lsqr(
            &mut dense(2, 2, &[1., 2., 2., 4.]),
            &[3., 6.],
            options(),
            || false,
        )
        .unwrap();
        assert_solution(&rank_deficient, &[0.6, 1.2]);
        let underdetermined = solve_lsqr(
            &mut dense(2, 3, &[1., 0., 1., 0., 1., 1.]),
            &[1., 2.],
            options(),
            || false,
        )
        .unwrap();
        assert_solution(&underdetermined, &[0., 1., 1.]);
        let null = solve_lsqr(&mut dense(2, 2, &[0.; 4]), &[3., 4.], options(), || false).unwrap();
        assert_solution(&null, &[0., 0.]);
        assert_eq!(null.residual_norm, 5.);
        assert_eq!(null.normal_residual_norm, Some(0.));
    }
    #[test]
    fn limits_cancellation_and_zero_rhs_have_truthful_results() {
        let mut a = dense(3, 3, &[1., 0., 0., 0., 2., 0., 0., 0., 4.]);
        let b = [1., 1., 1.];
        let mut limits = options();
        limits.max_iterations = 0;
        let zero_budget = solve_lsqr(&mut a, &b, limits, || false).unwrap();
        assert_eq!(zero_budget.stop, LsqrStop::IterationLimit);
        assert_eq!(zero_budget.iterations, 0);
        limits.max_iterations = 1;
        let one = solve_lsqr(&mut a, &b, limits, || false).unwrap();
        assert_eq!(one.stop, LsqrStop::IterationLimit);
        assert_eq!(one.iterations, 1);
        limits = options();
        limits.condition_limit = 1.;
        let conditioned = solve_lsqr(&mut a, &b, limits, || false).unwrap();
        assert_eq!(conditioned.stop, LsqrStop::ConditionLimit);
        for cancel_at in 1..=5 {
            let mut polls = 0;
            let stopped = solve_lsqr(&mut a, &b, options(), || {
                polls += 1;
                polls == cancel_at
            })
            .unwrap();
            assert_eq!(stopped.stop, LsqrStop::Cancelled);
            let actual: Vec<_> = stopped
                .solution
                .iter()
                .zip([1., 2., 4.])
                .zip(b)
                .map(|((x, diag), b)| b - diag * x)
                .collect();
            assert!((stopped.residual_norm - norm(&actual).unwrap()).abs() < 1e-13);
            if let Some(normal) = stopped.normal_residual_norm {
                let actual_normal: Vec<_> = actual
                    .iter()
                    .zip([1., 2., 4.])
                    .map(|(x, diag)| x * diag)
                    .collect();
                assert!((normal - norm(&actual_normal).unwrap()).abs() < 1e-12);
            }
        }
        let initial = solve_lsqr(&mut a, &b, options(), || true).unwrap();
        assert_eq!(initial.normal_residual_norm, None);
        let calls = a.calls;
        let zero = solve_lsqr(&mut a, &[0.; 3], options(), || false).unwrap();
        assert_eq!(zero.stop, LsqrStop::ResidualTolerance);
        assert_eq!(a.calls, calls);
    }
    #[test]
    fn cancellation_latched_by_operator_precedes_convergence() {
        use std::{cell::Cell, rc::Rc};
        struct Latching {
            inner: Dense,
            cancel: Rc<Cell<bool>>,
            latch_at: usize,
        }
        impl LeastSquaresOperator for Latching {
            fn rows(&self) -> usize {
                self.inner.rows()
            }
            fn columns(&self) -> usize {
                self.inner.columns()
            }
            fn apply(&mut self, x: &[f64], y: &mut [f64]) -> Result<(), String> {
                self.inner.apply(x, y)?;
                if self.inner.calls == self.latch_at {
                    self.cancel.set(true);
                }
                Ok(())
            }
            fn apply_adjoint(&mut self, y: &[f64], x: &mut [f64]) -> Result<(), String> {
                self.inner.apply_adjoint(y, x)?;
                if self.inner.calls == self.latch_at {
                    self.cancel.set(true);
                }
                Ok(())
            }
        }
        // Call 1 initializes the transpose; calls 4/5 evaluate actual residuals.
        for latch_at in [1, 4, 5] {
            let cancel = Rc::new(Cell::new(false));
            let mut operator = Latching {
                inner: dense(2, 2, &[1., 0., 0., 1.]),
                cancel: cancel.clone(),
                latch_at,
            };
            let result = solve_lsqr(&mut operator, &[1., 2.], options(), || cancel.get()).unwrap();
            assert_eq!(
                result.stop,
                LsqrStop::Cancelled,
                "latch {latch_at}: {result:?}"
            );
            let residual = [1. - result.solution[0], 2. - result.solution[1]];
            let actual_norm = norm(&residual).unwrap();
            assert!((result.residual_norm - actual_norm).abs() < 1e-13);
            assert!((result.normal_residual_norm.unwrap() - actual_norm).abs() < 1e-13);
        }
    }

    #[test]
    fn storage_and_invalid_inputs_fail_before_operator_evaluation() {
        let mut a = dense(3, 2, &[1., 0., 1., 1., 0., 1.]);
        let bytes = (4 * 2 + 2 * 3) * 8;
        let mut limits = options();
        limits.max_buffer_bytes = bytes - 1;
        assert!(solve_lsqr(&mut a, &[1., 0., -1.], limits, || false).is_err());
        assert_eq!(a.calls, 0);
        limits.max_buffer_bytes = bytes;
        assert_eq!(
            solve_lsqr(&mut a, &[1., 0., -1.], limits, || false)
                .unwrap()
                .buffer_bytes,
            bytes
        );
        for invalid in [f64::NAN, f64::INFINITY, -1., 2.] {
            limits.atol = invalid;
            assert!(solve_lsqr(&mut a, &[1., 0., -1.], limits, || false).is_err());
        }
        assert!(solve_lsqr(&mut a, &[f64::NAN; 3], options(), || false).is_err());
        assert!(solve_lsqr(&mut a, &[1.; 2], options(), || false).is_err());
        a.columns = usize::MAX;
        assert!(solve_lsqr(&mut a, &[1.; 3], options(), || false).is_err());
    }
    #[test]
    fn explicit_right_preconditioner_preserves_inverse_mapping_and_transpose() {
        struct RightMapped;
        impl LeastSquaresOperator for RightMapped {
            fn rows(&self) -> usize {
                3
            }
            fn columns(&self) -> usize {
                2
            }
            fn apply(&mut self, g: &[f64], output: &mut [f64]) -> Result<(), String> {
                // A = [R; 0], R = [[2,1],[0,3]], x = R^-1*g.
                let x = [(g[0] - g[1] / 3.) / 2., g[1] / 3.];
                output.copy_from_slice(&[2. * x[0] + x[1], 3. * x[1], 0.]);
                Ok(())
            }
            fn apply_adjoint(&mut self, y: &[f64], output: &mut [f64]) -> Result<(), String> {
                // R^-T*A^T*y, not R^-1*A^T*y.
                let gradient = [2. * y[0], y[0] + 3. * y[1]];
                output[0] = gradient[0] / 2.;
                output[1] = (gradient[1] - output[0]) / 3.;
                Ok(())
            }
        }
        let rhs = [1., 2., 0.1];
        let result = solve_lsqr(&mut RightMapped, &rhs, options(), || false).unwrap();
        assert_solution(&result, &[1., 2.]);
        let physical = [
            (result.solution[0] - result.solution[1] / 3.) / 2.,
            result.solution[1] / 3.,
        ];
        let original = solve_lsqr(
            &mut dense(3, 2, &[2., 1., 0., 3., 0., 0.]),
            &rhs,
            options(),
            || false,
        )
        .unwrap();
        assert_solution(&original, &physical);
        assert!((physical[0] - 1. / 6.).abs() < 1e-13);
        assert!((physical[1] - 2. / 3.).abs() < 1e-13);
        assert!((result.residual_norm - 0.1).abs() < 1e-13);
    }

    #[test]
    fn operator_errors_and_nonfinite_outputs_never_return_an_accepted_iterate() {
        struct Invalid {
            error: bool,
            forward: bool,
        }
        impl LeastSquaresOperator for Invalid {
            fn rows(&self) -> usize {
                2
            }
            fn columns(&self) -> usize {
                2
            }
            fn apply(&mut self, _: &[f64], output: &mut [f64]) -> Result<(), String> {
                if self.error {
                    return Err("injected operator error".into());
                }
                output.fill(f64::NAN);
                Ok(())
            }
            fn apply_adjoint(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), String> {
                if self.forward {
                    output.copy_from_slice(input);
                    return Ok(());
                }
                if self.error {
                    return Err("injected adjoint error".into());
                }
                output.fill(f64::INFINITY);
                Ok(())
            }
        }
        for error in [true, false] {
            for forward in [true, false] {
                assert!(
                    solve_lsqr(
                        &mut Invalid { error, forward },
                        &[1., 2.],
                        options(),
                        || false
                    )
                    .is_err()
                );
            }
        }
        let mut a = dense(2, 2, &[1., 0., 0., 1.]);
        assert!(solve_lsqr(&mut a, &[f64::MAX; 2], options(), || false).is_err());
    }

    #[test]
    fn polynomial_operator_fit_predicts_independent_time_domain_data_with_guard_noise() {
        use super::super::polynomial_convolution::PolynomialConvolutionOperator;
        let input: Vec<f64> = (0..63)
            .map(|i| ((i * 17 % 31) as f64 - 15.) / 17.)
            .collect();
        let expected_taps = [0.4, -0.2, 0.05, -0.13, 0.07, 0.03];
        let mut rhs = vec![0.; 80];
        // Independent direct convolution, including deliberately unmodellable guard noise.
        for (order, taps) in expected_taps.as_chunks::<3>().0.iter().enumerate() {
            for (delay, tap) in taps.iter().enumerate() {
                for (i, x) in input.iter().enumerate() {
                    rhs[i + delay] += tap * x.powi(order as i32 + 1);
                }
            }
        }
        rhs[79] = 0.02;
        let mut operator =
            PolynomialConvolutionOperator::new(&input, 2, 3, 80, 128, 1 << 20).unwrap();
        let result = solve_lsqr(&mut operator, &rhs, options(), || false).unwrap();
        assert_solution(&result, &expected_taps);
        assert!((result.residual_norm - 0.02).abs() < 1e-13);
        assert!(result.normal_residual_norm.unwrap() < 1e-11);
    }
}

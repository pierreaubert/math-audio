//! Upper-triangular right maps for offline matrix-free least squares.
//!
//! A supplied upper-triangular factor R defines B = A*R^-1. Its transpose is
//! B^T = R^-T*A^T, and a fitted B-coordinate vector g maps back to x = R^-1*g.
//! This module applies that map; it does not construct QR, establish rank,
//! bound conditioning, or validate the provenance of R. The ESS estimator still
//! requires independent input-only factor construction and conditioning evidence.
//! Dense factor storage is explicitly capped; this is not a scalable QR builder.

// Rust guideline compliant 2026-02-21
use super::lsqr::LeastSquaresOperator;

/// Reusable right-preconditioned operator with a caller-supplied triangular factor.
///
/// Factor entries are row-major, with exact zeros below the diagonal. Construction
/// validates finite entries and nonzero diagonals, rather than certifying numerical
/// conditioning. Applications allocate no additional buffers in this wrapper.
#[derive(Debug)]
pub struct UpperTriangularRightOperator<O> {
    operator: O,
    factor: Vec<f64>,
    work: Vec<f64>,
    coefficients: Vec<f64>,
    dimension: usize,
    buffer_bytes: usize,
}

fn solve_factor(
    factor: &[f64],
    work: &mut [f64],
    input: &[f64],
    transpose: bool,
) -> Result<(), String> {
    let dimension = work.len();
    if input.len() != dimension || input.iter().any(|x| !x.is_finite()) {
        return Err("triangular solve input dimension or finiteness mismatch".into());
    }
    work.copy_from_slice(input);
    for step in 0..dimension {
        let row = if transpose {
            step
        } else {
            dimension - 1 - step
        };
        let columns = if transpose {
            0..row
        } else {
            row + 1..dimension
        };
        // Neumaier accumulation retains cancellation lost by repeated subtraction.
        // A fused multiply-add also retains each finite product's rounding error.
        // This improves accuracy; it does not certify the factor's conditioning.
        let mut value = input[row];
        let mut correction = 0.0;
        for column in columns {
            let index = if transpose {
                column * dimension + row
            } else {
                row * dimension + column
            };
            let coefficient = -factor[index];
            let product = coefficient * work[column];
            let total = value + product;
            let addition_error = if value.abs() >= product.abs() {
                (value - total) + product
            } else {
                (product - total) + value
            };
            correction += addition_error + coefficient.mul_add(work[column], -product);
            value = total;
        }
        work[row] = (value + correction) / factor[row * dimension + row];
        if !work[row].is_finite() {
            return Err("triangular solve produced a nonfinite coefficient".into());
        }
    }
    Ok(())
}

impl<O: LeastSquaresOperator> UpperTriangularRightOperator<O> {
    /// Bind a triangular right map within an explicit vector-storage cap.
    ///
    /// `max_buffer_bytes` covers the stored factor and two coefficient vectors.
    /// It excludes the underlying operator, caller input/factor copies, solver
    /// vectors, allocator overhead, and temporary storage in the underlying map.
    /// The factor must be independently validated for the intended fit.
    ///
    /// # Errors
    /// Rejects empty/mismatched dimensions, storage overflow or cap violations,
    /// nonfinite entries, nonzero lower-triangular entries, and zero diagonals.
    pub fn new(operator: O, factor: Vec<f64>, max_buffer_bytes: usize) -> Result<Self, String> {
        let dimension = operator.columns();
        if dimension == 0 || operator.rows() == 0 {
            return Err("triangular right map requires nonempty dimensions".into());
        }
        let entries = dimension
            .checked_mul(dimension)
            .ok_or("triangular factor dimension overflow")?;
        if factor.len() != entries {
            return Err("triangular factor dimensions mismatch".into());
        }
        let buffer_bytes = entries
            .checked_add(
                dimension
                    .checked_mul(2)
                    .ok_or("triangular workspace dimension overflow")?,
            )
            .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
            .ok_or("triangular workspace byte count overflow")?;
        // Account for Vec spare capacity because ownership retains it.
        let spare_bytes = factor
            .capacity()
            .checked_sub(factor.len())
            .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
            .ok_or("triangular factor capacity overflow")?;
        let buffer_bytes = buffer_bytes
            .checked_add(spare_bytes)
            .ok_or("triangular vector byte count overflow")?;
        if buffer_bytes > max_buffer_bytes || buffer_bytes > isize::MAX as usize {
            return Err("triangular right-map vector byte limit exceeded".into());
        }
        for row in 0..dimension {
            for column in 0..dimension {
                let value = factor[row * dimension + column];
                if !value.is_finite()
                    || (column < row && value != 0.0)
                    || (column == row && value == 0.0)
                {
                    return Err(
                        "triangular factor is nonfinite, singular, or not upper triangular".into(),
                    );
                }
            }
        }
        let work = vec![0.; dimension];
        let coefficients = vec![0.; dimension];
        let buffer_bytes = factor
            .capacity()
            .checked_add(work.capacity())
            .and_then(|n| n.checked_add(coefficients.capacity()))
            .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
            .ok_or("triangular actual vector byte count overflow")?;
        if buffer_bytes > max_buffer_bytes || buffer_bytes > isize::MAX as usize {
            return Err("triangular actual vector byte limit exceeded".into());
        }
        Ok(Self {
            operator,
            factor,
            work,
            coefficients,
            dimension,
            buffer_bytes,
        })
    }

    /// Return retained vector bytes, excluding the underlying operator.
    #[must_use]
    pub fn buffer_bytes(&self) -> usize {
        self.buffer_bytes
    }

    /// Map fitted right-map coordinates back to the underlying operator's coefficients.
    ///
    /// The output is preserved on error. These are base-operator coordinates;
    /// any additional order scaling remains the caller's responsibility.
    ///
    /// # Errors
    /// Rejects mismatched dimensions, nonfinite input, or solve overflow.
    pub fn map_coefficients(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), String> {
        if output.len() != self.dimension {
            return Err("triangular coefficient output dimensions mismatch".into());
        }
        solve_factor(&self.factor, &mut self.work, input, false)?;
        output.copy_from_slice(&self.work);
        Ok(())
    }
}

impl<O: LeastSquaresOperator> LeastSquaresOperator for UpperTriangularRightOperator<O> {
    fn rows(&self) -> usize {
        self.operator.rows()
    }
    fn columns(&self) -> usize {
        self.dimension
    }
    fn apply(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), String> {
        if output.len() != self.rows() {
            return Err("right-map forward output dimensions mismatch".into());
        }
        solve_factor(&self.factor, &mut self.work, input, false)?;
        self.operator.apply(&self.work, output)
    }
    fn apply_adjoint(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), String> {
        if input.len() != self.rows() || output.len() != self.dimension {
            return Err("right-map adjoint dimensions mismatch".into());
        }
        if input.iter().any(|x| !x.is_finite()) {
            return Err("right-map adjoint input is nonfinite".into());
        }
        self.operator.apply_adjoint(input, &mut self.coefficients)?;
        solve_factor(&self.factor, &mut self.work, &self.coefficients, true)?;
        output.copy_from_slice(&self.work);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::lsqr::{LsqrOptions, LsqrStop, solve_lsqr};

    #[derive(Debug)]
    struct Dense {
        rows: usize,
        columns: usize,
        values: Vec<f64>,
    }
    impl LeastSquaresOperator for Dense {
        fn rows(&self) -> usize {
            self.rows
        }
        fn columns(&self) -> usize {
            self.columns
        }
        fn apply(&mut self, x: &[f64], y: &mut [f64]) -> Result<(), String> {
            for (row, value) in self.values.chunks_exact(self.columns).zip(y) {
                *value = row.iter().zip(x).map(|(a, b)| a * b).sum();
            }
            Ok(())
        }
        fn apply_adjoint(&mut self, y: &[f64], x: &mut [f64]) -> Result<(), String> {
            x.fill(0.);
            for (row, input) in self.values.chunks_exact(self.columns).zip(y) {
                for (value, a) in x.iter_mut().zip(row) {
                    *value += a * input;
                }
            }
            Ok(())
        }
    }
    fn operator() -> Dense {
        Dense {
            rows: 3,
            columns: 2,
            values: vec![2., 1., 0., 3., 0., 0.],
        }
    }
    #[test]
    fn right_map_adjoint_and_fitted_coefficients_match_independent_triangular_algebra() {
        // A = [R; 0] makes B an independently known embedded identity.
        let mut mapped =
            UpperTriangularRightOperator::new(operator(), vec![2., 1., 0., 3.], 64).unwrap();
        assert_eq!(mapped.buffer_bytes(), 64);
        assert_eq!(
            mapped.buffer_bytes(),
            (mapped.factor.capacity() + mapped.work.capacity() + mapped.coefficients.capacity())
                * std::mem::size_of::<f64>()
        );
        let g = [0.7, -1.2];
        let mut forward = [0.; 3];
        mapped.apply(&g, &mut forward).unwrap();
        assert!((forward[0] - g[0]).abs() < 1e-14);
        assert!((forward[1] - g[1]).abs() < 1e-14);
        assert_eq!(forward[2], 0.);
        let mut adjoint = [0.; 2];
        mapped
            .apply_adjoint(&[0.3, -0.9, 1e100], &mut adjoint)
            .unwrap();
        assert!((adjoint[0] - 0.3).abs() < 1e-14);
        assert!((adjoint[1] + 0.9).abs() < 1e-14);
        let result = solve_lsqr(
            &mut mapped,
            &[1., 2., 0.1],
            LsqrOptions {
                atol: 1e-13,
                btol: 1e-13,
                condition_limit: 1e8,
                max_iterations: 10,
                max_buffer_bytes: 1024,
            },
            || false,
        )
        .unwrap();
        assert!(matches!(
            result.stop,
            LsqrStop::NormalResidualTolerance | LsqrStop::ResidualTolerance
        ));
        let mut actual = [0.; 2];
        mapped
            .map_coefficients(&result.solution, &mut actual)
            .unwrap();
        assert!((actual[0] - 1. / 6.).abs() < 1e-13);
        assert!((actual[1] - 2. / 3.).abs() < 1e-13);
        assert!((result.residual_norm - 0.1).abs() < 1e-13);
    }
    #[test]
    fn signed_factors_preserve_forward_and_transpose_identities() {
        let factor = vec![-2., 0.4, -0.7, 0., 3., 0.2, 0., 0., -0.5];
        let base = Dense {
            rows: 3,
            columns: 3,
            values: vec![1., 0., 0., 0., 1., 0., 0., 0., 1.],
        };
        let mut mapped = UpperTriangularRightOperator::new(base, factor.clone(), 1024).unwrap();
        let g = [0.8, -0.4, 0.3];
        let y = [-0.2, 0.5, 0.9];
        let mut x = [0.; 3];
        mapped.apply(&g, &mut x).unwrap();
        // Direct triangular multiplication checks the solve without reusing its traversal.
        for (row, expected) in factor.chunks_exact(3).zip(g) {
            let actual: f64 = row.iter().zip(x).map(|(a, b)| a * b).sum();
            assert!((actual - expected).abs() < 1e-14);
        }
        let mut transpose = [0.; 3];
        mapped.apply_adjoint(&y, &mut transpose).unwrap();
        for column in 0..3 {
            let actual: f64 = (0..3)
                .map(|row| factor[row * 3 + column] * transpose[row])
                .sum();
            assert!((actual - y[column]).abs() < 1e-14);
        }
        let left: f64 = x.iter().zip(y).map(|(a, b)| a * b).sum();
        let right: f64 = g.iter().zip(transpose).map(|(a, b)| a * b).sum();
        assert!((left - right).abs() < 1e-14);
    }
    #[test]
    fn triangular_rows_retain_unit_terms_between_large_signed_products() {
        // These integer equations have exactly representable solutions. Ordinary
        // sequential subtraction loses the unit term between opposite 2^53 terms.
        let large = 9007199254740992.0;
        let factor = [
            1., large, 1., -large, 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        ];
        let mut work = [0.; 4];
        solve_factor(&factor, &mut work, &[0., 1., 1., 1.], false).unwrap();
        assert_eq!(work, [-1., 1., 1., 1.]);
        let transpose_factor = [
            1., 0., 0., large, 0., 1., 0., 1., 0., 0., 1., -large, 0., 0., 0., 1.,
        ];
        solve_factor(&transpose_factor, &mut work, &[1., 1., 1., 0.], true).unwrap();
        assert_eq!(work, [1., 1., 1., -1.]);
    }
    #[test]
    fn triangular_rows_retain_product_rounding_residuals() {
        // (1 + 2^-27)*(1 - 2^-27) = 1 - 2^-54 exactly. The rounded
        // product alone is one; the solve must retain the nonzero residual.
        let a = 1. + 2.0_f64.powi(-27);
        let b = 1. - 2.0_f64.powi(-27);
        let residual = 2.0_f64.powi(-54);
        let factor = [1., a, 0., 1.];
        let mut work = [0.; 2];
        solve_factor(&factor, &mut work, &[1., b], false).unwrap();
        assert_eq!(work, [residual, b]);
        solve_factor(&factor, &mut work, &[b, 1.], true).unwrap();
        assert_eq!(work, [b, residual]);
    }
    #[test]
    fn invalid_factors_limits_and_overflow_preserve_output() {
        for factor in [
            vec![2., 1., 0.],
            vec![2., 1., 0., 0.],
            vec![2., 1., 0.1, 3.],
            vec![2., f64::NAN, 0., 3.],
        ] {
            assert!(UpperTriangularRightOperator::new(operator(), factor, 1024).is_err());
        }
        assert!(UpperTriangularRightOperator::new(operator(), vec![2., 1., 0., 3.], 63).is_err());
        let mut spare = Vec::with_capacity(100);
        spare.extend([2., 1., 0., 3.]);
        assert!(UpperTriangularRightOperator::new(operator(), spare, 64).is_err());
        let mut mapped =
            UpperTriangularRightOperator::new(operator(), vec![1e-300, 1., 0., 3.], 1024).unwrap();
        let mut output = [7.; 2];
        assert!(
            mapped
                .map_coefficients(&[f64::MAX, 1.], &mut output)
                .is_err()
        );
        assert_eq!(output, [7.; 2]);
        assert!(
            mapped
                .map_coefficients(&[f64::NAN, 1.], &mut output)
                .is_err()
        );
        assert_eq!(output, [7.; 2]);
        assert!(mapped.map_coefficients(&[1.], &mut output).is_err());
        assert_eq!(output, [7.; 2]);
        assert!(mapped.apply_adjoint(&[f64::NAN; 3], &mut output).is_err());
        assert_eq!(output, [7.; 2]);
    }
}

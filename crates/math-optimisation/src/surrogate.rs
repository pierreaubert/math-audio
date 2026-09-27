//! Radial-basis-function (RBF) surrogates for expensive black-box optimisation.
//!
//! An [`RbfSurrogate`] interpolates scattered observations `(x, y)` as a
//! weighted sum of radially symmetric kernels plus a linear polynomial tail,
//! solved as one augmented linear system. Three kernel families are
//! supported ([`RbfKind`]); [`RbfSurrogate::fit_auto`] picks between them by
//! repeated holdout validation.
//!
//! Callers should fit in normalised coordinates (the kernels use raw
//! Euclidean distances, so unscaled axes would dominate the shape).

use crate::error::{DEError, Result};
use nalgebra::{DMatrix, DVector};
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

/// RBF kernel family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RbfKind {
    /// Cubic `φ(r) = r^3`. Parameter-free, good default for smooth data.
    Cubic,
    /// Gaussian `φ(r) = exp(-(εr)^2)` with a median-distance shape.
    Gaussian,
    /// Multiquadric `φ(r) = sqrt(1 + (εr)^2)` with a median-distance shape.
    Multiquadric,
}

impl RbfKind {
    fn all() -> [RbfKind; 3] {
        [RbfKind::Cubic, RbfKind::Gaussian, RbfKind::Multiquadric]
    }
}

/// Interpolating RBF surrogate with a linear polynomial tail.
///
/// The model is `s(x) = Σ w_i φ(||x - c_i||) + a_0 + a·x`, fitted by solving
/// the augmented interpolation system with a small diagonal nugget for
/// numerical stability.
#[derive(Debug, Clone)]
pub struct RbfSurrogate {
    kind: RbfKind,
    dim: usize,
    centers: Vec<Array1<f64>>,
    weights: Vec<f64>,
    /// Linear tail coefficients `[a_0, a_1, ..., a_dim]`.
    tail: Vec<f64>,
    /// Shape parameter for Gaussian/Multiquadric kernels.
    epsilon: f64,
}

impl RbfSurrogate {
    /// Fit an interpolating surrogate of the given kernel family.
    ///
    /// Requires at least `dim + 1` finite observations with consistent
    /// dimensions. Returns an error when the interpolation system cannot be
    /// solved (e.g. fully degenerate point sets).
    pub fn fit(xs: &[Array1<f64>], ys: &[f64], kind: RbfKind) -> Result<Self> {
        validate_training_data(xs, ys)?;
        let dim = xs[0].len();
        if xs.len() < dim + 1 {
            return Err(DEError::InvalidConfig {
                message: format!(
                    "RBF fit needs at least dim + 1 = {} points, got {}",
                    dim + 1,
                    xs.len()
                ),
            });
        }
        let epsilon = shape_parameter(xs, kind);
        let (weights, tail) = solve_interpolation(xs, ys, kind, epsilon)?;
        Ok(Self {
            kind,
            dim,
            centers: xs.to_vec(),
            weights,
            tail,
            epsilon,
        })
    }

    /// Fit with automatic kernel selection.
    ///
    /// Each kernel family is scored by repeated holdout validation (3
    /// repeats, ~20% held out) and the lowest-RMSE model refit on all data
    /// is returned. With fewer than `dim + 2` points there is nothing to
    /// validate against, so a cubic model is fitted directly.
    pub fn fit_auto(xs: &[Array1<f64>], ys: &[f64]) -> Result<Self> {
        validate_training_data(xs, ys)?;
        let dim = xs[0].len();
        if xs.len() < dim + 2 {
            return Self::fit(xs, ys, RbfKind::Cubic);
        }
        let holdout = (xs.len() / 5).max(1).min(xs.len() - (dim + 1));
        let mut best: Option<(RbfKind, f64)> = None;
        for kind in RbfKind::all() {
            let rmse = holdout_rmse(xs, ys, kind, holdout, 3)?;
            let better = best.map(|(_, b)| rmse < b).unwrap_or(true);
            if better {
                best = Some((kind, rmse));
            }
        }
        let (kind, _) = best.expect("at least one kernel family");
        Self::fit(xs, ys, kind)
    }

    /// Predict the surrogate value at `x`.
    ///
    /// # Panics
    ///
    /// Panics when `x` has a different dimension than the training data.
    pub fn predict(&self, x: &Array1<f64>) -> f64 {
        assert!(
            x.len() == self.dim,
            "RBF predict dimension mismatch: expected {}, got {}",
            self.dim,
            x.len()
        );
        let mut value = self.tail[0];
        for j in 0..self.dim {
            value += self.tail[j + 1] * x[j];
        }
        for (center, weight) in self.centers.iter().zip(self.weights.iter()) {
            value += weight * kernel(self.kind, self.epsilon, distance(x, center));
        }
        value
    }

    /// Predict the surrogate value at each point in `xs`.
    pub fn predict_many(&self, xs: &[Array1<f64>]) -> Vec<f64> {
        xs.iter().map(|x| self.predict(x)).collect()
    }

    /// The kernel family of this surrogate.
    pub fn kind(&self) -> RbfKind {
        self.kind
    }

    /// Number of interpolation centers (training points).
    pub fn n_centers(&self) -> usize {
        self.centers.len()
    }
}

fn validate_training_data(xs: &[Array1<f64>], ys: &[f64]) -> Result<()> {
    if xs.is_empty() {
        return Err(DEError::InvalidConfig {
            message: "RBF fit needs at least one training point".to_string(),
        });
    }
    if xs.len() != ys.len() {
        return Err(DEError::InvalidConfig {
            message: format!(
                "RBF training size mismatch: {} points vs {} values",
                xs.len(),
                ys.len()
            ),
        });
    }
    let dim = xs[0].len();
    if dim == 0 {
        return Err(DEError::InvalidConfig {
            message: "RBF training points must be non-empty".to_string(),
        });
    }
    for (i, x) in xs.iter().enumerate() {
        if x.len() != dim {
            return Err(DEError::InvalidConfig {
                message: format!("RBF point {i} has dimension {}, expected {dim}", x.len()),
            });
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(DEError::InvalidConfig {
                message: format!("RBF point {i} contains non-finite values"),
            });
        }
    }
    if ys.iter().any(|v| !v.is_finite()) {
        return Err(DEError::InvalidConfig {
            message: "RBF training values must be finite".to_string(),
        });
    }
    Ok(())
}

fn distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(u, v)| (u - v) * (u - v))
        .sum::<f64>()
        .sqrt()
}

fn kernel(kind: RbfKind, epsilon: f64, r: f64) -> f64 {
    match kind {
        RbfKind::Cubic => r * r * r,
        RbfKind::Gaussian => (-(epsilon * r).powi(2)).exp(),
        RbfKind::Multiquadric => (1.0 + (epsilon * r).powi(2)).sqrt(),
    }
}

/// Median-distance shape heuristic: `ε = 1 / median(||x_i - x_j||)`.
fn shape_parameter(xs: &[Array1<f64>], kind: RbfKind) -> f64 {
    if matches!(kind, RbfKind::Cubic) {
        return 1.0;
    }
    let mut distances = Vec::new();
    for i in 0..xs.len() {
        for other in xs.iter().skip(i + 1) {
            distances.push(distance(&xs[i], other));
        }
    }
    if distances.is_empty() {
        return 1.0;
    }
    distances.sort_by(f64::total_cmp);
    let median = distances[distances.len() / 2];
    if median > 0.0 { 1.0 / median } else { 1.0 }
}

/// Solve the augmented interpolation system `[[Φ P] [Pᵀ 0]] [w a]ᵀ = [y 0]ᵀ`
/// with a diagonal nugget, retrying with a larger nugget on failure.
fn solve_interpolation(
    xs: &[Array1<f64>],
    ys: &[f64],
    kind: RbfKind,
    epsilon: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = xs.len();
    let dim = xs[0].len();
    let tail_len = dim + 1;
    let size = n + tail_len;

    // Assemble the augmented matrix once; only the nugget changes retries.
    let mut base = DMatrix::<f64>::zeros(size, size);
    let mut kernel_scale = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let value = kernel(kind, epsilon, distance(&xs[i], &xs[j]));
            if i != j {
                kernel_scale = kernel_scale.max(value.abs());
            }
            base[(i, j)] = value;
        }
        base[(i, n)] = 1.0;
        for d in 0..dim {
            base[(i, n + 1 + d)] = xs[i][d];
            base[(n + 1 + d, i)] = xs[i][d];
        }
        base[(n, i)] = 1.0;
    }
    let mut rhs = DVector::<f64>::zeros(size);
    for (i, &y) in ys.iter().enumerate() {
        rhs[i] = y;
    }

    // `RbfKind::Cubic` has a zero diagonal, so even well-spread data needs a
    // small nugget; degenerate data (duplicate centers) needs a large one.
    let unit = 1e-10 * kernel_scale.max(1.0);
    for attempt in 0..4 {
        let nugget = unit * 1e3_f64.powi(attempt);
        let mut system = base.clone();
        for i in 0..n {
            system[(i, i)] += nugget;
        }
        if let Some(solution) = system.lu().solve(&rhs)
            && solution.iter().all(|v| v.is_finite())
        {
            let weights = solution.rows(0, n).iter().copied().collect();
            let tail = solution.rows(n, tail_len).iter().copied().collect();
            return Ok((weights, tail));
        }
    }
    Err(DEError::InvalidConfig {
        message: format!(
            "RBF interpolation system is singular for {n} points (all nugget retries failed)"
        ),
    })
}

/// Mean holdout RMSE of a kernel family over `repeats` seeded shuffles.
fn holdout_rmse(
    xs: &[Array1<f64>],
    ys: &[f64],
    kind: RbfKind,
    holdout: usize,
    repeats: usize,
) -> Result<f64> {
    let n = xs.len();
    let mut total_squared = 0.0;
    let mut total_count = 0usize;
    for repeat in 0..repeats {
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut StdRng::seed_from_u64(repeat as u64));
        let (test, train) = indices.split_at(holdout);
        let train_xs: Vec<Array1<f64>> = train.iter().map(|&i| xs[i].clone()).collect();
        let train_ys: Vec<f64> = train.iter().map(|&i| ys[i]).collect();
        // A fold that fails to fit disqualifies the family.
        let model = RbfSurrogate::fit(&train_xs, &train_ys, kind)?;
        for &i in test {
            let error = model.predict(&xs[i]) - ys[i];
            total_squared += error * error;
            total_count += 1;
        }
    }
    Ok((total_squared / total_count as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    fn sample_2d(seed: u64, count: usize) -> Vec<Array1<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..count)
            .map(|_| Array1::from(vec![rng.random::<f64>(), rng.random::<f64>()]))
            .collect()
    }

    fn smooth_2d(x: &Array1<f64>) -> f64 {
        (2.0 * x[0]).sin() + (3.0 * x[1]).cos() + x[0] * x[1]
    }

    #[test]
    fn cubic_interpolates_training_points() {
        let xs = sample_2d(1, 25);
        let ys: Vec<f64> = xs.iter().map(smooth_2d).collect();
        let model = RbfSurrogate::fit(&xs, &ys, RbfKind::Cubic).expect("fit should succeed");
        assert_eq!(model.n_centers(), 25);
        for (x, y) in xs.iter().zip(ys.iter()) {
            assert!(
                (model.predict(x) - y).abs() < 1e-8,
                "interpolation error too large"
            );
        }
    }

    #[test]
    fn every_kernel_generalises_on_smooth_data() {
        let xs = sample_2d(2, 40);
        let ys: Vec<f64> = xs.iter().map(smooth_2d).collect();
        let test = sample_2d(3, 12);
        for kind in RbfKind::all() {
            let model = RbfSurrogate::fit(&xs, &ys, kind).expect("fit should succeed");
            let rmse = holdout_error(&model, &test);
            assert!(rmse < 0.05, "{kind:?} holdout RMSE too large: {rmse}");
        }
    }

    #[test]
    fn fit_auto_selects_a_working_model() {
        let xs = sample_2d(4, 30);
        let ys: Vec<f64> = xs.iter().map(smooth_2d).collect();
        let model = RbfSurrogate::fit_auto(&xs, &ys).expect("fit_auto should succeed");
        assert_eq!(model.n_centers(), 30);
        let rmse = holdout_error(&model, &sample_2d(5, 12));
        assert!(rmse < 0.05, "selected model RMSE too large: {rmse}");
    }

    #[test]
    fn fit_auto_falls_back_to_cubic_on_tiny_samples() {
        // 2D needs dim + 1 = 3 points to fit, dim + 2 = 4 to validate.
        let xs = sample_2d(6, 3);
        let ys: Vec<f64> = xs.iter().map(smooth_2d).collect();
        let model = RbfSurrogate::fit_auto(&xs, &ys).expect("fallback fit should succeed");
        assert_eq!(model.kind(), RbfKind::Cubic);
    }

    #[test]
    fn fit_rejects_bad_training_data() {
        let xs = sample_2d(7, 5);
        let ys = vec![0.0; 5];
        assert!(RbfSurrogate::fit(&xs, &ys[..4], RbfKind::Cubic).is_err());
        let mut bad_dim = xs.clone();
        bad_dim[2] = Array1::from(vec![0.0, 0.0, 0.0]);
        assert!(RbfSurrogate::fit(&bad_dim, &ys, RbfKind::Cubic).is_err());
        let mut bad_y = ys.clone();
        bad_y[1] = f64::NAN;
        assert!(RbfSurrogate::fit(&xs, &bad_y, RbfKind::Cubic).is_err());
        assert!(RbfSurrogate::fit(&xs[..2], &ys[..2], RbfKind::Cubic).is_err());
        assert!(RbfSurrogate::fit(&[], &[], RbfKind::Cubic).is_err());
    }

    #[test]
    fn duplicate_points_still_fit() {
        let mut xs = sample_2d(8, 9);
        xs.push(xs[0].clone());
        let ys: Vec<f64> = xs.iter().map(smooth_2d).collect();
        let model = RbfSurrogate::fit(&xs, &ys, RbfKind::Cubic).expect("nugget should save fit");
        for x in &xs {
            assert!(model.predict(x).is_finite());
        }
    }

    #[test]
    fn predict_many_matches_predict() {
        let xs = sample_2d(9, 10);
        let ys: Vec<f64> = xs.iter().map(smooth_2d).collect();
        let model = RbfSurrogate::fit(&xs, &ys, RbfKind::Gaussian).expect("fit should succeed");
        let test = sample_2d(10, 4);
        let batch = model.predict_many(&test);
        for (x, &b) in test.iter().zip(batch.iter()) {
            assert_eq!(b, model.predict(x));
        }
    }

    fn holdout_error(model: &RbfSurrogate, test: &[Array1<f64>]) -> f64 {
        let sum: f64 = test
            .iter()
            .map(|x| (model.predict(x) - smooth_2d(x)).powi(2))
            .sum();
        (sum / test.len() as f64).sqrt()
    }
}

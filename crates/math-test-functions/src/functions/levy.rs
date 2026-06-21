//! Levy test function

use ndarray::Array1;

/// Levy function - multimodal function (generalized version)
/// Global minimum: f(x) = 0 at x = (1, 1, ..., 1)
/// Bounds: x_i in [-10, 10]
pub fn levy(x: &Array1<f64>) -> f64 {
    use std::f64::consts::PI;

    let n = x.len();
    if n == 0 {
        return 0.0;
    }

    let w0 = 1.0 + (x[0] - 1.0) / 4.0;
    let first_term = (PI * w0).sin().powi(2);

    let middle_sum: f64 = x
        .iter()
        .take(n.saturating_sub(1))
        .map(|&xi: &f64| {
            let wi: f64 = 1.0 + (xi - 1.0) / 4.0;
            (wi - 1.0).powi(2) * (1.0 + 10.0 * (PI * wi + 1.0).sin().powi(2))
        })
        .sum();

    let wn = 1.0 + (x[n - 1] - 1.0) / 4.0;
    let last_term = (wn - 1.0).powi(2) * (1.0 + (2.0 * PI * wn).sin().powi(2));

    first_term + middle_sum + last_term
}

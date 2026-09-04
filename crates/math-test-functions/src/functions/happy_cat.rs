//! Happy Cat test function

use ndarray::Array1;

/// Deprecated duplicate of [`happycat`](super::happycat::happycat).
///
/// The previous body used `((s-n).powi(2)).powf(0.25)` (= `|s-n|^0.5`),
/// which disagrees with the reference definition
/// `f(x) = |sum(x_i^2) - n|^0.25 + (0.5*sum(x_i^2) + sum(x_i))/n + 0.5`.
/// Kept for backwards compatibility; delegates to [`happycat`](super::happycat::happycat).
#[deprecated(note = "duplicate of `happycat` with a corrected exponent; use `happycat` instead")]
pub fn happy_cat(x: &Array1<f64>) -> f64 {
    super::happycat::happycat(x)
}

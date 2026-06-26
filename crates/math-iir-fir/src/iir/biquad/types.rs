use super::Biquad;

/// Parametric EQ filter chain: a vector of (gain, Biquad) pairs.
///
/// Each element is a tuple of:
/// - `T`: The linear gain multiplier for this stage
/// - `Biquad<T>`: The biquad filter for this stage
pub type Peq<T = f64> = Vec<(T, Biquad<T>)>;

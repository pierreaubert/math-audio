use crate::traits::FilterFloat;

/// Biquad filter coefficients for external interpolation.
#[derive(Debug, Clone, Copy)]
pub struct BiquadCoefficients<T: FilterFloat = f64> {
    /// Feedforward coefficient b0
    pub b0: T,
    /// Feedforward coefficient b1
    pub b1: T,
    /// Feedforward coefficient b2
    pub b2: T,
    /// Feedback coefficient a1 (normalized, a0=1)
    pub a1: T,
    /// Feedback coefficient a2 (normalized, a0=1)
    pub a2: T,
}

impl<T: FilterFloat> BiquadCoefficients<T> {
    /// Linearly interpolate between two sets of coefficients.
    ///
    /// `t` ranges from 0.0 (fully `self`) to 1.0 (fully `other`).
    /// For real, stable, normalized second-order denominators, this interval
    /// remains stable: the Schur/Jury stability region in `(a1, a2)` is convex.
    /// Values outside the documented interval are extrapolation and do not have
    /// that guarantee.
    #[inline(always)]
    pub fn lerp(&self, other: &BiquadCoefficients<T>, t: T) -> BiquadCoefficients<T> {
        BiquadCoefficients {
            b0: self.b0 + (other.b0 - self.b0) * t,
            b1: self.b1 + (other.b1 - self.b1) * t,
            b2: self.b2 + (other.b2 - self.b2) * t,
            a1: self.a1 + (other.a1 - self.a1) * t,
            a2: self.a2 + (other.a2 - self.a2) * t,
        }
    }
}

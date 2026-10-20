//! Physical-coordinate realization of the exact allpass-substituted transfer.

// Rust guideline compliant 2026-02-21
use super::{Biquad, BiquadFilterType, WarpedBiquad};
use crate::traits::{FilterFloat, lit};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub(super) struct PhysicalState<T: FilterFloat> {
    pub(super) c: [T; 5],
    z: [T; 2],
    peak_residual_gain: Option<T>,
}

impl<T: FilterFloat> PhysicalState<T> {
    pub(super) fn from_history(filter: &WarpedBiquad<T>) -> Self {
        let c = physical_coefficients(filter);
        let [b0, b1, b2, a1, a2] = c;
        let l = filter.lambda;
        let two: T = lit(2.0);
        let fac = (T::one() - l) * (T::one() + l);
        // Inverse rational substitution expresses the old four allpass states
        // in the two observable TDF-II coordinates. No history is discarded.
        // c2=(l²*b0+l*b1+b2)/(1-l²), and likewise for denominator d2.
        let c2 = (l * l).mul_add(b0, l.mul_add(b1, b2)) / fac;
        let d2 = l.mul_add(l, l.mul_add(a1, a2)) / fac;
        let c1 = b1 + two * l * b0 + l * c2;
        let d1 = a1 + two * l + l * d2;
        let z1 = c1 * filter.ap_x1_s + c2 * filter.ap_x2_s - d1 * filter.s_y1 - d2 * filter.s_y2;
        let z2 = -l * z1 + fac * (c2 * filter.ap_x1_s - d2 * filter.s_y1);
        Self {
            c,
            z: [z1, z2],
            peak_residual_gain: (filter.filter_type == BiquadFilterType::PeakMatched)
                .then_some(b0 - T::one()),
        }
    }

    #[inline]
    pub(super) fn process(&mut self, x: T) -> T {
        let [b0, b1, b2, a1, a2] = self.c;
        if let Some(k) = self.peak_residual_gain {
            // H=1+k*(1-z^-2)/D. Keeping dry separate avoids cancellation
            // between large matched-peak numerator terms at unity endpoints.
            let residual = k * x + self.z[0];
            self.z[0] = -a1 * residual + self.z[1];
            self.z[1] = -k * x - a2 * residual;
            return x + residual;
        }
        let y = b0 * x + self.z[0];
        self.z[0] = b1 * x - a1 * y + self.z[1];
        self.z[1] = b2 * x - a2 * y;
        y
    }
}

pub(super) fn physical_coefficients<T: FilterFloat>(f: &WarpedBiquad<T>) -> [T; 5] {
    // Keep Biquad::new's existing frequency/rate/gain fallback policy. In
    // particular, a corner above a new Nyquist must not create a negative
    // matched-pole decay when evaluated through tan(w/2).
    let native = Biquad::new(f.filter_type, f.freq, f.srate, f.q, f.db_gain);
    if f.filter_type != BiquadFilterType::PeakMatched || f.lambda == T::zero() {
        // The bilinear frequency variable and the corner prewarp receive the
        // same (1+l)/(1-l) scale, which cancels exactly. Optimized shelves
        // have the same fixed-slope minimum-phase transfer as these shelves.
        let c = native.coefficients();
        return [c.b0, c.b1, c.b2, c.a1, c.a2];
    }
    let one = T::one();
    let two: T = lit(2.0);
    let l = f.lambda;
    let w = two * T::PI() * native.freq / native.srate;
    // Half-angle prewarp avoids subtraction in omega+2*atan(correction),
    // especially for a low corner and lambda near -1.
    let k = (w / two).tan() * (one + l) / (one - l);
    let warped_w = two * k.atan();
    let sigma = warped_w / (two * native.q);
    let radius = (-sigma).exp();
    let radius_minus_one = (-sigma).exp_m1();
    let sin_half_sq = (warped_w / two).sin().powi(2);
    let sin_w = warped_w.sin();
    // Transform a boost pole p through (p+l)/(1+l*p), retaining small
    // differences with exp_m1 and half-angle terms rather than 1-cos(w).
    let nr = radius_minus_one + (one + l) - two * radius * sin_half_sq;
    let ni = radius * sin_w;
    let dr = (one + l) + l * radius_minus_one - two * l * radius * sin_half_sq;
    let di = l * ni;
    let power = dr * dr + di * di;
    let re = (nr * dr + ni * di) / power;
    let im = (ni * dr - nr * di) / power;
    let a1 = -two * re;
    let a2 = re * re + im * im;
    // The unique minimum-phase numerator has unity endpoints and prescribed
    // physical center gain. This is the same spectral factor as substituting
    // the allpass into the prototype, without rounded near-unit coefficients.
    let gain = lit::<T>(10.0).powf(native.db_gain.abs() / lit(20.0));
    let gain_sq = gain * gain;
    let p = one + a2;
    let real_over_sin = (p * w.cos() + a1) / w.sin();
    let d = (gain_sq * (one - a2).powi(2) + (gain_sq - one) * real_over_sin * real_over_sin).sqrt();
    let b0 = (p + d) / two;
    let b2 = (p - d) / two;
    if native.db_gain < T::zero() {
        [one / b0, a1 / b0, a2 / b0, a1 / b0, b2 / b0]
    } else if native.db_gain == T::zero() {
        [one, a1, a2, a1, a2]
    } else {
        [b0, a1, b2, a1, a2]
    }
}

#[cfg(test)]
mod tests;

//! Warped biquad IIR filter for perceptual room EQ.
//!
//! A warped biquad replaces each unit delay (z^-1) in a standard biquad with
//! a first-order allpass section: A(z) = (z^-1 - λ)/(1 - λz^-1).
//!
//! The prototype corner is prewarped before allpass substitution. For bilinear
//! filter families these frequency scales cancel, preserving the physical analog
//! transfer. Matched-peak pole placement can still depend on lambda.
//!
//! Processing uses equivalent physical-coordinate coefficients to avoid a nearly
//! cancelling implicit solve. Original allpass histories are retained for state
//! conversion on parameter changes and restoration of earlier serialized filters.

use crate::iir::biquad::{Biquad, BiquadFilterType};
use crate::traits::{FilterFloat, lit};
use ndarray::Array1;
use num_complex::Complex;

mod canonical;
mod misc;
use canonical::{PhysicalState, physical_coefficients};
#[cfg(test)]
mod tests;

pub use misc::*;

use misc::allpass_step;
use misc::compute_implicit_consts;

/// Allpass-substituted biquad with a numerically conditioned physical realization.
///
/// The transfer is the prototype biquad evaluated at
/// `A(z)=(z^-1-lambda)/(1-lambda*z^-1)`. The prototype corner uses the same
/// frequency mapping, so bilinear families preserve their physical analog
/// response. Zero lambda gives the standard biquad transfer.
///
/// A normalized physical TDF-II realization avoids loss of precision in the
/// implicit denominator `1-a1*lambda+a2*lambda^2`. Input and output allpass
/// histories remain populated, and parameter updates map those histories to the
/// new physical coordinates without resetting them. This preserves the implicit
/// realization's stateful update convention, not a promise of click-free edits.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct WarpedBiquad<T: FilterFloat = f64> {
    /// Prototype biquad coefficients (designed at the warped center frequency)
    a1: T,
    a2: T,
    b0: T,
    b1: T,
    b2: T,
    /// Warping coefficient λ
    pub lambda: T,
    /// Legacy serialized implicit denominator cache: 1/(1 - a1·λ + a2·λ²)
    inv_denom: T,
    /// Precomputed (1 − λ²)
    one_minus_lam_sq: T,
    /// Filter type
    pub filter_type: BiquadFilterType,
    /// Center frequency in Hz (physical domain)
    pub freq: T,
    /// Sample rate in Hz
    pub srate: T,
    /// Q factor
    pub q: T,
    /// Gain in dB (for peak/shelf types)
    pub db_gain: T,
    // Allpass internal states for the input (x) chain.
    // With λ=0 these degenerate to x[n-1] and x[n-2].
    ap_x1_s: T,
    ap_x2_s: T,
    // Allpass internal states for the output (y) chain.
    // With λ=0: s_y1 = y[n-1], s_y2 = y[n-2] (standard DF-I feedback).
    s_y1: T,
    s_y2: T,
    // Absent in older serialized states; restore lazily from their histories.
    #[serde(default)]
    physical: Option<PhysicalState<T>>,
}

impl<T: FilterFloat> WarpedBiquad<T> {
    /// Creates a new warped biquad filter.
    ///
    /// # Arguments
    ///
    /// * `filter_type` - The biquad filter topology to apply
    /// * `freq`        - Center frequency in Hz (physical domain)
    /// * `srate`       - Sample rate in Hz
    /// * `q`           - Q factor (0.0 uses the same defaults as [`Biquad::new`])
    /// * `db_gain`     - Gain in dB (peak/shelf types only)
    /// * `lambda`      - Warping coefficient; use [`bark_lambda`] for Bark-scale,
    ///   or 0.0 for standard biquad behavior
    pub fn new(
        filter_type: BiquadFilterType,
        freq: T,
        srate: T,
        q: T,
        db_gain: T,
        lambda: T,
    ) -> Self {
        let zero = T::zero();

        let freq_warped = if lambda == zero {
            freq
        } else {
            warp_frequency(freq, srate, lambda)
        };

        let proto = Biquad::new(filter_type, freq_warped, srate, q, db_gain);
        let coeffs = proto.coefficients();
        let (inv_denom, one_minus_lam_sq) = compute_implicit_consts(coeffs.a1, coeffs.a2, lambda);

        let mut filter = WarpedBiquad {
            a1: coeffs.a1,
            a2: coeffs.a2,
            b0: coeffs.b0,
            b1: coeffs.b1,
            b2: coeffs.b2,
            lambda,
            inv_denom,
            one_minus_lam_sq,
            filter_type,
            freq,
            srate,
            q: proto.q, // clamped/defaulted Q from Biquad::new
            db_gain,
            ap_x1_s: zero,
            ap_x2_s: zero,
            s_y1: zero,
            s_y2: zero,
            physical: None,
        };
        filter.restore_physical();
        filter
    }

    /// Update filter parameters and recompute coefficients without resetting state.
    pub fn update_params(
        &mut self,
        filter_type: BiquadFilterType,
        freq: T,
        srate: T,
        q: T,
        db_gain: T,
        lambda: T,
    ) {
        self.filter_type = filter_type;
        self.freq = freq;
        self.srate = srate;
        self.db_gain = db_gain;
        self.lambda = lambda;

        let freq_warped = if lambda == T::zero() {
            freq
        } else {
            warp_frequency(freq, srate, lambda)
        };

        let proto = Biquad::new(filter_type, freq_warped, srate, q, db_gain);
        self.q = proto.q;
        let coeffs = proto.coefficients();
        self.a1 = coeffs.a1;
        self.a2 = coeffs.a2;
        self.b0 = coeffs.b0;
        self.b1 = coeffs.b1;
        self.b2 = coeffs.b2;
        let (inv_denom, one_minus_lam_sq) = compute_implicit_consts(coeffs.a1, coeffs.a2, lambda);
        self.inv_denom = inv_denom;
        self.one_minus_lam_sq = one_minus_lam_sq;
        self.restore_physical();
    }

    fn restore_physical(&mut self) {
        self.physical = Some(PhysicalState::from_history(self));
    }

    /// Processes one sample while preserving the original allpass histories.
    #[inline]
    pub fn process(&mut self, x: T) -> T {
        if self.physical.is_none() {
            self.restore_physical();
        }
        let y = self
            .physical
            .as_mut()
            .expect("physical state initialized")
            .process(x);
        let xd1 = allpass_step(x, &mut self.ap_x1_s, self.lambda);
        allpass_step(xd1, &mut self.ap_x2_s, self.lambda);
        let yd1 = allpass_step(y, &mut self.s_y1, self.lambda);
        allpass_step(yd1, &mut self.s_y2, self.lambda);
        y
    }

    /// Processes a block with the same state convention as sample processing.
    pub fn process_block(&mut self, samples: &mut [T]) {
        for sample in samples {
            *sample = self.process(*sample);
        }
    }

    /// Evaluates the physical transfer at frequency `f` in Hz.
    pub fn complex_response(&self, f: T) -> Complex<T> {
        let [b0, b1, b2, a1, a2] = self
            .physical
            .as_ref()
            .map_or_else(|| physical_coefficients(self), |state| state.c);
        let omega = lit::<T>(2.0) * T::PI() * f / self.srate;
        let z = Complex::<T>::from_polar(T::one(), -omega);
        let denominator = Complex::new(T::one(), T::zero()) + z * a1 + z * z * a2;
        if self.filter_type == BiquadFilterType::PeakMatched {
            return Complex::new(T::one(), T::zero())
                + (Complex::new(T::one(), T::zero()) - z * z) * (b0 - T::one()) / denominator;
        }
        (Complex::new(b0, T::zero()) + z * b1 + z * z * b2) / denominator
    }

    /// Magnitude response at a single physical frequency `f`.
    pub fn result(&self, f: T) -> T {
        self.complex_response(f).norm()
    }

    /// Response in dB at a single physical frequency `f`.
    pub fn log_result(&self, f: T) -> T {
        let mag = self.result(f);
        if mag > T::zero() {
            lit::<T>(20.0) * mag.log10()
        } else {
            lit(-200.0)
        }
    }

    /// Vectorized dB response over an array of physical frequencies.
    pub fn np_log_result(&self, freqs: &Array1<T>) -> Array1<T> {
        freqs.mapv(|f| self.log_result(f))
    }

    /// Returns the normalized prototype coefficients as `(a1, a2, b0, b1, b2)`.
    pub fn coefficients(&self) -> (T, T, T, T, T) {
        (self.a1, self.a2, self.b0, self.b1, self.b2)
    }
}

impl<T: FilterFloat> std::fmt::Display for WarpedBiquad<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "WarpedBiquad(Type:{},Freq:{:.1},Rate:{:.1},Q:{:.1},Gain:{:.1},Lambda:{:.4})",
            self.filter_type.short_name(),
            self.freq,
            self.srate,
            self.q,
            self.db_gain,
            self.lambda,
        )
    }
}

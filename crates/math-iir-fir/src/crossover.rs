//! Selectable second- and fourth-order IIR crossovers.
//!
//! Frequency is in Hz. Butterworth branches are -3 dB at this frequency;
//! All other families have -6 dB branches. Second-order high outputs are polarity
//! inverted so callers can sum the returned bands directly. Linkwitz–Riley and
//! Neville–Thiele have flat summed magnitudes, with phase rotation. Cascaded
//! Bessel order 2 equals LR2; order 4 has a summation dip (about 1.37 dB at
//! crossover) and does not retain the maximally flat analog Bessel group delay
//! after the bilinear transform.
//!
//! Neville–Thiele supports order 4 only, with k = 0.5. Its stopband zeros
//! occur at half and twice the crossover frequency in the prewarped analog
//! domain. Digital notch frequencies are fs/pi * atan(k * tan(pi*fc/fs))
//! and fs/pi * atan(tan(pi*fc/fs)/k). Beyond these zeros, attenuation
//! approaches 12 dB/octave, despite the fourth-order denominator.
//!
//! ```
//! use math_audio_iir_fir::{Crossover, CrossoverFamily, CrossoverOrder};
//!
//! let mut crossover = Crossover::<f64>::new(
//!     CrossoverFamily::LinkwitzRiley, CrossoverOrder::Second,
//!     1000.0, 48000.0, 2,
//! );
//! let (low, high) = crossover.process(1.0, 0);
//! let recombined = low + high;
//! assert!(recombined.is_finite());
//! ```

use crate::traits::lit;
use crate::{Biquad, BiquadCoefficients, BiquadFilterType, FilterFloat};

/// Crossover response family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossoverFamily {
    /// Maximally flat individual passbands.
    Butterworth,
    /// Cascaded Butterworth filters with a flat summed magnitude.
    LinkwitzRiley,
    /// Two cascaded half-order Bessel filters, each normalized to -3 dB.
    CascadedBessel,
    /// Fourth-order Neville–Thiele with notch parameter k = 0.5.
    NevilleThiele,
}

/// Total order of each output branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossoverOrder {
    /// Two poles per branch (12 dB/octave).
    Second,
    /// Four poles per branch (24 dB/octave for all-pole families).
    /// Neville–Thiele has stopband notches and a 12 dB/octave asymptotic slope.
    Fourth,
}

#[derive(Debug, Clone)]
struct Section<T: FilterFloat> {
    filter: Biquad<T>,
    coefficients: Option<BiquadCoefficients<T>>,
}

impl<T: FilterFloat> Section<T> {
    #[inline]
    fn process(&mut self, sample: T) -> T {
        match &self.coefficients {
            Some(c) => self.filter.process_with_coefficients(sample, c),
            None => self.filter.process(sample),
        }
    }

    #[cfg(test)]
    fn complex_response(&self, frequency: T) -> num_complex::Complex<T> {
        let c = self
            .coefficients
            .unwrap_or_else(|| self.filter.coefficients());
        let z = num_complex::Complex::from_polar(
            T::one(),
            -lit::<T>(2.0) * T::PI() * frequency / self.filter.srate,
        );
        (num_complex::Complex::new(c.b0, T::zero()) + z * c.b1 + z * z * c.b2)
            / (num_complex::Complex::new(T::one(), T::zero()) + z * c.a1 + z * z * c.a2)
    }
}

/// Multichannel crossover with independent channel state and no processing allocations.
///
/// Uses one or two biquads per branch, with no lookahead or buffering latency.
/// Coefficient changes preserve state but are not smoothed and may cause transients.
#[derive(Debug, Clone)]
pub struct Crossover<T: FilterFloat = f64> {
    low: Vec<Vec<Section<T>>>,
    high: Vec<Vec<Section<T>>>,
    family: CrossoverFamily,
    order: CrossoverOrder,
    frequency: T,
    sample_rate: T,
}

impl<T: FilterFloat> Crossover<T> {
    /// Construct a crossover. Panics unless frequency is finite and strictly
    /// between zero and Nyquist, sample rate is finite and positive, and channels > 0.
    /// Also panics for Neville–Thiele with any order other than fourth.
    pub fn new(
        family: CrossoverFamily,
        order: CrossoverOrder,
        frequency: T,
        sample_rate: T,
        channels: usize,
    ) -> Self {
        Self::validate(frequency, sample_rate);
        assert!(channels > 0, "channels must be positive");
        assert!(
            family != CrossoverFamily::NevilleThiele || order == CrossoverOrder::Fourth,
            "Neville–Thiele supports only fourth order"
        );
        let make = |kind| {
            (0..channels)
                .map(|_| {
                    Self::qs(family, order)
                        .iter()
                        .enumerate()
                        .map(|(stage, &q)| {
                            let f =
                                Self::stage_frequency(family, order, kind, frequency, sample_rate);
                            let filter = Biquad::new(kind, f, sample_rate, lit(q), T::zero());
                            let coefficients = Self::notch_coefficients(family, stage, &filter);
                            Section {
                                filter,
                                coefficients,
                            }
                        })
                        .collect()
                })
                .collect()
        };
        Self {
            low: make(BiquadFilterType::Lowpass),
            high: make(BiquadFilterType::Highpass),
            family,
            order,
            frequency,
            sample_rate,
        }
    }

    fn validate(frequency: T, sample_rate: T) {
        assert!(
            sample_rate.is_finite() && sample_rate > T::zero(),
            "invalid sample rate"
        );
        assert!(
            frequency.is_finite() && frequency > T::zero() && frequency < sample_rate / lit(2.0),
            "frequency must be between zero and Nyquist"
        );
    }

    fn qs(family: CrossoverFamily, order: CrossoverOrder) -> &'static [f64] {
        match (family, order) {
            (CrossoverFamily::Butterworth, CrossoverOrder::Second) => {
                &[std::f64::consts::FRAC_1_SQRT_2]
            }
            (CrossoverFamily::Butterworth, CrossoverOrder::Fourth) => {
                &[0.541196100146197, 1.306562964876377]
            }
            (
                CrossoverFamily::LinkwitzRiley | CrossoverFamily::CascadedBessel,
                CrossoverOrder::Second,
            ) => &[0.5],
            (CrossoverFamily::CascadedBessel, CrossoverOrder::Fourth) => &[0.5773502691896258; 2],
            (CrossoverFamily::NevilleThiele, CrossoverOrder::Fourth) => &[0.816496580927726; 2],
            (CrossoverFamily::NevilleThiele, CrossoverOrder::Second) => {
                unreachable!("unsupported order")
            }
            (CrossoverFamily::LinkwitzRiley, CrossoverOrder::Fourth) => {
                &[std::f64::consts::FRAC_1_SQRT_2; 2]
            }
        }
    }

    // A second-order Bessel prototype is 1/(s² + sqrt(3)s + 1).
    // Its -3 dB point is sqrt((sqrt(5)-1)/2). Scale in the analog
    // prewarped domain so LP and HP both cross at -6 dB after cascading.
    fn stage_frequency(
        family: CrossoverFamily,
        order: CrossoverOrder,
        kind: BiquadFilterType,
        frequency: T,
        sample_rate: T,
    ) -> T {
        if family != CrossoverFamily::CascadedBessel || order != CrossoverOrder::Fourth {
            return frequency;
        }
        let scale: T = lit(1.272019649514069);
        let warped = (T::PI() * frequency / sample_rate).tan();
        let scaled = if kind == BiquadFilterType::Lowpass {
            warped * scale
        } else {
            warped / scale
        };
        sample_rate * scaled.atan() / T::PI()
    }

    // With s normalized to the prewarped crossover frequency, NT4 is
    // L=(1+k²s²)/D², H=s²(s²+k²)/D², D=s²+a*s+1,
    // k=1/2, a=sqrt(2*(1-k²)). Thus L+H=D(-s)/D(s).
    // One ordinary LP/HP section followed by a mixed-numerator section
    // realizes the notches without adding poles or subtracting output bands.
    fn notch_coefficients(
        family: CrossoverFamily,
        stage: usize,
        filter: &Biquad<T>,
    ) -> Option<BiquadCoefficients<T>> {
        if family != CrossoverFamily::NevilleThiele || stage == 0 {
            return None;
        }
        let opposite = if filter.filter_type == BiquadFilterType::Lowpass {
            BiquadFilterType::Highpass
        } else {
            BiquadFilterType::Lowpass
        };
        let other =
            Biquad::new(opposite, filter.freq, filter.srate, filter.q, T::zero()).coefficients();
        let mut c = filter.coefficients();
        let k2 = lit::<T>(0.25);
        c.b0 += k2 * other.b0;
        c.b1 += k2 * other.b1;
        c.b2 += k2 * other.b2;
        Some(c)
    }

    /// Process one channel. Returns `(low, high)` with summing polarity applied.
    #[inline]
    pub fn process(&mut self, sample: T, channel: usize) -> (T, T) {
        let low = self.low[channel]
            .iter_mut()
            .fold(sample, |x, b| b.process(x));
        let high = self.high[channel]
            .iter_mut()
            .fold(sample, |x, b| b.process(x));
        (
            low,
            if self.order == CrossoverOrder::Second {
                -high
            } else {
                high
            },
        )
    }

    /// Process a frame; all slices must have exactly [`Self::channels`] entries.
    pub fn process_frame(&mut self, input: &[T], low: &mut [T], high: &mut [T]) {
        assert_eq!(input.len(), self.channels());
        assert_eq!(low.len(), self.channels());
        assert_eq!(high.len(), self.channels());
        for ch in 0..input.len() {
            (low[ch], high[ch]) = self.process(input[ch], ch);
        }
    }

    /// Update frequency, preserving state. Uses the same validation as [`Self::new`].
    pub fn set_frequency(&mut self, frequency: T) {
        Self::validate(frequency, self.sample_rate);
        for (bank, kind) in [
            (&mut self.low, BiquadFilterType::Lowpass),
            (&mut self.high, BiquadFilterType::Highpass),
        ] {
            for channel in bank {
                for (stage, (filter, &q)) in channel
                    .iter_mut()
                    .zip(Self::qs(self.family, self.order))
                    .enumerate()
                {
                    let f = Self::stage_frequency(
                        self.family,
                        self.order,
                        kind,
                        frequency,
                        self.sample_rate,
                    );
                    filter
                        .filter
                        .update_params(kind, f, self.sample_rate, lit(q), T::zero());
                    filter.coefficients =
                        Self::notch_coefficients(self.family, stage, &filter.filter);
                }
            }
        }
        self.frequency = frequency;
    }

    /// Clear all channel states.
    pub fn reset(&mut self) {
        for channel in self.low.iter_mut().chain(&mut self.high) {
            for filter in channel {
                filter.filter.reset();
            }
        }
    }

    /// Crossover frequency in Hz.
    pub fn frequency(&self) -> T {
        self.frequency
    }
    /// Number of independently processed channels.
    pub fn channels(&self) -> usize {
        self.low.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    // Independent analog prototypes, evaluated with a prewarped bilinear map.
    fn prototype(
        family: CrossoverFamily,
        f: f64,
        fc: f64,
        fs: f64,
    ) -> (Complex<f64>, Complex<f64>) {
        let w = (std::f64::consts::PI * f / fs).tan() / (std::f64::consts::PI * fc / fs).tan();
        let s = Complex::new(0.0, w);
        match family {
            CrossoverFamily::NevilleThiele => {
                let d = (s * s + 1.5_f64.sqrt() * s + 1.0).powu(2);
                ((1.0 + 0.25 * s * s) / d, s * s * (s * s + 0.25) / d)
            }
            CrossoverFamily::CascadedBessel => {
                // Delay-normalized Bessel polynomial p(s)=s²+3s+3.
                // Solve |3/p(j*w3)|²=1/2 for its -3 dB frequency.
                let w3 = ((45.0_f64.sqrt() - 3.0) / 2.0).sqrt();
                let lp_s = s * w3;
                let hp_s = w3 / s;
                (
                    (3.0 / (lp_s * lp_s + 3.0 * lp_s + 3.0)).powu(2),
                    (3.0 / (hp_s * hp_s + 3.0 * hp_s + 3.0)).powu(2),
                )
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn new_families_match_analog_prototypes_and_impulse_dft() {
        for family in [
            CrossoverFamily::NevilleThiele,
            CrossoverFamily::CascadedBessel,
        ] {
            for fs in [44100.0, 96000.0] {
                let fc = fs * 0.1;
                let mut xo = Crossover::new(family, CrossoverOrder::Fourth, fc, fs, 2);
                let impulse: Vec<_> = (0..4096)
                    .map(|i| xo.process(if i == 0 { 1.0 } else { 0.0 }, 0))
                    .collect();
                for f in [fs * 0.005, fc, fs * 0.2, fs * 0.45] {
                    let expected = prototype(family, f, fc, fs);
                    let mut measured = (Complex::new(0.0, 0.0), Complex::new(0.0, 0.0));
                    for (i, &(l, h)) in impulse.iter().enumerate() {
                        let z = Complex::from_polar(
                            1.0,
                            -2.0 * std::f64::consts::PI * f * i as f64 / fs,
                        );
                        measured.0 += l * z;
                        measured.1 += h * z;
                    }
                    assert!((measured.0 - expected.0).norm() < 1e-10, "{family:?}, {f}");
                    assert!((measured.1 - expected.1).norm() < 1e-10, "{family:?}, {f}");
                }
                assert!(impulse.last().unwrap().0.abs() < 1e-20);
                assert!(impulse.last().unwrap().1.abs() < 1e-20);
                assert_eq!(xo.process(0.0, 1), (0.0, 0.0));
                xo.set_frequency(fs * 0.4);
                xo.reset();
                let mut fresh = Crossover::new(family, CrossoverOrder::Fourth, fs * 0.4, fs, 2);
                for i in 0..100 {
                    let x = if i == 0 { 1.0 } else { 0.0 };
                    assert_eq!(xo.process(x, 0), fresh.process(x, 0));
                }
            }
        }
    }

    #[test]
    fn nt4_notches_and_bessel_sum() {
        let fc = 4000.0;
        let fs = 48000.0;
        let nt = Crossover::new(
            CrossoverFamily::NevilleThiele,
            CrossoverOrder::Fourth,
            fc,
            fs,
            1,
        );
        let warped = (std::f64::consts::PI * fc / fs).tan();
        for (bank, ratio) in [(&nt.low, 2.0), (&nt.high, 0.5)] {
            let notch = fs / std::f64::consts::PI * (warped * ratio).atan();
            let response = bank[0]
                .iter()
                .fold(Complex::new(1.0, 0.0), |a, b| a * b.complex_response(notch));
            assert!(response.norm() < 1e-12);
        }
        let (l, h) = prototype(CrossoverFamily::CascadedBessel, fc, fc, fs);
        let sum_db = 20.0 * (l + h).norm().log10();
        assert!((sum_db + 1.37).abs() < 0.01, "{sum_db}");
        let mut bessel = Crossover::new(
            CrossoverFamily::CascadedBessel,
            CrossoverOrder::Second,
            fc,
            fs,
            1,
        );
        let mut lr = Crossover::new(
            CrossoverFamily::LinkwitzRiley,
            CrossoverOrder::Second,
            fc,
            fs,
            1,
        );
        for i in 0..100 {
            assert_eq!(bessel.process(i as f64, 0), lr.process(i as f64, 0));
        }
    }

    #[test]
    fn new_families_f32_remain_finite() {
        for family in [
            CrossoverFamily::CascadedBessel,
            CrossoverFamily::NevilleThiele,
        ] {
            for fc in [20.0, 1000.0, 20000.0] {
                let mut xo = Crossover::<f32>::new(family, CrossoverOrder::Fourth, fc, 48000.0, 1);
                for i in 0..20000 {
                    let (l, h) = xo.process(if i == 0 { 1.0 } else { 0.0 }, 0);
                    assert!(l.is_finite() && h.is_finite());
                    assert!(l.abs() < 2.0 && h.abs() < 2.0);
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "Neville–Thiele supports only fourth order")]
    fn nt_rejects_second_order() {
        Crossover::new(
            CrossoverFamily::NevilleThiele,
            CrossoverOrder::Second,
            1000.0,
            48000.0,
            1,
        );
    }

    #[test]
    fn response_and_summing_polarity() {
        for family in [
            CrossoverFamily::Butterworth,
            CrossoverFamily::LinkwitzRiley,
            CrossoverFamily::CascadedBessel,
            CrossoverFamily::NevilleThiele,
        ] {
            for order in [CrossoverOrder::Second, CrossoverOrder::Fourth] {
                if family == CrossoverFamily::NevilleThiele && order == CrossoverOrder::Second {
                    continue;
                }
                let xo = Crossover::new(family, order, 1000.0, 48000.0, 1);
                let response = |bank: &[Section<f64>], f| {
                    bank.iter()
                        .fold(Complex::new(1.0, 0.0), |a, b| a * b.complex_response(f))
                };
                let expected = if family == CrossoverFamily::Butterworth {
                    std::f64::consts::FRAC_1_SQRT_2
                } else {
                    0.5
                };
                assert!((response(&xo.low[0], 1000.0).norm() - expected).abs() < 1e-10);
                assert!((response(&xo.high[0], 1000.0).norm() - expected).abs() < 1e-10);
                assert!((response(&xo.low[0], 0.0).norm() - 1.0).abs() < 1e-10);
                assert!(response(&xo.high[0], 0.0).norm() < 1e-10);
                assert!(response(&xo.low[0], 24000.0).norm() < 1e-10);
                assert!((response(&xo.high[0], 24000.0).norm() - 1.0).abs() < 1e-10);
                if matches!(
                    family,
                    CrossoverFamily::LinkwitzRiley | CrossoverFamily::NevilleThiele
                ) {
                    for f in [10.0, 100.0, 500.0, 1000.0, 2000.0, 10000.0, 23000.0] {
                        let sign = if order == CrossoverOrder::Second {
                            -1.0
                        } else {
                            1.0
                        };
                        assert!(
                            ((response(&xo.low[0], f) + sign * response(&xo.high[0], f)).norm()
                                - 1.0)
                                .abs()
                                < 1e-10
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn lr4_matches_existing_processor() {
        let mut xo = Crossover::new(
            CrossoverFamily::LinkwitzRiley,
            CrossoverOrder::Fourth,
            1234.0,
            48000.0,
            2,
        );
        let mut old = crate::Lr4Crossover::new(1234.0, 48000.0, 2);
        for i in 0..4096 {
            let x = if i == 0 { 1.0 } else { 0.0 };
            assert_eq!(xo.process(x, 0), old.process(x, 0));
            assert_eq!(xo.process(0.0, 1), (0.0, 0.0));
        }
        xo.set_frequency(2345.0);
        old.set_frequency(2345.0);
        assert_eq!(xo.process(1.0, 0), old.process(1.0, 0));
        xo.reset();
        assert_eq!(xo.process(0.0, 0), (0.0, 0.0));
    }

    #[test]
    fn f32_frame_and_reset() {
        let mut xo = Crossover::<f32>::new(
            CrossoverFamily::Butterworth,
            CrossoverOrder::Fourth,
            1000.0,
            48000.0,
            2,
        );
        let (mut low, mut high) = ([0.0; 2], [0.0; 2]);
        for _ in 0..1000 {
            xo.process_frame(&[1.0, 0.0], &mut low, &mut high);
        }
        assert!((low[0] - 1.0).abs() < 1e-4);
        assert!(high[0].abs() < 1e-4);
        assert_eq!((low[1], high[1]), (0.0, 0.0));
        xo.reset();
        assert_eq!(xo.process(0.0, 0), (0.0, 0.0));
    }

    #[test]
    fn invalid_frequency_preserves_configuration() {
        let mut xo = Crossover::new(
            CrossoverFamily::Butterworth,
            CrossoverOrder::Second,
            1000.0,
            48000.0,
            1,
        );
        for f in [0.0, -1.0, 24000.0, f64::INFINITY, f64::NAN] {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| xo.set_frequency(f)))
                    .is_err()
            );
            assert_eq!(xo.frequency(), 1000.0);
        }
    }
}

// ============================================================================
// LR8 Crossover — Linkwitz-Riley 8th-order crossover utility
// ============================================================================

use crate::traits::{FilterFloat, lit};
use crate::{Biquad, BiquadFilterType};

/// Q values for a 4th-order Butterworth filter (used to build LR8).
///
/// LR8 = two cascaded Butterworth-4 filters, each needing two biquad sections
/// with these Q values. Total: 4 biquad sections per lowpass/highpass.
fn butterworth4_q_values<T: FilterFloat>() -> [T; 2] {
    // Q1 = 1 / (2 * sin(π/8))  ≈ 1.30656
    // Q2 = 1 / (2 * sin(3π/8)) ≈ 0.54120
    let pi = T::PI();
    let eight: T = lit(8.0);
    let two: T = lit(2.0);

    let q1 = T::one() / (two * (pi / eight).sin());
    let q2 = T::one() / (two * (lit::<T>(3.0) * pi / eight).sin());

    [q1, q2]
}

/// A single Linkwitz-Riley 8th-order crossover point.
///
/// Generic over [`FilterFloat`] (`f32` or `f64`, default `f64`).
///
/// Splits a signal into low and high bands with -48 dB/octave slopes.
/// The two outputs sum to unity (flat magnitude response) with zero
/// phase shift at the crossover frequency.
///
/// Constructed from two cascaded 4th-order Butterworth filters (4 biquad
/// sections each for lowpass and highpass).
#[derive(Debug, Clone)]
pub struct Lr8Crossover<T: FilterFloat = f64> {
    /// Four cascaded lowpass biquads per channel.
    lowpass: Vec<[Biquad<T>; 4]>,
    /// Four cascaded highpass biquads per channel.
    highpass: Vec<[Biquad<T>; 4]>,
    freq: T,
    sample_rate: T,
    channels: usize,
}

impl<T: FilterFloat> Lr8Crossover<T> {
    /// Create a new LR8 crossover at the given frequency.
    pub fn new(freq: T, sample_rate: T, channels: usize) -> Self {
        let [q1, q2] = butterworth4_q_values::<T>();

        let make_stages = |filter_type: BiquadFilterType| -> [Biquad<T>; 4] {
            [
                // First Butterworth-4: sections with q1, q2
                Biquad::new(filter_type, freq, sample_rate, q1, T::zero()),
                Biquad::new(filter_type, freq, sample_rate, q2, T::zero()),
                // Second Butterworth-4: sections with q1, q2
                Biquad::new(filter_type, freq, sample_rate, q1, T::zero()),
                Biquad::new(filter_type, freq, sample_rate, q2, T::zero()),
            ]
        };

        let lowpass: Vec<[Biquad<T>; 4]> = (0..channels)
            .map(|_| make_stages(BiquadFilterType::Lowpass))
            .collect();
        let highpass: Vec<[Biquad<T>; 4]> = (0..channels)
            .map(|_| make_stages(BiquadFilterType::Highpass))
            .collect();

        Self {
            lowpass,
            highpass,
            freq,
            sample_rate,
            channels,
        }
    }

    /// Process one sample for a given channel. Returns `(low, high)`.
    #[inline]
    pub fn process(&mut self, sample: T, channel: usize) -> (T, T) {
        let mut low = sample;
        for stage in &mut self.lowpass[channel] {
            low = stage.process(low);
        }

        let mut high = sample;
        for stage in &mut self.highpass[channel] {
            high = stage.process(high);
        }

        (low, high)
    }

    /// Process one interleaved frame. `input` has `channels` samples.
    /// `low` and `high` outputs each have `channels` samples.
    #[inline]
    pub fn process_frame(&mut self, input: &[T], low: &mut [T], high: &mut [T]) {
        debug_assert_eq!(input.len(), self.channels);
        debug_assert_eq!(low.len(), self.channels);
        debug_assert_eq!(high.len(), self.channels);
        for ch in 0..self.channels {
            let (l, h) = self.process(input[ch], ch);
            low[ch] = l;
            high[ch] = h;
        }
    }

    /// Update the crossover frequency. Recomputes all filter coefficients
    /// without resetting filter state (click-free).
    pub fn set_frequency(&mut self, freq: T) {
        if (freq - self.freq).abs() < lit(0.1) {
            return;
        }
        self.freq = freq;
        let [q1, q2] = butterworth4_q_values::<T>();
        let qs = [q1, q2, q1, q2];

        for ch in 0..self.channels {
            for (stage, &q) in qs.iter().enumerate() {
                self.lowpass[ch][stage].update_params(
                    BiquadFilterType::Lowpass,
                    freq,
                    self.sample_rate,
                    q,
                    T::zero(),
                );
                self.highpass[ch][stage].update_params(
                    BiquadFilterType::Highpass,
                    freq,
                    self.sample_rate,
                    q,
                    T::zero(),
                );
            }
        }
    }

    /// Reset all filter states to zero without reallocating channel storage.
    pub fn reset(&mut self) {
        let [q1, q2] = butterworth4_q_values::<T>();
        let qs = [q1, q2, q1, q2];
        for ch in 0..self.channels {
            for (stage, &q) in qs.iter().enumerate() {
                self.lowpass[ch][stage] = Biquad::new(
                    BiquadFilterType::Lowpass,
                    self.freq,
                    self.sample_rate,
                    q,
                    T::zero(),
                );
                self.highpass[ch][stage] = Biquad::new(
                    BiquadFilterType::Highpass,
                    self.freq,
                    self.sample_rate,
                    q,
                    T::zero(),
                );
            }
        }
    }

    /// Reset state and set the exact cutoff, including changes below the live-update threshold.
    pub fn reset_at_frequency(&mut self, freq: T) {
        self.freq = freq;
        self.reset();
    }

    /// Re-initialize for a new sample rate and/or channel count.
    pub fn reinit(&mut self, freq: T, sample_rate: T, channels: usize) {
        *self = Self::new(freq, sample_rate, channels);
    }

    /// Get the crossover frequency.
    pub fn frequency(&self) -> T {
        self.freq
    }

    /// Get the number of channels.
    pub fn channels(&self) -> usize {
        self.channels
    }
}

/// A multi-band LR8 crossover that splits a signal into N bands.
///
/// Generic over [`FilterFloat`] (`f32` or `f64`, default `f64`).
/// Uses N-1 crossover points to create N frequency bands.
#[derive(Debug, Clone)]
pub struct MultibandLr8Crossover<T: FilterFloat = f64> {
    crossovers: Vec<Lr8Crossover<T>>,
    scratch: Vec<T>,
    carry: Vec<T>,
}

impl<T: FilterFloat> MultibandLr8Crossover<T> {
    /// Create a multiband crossover with the given crossover frequencies.
    ///
    /// Frequencies must be sorted in ascending order. Creates `freqs.len() + 1` bands.
    pub fn new(freqs: &[T], sample_rate: T, channels: usize) -> Self {
        let crossovers = freqs
            .iter()
            .map(|&f| Lr8Crossover::new(f, sample_rate, channels))
            .collect();
        Self {
            crossovers,
            scratch: vec![T::zero(); channels],
            carry: vec![T::zero(); channels],
        }
    }

    /// Process one interleaved frame into bands.
    ///
    /// `input` has `channels` samples.
    /// `bands` is a slice of band outputs, each having `channels` samples.
    /// `bands.len()` must equal `num_bands()`.
    pub fn process_frame(&mut self, input: &[T], bands: &mut [&mut [T]]) {
        debug_assert_eq!(bands.len(), self.num_bands());

        if self.crossovers.is_empty() {
            bands[0].copy_from_slice(input);
            return;
        }

        self.crossovers[0].process_frame(input, bands[0], &mut self.scratch);

        #[allow(clippy::needless_range_loop)]
        for i in 1..self.crossovers.len() {
            self.carry.copy_from_slice(&self.scratch);
            self.crossovers[i].process_frame(&self.carry, bands[i], &mut self.scratch);
        }

        let last = bands.len() - 1;
        bands[last].copy_from_slice(&self.scratch);
    }

    /// Number of output bands.
    pub fn num_bands(&self) -> usize {
        self.crossovers.len() + 1
    }

    /// Update a crossover frequency by index.
    pub fn set_frequency(&mut self, index: usize, freq: T) {
        if index < self.crossovers.len() {
            self.crossovers[index].set_frequency(freq);
        }
    }

    /// Reset all crossover filter states.
    pub fn reset(&mut self) {
        for xo in &mut self.crossovers {
            xo.reset();
        }
    }

    /// Reset every crossover at exact frequencies while preserving allocated storage.
    ///
    /// # Panics
    /// Panics if the frequency count differs from the prepared crossover count.
    pub fn reset_at_frequencies(&mut self, frequencies: &[T]) {
        assert_eq!(
            frequencies.len(),
            self.crossovers.len(),
            "reset frequency count must match prepared crossovers"
        );
        for (crossover, &frequency) in self.crossovers.iter_mut().zip(frequencies) {
            crossover.reset_at_frequency(frequency);
        }
        self.scratch.fill(T::zero());
        self.carry.fill(T::zero());
    }

    /// Re-initialize for new frequencies, sample rate, and/or channel count.
    pub fn reinit(&mut self, freqs: &[T], sample_rate: T, channels: usize) {
        *self = Self::new(freqs, sample_rate, channels);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lr8_basic() {
        let mut xo = Lr8Crossover::new(1000.0, 48000.0, 1);
        for i in 0..1000 {
            let sample = (i as f64 * 0.1).sin();
            let (low, high) = xo.process(sample, 0);
            assert!(low.is_finite());
            assert!(high.is_finite());
        }
    }

    #[test]
    fn test_lr8_unity_sum_dc() {
        let mut xo = Lr8Crossover::new(1000.0, 48000.0, 1);
        let mut sum_low = 0.0f64;
        let mut sum_high = 0.0f64;
        let n = 10000;
        for _ in 0..n {
            let (low, high) = xo.process(1.0, 0);
            sum_low += low;
            sum_high += high;
        }
        let avg_low = sum_low / n as f64;
        let avg_high = sum_high / n as f64;
        assert!(avg_low > 0.9, "avg_low: {}", avg_low);
        assert!(avg_high.abs() < 0.1, "avg_high: {}", avg_high);
    }

    #[test]
    fn test_lr8_steeper_than_lr4() {
        // LR8 should attenuate a signal far from the crossover frequency
        // more than LR4 does. Test with a high-frequency sine through
        // the lowpass half at a low crossover point.
        let freq = 500.0;
        let sr = 48000.0;
        let mut lr4 = crate::Lr4Crossover::new(freq, sr, 1);
        let mut lr8 = Lr8Crossover::new(freq, sr, 1);

        // Generate a 10 kHz tone (well above 500 Hz crossover)
        let tone_freq = 10000.0;
        let n = 4000;
        let mut lr4_energy = 0.0f64;
        let mut lr8_energy = 0.0f64;

        for i in 0..n {
            let t = i as f64 / sr;
            let sample = (2.0 * std::f64::consts::PI * tone_freq * t).sin();

            let (low4, _) = lr4.process(sample, 0);
            let (low8, _) = lr8.process(sample, 0);

            // Skip transient (first 500 samples)
            if i > 500 {
                lr4_energy += low4 * low4;
                lr8_energy += low8 * low8;
            }
        }

        assert!(
            lr8_energy < lr4_energy,
            "LR8 lowpass should attenuate more than LR4: lr8={} lr4={}",
            lr8_energy,
            lr4_energy
        );
    }

    #[test]
    fn test_multiband_three_bands() {
        let mut mb = MultibandLr8Crossover::new(&[500.0, 5000.0], 48000.0, 1);
        assert_eq!(mb.num_bands(), 3);

        let mut band0 = [0.0f64; 1];
        let mut band1 = [0.0f64; 1];
        let mut band2 = [0.0f64; 1];

        for i in 0..10000 {
            let sample = (i as f64 * 0.01).sin();
            let input = [sample];
            mb.process_frame(
                &input,
                &mut [&mut band0[..], &mut band1[..], &mut band2[..]],
            );
        }
        assert!(band0[0].is_finite());
        assert!(band1[0].is_finite());
        assert!(band2[0].is_finite());
    }

    #[test]
    fn test_set_frequency() {
        let mut xo = Lr8Crossover::new(1000.0_f64, 48000.0, 2);
        xo.set_frequency(2000.0);
        assert!((xo.frequency() - 2000.0).abs() < 0.1);
    }

    #[test]
    fn test_reset() {
        let mut xo = Lr8Crossover::new(1000.0_f64, 48000.0, 1);
        for _ in 0..100 {
            xo.process(1.0, 0);
        }
        xo.reset();
        let (low, high) = xo.process(0.0, 0);
        assert!(low.abs() < 0.001);
        assert!(high.abs() < 0.001);
    }

    #[test]
    fn test_lr8_f32() {
        let mut xo = Lr8Crossover::<f32>::new(1000.0, 48000.0, 1);
        let (low, high) = xo.process(1.0f32, 0);
        assert!(low.is_finite());
        assert!(high.is_finite());
    }

    #[test]
    fn reset_at_sub_threshold_frequency_matches_fresh_crossover() {
        let target = 1000.05_f32;
        let mut reset = Lr8Crossover::new(1000.0_f32, 48_000.0, 2);
        for _ in 0..256 {
            reset.process(0.4, 0);
            reset.process(-0.3, 1);
        }
        reset.reset_at_frequency(target);
        let mut fresh = Lr8Crossover::new(target, 48_000.0, 2);
        assert_eq!(reset.frequency(), target);
        for frame in 0..128 {
            let sample = (frame as f32 * 0.073).sin();
            for channel in 0..2 {
                assert_eq!(
                    reset.process(sample, channel),
                    fresh.process(sample, channel)
                );
            }
        }
    }
}

#[cfg(test)]
mod reset_tests {
    use super::*;

    fn check_reset<T: FilterFloat>() {
        for rate in [44_100.0, 48_000.0, 96_000.0] {
            let initial: T = lit(1_000.0);
            let requested: T = lit(1_000.000_5);
            let mut actual = Lr8Crossover::new(initial, lit(rate), 2);
            for index in 0..128 {
                actual.process(lit((index as f64 * 0.071).sin()), index % 2);
            }
            let low_storage = actual.lowpass.as_ptr();
            let high_storage = actual.highpass.as_ptr();
            actual.reset_at_frequency(requested);
            assert_eq!(actual.lowpass.as_ptr(), low_storage);
            assert_eq!(actual.highpass.as_ptr(), high_storage);
            assert_eq!(actual.frequency(), requested);
            let mut expected = Lr8Crossover::new(requested, lit(rate), 2);
            for index in 0..512 {
                let sample = lit((index as f64 * 0.037).cos());
                assert_eq!(
                    actual.process(sample, index % 2),
                    expected.process(sample, index % 2)
                );
            }
            actual.reset();
            expected.reset_at_frequency(requested);
            assert_eq!(actual.lowpass.as_ptr(), low_storage);
            assert_eq!(actual.highpass.as_ptr(), high_storage);
            assert_eq!(actual.process(T::one(), 0), expected.process(T::one(), 0));
        }
    }

    #[test]
    fn exact_reset_replays_fresh_filters_and_preserves_storage() {
        check_reset::<f32>();
        check_reset::<f64>();
    }

    #[test]
    fn multiband_reset_replays_fresh_filters_and_preserves_storage() {
        let mut actual = MultibandLr8Crossover::<f32>::new(&[300.0, 3_000.0], 48_000.0, 2);
        let mut a = [[0.0; 2]; 3];
        for _ in 0..128 {
            actual.process_frame(
                &[0.25, -0.3],
                &mut a
                    .iter_mut()
                    .map(|band| band.as_mut_slice())
                    .collect::<Vec<_>>(),
            );
        }
        let storage = (
            actual.crossovers.as_ptr(),
            actual.scratch.as_ptr(),
            actual.carry.as_ptr(),
        );
        let frequencies = [301.0, 2_999.0];
        actual.reset_at_frequencies(&frequencies);
        assert_eq!(
            storage,
            (
                actual.crossovers.as_ptr(),
                actual.scratch.as_ptr(),
                actual.carry.as_ptr()
            )
        );
        let mut expected = MultibandLr8Crossover::new(&frequencies, 48_000.0, 2);
        let mut b = [[0.0; 2]; 3];
        for index in 0..512 {
            let input = [(index as f32 * 0.037).sin(), (index as f32 * 0.071).cos()];
            actual.process_frame(
                &input,
                &mut a
                    .iter_mut()
                    .map(|band| band.as_mut_slice())
                    .collect::<Vec<_>>(),
            );
            expected.process_frame(
                &input,
                &mut b
                    .iter_mut()
                    .map(|band| band.as_mut_slice())
                    .collect::<Vec<_>>(),
            );
            assert_eq!(a, b);
        }
    }

    #[test]
    #[should_panic(expected = "reset frequency count")]
    fn multiband_reset_requires_the_prepared_frequency_count() {
        MultibandLr8Crossover::<f64>::new(&[300.0, 3_000.0], 48_000.0, 1)
            .reset_at_frequencies(&[301.0]);
    }
}

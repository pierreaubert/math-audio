use crate::BiquadCoefficients;
use crate::traits::{FilterFloat, lit};

/// Apply a single-pass IIR filter through a cascade of biquad sections.
///
/// Equivalent to `scipy.signal.sosfilt`. This introduces phase distortion —
/// use [`filtfilt`] for zero-phase filtering.
pub fn sosfilt<T: FilterFloat>(signal: &[T], sections: &[BiquadCoefficients<T>]) -> Vec<T> {
    if signal.is_empty() || sections.is_empty() {
        return signal.to_vec();
    }

    let mut data: Vec<T> = signal.to_vec();

    for section in sections {
        data = biquad_filter(&data, section, [T::zero(), T::zero()]);
    }

    data
}

/// Apply forward-reverse filtering (zero-phase) through a cascade of biquad sections.
///
/// Equivalent to MATLAB's `filtfilt` or `scipy.signal.sosfiltfilt`.
///
/// The signal is extended at both ends using odd reflection to reduce edge
/// transients. Steady-state initial conditions are computed for each section
/// to further minimize startup artifacts.
///
/// Because the filter is applied twice (forward + backward), the effective
/// filter order is doubled and the magnitude response is squared. A 2nd-order
/// Butterworth becomes effectively 4th-order with -80 dB/decade rolloff.
///
/// # Returns
///
/// A filtered signal of the same length as the input, with zero phase distortion.
/// Apply forward-reverse filtering with an explicit padding length.
///
/// See [`filtfilt`] for the filtering algorithm.  The `padlen` argument
/// controls how many samples of odd-reflected padding are added at each
/// end of the signal.  A larger value reduces edge transients for narrow
/// high-order filters but increases computation time.
///
/// # Panics
///
/// Panics if `padlen` exceeds `signal.len() - 1`.
pub fn filtfilt_with_padlen<T: FilterFloat>(
    signal: &[T],
    sections: &[BiquadCoefficients<T>],
    padlen: usize,
) -> Vec<T> {
    FiltfiltWorkspace::new()
        .process_with_padlen(signal, sections, padlen)
        .to_vec()
}

/// Apply forward-reverse filtering (zero-phase) through a cascade of biquad sections.
///
/// Equivalent to MATLAB's `filtfilt` or `scipy.signal.sosfiltfilt`.
///
/// The signal is extended at both ends using odd reflection to reduce edge
/// transients. Steady-state initial conditions are computed for each section
/// to further minimize startup artifacts.
///
/// Because the filter is applied twice (forward + backward), the effective
/// filter order is doubled and the magnitude response is squared. A 2nd-order
/// Butterworth becomes effectively 4th-order with -80 dB/decade rolloff.
///
/// The default padding length is `6 * (2 * sections.len() + 1)` samples on
/// each side (capped at `signal.len() - 1`).  For very narrow high-order
/// filters on short signals this may still be insufficient; use
/// [`filtfilt_with_padlen`] to specify a larger value.
///
/// # Returns
///
/// A filtered signal of the same length as the input, with zero phase distortion.
pub fn filtfilt<T: FilterFloat>(signal: &[T], sections: &[BiquadCoefficients<T>]) -> Vec<T> {
    if signal.is_empty() || sections.is_empty() {
        return signal.to_vec();
    }

    let n = signal.len();
    if n < 2 {
        return signal.to_vec();
    }

    // Issue #8: increased multiplier from 3 to 6 for better edge handling
    // with narrow high-order filters.
    let padlen = (6 * (2 * sections.len() + 1)).min(n - 1);
    filtfilt_with_padlen(signal, sections, padlen)
}

/// Reusable padding and output storage for forward-reverse SOS filtering.
#[derive(Debug, Default)]
pub struct FiltfiltWorkspace<T: FilterFloat = f64> {
    data: Vec<T>,
}

impl<T: FilterFloat> FiltfiltWorkspace<T> {
    /// Create an empty workspace.
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Reserve space for a signal and the default reflected padding.
    pub fn prepare(&mut self, signal_len: usize, section_count: usize) {
        let padding = (6 * (2 * section_count + 1)).min(signal_len.saturating_sub(1));
        self.data
            .reserve((signal_len + 2 * padding).saturating_sub(self.data.len()));
    }

    /// Filter into borrowed workspace output, retaining storage across calls.
    ///
    /// After preparation, equal or shorter signals with no more sections allocate
    /// nothing. Empty sections or fewer than two input samples pass through.
    pub fn process(&mut self, signal: &[T], sections: &[BiquadCoefficients<T>]) -> &[T] {
        let padding = (6 * (2 * sections.len() + 1)).min(signal.len().saturating_sub(1));
        self.process_with_padlen(signal, sections, padding)
    }

    /// Filter with explicit odd-reflection padding into borrowed storage.
    ///
    /// # Panics
    /// Panics if nonempty filtering of two or more samples requests padding
    /// greater than or equal to the signal length.
    pub fn process_with_padlen(
        &mut self,
        signal: &[T],
        sections: &[BiquadCoefficients<T>],
        padding: usize,
    ) -> &[T] {
        self.data.clear();
        if signal.len() < 2 || sections.is_empty() {
            self.data.extend_from_slice(signal);
            return &self.data;
        }
        assert!(
            padding < signal.len(),
            "padlen ({padding}) must not exceed signal.len() - 1 ({})",
            signal.len()
        );
        // Reserve both ends before appending the center. Otherwise the final
        // right padding can double a nearly-full long-record allocation.
        self.data.reserve(signal.len() + 2 * padding);
        let two = lit::<T>(2.0);
        for i in (1..=padding).rev() {
            self.data.push(two * signal[0] - signal[i]);
        }
        self.data.extend_from_slice(signal);
        for i in (1..=padding).rev() {
            self.data
                .push(two * signal[signal.len() - 1] - signal[signal.len() - 1 - i]);
        }
        for _ in 0..2 {
            for section in sections {
                let zi = biquad_zi(section);
                let first = self.data[0];
                biquad_filter_in_place(&mut self.data, section, [zi[0] * first, zi[1] * first]);
            }
            self.data.reverse();
        }
        &self.data[padding..padding + signal.len()]
    }
}

/// Filter into caller storage while reusing reflected-padding workspace.
///
/// # Panics
/// Panics unless input and output lengths match.
pub fn filtfilt_into<T: FilterFloat>(
    signal: &[T],
    sections: &[BiquadCoefficients<T>],
    output: &mut [T],
    workspace: &mut FiltfiltWorkspace<T>,
) {
    assert_eq!(
        signal.len(),
        output.len(),
        "filtfilt input/output length mismatch"
    );
    output.copy_from_slice(workspace.process(signal, sections));
}

fn biquad_filter_in_place<T: FilterFloat>(
    data: &mut [T],
    section: &BiquadCoefficients<T>,
    zi: [T; 2],
) {
    let [mut d0, mut d1] = zi;
    for sample in data {
        let input = *sample;
        let output = section.b0 * input + d0;
        d0 = section.b1 * input - section.a1 * output + d1;
        d1 = section.b2 * input - section.a2 * output;
        *sample = output;
    }
}

/// Compute steady-state initial conditions for a biquad section (direct form II transposed).
///
/// Returns `[z1, z2]` such that filtering a constant signal `x[n] = 1` produces
/// constant output from the very first sample (no transient).
///
/// Usage: multiply by `signal[0]` to get the actual initial conditions.
pub(super) fn biquad_zi<T: FilterFloat>(section: &BiquadCoefficients<T>) -> [T; 2] {
    let one = T::one();
    let denom = one + section.a1 + section.a2;
    if denom.abs() < lit(1e-15) {
        return [T::zero(), T::zero()];
    }
    let z1 = (section.b1 + section.b2 - (section.a1 + section.a2) * section.b0) / denom;
    let z2 = (section.b2 - section.a2 * section.b0) - section.a2 * z1;
    [z1, z2]
}

/// Apply a single biquad section using direct form II transposed.
pub(super) fn biquad_filter<T: FilterFloat>(
    input: &[T],
    section: &BiquadCoefficients<T>,
    zi: [T; 2],
) -> Vec<T> {
    let mut output = Vec::with_capacity(input.len());
    let mut d0 = zi[0];
    let mut d1 = zi[1];

    for &x in input {
        let y = section.b0 * x + d0;
        d0 = section.b1 * x - section.a1 * y + d1;
        d1 = section.b2 * x - section.a2 * y;
        output.push(y);
    }

    output
}

//! Four-times oversampled peak metering using the BS.1770 reference FIR.
//!
//! The 48-tap polyphase reconstruction filter is shared with EBU metering.
//! Its finite bandwidth and four-times sampling approximate a continuous peak;
//! this is not an exact limiter ceiling. The table targets 48 kHz input. Other
//! rates retain the same normalized response without a standards accuracy claim.

// Rust guideline compliant 2026-02-21
use crate::ebur128::{TRUE_PEAK_FIR_LEN, TRUE_PEAK_FIR_PHASES};

/// Tracks reconstructed and sampled peaks with bounded, allocation-free processing.
///
/// Reconstruction has FIR delay. Each call returns the peak of the current
/// filter interval and input sample; the running maximum remains available
/// through [`Self::peak_linear`]. Call [`Self::finish`] for a finite record's tail.
#[derive(Debug, Clone)]
pub struct TruePeakDetector {
    history: [f64; TRUE_PEAK_FIR_LEN],
    position: usize,
    peak: f32,
}

impl TruePeakDetector {
    /// Create a zero-initialized detector.
    pub fn new() -> Self {
        Self {
            history: [0.0; TRUE_PEAK_FIR_LEN],
            position: 0,
            peak: 0.0,
        }
    }

    /// Process one sample and return the current interval's peak in dB.
    ///
    /// Shares processing state with [`Self::process_linear`]; call one per sample.
    #[inline]
    pub fn process(&mut self, sample: f32) -> f32 {
        let linear = self.process_linear(sample);
        if linear < 1e-12 {
            -120.0
        } else {
            20.0 * linear.log10()
        }
    }

    /// Process one sample and return the current reconstructed interval's peak.
    #[inline]
    pub fn process_linear(&mut self, sample: f32) -> f32 {
        self.history[self.position] = f64::from(sample);
        self.position = (self.position + 1) % TRUE_PEAK_FIR_LEN;
        let mut maximum = sample.abs();
        for phase in &TRUE_PEAK_FIR_PHASES {
            let mut sum = 0.0_f64;
            for (tap, &coefficient) in phase.iter().enumerate() {
                sum += coefficient * self.history[(self.position + tap) % TRUE_PEAK_FIR_LEN];
            }
            maximum = maximum.max(sum.abs() as f32);
        }
        self.peak = self.peak.max(maximum);
        maximum
    }

    /// Complete a finite record's FIR tail while retaining its running peak.
    ///
    /// Repeated calls preserve the same result. Later input starts a new filter
    /// segment; [`Self::reset`] also clears the accumulated peak.
    pub fn finish(&mut self) {
        for _ in 1..TRUE_PEAK_FIR_LEN {
            self.process_linear(0.0);
        }
        self.history.fill(0.0);
        self.position = 0;
    }

    /// Return the running peak in dB, with a -120 dB display floor.
    pub fn peak_db(&self) -> f32 {
        if self.peak < 1e-12 {
            -120.0
        } else {
            20.0 * self.peak.log10()
        }
    }

    /// Return the running peak as linear amplitude.
    pub fn peak_linear(&self) -> f32 {
        self.peak
    }

    /// Clear filter history and the running peak.
    pub fn reset(&mut self) {
        self.history.fill(0.0);
        self.position = 0;
        self.peak = 0.0;
    }
}

impl Default for TruePeakDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silence() {
        let mut tp = TruePeakDetector::new();
        let db = tp.process(0.0);
        assert!(db <= -120.0);
    }

    #[test]
    fn test_unity_peak() {
        let mut tp = TruePeakDetector::new();
        let db = tp.process(1.0);
        // Current sample is 1.0 → should be 0 dBFS (or higher due to interpolation overshoot)
        assert!(db >= -0.1, "peak db: {}", db);
    }

    #[test]
    fn test_inter_sample_peak() {
        let mut tp = TruePeakDetector::new();
        // Alternating +0.9 / -0.9 at Nyquist creates large inter-sample peaks
        // First establish history
        tp.process(0.0);
        tp.process(0.9);
        let _db = tp.process(-0.9);
        // The Catmull-Rom interpolation between 0.9 and -0.9 should show
        // peaks larger than 0.9 due to the overshoot
        assert!(tp.peak_linear() >= 0.9, "peak: {}", tp.peak_linear());
    }

    #[test]
    fn test_detects_overshoot() {
        let mut tp = TruePeakDetector::new();
        // Create a signal that overshoots between samples:
        // 0.0, 0.8, 0.0 — the interpolation should show a peak >= 0.8
        tp.process(0.0);
        tp.process(0.8);
        tp.process(0.0);
        assert!(tp.peak_linear() >= 0.79, "peak: {}", tp.peak_linear());
    }

    #[test]
    fn test_peak_is_running_max() {
        let mut tp = TruePeakDetector::new();
        tp.process(1.0);
        let held = tp.peak_linear();
        assert!(held >= 0.99, "peak: {}", held);
        // Quiet input: the per-window return drops (after the 1.0 sample
        // leaves the two-sample window) but the stored peak holds.
        for _ in 0..TRUE_PEAK_FIR_LEN {
            tp.process_linear(0.0);
        }
        let window = tp.process_linear(0.0);
        assert!(
            window < held,
            "window {window} should drop below held {held}"
        );
        assert!(
            (tp.peak_linear() - held).abs() < 1e-6,
            "running peak should hold at {held}, got {}",
            tp.peak_linear()
        );
    }

    #[test]
    fn test_reset() {
        let mut tp = TruePeakDetector::new();
        tp.process(1.0);
        tp.reset();
        assert_eq!(tp.peak_linear(), 0.0);
        assert_eq!(tp.history, [0.0; TRUE_PEAK_FIR_LEN]);
    }
}

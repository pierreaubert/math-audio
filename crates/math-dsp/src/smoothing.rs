// ============================================================================
// Parameter Smoothing
// ============================================================================

/// Simple one-pole smoothing filter for control parameters to prevent zipper noise.
#[derive(Debug, Clone, Copy)]
pub struct Smoother {
    target: f32,
    current: f32,
    coeff: f32,
}

#[allow(dead_code)]
impl Smoother {
    /// Create a new smoother
    /// time_ms: Smoothing time constant (e.g., 10ms - 50ms)
    pub fn new(value: f32, time_ms: f32, sample_rate: impl Into<f64>) -> Self {
        let coeff = Self::calculate_coeff(time_ms, sample_rate.into());
        Self {
            target: value,
            current: value,
            coeff,
        }
    }

    fn calculate_coeff(time_ms: f32, sample_rate: f64) -> f32 {
        if time_ms <= 0.0 || !sample_rate.is_finite() || sample_rate <= 0.0 {
            0.0
        } else {
            // Standard one-pole coeff: e^(-1 / (tau * fs))
            // time_ms is roughly time to reach ~63% of target
            (-1.0_f64 / (f64::from(time_ms) * 0.001 * sample_rate)).exp() as f32
        }
    }

    pub fn set_time(&mut self, time_ms: f32, sample_rate: impl Into<f64>) {
        self.coeff = Self::calculate_coeff(time_ms, sample_rate.into());
    }

    /// Set new target value
    pub fn set_target(&mut self, value: f32) {
        self.target = value;
        // If smoothing is disabled (coeff = 0), jump immediately
        if self.coeff == 0.0 {
            self.current = value;
        }
    }

    /// Process N sample steps at once (updates current value)
    #[inline]
    pub fn next_n(&mut self, n: usize) -> f32 {
        if self.coeff == 0.0 || (self.current - self.target).abs() < 1e-5 || n == 0 {
            self.current = self.target;
        } else {
            // current = target + coeff^n * (current - target)
            let block_coeff = self.coeff.powi(n as i32);
            self.current = self.target + block_coeff * (self.current - self.target);
        }
        self.current
    }

    /// Process one sample step (updates current value)
    #[inline]
    pub fn advance(&mut self) -> f32 {
        self.next_n(1)
    }

    /// Get current smoothed value
    #[inline]
    pub fn current(&self) -> f32 {
        self.current
    }

    /// Get target value
    #[inline]
    pub fn target(&self) -> f32 {
        self.target
    }

    /// Process one sample step (updates current value) - per-sample smoothing
    /// Returns the smoothed value for this sample
    #[inline]
    pub fn process_sample(&mut self, sample: f32) -> f32 {
        // Apply smoothing to parameter changes, then process input with smoothed gain
        // This gives smooth parameter transitions AND smooth gain application
        if (self.current - self.target).abs() < 1e-5 {
            self.current = self.target;
        } else {
            self.current = self.target + self.coeff * (self.current - self.target);
        }
        sample * self.current
    }

    /// Reset to value immediately
    pub fn reset(&mut self, value: f32) {
        self.target = value;
        self.current = value;
    }
}

/// Linear smoother for control parameters.
/// Changes the value by a constant amount per sample.
#[derive(Debug, Clone, Copy)]
pub struct LinearSmoother {
    target: f32,
    current: f32,
    step: f32,
    sample_rate: f64,
    time_ms: f32,
}

impl LinearSmoother {
    pub fn new(value: f32, time_ms: f32, sample_rate: impl Into<f64>) -> Self {
        Self {
            target: value,
            current: value,
            step: 0.0,
            sample_rate: sample_rate.into(),
            time_ms,
        }
    }

    pub fn set_target(&mut self, value: f32) {
        self.target = value;
        if self.time_ms <= 0.0 {
            self.current = value;
            self.step = 0.0;
        } else {
            let samples = (f64::from(self.time_ms) * 0.001 * self.sample_rate).max(1.0);
            self.step = (f64::from(self.target - self.current) / samples) as f32;
        }
    }

    #[inline]
    pub fn advance(&mut self) -> f32 {
        self.next_n(1)
    }

    #[inline]
    pub fn next_n(&mut self, n: usize) -> f32 {
        if n == 0 {
            return self.current;
        }
        let total_step = self.step * n as f32;
        if (self.current - self.target).abs() <= total_step.abs() {
            self.current = self.target;
            self.step = 0.0;
        } else {
            self.current += total_step;
        }
        self.current
    }

    #[allow(dead_code)]
    pub fn reset(&mut self, value: f32) {
        self.target = value;
        self.current = value;
        self.step = 0.0;
    }

    #[allow(dead_code)]
    pub fn current(&self) -> f32 {
        self.current
    }

    pub fn target(&self) -> f32 {
        self.target
    }
}

/// Logarithmic smoother for frequency or dB parameters.
/// Changes the value by a constant ratio per sample.
#[derive(Debug, Clone, Copy)]
pub struct LogSmoother {
    target: f32,
    current: f32,
    ratio: f32,
    sample_rate: f64,
    time_ms: f32,
}

impl LogSmoother {
    pub fn new(value: f32, time_ms: f32, sample_rate: impl Into<f64>) -> Self {
        Self {
            target: value.max(1e-7),
            current: value.max(1e-7),
            ratio: 1.0,
            sample_rate: sample_rate.into(),
            time_ms,
        }
    }

    pub fn set_target(&mut self, value: f32) {
        self.target = value.max(1e-7);
        if self.time_ms <= 0.0 {
            self.current = self.target;
            self.ratio = 1.0;
        } else {
            let samples = (f64::from(self.time_ms) * 0.001 * self.sample_rate).max(1.0);
            // target = current * ratio^samples  => ratio = (target/current)^(1/samples)
            self.ratio = (f64::from(self.target / self.current).powf(1.0 / samples)) as f32;
        }
    }

    #[inline]
    pub fn advance(&mut self) -> f32 {
        self.next_n(1)
    }

    #[inline]
    pub fn next_n(&mut self, n: usize) -> f32 {
        if self.ratio == 1.0 || n == 0 {
            return self.current;
        }

        let new_log = self.current.ln() + self.ratio.ln() * n as f32;
        self.current = new_log.exp().clamp(1e-7, 1e6);

        // Check if we passed the target
        if (self.ratio > 1.0 && self.current >= self.target)
            || (self.ratio < 1.0 && self.current <= self.target)
        {
            self.current = self.target;
            self.ratio = 1.0;
        }

        self.current
    }

    #[allow(dead_code)]
    pub fn reset(&mut self, value: f32) {
        self.target = value.max(1e-7);
        self.current = self.target;
        self.ratio = 1.0;
    }

    #[allow(dead_code)]
    pub fn current(&self) -> f32 {
        self.current
    }

    pub fn target(&self) -> f32 {
        self.target
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_rates_track_the_previous_f32_coefficient_formula() {
        for rate in [1_000_u32, 48_000, 192_000] {
            for time_ms in [0.1_f32, 10.0, 1_000.0] {
                let old_coeff = (-1.0_f32 / (time_ms * 0.001 * rate as f32)).exp();
                let new_coeff = Smoother::calculate_coeff(time_ms, f64::from(rate));
                // Both values are f32 coefficients. The new calculation carries
                // the rate in f64 but keeps the previous one-pole response.
                assert!((new_coeff - old_coeff).abs() <= 2.0 * f32::EPSILON);
            }
        }
    }

    #[test]
    fn fractional_rate_changes_the_smoothing_response() {
        let integral = Smoother::calculate_coeff(1.0, 1_000.0);
        let fractional = Smoother::calculate_coeff(1.0, 1_000.25);
        assert!(fractional > integral);
        assert!((fractional - integral) > 1e-5);
    }

    #[test]
    fn test_exponential_smoother() {
        let mut s = Smoother::new(0.0, 10.0, 1000); // 10ms at 1kHz = 10 samples
        s.set_target(1.0);

        let first = s.advance();
        assert!(first > 0.0 && first < 1.0);

        for _ in 0..100 {
            s.advance();
        }
        assert!((s.current() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_linear_smoother() {
        let mut s = LinearSmoother::new(0.0, 10.0, 1000); // 10 samples
        s.set_target(1.0);

        assert!((s.advance() - 0.1).abs() < 1e-6);
        assert!((s.advance() - 0.2).abs() < 1e-6);

        for _ in 0..7 {
            s.advance();
        }
        assert!((s.advance() - 1.0).abs() < 1e-6);
        assert!((s.advance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn linear_smoother_uses_fractional_rate_for_elapsed_frames() {
        let mut smoother = LinearSmoother::new(0.0, 1_000.0, 1_234.567_8);
        smoother.set_target(1.0);
        let first = smoother.advance();
        let expected = (1.0 / 1_234.567_8_f64) as f32;
        assert!((first - expected).abs() <= f32::EPSILON);
        let truncated_rate = (1.0 / 1_234.0_f64) as f32;
        assert_ne!(first, truncated_rate);
    }

    #[test]
    fn test_log_smoother() {
        let mut s = LogSmoother::new(100.0, 10.0, 1000); // 10 samples
        s.set_target(1000.0);

        let first = s.advance();
        // 100 * (1000/100)^(1/10) = 100 * 10^0.1 approx 125.89
        assert!((first - 125.89).abs() < 0.1);

        for _ in 0..8 {
            s.advance();
        }
        assert!((s.advance() - 1000.0).abs() < 1e-3);
        assert!((s.advance() - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn log_smoother_uses_fractional_rate_for_elapsed_frames() {
        let mut smoother = LogSmoother::new(1.0, 1_000.0, 1_234.567_8);
        smoother.set_target(2.0);
        let first = smoother.advance();
        let expected = (2.0_f64.powf(1.0 / 1_234.567_8)) as f32;
        assert!((first - expected).abs() <= 2.0 * f32::EPSILON);
        let truncated_rate = (2.0_f64.powf(1.0 / 1_234.0)) as f32;
        assert_ne!(first, truncated_rate);
    }

    #[test]
    fn test_log_smoother_large_block_stays_finite() {
        let mut s = LogSmoother::new(1e-7, 20.0, 48000);
        s.set_target(1e6);

        let value = s.next_n(4096);
        assert!(
            value.is_finite(),
            "large-block log smoothing must not overflow to inf"
        );
        assert!(
            value <= 1e6,
            "large-block log smoothing should clamp at target range, got {value}"
        );
    }

    #[test]
    fn test_smoother_sample_rate_zero_no_panic() {
        // sample_rate=0 should produce coeff=0 (instant jump, no smoothing)
        let mut s = Smoother::new(1.0, 50.0, 0);
        assert_eq!(s.current(), 1.0);
        s.set_target(2.0);
        // With coeff=0, set_target jumps immediately
        assert_eq!(s.current(), 2.0);
        // advance should also be stable
        let val = s.advance();
        assert_eq!(val, 2.0);
        assert!(!val.is_nan());
        assert!(!val.is_infinite());
    }

    #[test]
    fn test_smoother_set_time_sample_rate_zero() {
        let mut s = Smoother::new(1.0, 10.0, 48000);
        s.set_time(10.0, 0);
        s.set_target(5.0);
        // coeff=0 → instant jump
        assert_eq!(s.current(), 5.0);
    }
}

use std::f32::consts::TAU;

use crate::process::flush_denormal;

/// A first-order DC blocker used as a final safety stage for coloration.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DcBlocker {
    coefficient: f32,
    x_previous: f32,
    y_previous: f32,
}

impl DcBlocker {
    pub(crate) fn new(sample_rate: f32) -> Self {
        let coefficient = (-TAU * 20.0 / sample_rate).exp();
        Self {
            coefficient,
            x_previous: 0.0,
            y_previous: 0.0,
        }
    }

    #[inline]
    pub(crate) fn process(&mut self, input: f32) -> f32 {
        let output = input - self.x_previous + self.coefficient * self.y_previous;
        self.x_previous = input;
        if output.is_finite() {
            self.y_previous = flush_denormal(output);
            self.y_previous
        } else {
            self.y_previous = 0.0;
            0.0
        }
    }

    pub(crate) fn reset(&mut self) {
        self.x_previous = 0.0;
        self.y_previous = 0.0;
    }
}

/// A one-pole low-pass branch for generic Hammerstein models.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OnePoleLowpass {
    coefficient: f32,
    state: f32,
}

impl OnePoleLowpass {
    pub(crate) fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        let coefficient = if cutoff_hz <= 0.0 {
            0.0
        } else {
            (-TAU * cutoff_hz / sample_rate).exp()
        };
        Self {
            coefficient,
            state: 0.0,
        }
    }

    #[inline]
    pub(crate) fn process(&mut self, input: f32) -> f32 {
        let output = (1.0 - self.coefficient) * input + self.coefficient * self.state;
        if output.is_finite() {
            self.state = flush_denormal(output);
            self.state
        } else {
            self.state = 0.0;
            0.0
        }
    }

    pub(crate) fn reset(&mut self) {
        self.state = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_blocker_clears_invalid_history_before_returning() {
        let mut blocker = DcBlocker::new(48_000.0);
        blocker.y_previous = f32::MAX;
        blocker.coefficient = f32::MAX;
        assert_eq!(blocker.process(1.0), 0.0);
        assert!(blocker.process(0.0).is_finite());
        blocker.reset();
        assert_eq!(blocker.process(0.0), 0.0);
    }

    #[test]
    fn lowpass_clears_invalid_history_before_returning() {
        let mut filter = OnePoleLowpass::new(1_000.0, 48_000.0);
        filter.state = f32::MAX;
        filter.coefficient = f32::MAX;
        assert_eq!(filter.process(1.0), 0.0);
        assert_eq!(filter.process(0.0), 0.0);
    }
}

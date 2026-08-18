use super::consts::TRUE_PEAK_FIR_LEN;
use super::consts::TRUE_PEAK_FIR_PHASES;

pub(super) struct TruePeakDetector {
    /// Per-channel circular history of the last `TRUE_PEAK_FIR_LEN` samples.
    pub(super) history: Vec<[f64; TRUE_PEAK_FIR_LEN]>,
    /// Per-channel write position; the slot it points at holds the oldest
    /// sample once the buffer is full (zeros before that).
    pub(super) pos: Vec<usize>,
    pub(super) peak: Vec<f64>,
    pub(super) prev_peak: Vec<f64>,
}

impl TruePeakDetector {
    pub(super) fn new(channels: usize) -> Self {
        Self {
            history: vec![[0.0; TRUE_PEAK_FIR_LEN]; channels],
            pos: vec![0; channels],
            peak: vec![0.0; channels],
            prev_peak: vec![0.0; channels],
        }
    }

    pub(super) fn process_frame(&mut self, ch: usize, sample: f64) {
        // Circular write: overwrite the oldest slot instead of shifting.
        let h = &mut self.history[ch];
        let start = self.pos[ch];
        h[start] = sample;
        self.pos[ch] = if start + 1 == TRUE_PEAK_FIR_LEN {
            0
        } else {
            start + 1
        };

        // Evaluate 4 polyphase outputs. Coefficient i pairs with the i-th
        // oldest sample: slot (pos + i) mod LEN, i.e. the same oldest-to-
        // newest summation order as the previous shift-register version.
        for phase in &TRUE_PEAK_FIR_PHASES {
            let mut sum = 0.0;
            let mut idx = self.pos[ch];
            for &coeff in phase {
                sum += coeff * h[idx];
                idx += 1;
                if idx == TRUE_PEAK_FIR_LEN {
                    idx = 0;
                }
            }
            let abs_val = sum.abs();
            if abs_val > self.peak[ch] {
                self.peak[ch] = abs_val;
            }
            if abs_val > self.prev_peak[ch] {
                self.prev_peak[ch] = abs_val;
            }
        }
    }

    pub(super) fn reset(&mut self) {
        for h in &mut self.history {
            h.fill(0.0);
        }
        self.pos.fill(0);
        self.peak.fill(0.0);
        self.prev_peak.fill(0.0);
    }
}

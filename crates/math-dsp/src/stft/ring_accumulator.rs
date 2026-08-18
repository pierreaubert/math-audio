/// Sample accumulator with hop-based triggering.
/// Accumulates samples into a circular buffer and signals when `hop_size`
/// new samples have been written (and the buffer has been filled at least once).
pub struct RingAccumulator {
    pub(super) buffer: Vec<f32>,
    pub(super) write_pos: usize,
    pub(super) samples_since_trigger: usize,
    pub(super) filled: bool,
    pub(super) window_size: usize,
    pub(super) hop_size: usize,
}

impl RingAccumulator {
    /// Create a new ring accumulator.
    ///
    /// # Panics
    /// Panics if `window_size` or `hop_size` is zero.
    pub fn new(window_size: usize, hop_size: usize) -> Self {
        assert!(
            window_size > 0,
            "RingAccumulator: window_size must be greater than zero"
        );
        assert!(
            hop_size > 0,
            "RingAccumulator: hop_size must be greater than zero"
        );
        Self {
            buffer: vec![0.0; window_size],
            write_pos: 0,
            samples_since_trigger: 0,
            filled: false,
            window_size,
            hop_size,
        }
    }

    /// Push a single sample. Returns `true` when `hop_size` samples have
    /// accumulated since the last trigger (and the buffer is full).
    pub fn push(&mut self, sample: f32) -> bool {
        self.buffer[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % self.window_size;
        self.samples_since_trigger += 1;

        if !self.filled && self.samples_since_trigger >= self.window_size {
            self.filled = true;
        }

        if self.filled && self.samples_since_trigger >= self.hop_size {
            self.samples_since_trigger = 0;
            true
        } else {
            false
        }
    }

    /// Copy the current window (oldest-first) into `dest`.
    /// `dest` must be at least `window_size` long.
    /// Uses two contiguous copies instead of per-element modulo.
    pub fn read_window(&self, dest: &mut [f32]) {
        debug_assert!(dest.len() >= self.window_size);
        let start = self.write_pos; // oldest sample
        let first_len = self.window_size - start;
        dest[..first_len].copy_from_slice(&self.buffer[start..]);
        if start > 0 {
            dest[first_len..self.window_size].copy_from_slice(&self.buffer[..start]);
        }
    }

    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
        self.samples_since_trigger = 0;
        self.filled = false;
    }
}

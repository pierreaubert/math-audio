use std::path::Path;

/// Policy for frequencies outside a microphone calibration table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompensationOutOfRange {
    /// Apply no correction outside the measured range.
    Zero,
    /// Hold the nearest calibration point.
    Clamp,
    /// Return an error instead of silently extrapolating.
    Error,
}

/// Microphone compensation data (frequency response correction)
#[derive(Debug, Clone)]
pub struct MicrophoneCompensation {
    /// Frequency points in Hz
    pub frequencies: Vec<f32>,
    /// SPL deviation in dB (positive = mic is louder, negative = mic is quieter)
    pub spl_db: Vec<f32>,
}

impl MicrophoneCompensation {
    /// Construct calibration data from frequency/dB vectors.
    pub fn new(frequencies: Vec<f32>, spl_db: Vec<f32>) -> Result<Self, String> {
        if frequencies.is_empty() || frequencies.len() != spl_db.len() {
            return Err(
                "MicrophoneCompensation::new: frequency and dB vectors must match and be non-empty"
                    .to_string(),
            );
        }
        if frequencies
            .iter()
            .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
            || spl_db.iter().any(|value| !value.is_finite())
        {
            return Err("MicrophoneCompensation::new: calibration points must be finite and frequencies positive".to_string());
        }
        if frequencies.windows(2).any(|window| window[1] <= window[0]) {
            return Err(
                "MicrophoneCompensation::new: frequencies must be strictly increasing".to_string(),
            );
        }
        Ok(Self {
            frequencies,
            spl_db,
        })
    }

    /// Apply pre-compensation to a sweep signal
    ///
    /// For log sweeps, this modulates the amplitude based on the instantaneous frequency
    /// to pre-compensate for the microphone's response.
    ///
    /// # Arguments
    /// * `signal` - The sweep signal to compensate
    /// * `start_freq` - Start frequency of the sweep in Hz
    /// * `end_freq` - End frequency of the sweep in Hz
    /// * `sample_rate` - Sample rate in Hz
    /// * `inverse` - If true, applies inverse compensation (boost where mic is weak)
    ///
    /// # Returns
    /// Pre-compensated signal
    pub fn apply_to_sweep(
        &self,
        signal: &[f32],
        start_freq: f32,
        end_freq: f32,
        sample_rate: u32,
        inverse: bool,
    ) -> Vec<f32> {
        if sample_rate == 0
            || !start_freq.is_finite()
            || !end_freq.is_finite()
            || start_freq <= 0.0
            || end_freq <= 0.0
            || signal.is_empty()
        {
            return signal.to_vec();
        }
        let duration = signal.len() as f32 / sample_rate as f32;
        let mut compensated = Vec::with_capacity(signal.len());

        // Debug: print some sample points
        let debug_points = [0, signal.len() / 4, signal.len() / 2, 3 * signal.len() / 4];

        for (i, &sample) in signal.iter().enumerate() {
            let t = i as f32 / sample_rate as f32;

            // Compute instantaneous frequency for log sweep
            // f(t) = f0 * exp(t * ln(f1/f0) / T)
            let freq = start_freq * ((t * (end_freq / start_freq).ln()) / duration).exp();

            // Get compensation at this frequency (in dB)
            let comp_db = self.interpolate_at(freq);

            // Apply inverse or direct compensation
            let gain_db = if inverse { -comp_db } else { comp_db };

            // Convert dB to linear gain
            let gain = 10_f32.powf(gain_db / 20.0);

            // Debug output for sample points
            if debug_points.contains(&i) {
                log::debug!(
                    "[apply_to_sweep] t={:.3}s, freq={:.1}Hz, comp_db={:.2}dB, gain_db={:.2}dB, gain={:.3}x",
                    t,
                    freq,
                    comp_db,
                    gain_db,
                    gain
                );
            }

            compensated.push(sample * gain);
        }

        log::debug!(
            "[apply_to_sweep] Processed {} samples, duration={:.2}s",
            signal.len(),
            duration
        );
        compensated
    }

    /// Load microphone compensation from a CSV or TXT file
    ///
    /// File format:
    /// - CSV: frequency_hz,spl_db (with or without header, comma-separated)
    /// - TXT: freq spl (space/tab-separated, no header assumed)
    pub fn from_file(path: &Path) -> Result<Self, String> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        log::debug!("[MicrophoneCompensation] Loading from {:?}", path);

        let file = File::open(path)
            .map_err(|e| format!("Failed to open compensation file {:?}: {}", path, e))?;
        let reader = BufReader::new(file);

        // Determine if this is a .txt file (no header expected)
        let is_txt_file = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase() == "txt")
            .unwrap_or(false);

        if is_txt_file {
            log::info!(
                "[MicrophoneCompensation] Detected .txt file - assuming space/tab-separated without header"
            );
        }

        let mut frequencies = Vec::new();
        let mut spl_db = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| format!("Failed to read line {}: {}", line_num + 1, e))?;
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // For CSV files, skip header line
            if !is_txt_file && line.starts_with("frequency") {
                continue;
            }

            // For TXT files, skip lines that don't start with a number
            if is_txt_file {
                let first_char = line.chars().next().unwrap_or(' ');
                if !first_char.is_ascii_digit() && first_char != '-' && first_char != '+' {
                    log::info!(
                        "[MicrophoneCompensation] Skipping non-numeric line {}: '{}'",
                        line_num + 1,
                        line
                    );
                    continue;
                }
            }

            // Parse based on file type with auto-detection for TXT
            let parts: Vec<&str> = if is_txt_file {
                // TXT: Try to auto-detect separator
                // First, try comma (in case it's mislabeled CSV)
                let comma_parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if comma_parts.len() >= 2
                    && comma_parts[0].parse::<f32>().is_ok()
                    && comma_parts[1].parse::<f32>().is_ok()
                {
                    comma_parts
                } else {
                    // Try tab
                    let tab_parts: Vec<&str> = line.split('\t').map(|s| s.trim()).collect();
                    if tab_parts.len() >= 2
                        && tab_parts[0].parse::<f32>().is_ok()
                        && tab_parts[1].parse::<f32>().is_ok()
                    {
                        tab_parts
                    } else {
                        // Fall back to whitespace
                        line.split_whitespace().collect()
                    }
                }
            } else {
                // CSV: comma separated
                line.split(',').collect()
            };

            if parts.len() < 2 {
                let separator = if is_txt_file {
                    "separator (comma/tab/space)"
                } else {
                    "comma"
                };
                return Err(format!(
                    "Invalid format at line {}: expected {} with 2+ values but got '{}'",
                    line_num + 1,
                    separator,
                    line
                ));
            }

            let freq: f32 = parts[0]
                .trim()
                .parse()
                .map_err(|e| format!("Invalid frequency at line {}: {}", line_num + 1, e))?;
            let spl: f32 = parts[1]
                .trim()
                .parse()
                .map_err(|e| format!("Invalid SPL at line {}: {}", line_num + 1, e))?;

            if !freq.is_finite() || freq <= 0.0 || !spl.is_finite() {
                return Err(format!(
                    "Invalid non-finite or non-positive calibration point at line {}",
                    line_num + 1
                ));
            }

            frequencies.push(freq);
            spl_db.push(spl);
        }

        if frequencies.is_empty() {
            return Err(format!("No compensation data found in {:?}", path));
        }

        // Validate that frequencies are sorted
        for i in 1..frequencies.len() {
            if frequencies[i] <= frequencies[i - 1] {
                return Err(format!(
                    "Frequencies must be strictly increasing: found {} after {} at line {}",
                    frequencies[i],
                    frequencies[i - 1],
                    i + 1
                ));
            }
        }

        log::info!(
            "[MicrophoneCompensation] Loaded {} calibration points: {:.1} Hz - {:.1} Hz",
            frequencies.len(),
            frequencies[0],
            frequencies[frequencies.len() - 1]
        );
        log::info!(
            "[MicrophoneCompensation] SPL range: {:.2} dB to {:.2} dB",
            spl_db.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            spl_db.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        );

        Ok(Self {
            frequencies,
            spl_db,
        })
    }

    /// Interpolate compensation value at a given frequency
    ///
    /// Uses linear interpolation in the dB domain against log-frequency.
    /// Returns 0.0 for frequencies outside the calibration range.
    pub fn interpolate_at(&self, freq: f32) -> f32 {
        if self.frequencies.is_empty() || !freq.is_finite() || freq <= 0.0 {
            return 0.0;
        }
        if freq < self.frequencies[0] || freq > self.frequencies[self.frequencies.len() - 1] {
            log::warn!(
                "[MicrophoneCompensation] {:.3} Hz is outside calibration range {:.3}..{:.3} Hz; applying 0 dB",
                freq,
                self.frequencies[0],
                self.frequencies[self.frequencies.len() - 1]
            );
            // Outside calibration range - no compensation for the legacy API.
            return 0.0;
        }

        // Find the two nearest points
        let idx = match self.frequencies.binary_search_by(|f| f.total_cmp(&freq)) {
            Ok(i) => return self.spl_db[i], // Exact match
            Err(i) => i,
        };

        if idx == 0 {
            return self.spl_db[0];
        }
        if idx >= self.frequencies.len() {
            return self.spl_db[self.frequencies.len() - 1];
        }

        // Calibration curves are sampled on a logarithmic frequency axis in
        // most microphone data sheets. Interpolate in log(Hz), while keeping
        // the correction itself in dB.
        let f0 = self.frequencies[idx - 1];
        let f1 = self.frequencies[idx];
        let s0 = self.spl_db[idx - 1];
        let s1 = self.spl_db[idx];

        let t = (freq.ln() - f0.ln()) / (f1.ln() - f0.ln());
        s0 + t * (s1 - s0)
    }

    /// Apply microphone correction to a measured frequency response.
    ///
    /// `spl_db` is the measured response and the calibration table describes
    /// the microphone's own deviation, so the deviation is subtracted.
    /// Frequencies outside the table use [`CompensationOutOfRange::Zero`].
    pub fn apply_to_response(&self, freqs: &[f32], spl_db: &[f32]) -> Vec<f32> {
        self.apply_to_response_with_policy(freqs, spl_db, CompensationOutOfRange::Zero)
            .unwrap_or_else(|_| spl_db.to_vec())
    }

    /// Apply microphone correction with an explicit out-of-range policy.
    pub fn apply_to_response_with_policy(
        &self,
        freqs: &[f32],
        spl_db: &[f32],
        policy: CompensationOutOfRange,
    ) -> Result<Vec<f32>, String> {
        if freqs.len() != spl_db.len() {
            return Err(format!(
                "apply_to_response: frequency count {} != response count {}",
                freqs.len(),
                spl_db.len()
            ));
        }
        if self.frequencies.is_empty() {
            return Ok(spl_db.to_vec());
        }
        let first = self.frequencies[0];
        let last = self.frequencies[self.frequencies.len() - 1];
        freqs
            .iter()
            .zip(spl_db)
            .map(|(&freq, &level)| {
                if !freq.is_finite() || freq <= 0.0 || !level.is_finite() {
                    return Err("apply_to_response: frequencies and levels must be finite".to_string());
                }
                let compensation = if (first..=last).contains(&freq) {
                    self.interpolate_at(freq)
                } else {
                    match policy {
                        CompensationOutOfRange::Zero => 0.0,
                        CompensationOutOfRange::Clamp => {
                            let clamped = freq.clamp(first, last);
                            self.interpolate_at(clamped)
                        }
                        CompensationOutOfRange::Error => {
                            return Err(format!(
                                "apply_to_response: frequency {freq:.3} Hz is outside calibration range {first:.3}..{last:.3} Hz"
                            ));
                        }
                    }
                };
                Ok(level - compensation)
            })
            .collect()
    }
}

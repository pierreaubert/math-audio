/// Configuration for standalone WAV buffer analysis
#[derive(Debug, Clone)]
pub struct WavAnalysisConfig {
    /// Number of output frequency points (default: 2000)
    pub num_points: usize,
    /// Minimum frequency in Hz (default: 20)
    pub min_freq: f32,
    /// Maximum frequency in Hz (default: 20000)
    pub max_freq: f32,
    /// FFT size (if None, auto-computed based on signal length and mode)
    pub fft_size: Option<usize>,
    /// Window overlap ratio for Welch's method (0.0-1.0, default: 0.5)
    pub overlap: f32,
    /// Use single FFT instead of Welch's method (better for sweeps and impulse responses)
    pub single_fft: bool,
    /// Apply pink compensation (+3 dB/octave) for log sweeps
    pub pink_compensation: bool,
    /// Use rectangular window instead of Hann
    pub no_window: bool,
    /// Desired in-room response slope in dB across the analysis frequency range.
    /// The correction is zero at `min_freq` and reaches this value at `max_freq`.
    pub room_slope_db: Option<f32>,
    /// Skip the in-room response slope correction for subwoofer measurements.
    pub subwoofer: bool,
}

impl Default for WavAnalysisConfig {
    fn default() -> Self {
        Self {
            num_points: 2000,
            min_freq: 20.0,
            max_freq: 20000.0,
            fft_size: None,
            overlap: 0.5,
            single_fft: false,
            pink_compensation: false,
            no_window: false,
            room_slope_db: None,
            subwoofer: false,
        }
    }
}

impl WavAnalysisConfig {
    /// Create config optimized for log sweep analysis
    pub fn for_log_sweep() -> Self {
        Self {
            single_fft: true,
            pink_compensation: true,
            no_window: true,
            ..Default::default()
        }
    }

    /// Create config optimized for impulse response analysis
    pub fn for_impulse_response() -> Self {
        Self {
            single_fft: true,
            ..Default::default()
        }
    }

    /// Create config for stationary signals (music, noise)
    pub fn for_stationary() -> Self {
        Self::default()
    }
}

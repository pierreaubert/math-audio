/// Filter types for biquad filters
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BiquadFilterType {
    /// Low-pass filter
    Lowpass,
    /// High-pass filter
    Highpass,
    /// High-pass filter
    HighpassVariableQ,
    /// Band-pass filter
    Bandpass,
    /// Peaking filter
    Peak,
    /// Notch filter
    Notch,
    /// Low-shelf filter
    Lowshelf,
    /// High-shelf filter
    Highshelf,
    /// All-pass filter
    AllPass,
    /// Minimum-phase low shelf with prescribed DC and Nyquist gains.
    ///
    /// Keeps fixed-slope poles and geometric midpoint gain. Uses a closed-form
    /// numerator split; Q does not change the shelf slope.
    LowshelfOrf,
    /// Minimum-phase high shelf with prescribed DC and Nyquist gains.
    ///
    /// Keeps fixed-slope poles and geometric midpoint gain. Uses a closed-form
    /// numerator split; Q does not change the shelf slope.
    HighshelfOrf,
    /// Minimum-phase matched peak with unity DC and Nyquist gain.
    ///
    /// Boost poles have radius `exp(-pi*f/(Q*fs))` and angle `2*pi*f/fs`.
    /// Numerator spectral factoring sets the requested center gain. Cuts invert
    /// the corresponding boost; zero gain is identity. This recipe does not
    /// guarantee an analog bandwidth match at every frequency.
    PeakMatched,
}

impl BiquadFilterType {
    /// Returns the short string representation of the filter type (e.g., "LP").
    pub fn short_name(&self) -> &'static str {
        match self {
            BiquadFilterType::Lowpass => "LP",
            BiquadFilterType::Highpass => "HP",
            BiquadFilterType::HighpassVariableQ => "HPQ",
            BiquadFilterType::Bandpass => "BP",
            BiquadFilterType::Peak => "PK",
            BiquadFilterType::Notch => "NO",
            BiquadFilterType::Lowshelf => "LS",
            BiquadFilterType::Highshelf => "HS",
            BiquadFilterType::AllPass => "AP",
            BiquadFilterType::LowshelfOrf => "LSO",
            BiquadFilterType::HighshelfOrf => "HSO",
            BiquadFilterType::PeakMatched => "PKM",
        }
    }

    /// Returns the long string representation of the filter type (e.g., "Lowpass").
    pub fn long_name(&self) -> &'static str {
        match self {
            BiquadFilterType::Lowpass => "Lowpass",
            BiquadFilterType::Highpass => "Highpass",
            BiquadFilterType::HighpassVariableQ => "HighpassVariableQ",
            BiquadFilterType::Bandpass => "Bandpass",
            BiquadFilterType::Peak => "Peak",
            BiquadFilterType::Notch => "Notch",
            BiquadFilterType::Lowshelf => "Lowshelf",
            BiquadFilterType::Highshelf => "Highshelf",
            BiquadFilterType::AllPass => "AllPass",
            BiquadFilterType::LowshelfOrf => "LowshelfOrf",
            BiquadFilterType::HighshelfOrf => "HighshelfOrf",
            BiquadFilterType::PeakMatched => "PeakMatched",
        }
    }
}

use super::fir_design_config::FirDesignConfig;

/// Generate an FIR filter to match a target frequency response
///
/// This function takes a target magnitude response (in dB) at specified frequencies
/// and generates FIR coefficients that approximate that response.
///
/// `FirPhase::Kirkeby` is intentionally unsupported here because Kirkeby
/// regularized inversion requires both a measurement and a target response.
/// Use `generate_kirkeby_correction` for that workflow.
///
/// # Arguments
/// * `freqs` - Frequency points in Hz (must be positive, sorted ascending)
/// * `magnitude_db` - Target magnitude in dB at each frequency point
/// * `config` - FIR design configuration
///
/// # Returns
/// * Vector of FIR coefficients
///
/// # Panics
/// Panics if `freqs` and `magnitude_db` have different lengths, if either
/// slice is empty, if any frequency or magnitude is non-finite, if any
/// frequency is non-positive, or if frequencies are not strictly increasing.
pub fn generate_fir_from_response(
    freqs: &[f64],
    magnitude_db: &[f64],
    config: &FirDesignConfig,
) -> Vec<f64> {
    super::context::FirDesignContext::new().generate_fir_from_response(freqs, magnitude_db, config)
}

/// Generate Kirkeby regularized FIR correction filter
///
/// Kirkeby inversion uses frequency-dependent regularization to create a stable
/// inverse filter that doesn't over-boost deep nulls (common in room measurements).
///
/// # Arguments
/// * `meas_freqs` - Measurement frequency points in Hz
/// * `meas_magnitude_db` - Measurement magnitude in dB
/// * `meas_phase_deg` - Measurement phase in degrees (optional, uses 0 if None)
/// * `target_magnitude_db` - Target magnitude in dB at meas_freqs points
/// * `config` - FIR design configuration
///
/// # Returns
/// * Vector of FIR coefficients
pub fn generate_kirkeby_correction(
    meas_freqs: &[f64],
    meas_magnitude_db: &[f64],
    meas_phase_deg: Option<&[f64]>,
    target_magnitude_db: &[f64],
    config: &FirDesignConfig,
) -> Vec<f64> {
    super::context::FirDesignContext::new().generate_kirkeby_correction(
        meas_freqs,
        meas_magnitude_db,
        meas_phase_deg,
        target_magnitude_db,
        config,
    )
}

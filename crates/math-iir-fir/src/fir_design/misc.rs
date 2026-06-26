use std::path::Path;

/// Save FIR coefficients to a WAV file (32-bit float mono)
///
/// # Arguments
/// * `coeffs` - FIR coefficients
/// * `sample_rate` - Sample rate in Hz
/// * `path` - Output file path
pub fn save_fir_to_wav(
    coeffs: &[f64],
    sample_rate: u32,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in coeffs {
        writer.write_sample(sample as f32)?;
    }
    writer.finalize()?;

    Ok(())
}

/// Interpolate values from source frequencies to target frequencies using log-space
pub(super) fn interpolate_log_space(
    src_freqs: &[f64],
    src_values: &[f64],
    target_freqs: &[f64],
) -> Vec<f64> {
    let mut result = Vec::with_capacity(target_freqs.len());

    for &f in target_freqs {
        if f <= 0.0 {
            // DC: use first value or extrapolate
            result.push(src_values.first().copied().unwrap_or(0.0));
            continue;
        }

        let log_f = f.ln();

        // Find bracketing indices in source
        let mut lower_idx = 0;
        let mut upper_idx = src_freqs.len() - 1;

        // Binary search for position
        for (i, &sf) in src_freqs.iter().enumerate() {
            if sf <= f {
                lower_idx = i;
            }
            if sf >= f && i < upper_idx {
                upper_idx = i;
                break;
            }
        }

        if lower_idx == upper_idx || src_freqs[lower_idx] <= 0.0 || src_freqs[upper_idx] <= 0.0 {
            result.push(src_values[lower_idx]);
        } else {
            // Log-linear interpolation
            let log_f_low = src_freqs[lower_idx].ln();
            let log_f_high = src_freqs[upper_idx].ln();
            let t = (log_f - log_f_low) / (log_f_high - log_f_low);
            let interp_val =
                src_values[lower_idx] + t * (src_values[upper_idx] - src_values[lower_idx]);
            result.push(interp_val);
        }
    }

    result
}

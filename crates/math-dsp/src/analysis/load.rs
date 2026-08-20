use hound::WavReader;
use std::path::Path;

/// Full-scale divisor for integer PCM of the given bit depth.
///
/// hound sign-extends integer samples into `i32` without left-shifting, so
/// 16-bit audio peaks at `1 << 15` and 24-bit audio at `1 << 23` — dividing
/// by `i32::MAX` would make 16-bit files ~96 dB too quiet.
fn int_full_scale(bits_per_sample: u16) -> f32 {
    (1_i64 << (bits_per_sample - 1)) as f32
}

fn is_truncated_sample_error(error: &hound::Error) -> bool {
    match error {
        hound::Error::IoError(error) => {
            matches!(
                error.kind(),
                std::io::ErrorKind::Other | std::io::ErrorKind::UnexpectedEof
            ) && error.to_string().contains("enough bytes")
        }
        _ => false,
    }
}

/// Read all complete samples, tolerating a data chunk that extends past EOF.
fn read_samples<R: std::io::Read>(
    reader: &mut WavReader<R>,
    path: &Path,
) -> Result<Vec<f32>, String> {
    let spec = reader.spec();
    let mut truncated = false;
    let mut samples = Vec::new();

    match spec.sample_format {
        hound::SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                match sample {
                    Ok(sample) => samples.push(sample),
                    Err(error) if is_truncated_sample_error(&error) => {
                        truncated = true;
                        break;
                    }
                    Err(error) => return Err(format!("Failed to read samples: {}", error)),
                }
            }
        }
        hound::SampleFormat::Int => {
            let max_val = int_full_scale(spec.bits_per_sample);
            for sample in reader.samples::<i32>() {
                match sample {
                    Ok(sample) => samples.push(sample as f32 / max_val),
                    Err(error) if is_truncated_sample_error(&error) => {
                        truncated = true;
                        break;
                    }
                    Err(error) => return Err(format!("Failed to read samples: {}", error)),
                }
            }
        }
    }

    if truncated {
        eprintln!(
            "Warning: WAV file {:?} ends before its declared data chunk; using {} complete samples",
            path,
            samples.len()
        );
    }

    Ok(samples)
}

/// Load WAV file as mono and return samples with sample rate
pub(super) fn load_wav_mono_with_rate(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let mut reader =
        WavReader::open(path).map_err(|e| format!("Failed to open WAV file: {}", e))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let mut samples = read_samples(&mut reader, path)?;

    // Convert to mono by averaging channels
    let mono = if channels == 1 {
        samples
    } else {
        samples.truncate(samples.len() - samples.len() % channels);
        samples
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    Ok((mono, sample_rate))
}

/// Load a mono WAV file and convert to f32 samples
/// Load a WAV file and extract a specific channel or convert to mono
///
/// # Arguments
/// * `path` - Path to WAV file
/// * `channel_index` - Optional channel index to extract (0-based). If None, will average all channels for mono
pub(super) fn load_wav_mono_channel(
    path: &Path,
    channel_index: Option<usize>,
) -> Result<Vec<f32>, String> {
    let mut reader =
        WavReader::open(path).map_err(|e| format!("Failed to open WAV file: {}", e))?;

    let spec = reader.spec();
    let channels = spec.channels as usize;

    log::info!(
        "[load_wav_mono_channel] WAV file: {} channels, {} Hz, {:?} format",
        channels,
        spec.sample_rate,
        spec.sample_format
    );

    // Read all samples and convert to f32.
    let mut samples = read_samples(&mut reader, path)?;
    log::info!(
        "[load_wav_mono_channel] Read {} total samples",
        samples.len()
    );

    // Handle mono file - return as-is
    if channels == 1 {
        log::info!(
            "[load_wav_mono_channel] File is already mono, returning {} samples",
            samples.len()
        );
        return Ok(samples);
    }

    // Handle multi-channel file
    if let Some(ch_idx) = channel_index {
        // Extract specific channel
        if ch_idx >= channels {
            return Err(format!(
                "Channel index {} out of range (file has {} channels)",
                ch_idx, channels
            ));
        }
        log::info!(
            "[load_wav_mono_channel] Extracting channel {} from {} channels",
            ch_idx,
            channels
        );
        Ok(samples
            .chunks(channels)
            .map(|chunk| chunk[ch_idx])
            .collect())
    } else {
        samples.truncate(samples.len() - samples.len() % channels);
        // Average all channels to mono
        log::info!(
            "[load_wav_mono_channel] Averaging {} channels to mono",
            channels
        );
        Ok(samples
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect())
    }
}

/// Load a WAV file as mono (averages channels if multi-channel)
pub(super) fn load_wav_mono(path: &Path) -> Result<Vec<f32>, String> {
    load_wav_mono_channel(path, None)
}

#[cfg(test)]
mod tests {
    use super::load_wav_mono_with_rate;

    #[test]
    fn reads_complete_samples_from_truncated_data_chunk() {
        let path = std::env::temp_dir().join(format!(
            "math-audio-truncated-wav-{}.wav",
            std::process::id()
        ));
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(&path, spec).expect("create WAV");
        for sample in 0..100_i16 {
            writer.write_sample(sample).expect("write sample");
        }
        writer.finalize().expect("finalize WAV");

        let file_size = std::fs::metadata(&path).expect("stat WAV").len();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open WAV for truncation")
            .set_len(file_size - 2)
            .expect("truncate WAV");

        let (samples, sample_rate) = load_wav_mono_with_rate(&path).expect("read WAV");
        assert_eq!(sample_rate, 48_000);
        assert_eq!(samples.len(), 99);

        std::fs::remove_file(path).expect("remove test WAV");
    }
}

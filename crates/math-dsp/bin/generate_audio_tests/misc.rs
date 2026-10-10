pub(super) fn ensure_below_nyquist(freq: f32, sr: u32, label: &str) -> Result<(), String> {
    let nyquist = sr as f32 / 2.0;
    if freq >= nyquist {
        Err(format!(
            "Nyquist violation: {label} {freq} Hz >= {nyquist} Hz (skipped)"
        ))
    } else {
        Ok(())
    }
}

pub(super) fn ensure_sti_sample_rate(sr: u32) -> Result<(), String> {
    // The 8 kHz STI octave reaches 8000 * sqrt(2) Hz; lower rates narrow
    // it, so STI signals require the IEC reference minimum instead.
    if sr < math_audio_dsp::signals::STI_MIN_SAMPLE_RATE_HZ {
        Err(format!(
            "Nyquist violation: STI needs sr >= {} Hz for the 8 kHz octave, got {sr} Hz (skipped)",
            math_audio_dsp::signals::STI_MIN_SAMPLE_RATE_HZ
        ))
    } else {
        Ok(())
    }
}

pub(super) fn clip(sample: f32) -> f32 {
    sample.clamp(-1.0, 1.0)
}

/// Build a RIFF LIST INFO chunk containing the given tag pairs.
pub(super) fn build_info_chunk(tags: &[(&[u8; 4], &str)]) -> Vec<u8> {
    fn info_subchunk(id: &[u8; 4], value: &str) -> Vec<u8> {
        let mut v = value.as_bytes().to_vec();
        v.push(0); // null terminator
        if !v.len().is_multiple_of(2) {
            v.push(0); // RIFF word-alignment pad
        }
        let mut buf = Vec::with_capacity(8 + v.len());
        buf.extend_from_slice(id);
        buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
        buf.extend_from_slice(&v);
        buf
    }

    let sub_chunks: Vec<u8> = tags
        .iter()
        .flat_map(|(id, val)| info_subchunk(id, val))
        .collect();
    let list_data_len = 4 + sub_chunks.len(); // "INFO" + sub-chunks
    let mut chunk = Vec::with_capacity(8 + list_data_len);
    chunk.extend_from_slice(b"LIST");
    chunk.extend_from_slice(&(list_data_len as u32).to_le_bytes());
    chunk.extend_from_slice(b"INFO");
    chunk.extend_from_slice(&sub_chunks);
    chunk
}

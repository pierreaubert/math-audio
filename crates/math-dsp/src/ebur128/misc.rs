pub(super) fn channel_weight(ch: usize, num_channels: usize) -> f64 {
    // BS.1770-4 §5.1 general rule with libebur128's default channel map:
    // L, R, C (indices 0-2) -> 1.0; index 3 -> LFE, excluded (0.0);
    // indices 4-5 -> surrounds at 1.41 (+1.5 dB). Special cases: 4ch is
    // L, R, Ls, Rs (no centre, no LFE); 5ch is L, R, C, Ls, Rs.
    // Channels beyond 5.1 (index >= 6) are unused in the default map.
    if num_channels == 4 {
        // L, R, Ls, Rs
        match ch {
            0 | 1 => 1.0,
            2 | 3 => 1.41,
            _ => 0.0,
        }
    } else if num_channels == 5 {
        // L, R, C, Ls, Rs
        match ch {
            0..=2 => 1.0,  // L, R, C
            3 | 4 => 1.41, // Ls, Rs (surround +1.5 dB)
            _ => 0.0,
        }
    } else {
        // General case (mono, stereo, 6ch, 8ch, ...): 0,1,2 = L,R,C;
        // 3 = LFE (excluded); 4,5 = surrounds; beyond -> unused.
        match ch {
            0..=2 => 1.0,
            3 => 0.0,
            4 | 5 => 1.41,
            _ => 0.0,
        }
    }
}

/// Convert energy to loudness in LUFS: -0.691 + 10 × log10(energy).
pub fn energy_to_loudness(energy: f64) -> f64 {
    -0.691 + 10.0 * energy.log10()
}

/// Convert loudness in LUFS to energy.
pub(super) fn loudness_to_energy(lufs: f64) -> f64 {
    10.0_f64.powf((lufs + 0.691) / 10.0)
}

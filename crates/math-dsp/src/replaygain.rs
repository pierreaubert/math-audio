use crate::ebur128::{EbuR128, Mode, energy_to_loudness};
use serde::{Deserialize, Serialize};

/// ReplayGain 2.0 Reference Gain
///
/// See the [ReplayGain 2.0 specification][rg2spec] for details.
///
/// [rg2spec]: https://wiki.hydrogenaud.io/index.php?title=ReplayGain_2.0_specification#Reference_level
const REPLAYGAIN2_REFERENCE_LUFS: f64 = -18.0;

/// ReplayGain analysis result containing gain and peak information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayGainInfo {
    /// ReplayGain 2.0 Track Gain in dB
    pub gain: f64,
    /// ReplayGain 2.0 Track Peak (0.0 to 1.0+); the sample peak, not the
    /// oversampled true peak.
    pub peak: f64,
}

/// Extended ReplayGain data including EBU R128 gating block statistics.
/// Used for computing album-level gain by accumulating across tracks.
#[derive(Debug, Clone)]
pub struct ReplayGainTrackData {
    pub gain: f64,
    pub peak: f64,
    pub gating_block_count: u64,
    pub energy: f64,
}

/// Streaming ReplayGain analyser.
///
/// Feed decoded audio frames incrementally via [`add_frames_f32`](Self::add_frames_f32),
/// then call [`finalize`](Self::finalize) or [`finalize_extended`](Self::finalize_extended).
///
/// Runs the EBU R128 meter in integrated-loudness + sample-peak mode only
/// (`Mode::I | Mode::SAMPLE_PEAK`); the `peak` reported in the results is the
/// sample peak. The 4x-oversampling true-peak path is not enabled.
pub struct ReplayGainAnalyzer {
    ebur128: EbuR128,
    channels: u32,
}

impl ReplayGainAnalyzer {
    /// Create a new analyser for the given channel count and sample rate.
    pub fn new(channels: u32, sample_rate: u32) -> Result<Self, String> {
        let ebur128 = EbuR128::new(channels, sample_rate, Mode::I | Mode::SAMPLE_PEAK)
            .map_err(|e| format!("Failed to create EBU R128 analyzer: {:?}", e))?;
        Ok(Self { ebur128, channels })
    }

    /// Feed interleaved f32 frames (normalised to [-1, 1]).
    pub fn add_frames_f32(&mut self, samples: &[f32]) -> Result<(), String> {
        self.ebur128
            .add_frames_f32(samples)
            .map_err(|e| format!("Failed to add frames to EBU R128: {:?}", e))
    }

    /// Compute [`ReplayGainInfo`] from all frames fed so far.
    pub fn finalize(&self) -> Result<ReplayGainInfo, String> {
        let loudness = self
            .ebur128
            .loudness_global()
            .map_err(|e| format!("Failed to calculate loudness: {:?}", e))?;

        // Silent / fully gated-out tracks have loudness -inf; NaN input
        // propagates as NaN. Both would produce a non-finite gain (which
        // also cannot be serialized), so reject them like finalize_extended.
        if !loudness.is_finite() {
            return Err(format!(
                "Cannot compute ReplayGain: integrated loudness is {loudness} \
                 (silent or invalid input)"
            ));
        }

        let peak = self.peak()?;
        let gain = REPLAYGAIN2_REFERENCE_LUFS - loudness;

        log::debug!("[Replay Gain] Gain: {}dB Peak: {}dB", gain, peak);
        Ok(ReplayGainInfo { gain, peak })
    }

    /// Compute extended data including gating block count and energy.
    pub fn finalize_extended(&self) -> Result<ReplayGainTrackData, String> {
        let loudness = self
            .ebur128
            .loudness_global()
            .map_err(|e| format!("Failed to calculate loudness: {:?}", e))?;

        if !loudness.is_finite() {
            return Err(format!(
                "Cannot compute ReplayGain: integrated loudness is {loudness} \
                 (silent or invalid input)"
            ));
        }

        let peak = self.peak()?;
        let gain = REPLAYGAIN2_REFERENCE_LUFS - loudness;

        let (gating_block_count, energy) = self
            .ebur128
            .gating_block_count_and_energy()
            .ok_or_else(|| "Failed to get gating block count and energy".to_string())?;

        Ok(ReplayGainTrackData {
            gain,
            peak,
            gating_block_count,
            energy,
        })
    }

    /// 400 ms block energies above the absolute gate (-70 LUFS) for this
    /// track, for strict pooled album gating via [`compute_album_gain_pooled`].
    pub fn gating_blocks_above_absolute_gate(&self) -> Vec<f64> {
        self.ebur128.gating_blocks_above_absolute_gate()
    }

    fn peak(&self) -> Result<f64, String> {
        let mut peak = 0.0f64;
        for ch in 0..self.channels {
            let ch_peak = self
                .ebur128
                .sample_peak(ch)
                .map_err(|e| format!("Failed to get peak for channel {}: {:?}", ch, e))?;
            peak = peak.max(ch_peak);
        }
        Ok(peak)
    }
}

/// Compute album-level ReplayGain from accumulated per-track gating block data.
///
/// `tracks` contains `(peak, gating_block_count, energy)` for each track in the album.
/// Returns `(album_gain_db, album_peak)`.
///
/// Note: this pools per-track *pre-gated* energy sums (each track's own
/// relative gate was already applied). That is an approximation kept for
/// compatibility with persisted `(peak, count, energy)` tuples. The strict
/// BS.1770 album algorithm — pool all blocks above the absolute gate, then
/// apply one relative gate to the pool — is [`compute_album_gain_pooled`].
pub fn compute_album_gain(tracks: &[(f64, u64, f64)]) -> Option<(f64, f64)> {
    if tracks.is_empty() {
        return None;
    }

    let mut total_blocks: u64 = 0;
    let mut total_energy: f64 = 0.0;
    let mut album_peak: f64 = 0.0;

    for &(peak, blocks, energy) in tracks {
        total_blocks += blocks;
        total_energy += energy;
        album_peak = album_peak.max(peak);
    }

    if total_blocks == 0 {
        return None;
    }

    let album_loudness = energy_to_loudness(total_energy / total_blocks as f64);
    let album_gain = REPLAYGAIN2_REFERENCE_LUFS - album_loudness;

    Some((album_gain, album_peak))
}

/// Compute album-level ReplayGain with strict BS.1770 pooled gating.
///
/// All 400 ms blocks above the absolute gate (-70 LUFS) from every track are
/// pooled, then a single relative gate (pool mean - 10 LU) is applied to the
/// pool. `tracks` contains `(track_peak, blocks_above_absolute_gate)` per
/// track; obtain the blocks via
/// [`ReplayGainAnalyzer::gating_blocks_above_absolute_gate`].
///
/// Returns `(album_gain_db, album_peak)`, or `None` if no blocks remain.
pub fn compute_album_gain_pooled(tracks: &[(f64, Vec<f64>)]) -> Option<(f64, f64)> {
    if tracks.is_empty() {
        return None;
    }

    let mut album_peak: f64 = 0.0;
    let mut sum: f64 = 0.0;
    let mut count: u64 = 0;
    for (peak, blocks) in tracks {
        album_peak = album_peak.max(*peak);
        sum += blocks.iter().sum::<f64>();
        count += blocks.len() as u64;
    }
    if count == 0 {
        return None;
    }

    // One relative gate for the whole pool: pool mean - 10 LU.
    let rel_gate_energy = (sum / count as f64) * 0.1;
    let mut gated_sum: f64 = 0.0;
    let mut gated_count: u64 = 0;
    for (_, blocks) in tracks {
        for &e in blocks {
            if e > rel_gate_energy {
                gated_sum += e;
                gated_count += 1;
            }
        }
    }
    if gated_count == 0 {
        return None;
    }

    let album_loudness = energy_to_loudness(gated_sum / gated_count as f64);
    let album_gain = REPLAYGAIN2_REFERENCE_LUFS - album_loudness;

    Some((album_gain, album_peak))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replaygain_info_serialization() {
        let info = ReplayGainInfo {
            gain: -5.5,
            peak: 0.95,
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("-5.5"));
        assert!(json.contains("0.95"));
        let deserialized: ReplayGainInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.gain, info.gain);
        assert_eq!(deserialized.peak, info.peak);
    }

    #[test]
    fn test_compute_album_gain_empty() {
        assert!(compute_album_gain(&[]).is_none());
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = ReplayGainAnalyzer::new(2, 48000);
        assert!(analyzer.is_ok());
    }

    #[test]
    fn finalize_silent_track_returns_error_not_infinite_gain() {
        // A fully gated-out (silent) track has loudness -inf; the gain would
        // be +inf, which also cannot be serialized by serde_json. finalize
        // must return an error instead, mirroring finalize_extended.
        let mut analyzer = ReplayGainAnalyzer::new(2, 48000).unwrap();
        let silence = vec![0.0f32; 48000 * 2]; // 1 s stereo
        analyzer.add_frames_f32(&silence).unwrap();
        let result = analyzer.finalize();
        assert!(
            result.is_err(),
            "silent track must error, not yield gain={:?}",
            result.map(|info| info.gain)
        );
        assert!(analyzer.finalize_extended().is_err());
    }
}

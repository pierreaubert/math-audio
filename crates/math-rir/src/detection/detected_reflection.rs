use super::misc::angular_distance;
use super::misc::is_local_energy_peak;
use super::misc::median_of;
use super::types::DetectedReflection;
use crate::config::SsirConfig;
use rayon::prelude::*;

/// Detect early reflections using Local Energy Ratio (LER).
///
/// The RIR is divided into consecutive windows of `ler_window_ms` length.
/// Within each window, the median energy is computed. A reflection is detected
/// at the sample with maximum energy if that energy exceeds `energy_threshold`
/// times the median.
///
/// Detections within the direct sound window are discarded.
/// Consecutive pairs that are too close in both DOA and TOA are merged.
pub(crate) fn detect_reflections(
    rir: &[f32],
    direct_sound_toa: usize,
    doa_vectors: Option<&[[f32; 3]]>,
    config: &SsirConfig,
) -> Vec<DetectedReflection> {
    let window_len = config.ler_window_samples();
    let mixing_time = config.mixing_time_samples().min(rir.len());
    let (ds_pre, ds_post) = config.direct_sound_window_samples();

    // Direct sound exclusion zone
    let ds_start = direct_sound_toa.saturating_sub(ds_pre);
    let ds_end = (direct_sound_toa + ds_post).min(rir.len());

    // Number of analysis windows up to mixing time
    let num_windows = if window_len > 0 {
        mixing_time.div_ceil(window_len)
    } else {
        return Vec::new();
    };

    let mut raw_detections: Vec<(usize, f64)> = (0..num_windows)
        .into_par_iter()
        .flat_map(|i| {
            let win_start = i * window_len;
            let win_end = ((i + 1) * window_len).min(rir.len());
            if win_start >= rir.len() {
                return Vec::new();
            }

            // Compute energies in this window
            let mut energies: Vec<f64> = (win_start..win_end)
                .filter(|&j| j < ds_start || j >= ds_end)
                .map(|j| {
                    let s = rir[j] as f64;
                    s * s
                })
                .collect();

            if energies.is_empty() {
                return Vec::new();
            }

            // Compute median energy
            let median = median_of(&mut energies);

            // Threshold: energy must exceed `energy_threshold` times the median
            let threshold = config.energy_threshold * median;

            let mut detections = Vec::new();
            for (j, &sample) in rir.iter().enumerate().take(win_end).skip(win_start) {
                if j >= ds_start && j < ds_end {
                    continue;
                }
                let e = (sample as f64) * (sample as f64);
                if e > threshold && is_local_energy_peak(rir, j, win_start, win_end) {
                    detections.push((j, e));
                }
            }

            detections
        })
        .collect();

    // Sort by sample index
    raw_detections.sort_by_key(|&(idx, _)| idx);

    // Assign DOA vectors and build DetectedReflection list
    let mut reflections: Vec<DetectedReflection> = raw_detections
        .iter()
        .map(|&(toa, energy)| DetectedReflection {
            toa_sample: toa,
            peak_energy: energy,
            doa: doa_vectors.and_then(|doas| doas.get(toa).copied()),
        })
        .collect();

    // Validate consecutive pairs and merge if too close
    validate_and_merge(&mut reflections, config);

    reflections
}

/// Validate consecutive reflection pairs using DOA and TOA thresholds.
///
/// From Eq. (3) in the paper: a reflection R_{j+1} is retained only if
///   DELTA_DOA(j) >= lambda_DOA  AND  DELTA_TOA(j) > lambda_TOA
/// Otherwise it is merged with the previous reflection (keeping the one with higher energy).
pub(super) fn validate_and_merge(reflections: &mut Vec<DetectedReflection>, config: &SsirConfig) {
    if reflections.len() < 2 {
        return;
    }

    let toa_threshold = config.toa_threshold_samples();
    let doa_threshold_rad = config.doa_threshold_deg.to_radians();

    // In-place stack compaction: reflections[0..write] is the kept stack.
    // When a candidate is too close to the top, keep the higher-energy one
    // and continue checking against the new top (same semantics as the
    // original step-back loop, but without O(n²) Vec::remove shifts).
    let mut write = 0;
    for read in 0..reflections.len() {
        let mut candidate = read;
        while write > 0 {
            let prev = write - 1;
            let toa_diff = reflections[candidate]
                .toa_sample
                .saturating_sub(reflections[prev].toa_sample);

            let doa_diff = match (&reflections[prev].doa, &reflections[candidate].doa) {
                (Some(a), Some(b)) => angular_distance(a, b),
                _ => f64::MAX,
            };

            let spatially_distinct = doa_diff >= doa_threshold_rad;
            let temporally_distinct = toa_diff > toa_threshold;

            if spatially_distinct && temporally_distinct {
                break;
            }
            if reflections[candidate].peak_energy > reflections[prev].peak_energy {
                // Candidate replaces previous; pop previous and keep checking.
                write -= 1;
            } else {
                // Candidate is dominated; discard it.
                candidate = usize::MAX;
                break;
            }
        }
        if candidate != usize::MAX {
            if write != candidate {
                reflections.swap(write, candidate);
            }
            write += 1;
        }
    }
    reflections.truncate(write);
}

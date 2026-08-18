use super::misc::single_bin_dft;
use super::tone_phasor_window::tone_phase_phasors;
use super::types::TonePhaseResult;
use super::types::aggregate_tone_phase;
use std::f32::consts::PI;

/// Extract the phase of a single frequency bin from a time-domain
/// signal using a direct DFT sum, plus a half-split stability metric.
///
/// For `len(signal) = N`, the bin content at `freq_hz` is accumulated
/// with a sin-referenced projection (see `single_bin_dft`):
///
/// ```text
/// re = Σ_{k=0..N} s[k] · sin(ω·k), im = Σ_{k=0..N} s[k] · cos(ω·k)
/// ω = 2π·freq_hz/sample_rate
/// phase = atan2(im, re)   → pure sin(ω·k) reads 0°, cos(ω·k) reads +90°
/// magnitude = 2·√(re² + im²) / N
/// ```
///
/// The stability metric runs the same extraction over the first and
/// second halves independently and reports the wrapped phase
/// difference in degrees. A clean stationary tone gives ≈ 0;
/// modally-ringing or noisy bursts show ≫ 0.
pub fn extract_tone_phase(signal: &[f32], freq_hz: f32, sample_rate: u32) -> TonePhaseResult {
    if signal.len() < 4 || freq_hz <= 0.0 || sample_rate == 0 {
        return TonePhaseResult {
            phase_deg: 0.0,
            magnitude: 0.0,
            stability_deg: 0.0,
        };
    }
    let mid = signal.len() / 2;
    let (re_a, im_a) = single_bin_dft(&signal[..mid], freq_hz, sample_rate, 0);
    // The second half keeps the global time reference (k_offset = mid)
    // so both halves measure phase against the same t = 0 — otherwise
    // the split induces a spurious `mid·ω` phase jump on a stable tone.
    let (re_b, im_b) = single_bin_dft(&signal[mid..], freq_hz, sample_rate, mid);

    // The DFT sum is linear, so the full-signal projection equals the sum
    // of the two half accumulations — two passes suffice, not three.
    let re_full = re_a + re_b;
    let im_full = im_a + im_b;
    let magnitude_raw = (re_full * re_full + im_full * im_full).sqrt();
    let magnitude = 2.0 * magnitude_raw / signal.len() as f64;
    let phase_rad = im_full.atan2(re_full);

    let phase_a = im_a.atan2(re_a);
    let phase_b = im_b.atan2(re_b);
    // Wrap (phase_b − phase_a) to (−π, π] before taking |·|.
    let mut diff = phase_b - phase_a;
    while diff > PI as f64 {
        diff -= 2.0 * PI as f64;
    }
    while diff <= -(PI as f64) {
        diff += 2.0 * PI as f64;
    }

    TonePhaseResult {
        phase_deg: phase_rad.to_degrees(),
        magnitude,
        stability_deg: diff.abs().to_degrees(),
    }
}

/// Sub-window lock-in phase extraction.
///
/// Wraps [`tone_phase_phasors`] and aggregates the per-window
/// phasors into:
/// - `phase_deg`     — circular mean of per-window phase, in degrees,
///   wrapped to (−180°, 180°]. Computed from `atan2(ΣQᵢ, ΣIᵢ)` where the
///   I/Q come from each window with a shared `k_offset` so the phases
///   share a common time reference.
/// - `magnitude`     — mean peak amplitude, `2·√(I² + Q²) / window_len`,
///   across windows.
/// - `stability_deg` — circular standard deviation of per-window phase
///   (degrees). Equivalent to `√(−2·ln(R̄))` in radians, converted to
///   degrees, where `R̄` is the mean resultant length over all windows.
///   A stable lock returns ≈ 0; modal contamination or low SNR pushes
///   this above the 20° advisory threshold.
///
/// Returns `phase_deg = 0`, `magnitude = 0`, `stability_deg = 0` when
/// inputs are degenerate (see [`tone_phase_phasors`] for the rules).
pub fn extract_tone_phase_windowed(
    signal: &[f32],
    freq_hz: f32,
    sample_rate: u32,
    num_windows: usize,
) -> TonePhaseResult {
    let phasors = tone_phase_phasors(signal, freq_hz, sample_rate, num_windows);
    aggregate_tone_phase(&phasors)
}

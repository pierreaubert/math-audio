use super::misc::ensure_below_nyquist;
use super::misc::ensure_sti_sample_rate;
use math_audio_dsp::signals::gen_full_sti_signal;
use math_audio_dsp::signals::gen_log_sweep;
use math_audio_dsp::signals::gen_stipa_signal;
use math_audio_dsp::signals::gen_tone;
use math_audio_dsp::signals::gen_two_tone;

pub(super) fn gen_tone_checked(
    freq: f32,
    amp: f32,
    sr: u32,
    duration: f32,
) -> Result<Vec<f32>, String> {
    ensure_below_nyquist(freq, sr, "tone")?;
    Ok(gen_tone(freq, amp, sr, duration))
}

pub(super) fn gen_two_tone_checked(
    f1: f32,
    a1: f32,
    f2: f32,
    a2: f32,
    sr: u32,
    duration: f32,
) -> Result<Vec<f32>, String> {
    ensure_below_nyquist(f1, sr, "first tone")?;
    ensure_below_nyquist(f2, sr, "second tone")?;
    Ok(gen_two_tone(f1, a1, f2, a2, sr, duration))
}

pub(super) fn gen_log_sweep_checked(
    f_start: f32,
    f_end: f32,
    amp: f32,
    sr: u32,
    duration: f32,
) -> Result<Vec<f32>, String> {
    ensure_below_nyquist(f_end, sr, "sweep end")?;
    Ok(gen_log_sweep(f_start, f_end, amp, sr, duration))
}

pub(super) fn gen_stipa_checked(sr: u32, duration: f32) -> Result<Vec<f32>, String> {
    ensure_sti_sample_rate(sr)?;
    Ok(gen_stipa_signal(sr, duration))
}

pub(super) fn gen_full_sti_checked(
    sr: u32,
    segment_duration: f32,
    silence_gap: f32,
) -> Result<Vec<f32>, String> {
    ensure_sti_sample_rate(sr)?;
    Ok(gen_full_sti_signal(sr, segment_duration, silence_gap))
}

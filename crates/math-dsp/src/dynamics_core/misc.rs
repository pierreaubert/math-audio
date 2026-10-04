/// Time constant (ms) to one-pole coefficient.
#[inline]
pub(super) fn time_to_coeff(time_ms: f32, sample_rate: f64) -> f32 {
    if time_ms <= 0.0 {
        0.0
    } else {
        (-1.0 / (f64::from(time_ms) * 0.001 * sample_rate)).exp() as f32
    }
}

#[inline]
pub(super) fn hold_ms_to_samples(hold_ms: f32, sample_rate: f64) -> usize {
    (f64::from(hold_ms) * 0.001 * sample_rate) as usize
}

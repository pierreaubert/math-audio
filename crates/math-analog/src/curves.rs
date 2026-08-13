//! Reusable memoryless coloration curves.
//!
//! These functions are mathematical curve definitions, not claims about
//! physical hardware. SOTF wrappers may retain historical names and
//! parameter ranges while delegating the equations here.

/// Normalized `tanh` soft clipping used by the legacy Saturation plugin.
#[inline]
pub fn normalized_soft_clip(x: f32, drive: f32) -> f32 {
    let driven = x * drive;
    let tanh_drive = drive.tanh();
    if tanh_drive < 1e-6 {
        x
    } else {
        driven.tanh() / tanh_drive
    }
}

/// Odd-symmetric polynomial-like saturation curve.
#[inline]
pub fn tube_style(x: f32, drive: f32, exponent: f32) -> f32 {
    // Preserve the legacy f32 transfer exactly throughout its ordinary
    // operating range. Fall back to f64 only when an extreme finite input
    // would overflow an intermediate; this keeps extraction behavior stable
    // without allowing a finite input to produce a non-finite output.
    let legacy_driven = x * drive;
    if legacy_driven.is_finite() {
        let legacy_value = legacy_driven / (1.0 + legacy_driven.abs().powf(exponent));
        if legacy_value.is_finite() {
            return legacy_value;
        }
    }

    let driven = x as f64 * drive as f64;
    let value = driven / (1.0 + driven.abs().powf(exponent as f64));
    if value.is_finite() {
        value.clamp(f32::MIN as f64, f32::MAX as f64) as f32
    } else {
        (driven.signum() * f32::MAX as f64) as f32
    }
}

/// Memoryless exponential sigmoid used by the legacy Tape-style mode.
#[inline]
pub fn tape_style(x: f32, drive: f32) -> f32 {
    let driven = x * drive;
    driven.signum() * (1.0 - (-driven.abs() * 2.0).exp()) * 0.5
}

/// DC-centred, rail-normalized, bias-shifted asymmetric curve.
#[inline]
pub fn asymmetric_style(x: f32, drive: f32, tone: f32) -> f32 {
    let bias = 0.08 + 0.16 * (tone - 1.0).clamp(0.0, 2.0);
    let bias_tanh = bias.tanh();
    let centered = (x * drive + bias).tanh() - bias_tanh;
    if centered >= 0.0 {
        centered / (1.0 - bias_tanh)
    } else {
        centered / (1.0 + bias_tanh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_curves_are_odd() {
        for x in [-4.0, -0.5, 0.0, 0.25, 3.0] {
            assert_eq!(normalized_soft_clip(-x, 4.0), -normalized_soft_clip(x, 4.0));
            assert_eq!(tube_style(-x, 4.0, 1.7), -tube_style(x, 4.0, 1.7));
            assert_eq!(tape_style(-x, 4.0), -tape_style(x, 4.0));
        }
    }

    #[test]
    fn curves_remain_finite_for_extreme_finite_inputs() {
        for curve in [
            normalized_soft_clip(f32::MAX, 20.0),
            tube_style(f32::MAX, 20.0, 3.0),
            tape_style(f32::MAX, 20.0),
            asymmetric_style(f32::MAX, 20.0, 3.0),
        ] {
            assert!(curve.is_finite());
        }
    }

    #[test]
    fn tube_style_preserves_legacy_f32_transfer_in_normal_range() {
        for &(x, drive, exponent) in &[
            (-0.75, 1.0, 1.0),
            (-0.25, 4.0, 1.7),
            (0.125, 12.0, 2.5),
            (0.9, 20.0, 3.0),
        ] {
            let driven: f32 = x * drive;
            let expected = driven / (1.0 + driven.abs().powf(exponent));
            assert_eq!(tube_style(x, drive, exponent), expected);
        }
    }

    #[test]
    fn asymmetric_curve_has_a_controlled_even_component() {
        let positive = asymmetric_style(0.25, 4.0, 2.0);
        let negative = asymmetric_style(-0.25, 4.0, 2.0);
        assert!((positive + negative).abs() > 1e-4);
    }
}

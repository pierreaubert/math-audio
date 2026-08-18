//! SIMD helpers for bounded analog-style Chebyshev curves.
//!
//! Inputs are expected to be bounded (`z`, normally `tanh(drive*x)`).
//! Transcendental curves remain on their exact scalar/ADAA path; this helper
//! accelerates the polynomial recurrence and has a runtime-dispatched scalar
//! fallback.

/// Evaluate `T_order(z)` for a bounded slice.  AVX2/NEON are selected at
/// runtime where available; the scalar recurrence has the same operation
/// order and remains the portable fallback.
#[inline]
pub fn chebyshev_basis_simd(input: &[f32], output: &mut [f32], order: usize) {
    let length = input.len().min(output.len());
    if order == 0 {
        output[..length].fill(1.0);
        return;
    }
    if order == 1 {
        output[..length].copy_from_slice(&input[..length]);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        let simd_len = length / 8 * 8;
        // SAFETY: the runtime feature check establishes the AVX2 precondition.
        unsafe { chebyshev_recurrence_avx2(input, output, simd_len, order) };
        scalar_recurrence(
            &input[simd_len..length],
            &mut output[simd_len..length],
            order,
        );
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        scalar_recurrence(input, output, order);
    }

    #[cfg(target_arch = "aarch64")]
    {
        let simd_len = length / 4 * 4;
        // NEON is baseline SIMD for the supported AArch64 target.
        unsafe { chebyshev_recurrence_neon(input, output, simd_len, order) };
        scalar_recurrence(
            &input[simd_len..length],
            &mut output[simd_len..length],
            order,
        );
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar_recurrence(input, output, order);
}

/// Portable scalar reference for performance comparisons and verification.
#[inline]
pub fn chebyshev_basis_scalar(input: &[f32], output: &mut [f32], order: usize) {
    let length = input.len().min(output.len());
    if order == 0 {
        output[..length].fill(1.0);
    } else if order == 1 {
        output[..length].copy_from_slice(&input[..length]);
    } else {
        scalar_recurrence(&input[..length], &mut output[..length], order);
    }
}

/// Whether this process has a runtime SIMD implementation for the analog
/// polynomial helpers.
#[inline]
pub fn analog_simd_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    let available = std::is_x86_feature_detected!("avx2");
    #[cfg(target_arch = "aarch64")]
    let available = true;
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let available = false;
    available
}

/// Exact bounded soft-clip arithmetic with runtime SIMD dispatch.
#[inline]
pub fn softclip_static_simd(input: &[f32], output: &mut [f32]) {
    let length = input.len().min(output.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            let simd_len = length / 8 * 8;
            // SAFETY: the runtime feature check establishes the AVX2 precondition.
            unsafe { softclip_avx2(input, output, simd_len) };
            scalar_softclip(&input[simd_len..length], &mut output[simd_len..length]);
            return;
        }
        scalar_softclip(input, output);
    }
    #[cfg(target_arch = "aarch64")]
    {
        let simd_len = length / 4 * 4;
        // SAFETY: NEON is baseline SIMD for the supported AArch64 target.
        unsafe { softclip_neon(input, output, simd_len) };
        scalar_softclip(&input[simd_len..length], &mut output[simd_len..length]);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar_softclip(input, output);
}

/// Exact bounded hard-clip arithmetic with runtime SIMD dispatch.
#[inline]
pub fn hardclip_static_simd(input: &[f32], output: &mut [f32]) {
    let length = input.len().min(output.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            let simd_len = length / 8 * 8;
            // SAFETY: the runtime feature check establishes the AVX2 precondition.
            unsafe { hardclip_avx2(input, output, simd_len) };
            scalar_hardclip(&input[simd_len..length], &mut output[simd_len..length]);
            return;
        }
        scalar_hardclip(input, output);
    }
    #[cfg(target_arch = "aarch64")]
    {
        let simd_len = length / 4 * 4;
        // SAFETY: NEON is baseline SIMD for the supported AArch64 target.
        unsafe { hardclip_neon(input, output, simd_len) };
        scalar_hardclip(&input[simd_len..length], &mut output[simd_len..length]);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar_hardclip(input, output);
}

#[inline]
fn scalar_softclip(input: &[f32], output: &mut [f32]) {
    for (input, output) in input.iter().zip(output.iter_mut()) {
        *output = *input / (1.0 + input.abs());
    }
}

#[inline]
fn scalar_hardclip(input: &[f32], output: &mut [f32]) {
    for (input, output) in input.iter().zip(output.iter_mut()) {
        *output = input.clamp(-1.0, 1.0);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn softclip_avx2(input: &[f32], output: &mut [f32], simd_len: usize) {
    use std::arch::x86_64::*;
    for index in (0..simd_len).step_by(8) {
        unsafe {
            let value = _mm256_loadu_ps(input.as_ptr().add(index));
            let absolute = _mm256_andnot_ps(_mm256_set1_ps(-0.0), value);
            let result = _mm256_div_ps(value, _mm256_add_ps(_mm256_set1_ps(1.0), absolute));
            _mm256_storeu_ps(output.as_mut_ptr().add(index), result);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hardclip_avx2(input: &[f32], output: &mut [f32], simd_len: usize) {
    use std::arch::x86_64::*;
    for index in (0..simd_len).step_by(8) {
        unsafe {
            let value = _mm256_loadu_ps(input.as_ptr().add(index));
            let result = _mm256_min_ps(
                _mm256_max_ps(value, _mm256_set1_ps(-1.0)),
                _mm256_set1_ps(1.0),
            );
            _mm256_storeu_ps(output.as_mut_ptr().add(index), result);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn softclip_neon(input: &[f32], output: &mut [f32], simd_len: usize) {
    use std::arch::aarch64::*;
    for index in (0..simd_len).step_by(4) {
        unsafe {
            let value = vld1q_f32(input.as_ptr().add(index));
            let absolute = vabsq_f32(value);
            let result = vdivq_f32(value, vaddq_f32(vdupq_n_f32(1.0), absolute));
            vst1q_f32(output.as_mut_ptr().add(index), result);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hardclip_neon(input: &[f32], output: &mut [f32], simd_len: usize) {
    use std::arch::aarch64::*;
    for index in (0..simd_len).step_by(4) {
        unsafe {
            let value = vld1q_f32(input.as_ptr().add(index));
            let result = vminq_f32(vmaxq_f32(value, vdupq_n_f32(-1.0)), vdupq_n_f32(1.0));
            vst1q_f32(output.as_mut_ptr().add(index), result);
        }
    }
}

#[inline]
fn scalar_recurrence(input: &[f32], output: &mut [f32], order: usize) {
    for (z, result) in input.iter().zip(output.iter_mut()) {
        let mut previous = 1.0;
        let mut current = *z;
        for _ in 2..=order {
            let next = 2.0 * *z * current - previous;
            previous = current;
            current = next;
        }
        *result = current;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn chebyshev_recurrence_avx2(
    input: &[f32],
    output: &mut [f32],
    simd_len: usize,
    order: usize,
) {
    use std::arch::x86_64::*;
    for index in (0..simd_len).step_by(8) {
        unsafe {
            let z = _mm256_loadu_ps(input.as_ptr().add(index));
            let mut previous = _mm256_set1_ps(1.0);
            let mut current = z;
            for _ in 2..=order {
                let next = _mm256_sub_ps(
                    _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(2.0), z), current),
                    previous,
                );
                previous = current;
                current = next;
            }
            _mm256_storeu_ps(output.as_mut_ptr().add(index), current);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn chebyshev_recurrence_neon(
    input: &[f32],
    output: &mut [f32],
    simd_len: usize,
    order: usize,
) {
    use std::arch::aarch64::*;
    for index in (0..simd_len).step_by(4) {
        unsafe {
            let z = vld1q_f32(input.as_ptr().add(index));
            let mut previous = vdupq_n_f32(1.0);
            let mut current = z;
            for _ in 2..=order {
                let next = vsubq_f32(vmulq_f32(vmulq_f32(vdupq_n_f32(2.0), z), current), previous);
                previous = current;
                current = next;
            }
            vst1q_f32(output.as_mut_ptr().add(index), current);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chebyshev_simd_helper_matches_closed_forms_and_recurrence() {
        let input: Vec<f32> = (-32..32).map(|index| index as f32 / 32.0).collect();
        let mut output = vec![0.0; input.len()];
        for order in 0..=8 {
            chebyshev_basis_simd(&input, &mut output, order);
            for (z, result) in input.iter().zip(&output) {
                let mut previous = 1.0_f32;
                let mut current = *z;
                for _ in 2..=order {
                    let next = 2.0 * *z * current - previous;
                    previous = current;
                    current = next;
                }
                let expected = match order {
                    0 => 1.0,
                    1 => *z,
                    _ => current,
                };
                assert!((result - expected).abs() < 1e-6, "order={order} z={z}");
            }
        }
    }

    #[test]
    fn static_curve_simd_helpers_match_scalar_curves() {
        let input: Vec<f32> = (-64..64).map(|index| index as f32 / 16.0).collect();
        let mut soft = vec![0.0; input.len()];
        let mut hard = vec![0.0; input.len()];
        softclip_static_simd(&input, &mut soft);
        hardclip_static_simd(&input, &mut hard);
        for ((value, soft), hard) in input.iter().zip(&soft).zip(&hard) {
            assert!((*soft - *value / (1.0 + value.abs())).abs() < 1e-6);
            assert_eq!(*hard, value.clamp(-1.0, 1.0));
        }
    }
}

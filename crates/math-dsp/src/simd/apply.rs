/// SIMD-optimized gain application for a single gain value
#[inline]
pub fn apply_gain_simd(buffer: &mut [f32], gain: f32) {
    #[cfg(target_arch = "x86_64")]
    if super::has_avx2() {
        // SAFETY: the runtime probe establishes AVX2 support.
        unsafe { apply_gain_avx2(buffer, gain) };
        return;
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use std::arch::aarch64::*;
        let len = buffer.len();
        let gain_vec = unsafe { vdupq_n_f32(gain) };
        let simd_len = (len / 4) * 4;
        for i in (0..simd_len).step_by(4) {
            unsafe {
                let ptr = buffer.as_mut_ptr().add(i);
                let v = vld1q_f32(ptr);
                let res = vmulq_f32(v, gain_vec);
                vst1q_f32(ptr, res);
            }
        }
        for sample in buffer[simd_len..len].iter_mut() {
            *sample *= gain;
        }
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        for val in buffer.iter_mut() {
            *val *= gain;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_gain_avx2(buffer: &mut [f32], gain: f32) {
    use std::arch::x86_64::*;

    let len = buffer.len();
    let gain_vec = _mm256_set1_ps(gain);
    let simd_len = (len / 8) * 8;
    for i in (0..simd_len).step_by(8) {
        unsafe {
            let ptr = buffer.as_mut_ptr().add(i);
            let v = _mm256_loadu_ps(ptr);
            let res = _mm256_mul_ps(v, gain_vec);
            _mm256_storeu_ps(ptr, res);
        }
    }
    for sample in buffer.iter_mut().take(len).skip(simd_len) {
        *sample *= gain;
    }
}

/// SIMD-optimized per-channel gain application
///
/// Scales interleaved samples by per-channel gains: sample `i` (belonging to
/// channel `i % channels`) is multiplied by `gains[i % channels]`.
///
/// The buffer length does not need to be a multiple of `channels`: trailing
/// samples of a partial final frame are scaled with the gains of their
/// respective channels.
///
/// Malformed calls are tolerated as no-ops: if `channels == 0` or
/// `gains.len() < channels`, the buffer is left unchanged.
#[inline]
pub fn apply_per_channel_gain_simd(buffer: &mut [f32], channels: usize, gains: &[f32]) {
    if channels == 0 || gains.len() < channels {
        return;
    }

    let len = buffer.len();
    let num_frames = len / channels;
    let tail_start = num_frames * channels;

    // This is harder to SIMD generically for any channel count.
    // We prioritize common channel counts (1, 2, 6, 8) or use scalar for now.
    // For stereo (channels == 2), we can optimize easily.
    if channels == 2 {
        #[cfg(target_arch = "x86_64")]
        if super::has_avx2() {
            // SAFETY: the runtime probe establishes AVX2 support.
            unsafe { apply_stereo_gain_avx2(buffer, gains) };
            return;
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            use std::arch::aarch64::*;
            let gains_vec = unsafe {
                let g = [gains[0], gains[1], gains[0], gains[1]];
                vld1q_f32(g.as_ptr())
            };
            let simd_len = (num_frames / 2) * 2;
            for i in (0..simd_len).step_by(2) {
                unsafe {
                    let ptr = buffer.as_mut_ptr().add(i * 2);
                    let v = vld1q_f32(ptr);
                    let res = vmulq_f32(v, gains_vec);
                    vst1q_f32(ptr, res);
                }
            }
            for i in simd_len..num_frames {
                buffer[i * 2] *= gains[0];
                buffer[i * 2 + 1] *= gains[1];
            }
            for i in tail_start..len {
                buffer[i] *= gains[i - tail_start];
            }
            return;
        }
    }

    // Fallback scalar loop
    for frame in 0..num_frames {
        for ch in 0..channels {
            buffer[frame * channels + ch] *= gains[ch];
        }
    }
    // Trailing partial frame (len % channels != 0)
    for i in tail_start..len {
        buffer[i] *= gains[i - tail_start];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_stereo_gain_avx2(buffer: &mut [f32], gains: &[f32]) {
    use std::arch::x86_64::*;

    let len = buffer.len();
    let num_frames = len / 2;
    let tail_start = num_frames * 2;
    let gains_vec = _mm256_set_ps(
        gains[1], gains[0], gains[1], gains[0], gains[1], gains[0], gains[1], gains[0],
    );
    let simd_len = (num_frames / 4) * 4;
    for i in (0..simd_len).step_by(4) {
        unsafe {
            let ptr = buffer.as_mut_ptr().add(i * 2);
            let v = _mm256_loadu_ps(ptr);
            let res = _mm256_mul_ps(v, gains_vec);
            _mm256_storeu_ps(ptr, res);
        }
    }
    for i in simd_len..num_frames {
        buffer[i * 2] *= gains[0];
        buffer[i * 2 + 1] *= gains[1];
    }
    for i in tail_start..len {
        buffer[i] *= gains[i - tail_start];
    }
}

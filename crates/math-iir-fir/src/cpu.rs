//! Runtime CPU feature probes for architecture-specialized filter kernels.

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn has_avx2_fma() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

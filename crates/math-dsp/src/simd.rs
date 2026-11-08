mod analog;
mod apply;
mod complex;
mod flush;
mod misc;
mod scale;
mod scoped_ftz;
#[cfg(test)]
mod tests;
mod window;

pub use analog::*;
pub use apply::*;
pub use complex::*;
pub use flush::*;
pub use misc::*;
pub use scale::*;
pub use scoped_ftz::*;
pub use window::*;

// Runtime x86 feature probes used by the SIMD dispatchers in this module.
// `std::is_x86_feature_detected!` checks both CPUID and OS context-switch
// support and caches the result, so these are cheap enough to evaluate at
// each block-level dispatch point.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(super) fn has_avx2() -> bool {
    std::is_x86_feature_detected!("avx2")
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(super) fn has_avx2_fma() -> bool {
    has_avx2() && std::is_x86_feature_detected!("fma")
}

// AArch64 FCMA is an optional extension above the NEON baseline, so it gets
// the same runtime treatment as the optional x86 extensions. On target OSes
// without std feature detection, fall back to the compile-time baseline:
// the kernel may only run when the whole binary already requires FCMA.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) fn has_fcma() -> bool {
    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos",
        target_os = "watchos"
    ))]
    {
        std::arch::is_aarch64_feature_detected!("fcma")
    }
    #[cfg(not(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos",
        target_os = "watchos"
    )))]
    {
        cfg!(target_feature = "fcma")
    }
}

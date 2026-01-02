//! Utilities for handling floating-point denormals (subnormals).
//!
//! This module provides a scoped guard to enable "Flush-to-Zero" (FTZ) and
//! "Denormals-Are-Zero" (DAZ) modes on supported architectures.
//!
//! # Usage
//!
//! ```rust
//! use math_audio_iir_fir::denormals::ScopedFlushToZero;
//!
//! {
//!     let _guard = ScopedFlushToZero::new();
//!     // ... heavy floating point processing ...
//!     // Denormals will be treated as zero here
//! }
//! // Previous state restored
//! ```

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _MM_DENORMALS_ZERO_MASK, _MM_DENORMALS_ZERO_ON, _MM_FLUSH_ZERO_MASK, _MM_FLUSH_ZERO_ON,
    _mm_getcsr, _mm_setcsr,
};

/// A guard that enables Flush-to-Zero (FTZ) and Denormals-Are-Zero (DAZ)
/// when instantiated, and restores the previous state when dropped.
///
/// This is useful for optimizing audio processing loops where denormal numbers
/// (extremely small values close to zero) can cause significant performance penalties.
pub struct ScopedFlushToZero {
    #[cfg(target_arch = "x86_64")]
    original_csr: u32,
    #[cfg(target_arch = "aarch64")]
    original_fpcr: u64,
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    _dummy: (),
}

impl ScopedFlushToZero {
    /// Create a new guard that enables FTZ/DAZ mode on the current thread.
    pub fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let original_csr = _mm_getcsr();
            let mut new_csr = original_csr;
            // set Flush to Zero
            new_csr |= _MM_FLUSH_ZERO_ON;
            // set Denormals Are Zero
            new_csr |= _MM_DENORMALS_ZERO_ON;
            _mm_setcsr(new_csr);

            ScopedFlushToZero { original_csr }
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            let original_fpcr: u64;
            std::arch::asm!("mrs {}, fpcr", out(reg) original_fpcr);

            // Bit 24 is FZ (Flush-to-Zero)
            let new_fpcr = original_fpcr | (1 << 24);

            std::arch::asm!("msr fpcr, {}", in(reg) new_fpcr);

            ScopedFlushToZero { original_fpcr }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            ScopedFlushToZero { _dummy: () }
        }
    }
}

impl Drop for ScopedFlushToZero {
    fn drop(&mut self) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm_setcsr(self.original_csr);
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::asm!("msr fpcr, {}", in(reg) self.original_fpcr);
        }
    }
}

impl Default for ScopedFlushToZero {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scoped_flush_to_zero_creation() {
        let _guard = ScopedFlushToZero::new();
        // Just verify it doesn't crash on the current architecture
    }

    // It's hard to unit test the actual effect without generating denormals,
    // which the compiler might optimize away or handle differently.
    // We rely on the register manipulation logic being correct.
}

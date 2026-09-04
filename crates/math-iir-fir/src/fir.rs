//! FIR filter implementation with windowing functions

use crate::error::{FirError, FirResult};
use crate::traits::{FilterFloat, lit};
use ndarray::Array1;
use std::fmt;

mod design;
mod fir_filter_type;
mod misc;
#[cfg(test)]
mod tests;
mod types;
mod window_type;

pub use fir_filter_type::*;
pub use types::*;
pub use window_type::*;

use design::design_fir_bandpass;
use design::design_fir_bandstop;
use design::design_fir_highpass;
use design::design_fir_lowpass;

/// Represents a single FIR filter.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct Fir<T: FilterFloat = f64> {
    /// The type of filter
    pub filter_type: FirFilterType,
    /// Filter coefficients (taps)
    coeffs: Vec<T>,
    /// Sample rate in Hz
    pub srate: T,
    /// Cutoff frequency (or lower cutoff for bandpass/bandstop) in Hz
    pub freq: T,
    /// Upper cutoff frequency (for bandpass/bandstop) in Hz
    pub freq_upper: Option<T>,
    /// Window type used
    pub window: WindowType,
    /// Kaiser window beta parameter (if applicable)
    pub kaiser_beta: T,
    /// Doubled circular buffer for filter state. The upper half always holds a
    /// contiguous window of the last `n_taps` samples, so the convolution can be
    /// computed as a single slice dot-product without per-tap modulo indexing.
    state: Vec<T>,
    /// Current position in the circular buffer
    state_pos: usize,
    /// Whether the coefficients are symmetric, enabling the half-multiply fast path.
    #[serde(skip)]
    symmetric: bool,
    /// Temporary linear buffer used by block fast paths.
    /// Holds the previous `n_taps - 1` samples followed by the current input
    /// block so the convolution can be computed without circular wrap.
    #[serde(skip)]
    scratch: Vec<T>,
    /// Coefficients in reverse order, used by the linearized block dot-product.
    #[serde(skip)]
    coeffs_rev: Vec<T>,
}

/// Release-checked constructor validation helpers (shared by all `try_*`
/// constructors so debug and release builds reject the same inputs).
fn check_taps(n_taps: usize) -> FirResult<()> {
    if n_taps == 0 {
        return Err(FirError::InvalidTaps { n_taps });
    }
    Ok(())
}

fn check_srate<T: FilterFloat>(srate: T) -> FirResult<()> {
    if srate.partial_cmp(&T::zero()) != Some(std::cmp::Ordering::Greater) {
        return Err(FirError::InvalidSampleRate {
            sample_rate: srate.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

fn check_cutoff<T: FilterFloat>(cutoff: T, srate: T) -> FirResult<()> {
    let nyquist = srate / lit(2.0);
    if !(cutoff > T::zero() && cutoff < nyquist) {
        return Err(FirError::InvalidFrequency {
            freq: cutoff.to_f64().unwrap_or(f64::NAN),
            nyquist: nyquist.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

fn check_band<T: FilterFloat>(freq_low: T, freq_high: T, srate: T) -> FirResult<()> {
    let nyquist = srate / lit(2.0);
    if !(freq_low > T::zero()
        && freq_low < nyquist
        && freq_high > T::zero()
        && freq_high < nyquist
        && freq_low < freq_high)
    {
        return Err(FirError::InvalidBand {
            freq_low: freq_low.to_f64().unwrap_or(f64::NAN),
            freq_high: freq_high.to_f64().unwrap_or(f64::NAN),
            nyquist: nyquist.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

impl<T: FilterFloat> Fir<T> {
    /// Creates a new FIR filter with custom coefficients.
    ///
    /// # Arguments
    /// * `coeffs` - Filter coefficients (taps)
    /// * `srate` - Sample rate in Hz
    ///
    /// # Panics
    /// Panics in all build profiles (validated with [`FirError`]) if:
    /// - `coeffs` is empty
    /// - `srate` is not positive
    ///
    /// For a non-panicking variant, see [`Fir::try_new_custom`].
    pub fn new_custom(coeffs: Vec<T>, srate: T) -> Self {
        Self::try_new_custom(coeffs, srate).expect("Fir::new_custom: invalid parameters")
    }

    /// Fallible version of [`Fir::new_custom`].
    ///
    /// # Errors
    /// Returns [`FirError::EmptyCoeffs`] if `coeffs` is empty, or
    /// [`FirError::InvalidSampleRate`] if `srate` is not positive.
    pub fn try_new_custom(coeffs: Vec<T>, srate: T) -> FirResult<Self> {
        if coeffs.is_empty() {
            return Err(FirError::EmptyCoeffs);
        }
        check_srate(srate)?;

        let n_taps = coeffs.len();
        let symmetric = Self::coeffs_are_symmetric(&coeffs);
        Ok(Fir {
            filter_type: FirFilterType::Custom,
            coeffs,
            srate,
            freq: T::zero(),
            freq_upper: None,
            window: WindowType::Rectangular,
            kaiser_beta: T::zero(),
            state: vec![T::zero(); 2 * n_taps],
            state_pos: n_taps,
            symmetric,
            scratch: Vec::new(),
            coeffs_rev: Vec::new(),
        })
    }

    /// Creates a lowpass FIR filter using the windowed-sinc method.
    ///
    /// # Arguments
    /// * `n_taps` - Number of filter taps (must be odd)
    /// * `cutoff` - Cutoff frequency in Hz
    /// * `srate` - Sample rate in Hz
    /// * `window` - Window function to use
    /// * `kaiser_beta` - Beta parameter for Kaiser window (ignored for other windows)
    ///
    /// # Panics
    /// Panics in all build profiles (validated with [`FirError`]) if:
    /// - `n_taps` is zero
    /// - `srate` is not positive
    /// - `cutoff` is not positive or >= Nyquist frequency (srate/2)
    ///
    /// For a non-panicking variant, see [`Fir::try_lowpass`].
    pub fn lowpass(n_taps: usize, cutoff: T, srate: T, window: WindowType, kaiser_beta: T) -> Self {
        Self::try_lowpass(n_taps, cutoff, srate, window, kaiser_beta)
            .expect("Fir::lowpass: invalid parameters")
    }

    /// Fallible version of [`Fir::lowpass`].
    ///
    /// # Errors
    /// Returns [`FirError::InvalidTaps`] if `n_taps` is zero,
    /// [`FirError::InvalidSampleRate`] if `srate` is not positive, or
    /// [`FirError::InvalidFrequency`] if `cutoff` is not in `(0, Nyquist)`.
    pub fn try_lowpass(
        n_taps: usize,
        cutoff: T,
        srate: T,
        window: WindowType,
        kaiser_beta: T,
    ) -> FirResult<Self> {
        check_taps(n_taps)?;
        check_srate(srate)?;
        check_cutoff(cutoff, srate)?;

        let coeffs = design_fir_lowpass(n_taps, cutoff, srate, window, kaiser_beta);
        let n = coeffs.len();
        let symmetric = Self::coeffs_are_symmetric(&coeffs);
        Ok(Fir {
            filter_type: FirFilterType::Lowpass,
            coeffs,
            srate,
            freq: cutoff,
            freq_upper: None,
            window,
            kaiser_beta,
            state: vec![T::zero(); 2 * n],
            state_pos: n,
            symmetric,
            scratch: Vec::new(),
            coeffs_rev: Vec::new(),
        })
    }

    /// Creates a highpass FIR filter using spectral inversion of a lowpass filter.
    ///
    /// # Arguments
    /// * `n_taps` - Number of filter taps (must be odd)
    /// * `cutoff` - Cutoff frequency in Hz
    /// * `srate` - Sample rate in Hz
    /// * `window` - Window function to use
    /// * `kaiser_beta` - Beta parameter for Kaiser window (ignored for other windows)
    ///
    /// # Panics
    /// Panics in all build profiles (validated with [`FirError`]) if:
    /// - `n_taps` is zero
    /// - `srate` is not positive
    /// - `cutoff` is not positive or >= Nyquist frequency (srate/2)
    ///
    /// For a non-panicking variant, see [`Fir::try_highpass`].
    pub fn highpass(
        n_taps: usize,
        cutoff: T,
        srate: T,
        window: WindowType,
        kaiser_beta: T,
    ) -> Self {
        Self::try_highpass(n_taps, cutoff, srate, window, kaiser_beta)
            .expect("Fir::highpass: invalid parameters")
    }

    /// Fallible version of [`Fir::highpass`].
    ///
    /// # Errors
    /// Returns [`FirError::InvalidTaps`] if `n_taps` is zero,
    /// [`FirError::InvalidSampleRate`] if `srate` is not positive, or
    /// [`FirError::InvalidFrequency`] if `cutoff` is not in `(0, Nyquist)`.
    pub fn try_highpass(
        n_taps: usize,
        cutoff: T,
        srate: T,
        window: WindowType,
        kaiser_beta: T,
    ) -> FirResult<Self> {
        check_taps(n_taps)?;
        check_srate(srate)?;
        check_cutoff(cutoff, srate)?;

        let coeffs = design_fir_highpass(n_taps, cutoff, srate, window, kaiser_beta);
        let n = coeffs.len();
        let symmetric = Self::coeffs_are_symmetric(&coeffs);
        Ok(Fir {
            filter_type: FirFilterType::Highpass,
            coeffs,
            srate,
            freq: cutoff,
            freq_upper: None,
            window,
            kaiser_beta,
            state: vec![T::zero(); 2 * n],
            state_pos: n,
            symmetric,
            scratch: Vec::new(),
            coeffs_rev: Vec::new(),
        })
    }

    /// Creates a bandpass FIR filter by multiplying two sinc functions.
    ///
    /// # Arguments
    /// * `n_taps` - Number of filter taps (must be odd)
    /// * `freq_low` - Lower cutoff frequency in Hz
    /// * `freq_high` - Upper cutoff frequency in Hz
    /// * `srate` - Sample rate in Hz
    /// * `window` - Window function to use
    /// * `kaiser_beta` - Beta parameter for Kaiser window (ignored for other windows)
    ///
    /// # Panics
    /// Panics in all build profiles (validated with [`FirError`]) if:
    /// - `n_taps` is zero
    /// - `srate` is not positive
    /// - `freq_low` is not positive or >= Nyquist frequency (srate/2)
    /// - `freq_high` is not positive or >= Nyquist frequency (srate/2)
    /// - `freq_low` >= `freq_high`
    ///
    /// For a non-panicking variant, see [`Fir::try_bandpass`].
    pub fn bandpass(
        n_taps: usize,
        freq_low: T,
        freq_high: T,
        srate: T,
        window: WindowType,
        kaiser_beta: T,
    ) -> Self {
        Self::try_bandpass(n_taps, freq_low, freq_high, srate, window, kaiser_beta)
            .expect("Fir::bandpass: invalid parameters")
    }

    /// Fallible version of [`Fir::bandpass`].
    ///
    /// # Errors
    /// Returns [`FirError::InvalidTaps`] if `n_taps` is zero,
    /// [`FirError::InvalidSampleRate`] if `srate` is not positive, or
    /// [`FirError::InvalidBand`] if the band edges do not satisfy
    /// `0 < freq_low < freq_high < Nyquist`.
    pub fn try_bandpass(
        n_taps: usize,
        freq_low: T,
        freq_high: T,
        srate: T,
        window: WindowType,
        kaiser_beta: T,
    ) -> FirResult<Self> {
        check_taps(n_taps)?;
        check_srate(srate)?;
        check_band(freq_low, freq_high, srate)?;

        let coeffs = design_fir_bandpass(n_taps, freq_low, freq_high, srate, window, kaiser_beta);
        let n = coeffs.len();
        let symmetric = Self::coeffs_are_symmetric(&coeffs);
        Ok(Fir {
            filter_type: FirFilterType::Bandpass,
            coeffs,
            srate,
            freq: freq_low,
            freq_upper: Some(freq_high),
            window,
            kaiser_beta,
            state: vec![T::zero(); 2 * n],
            state_pos: n,
            symmetric,
            scratch: Vec::new(),
            coeffs_rev: Vec::new(),
        })
    }

    /// Creates a bandstop FIR filter using spectral inversion of a bandpass filter.
    ///
    /// # Arguments
    /// * `n_taps` - Number of filter taps (must be odd)
    /// * `freq_low` - Lower cutoff frequency in Hz
    /// * `freq_high` - Upper cutoff frequency in Hz
    /// * `srate` - Sample rate in Hz
    /// * `window` - Window function to use
    /// * `kaiser_beta` - Beta parameter for Kaiser window (ignored for other windows)
    ///
    /// # Panics
    /// Panics in all build profiles (validated with [`FirError`]) if:
    /// - `n_taps` is zero
    /// - `srate` is not positive
    /// - `freq_low` is not positive or >= Nyquist frequency (srate/2)
    /// - `freq_high` is not positive or >= Nyquist frequency (srate/2)
    /// - `freq_low` >= `freq_high`
    ///
    /// For a non-panicking variant, see [`Fir::try_bandstop`].
    pub fn bandstop(
        n_taps: usize,
        freq_low: T,
        freq_high: T,
        srate: T,
        window: WindowType,
        kaiser_beta: T,
    ) -> Self {
        Self::try_bandstop(n_taps, freq_low, freq_high, srate, window, kaiser_beta)
            .expect("Fir::bandstop: invalid parameters")
    }

    /// Fallible version of [`Fir::bandstop`].
    ///
    /// # Errors
    /// Returns [`FirError::InvalidTaps`] if `n_taps` is zero,
    /// [`FirError::InvalidSampleRate`] if `srate` is not positive, or
    /// [`FirError::InvalidBand`] if the band edges do not satisfy
    /// `0 < freq_low < freq_high < Nyquist`.
    pub fn try_bandstop(
        n_taps: usize,
        freq_low: T,
        freq_high: T,
        srate: T,
        window: WindowType,
        kaiser_beta: T,
    ) -> FirResult<Self> {
        check_taps(n_taps)?;
        check_srate(srate)?;
        check_band(freq_low, freq_high, srate)?;

        let coeffs = design_fir_bandstop(n_taps, freq_low, freq_high, srate, window, kaiser_beta);
        let n = coeffs.len();
        let symmetric = Self::coeffs_are_symmetric(&coeffs);
        Ok(Fir {
            filter_type: FirFilterType::Bandstop,
            coeffs,
            srate,
            freq: freq_low,
            freq_upper: Some(freq_high),
            window,
            kaiser_beta,
            state: vec![T::zero(); 2 * n],
            state_pos: n,
            symmetric,
            scratch: Vec::new(),
            coeffs_rev: Vec::new(),
        })
    }

    /// Returns the number of filter taps (coefficients).
    pub fn n_taps(&self) -> usize {
        self.coeffs.len()
    }

    /// Returns a reference to the filter coefficients.
    pub fn coeffs(&self) -> &[T] {
        &self.coeffs
    }

    /// Resets the filter state to zero.
    pub fn reset(&mut self) {
        self.state.fill(T::zero());
        self.state_pos = self.n_taps();
        self.scratch.clear();
        self.coeffs_rev.clear();
    }

    /// Processes a single audio sample through the filter.
    ///
    /// Denormal handling is caller-owned here: a per-sample FTZ guard would
    /// cost more than the convolution itself for small tap counts. For hot
    /// sample loops, wrap the loop in
    /// [`ScopedFlushToZero`](crate::denormals::ScopedFlushToZero) yourself, or
    /// prefer [`Fir::process_block`], which enables FTZ for the whole block.
    /// See the [FTZ policy](crate::denormals) for details.
    pub fn process(&mut self, x: T) -> T {
        let n_taps = self.coeffs.len();
        // Store input sample in circular buffer and its duplicate so the
        // convolution can read one contiguous slice.
        self.state[self.state_pos] = x;
        self.state[self.state_pos - n_taps] = x;

        // Compute output using convolution
        let y = if self.symmetric {
            self.compute_output_symmetric()
        } else {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            if std::mem::size_of::<T>() == 8 {
                unsafe { self.compute_output_f64_neon(self.state_pos) }
            } else {
                self.compute_output()
            }
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            if std::mem::size_of::<T>() == 8 {
                unsafe { self.compute_output_f64_avx2(self.state_pos) }
            } else {
                self.compute_output()
            }
            #[cfg(not(any(
                all(target_arch = "aarch64", target_feature = "neon"),
                all(target_arch = "x86_64", target_feature = "avx2"),
            )))]
            {
                self.compute_output()
            }
        };

        // Update circular buffer position (kept in the upper half of the
        // doubled state buffer).
        self.state_pos += 1;
        if self.state_pos == 2 * n_taps {
            self.state_pos = n_taps;
        }

        y
    }

    /// Processes a block of audio samples in-place.
    ///
    /// Enables Flush-to-Zero / Denormals-Are-Zero for the duration of the
    /// block via [`ScopedFlushToZero`](crate::denormals::ScopedFlushToZero)
    /// (restored on return), so long silent tails cannot stall on denormals.
    /// See the [FTZ policy](crate::denormals) for details.
    pub fn process_block(&mut self, samples: &mut [T]) {
        let _ftz = crate::denormals::ScopedFlushToZero::new();
        if self.symmetric {
            self.process_block_symmetric(samples);
        } else {
            self.process_block_general(samples);
        }
    }

    fn process_block_general(&mut self, samples: &mut [T]) {
        let n_taps = self.coeffs.len();

        for sample in samples.iter_mut() {
            let x = *sample;
            // Store input sample in circular buffer and its duplicate
            self.state[self.state_pos] = x;
            self.state[self.state_pos - n_taps] = x;

            // Compute output using convolution
            *sample = self.compute_output();

            // Update circular buffer position
            self.state_pos += 1;
            if self.state_pos == 2 * n_taps {
                self.state_pos = n_taps;
            }
        }
    }

    fn process_block_symmetric(&mut self, samples: &mut [T]) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if std::mem::size_of::<T>() == 8 {
            return unsafe { self.process_block_symmetric_f64_avx2(samples) };
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        if std::mem::size_of::<T>() == 8 {
            return unsafe { self.process_block_symmetric_f64_neon(samples) };
        }

        self.process_block_symmetric_linearized(samples);
    }

    fn ensure_coeffs_rev(&mut self) {
        if self.coeffs_rev.is_empty() {
            self.coeffs_rev = self.coeffs.iter().rev().copied().collect();
        }
    }

    /// Scalar linearized symmetric block convolution.
    ///
    /// Copies the previous `n_taps - 1` samples followed by the whole input
    /// block into a contiguous scratch buffer, then computes each output as a
    /// simple slice dot-product with the reversed coefficients.
    fn process_block_symmetric_linearized(&mut self, samples: &mut [T]) {
        let n_taps = self.coeffs.len();
        let len = samples.len();
        if len == 0 {
            return;
        }

        self.ensure_coeffs_rev();

        let scratch_len = n_taps - 1 + len;
        self.scratch.resize(scratch_len, T::zero());
        let prev_start = self.state_pos - n_taps + 1;
        self.scratch[..n_taps - 1].copy_from_slice(&self.state[prev_start..self.state_pos]);
        self.scratch[n_taps - 1..].copy_from_slice(samples);

        for (i, sample) in samples.iter_mut().enumerate().take(len) {
            let window = &self.scratch[i..i + n_taps];
            *sample = self
                .coeffs_rev
                .iter()
                .zip(window.iter())
                .map(|(&c, &s)| c * s)
                .sum();
        }

        let tail_start = len;
        self.state[1..n_taps].copy_from_slice(&self.scratch[tail_start..tail_start + n_taps - 1]);
        self.state[0] = T::zero();
        self.state_pos = n_taps;
    }

    #[inline(always)]
    fn compute_output(&self) -> T {
        let state_pos = self.state_pos;
        let n_taps = self.coeffs.len();
        let start = state_pos - n_taps + 1;

        // The newest sample is at `state_pos`, the oldest at `start`.
        // Coeffs[0] multiplies the newest, so one slice is reversed.
        self.coeffs
            .iter()
            .zip(self.state[start..=state_pos].iter().rev())
            .map(|(&c, &s)| c * s)
            .sum()
    }

    /// aarch64 NEON fast path for a single general (non-symmetric) output
    /// sample when `T` is `f64`.
    ///
    /// # Safety
    /// Must only be called when `T` is `f64` (checked by the caller via
    /// `size_of::<T>() == 8`).
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[inline(always)]
    unsafe fn compute_output_f64_neon(&self, state_pos: usize) -> T {
        // SAFETY: caller verified `T` is `f64`. All pointer accesses are within
        // the `coeffs`/`state` slices, and NEON is available on this target.
        unsafe {
            use std::arch::aarch64::*;

            let coeffs_ptr = self.coeffs.as_ptr() as *const f64;
            let state_ptr = self.state.as_ptr() as *const f64;
            let n_taps = self.coeffs.len();

            let mut acc0 = vdupq_n_f64(0.0);
            let mut acc1 = vdupq_n_f64(0.0);
            let mut i = 0;
            while i + 3 < n_taps {
                let c0 = vld1q_f64(coeffs_ptr.add(i));
                let s0_raw = vld1q_f64(state_ptr.add(state_pos - i - 1));
                let s0 = vextq_f64(s0_raw, s0_raw, 1);
                acc0 = vfmaq_f64(acc0, c0, s0);

                let c1 = vld1q_f64(coeffs_ptr.add(i + 2));
                let s1_raw = vld1q_f64(state_ptr.add(state_pos - i - 3));
                let s1 = vextq_f64(s1_raw, s1_raw, 1);
                acc1 = vfmaq_f64(acc1, c1, s1);

                i += 4;
            }
            let mut acc = vaddq_f64(acc0, acc1);

            while i + 1 < n_taps {
                let c = vld1q_f64(coeffs_ptr.add(i));
                let s_raw = vld1q_f64(state_ptr.add(state_pos - i - 1));
                let s = vextq_f64(s_raw, s_raw, 1);
                acc = vfmaq_f64(acc, c, s);
                i += 2;
            }
            let mut y = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);

            while i < n_taps {
                y = (*coeffs_ptr.add(i)).mul_add(*state_ptr.add(state_pos - i), y);
                i += 1;
            }

            std::mem::transmute_copy::<f64, T>(&y)
        }
    }

    /// x86_64 AVX2/FMA fast path for a single general (non-symmetric) output
    /// sample when `T` is `f64`.
    ///
    /// # Safety
    /// Must only be called when `T` is `f64` (checked by the caller via
    /// `size_of::<T>() == 8`) and when AVX2/FMA are available on the host.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn compute_output_f64_avx2(&self, state_pos: usize) -> T {
        // SAFETY: caller verified `T` is `f64`. All pointer accesses are within
        // the `coeffs`/`state` slices, and this function is compiled with
        // AVX2/FMA enabled.
        unsafe {
            use std::arch::x86_64::*;

            let coeffs_ptr = self.coeffs.as_ptr() as *const f64;
            let state_ptr = self.state.as_ptr() as *const f64;
            let n_taps = self.coeffs.len();

            let mut acc0 = _mm256_setzero_pd();
            let mut acc1 = _mm256_setzero_pd();
            let mut i = 0;
            while i + 7 < n_taps {
                let c0 = _mm256_loadu_pd(coeffs_ptr.add(i));
                let s0_raw = _mm256_loadu_pd(state_ptr.add(state_pos - i - 3));
                let s0 = _mm256_permute4x64_pd(s0_raw, 0x1b);
                acc0 = _mm256_fmadd_pd(c0, s0, acc0);

                let c1 = _mm256_loadu_pd(coeffs_ptr.add(i + 4));
                let s1_raw = _mm256_loadu_pd(state_ptr.add(state_pos - i - 7));
                let s1 = _mm256_permute4x64_pd(s1_raw, 0x1b);
                acc1 = _mm256_fmadd_pd(c1, s1, acc1);

                i += 8;
            }
            let mut acc = _mm256_add_pd(acc0, acc1);

            while i + 3 < n_taps {
                let c = _mm256_loadu_pd(coeffs_ptr.add(i));
                let s_raw = _mm256_loadu_pd(state_ptr.add(state_pos - i - 3));
                let s = _mm256_permute4x64_pd(s_raw, 0x1b);
                acc = _mm256_fmadd_pd(c, s, acc);
                i += 4;
            }

            let hi = _mm256_extractf128_pd(acc, 1);
            let lo = _mm256_castpd256_pd128(acc);
            let sum128 = _mm_add_pd(lo, hi);
            let sum64 = _mm_hadd_pd(sum128, sum128);
            let mut y = _mm_cvtsd_f64(sum64);

            while i < n_taps {
                y = (*coeffs_ptr.add(i)).mul_add(*state_ptr.add(state_pos - i), y);
                i += 1;
            }

            std::mem::transmute_copy::<f64, T>(&y)
        }
    }

    /// Computes one output sample exploiting symmetric coefficients.
    ///
    /// For a symmetric filter `h[i] == h[n_taps-1-i]` the convolution reduces to
    /// roughly half the multiplies: each coefficient pairs the newest and oldest
    /// samples in the current window.
    #[inline(always)]
    fn compute_output_symmetric(&self) -> T {
        let state_pos = self.state_pos;
        let n_taps = self.coeffs.len();
        let start = state_pos - n_taps + 1;
        let half = n_taps / 2;

        // On aarch64 use NEON FMA to keep the symmetric dot product in vector
        // registers (one rounding, one instruction per pair). FilterFloat is only
        // implemented for f32/f64, so size == 8 identifies f64 at runtime.
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        if std::mem::size_of::<T>() == 8 {
            return unsafe { self.compute_output_symmetric_f64(state_pos, start, half) };
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if std::mem::size_of::<T>() == 8 {
            return unsafe { self.compute_output_symmetric_f64_avx2(state_pos, start, half) };
        }

        // Pair the oldest and newest samples in the window; for symmetric
        // coefficients each pair shares one coefficient, halving the multiplies.
        let pair_sums = self.state[start..start + half]
            .iter()
            .zip(self.state[state_pos - half + 1..=state_pos].iter().rev())
            .map(|(&old, &new)| old + new);
        let mut y: T = self.coeffs[..half]
            .iter()
            .zip(pair_sums)
            .map(|(&c, s)| c * s)
            .sum();
        if n_taps % 2 == 1 {
            y += self.coeffs[half] * self.state[start + half];
        }
        y
    }

    /// aarch64 NEON fast path for [`compute_output_symmetric`] when `T` is `f64`.
    ///
    /// # Safety
    /// Must only be called when `T` is `f64` (checked by the caller via
    /// `size_of::<T>() == 8`).
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[inline(always)]
    unsafe fn compute_output_symmetric_f64(
        &self,
        state_pos: usize,
        start: usize,
        half: usize,
    ) -> T {
        // SAFETY: caller verified `T` is `f64`. All pointer accesses are within
        // the `coeffs`/`state` slices, and NEON is available on this target.
        unsafe {
            use std::arch::aarch64::*;

            let coeffs_ptr = self.coeffs.as_ptr() as *const f64;
            let state_ptr = self.state.as_ptr() as *const f64;

            // Use two independent vector accumulators and unroll by 4 pairs
            // per iteration to keep the NEON FMA pipeline full on cores with
            // multi-cycle FMA latency. Load consecutive coefficient/old-state
            // pairs with x2 loads to reduce load-unit pressure.
            let mut acc0 = vdupq_n_f64(0.0);
            let mut acc1 = vdupq_n_f64(0.0);
            let mut i = 0;
            while i + 3 < half {
                let c = vld1q_f64_x2(coeffs_ptr.add(i));
                let old = vld1q_f64_x2(state_ptr.add(start + i));
                let new0 = vld1q_f64(state_ptr.add(state_pos - i - 1));
                let new1 = vld1q_f64(state_ptr.add(state_pos - i - 3));
                let pair0 = vaddq_f64(old.0, vextq_f64(new0, new0, 1));
                let pair1 = vaddq_f64(old.1, vextq_f64(new1, new1, 1));
                acc0 = vfmaq_f64(acc0, c.0, pair0);
                acc1 = vfmaq_f64(acc1, c.1, pair1);
                i += 4;
            }
            let mut acc = vaddq_f64(acc0, acc1);

            // Vector tail: process the remaining pair (if any) with a single
            // accumulator so the scalar loop only handles the last odd pair.
            while i + 1 < half {
                let c = vld1q_f64(coeffs_ptr.add(i));
                let old = vld1q_f64(state_ptr.add(start + i));
                let new = vld1q_f64(state_ptr.add(state_pos - i - 1));
                let pair = vaddq_f64(old, vextq_f64(new, new, 1));
                acc = vfmaq_f64(acc, c, pair);
                i += 2;
            }
            let mut y = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);

            // Scalar tail for the final pair (kept as `mul_add` for one rounding).
            while i < half {
                let pair = *state_ptr.add(start + i) + *state_ptr.add(state_pos - i);
                y = (*coeffs_ptr.add(i)).mul_add(pair, y);
                i += 1;
            }
            let n_taps = self.coeffs.len();
            if n_taps % 2 == 1 {
                let c = *coeffs_ptr.add(half);
                let s = *state_ptr.add(start + half);
                y = c.mul_add(s, y);
            }
            // Caller guarantees T is f64, so the sizes match.
            std::mem::transmute_copy::<f64, T>(&y)
        }
    }

    /// x86_64 AVX2/FMA fast path for [`compute_output_symmetric`] when `T` is `f64`.
    ///
    /// # Safety
    /// Must only be called when `T` is `f64` (checked by the caller via
    /// `size_of::<T>() == 8`) and when AVX2/FMA are available on the host.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn compute_output_symmetric_f64_avx2(
        &self,
        state_pos: usize,
        start: usize,
        half: usize,
    ) -> T {
        // SAFETY: caller verified `T` is `f64`. All pointer accesses are within
        // the `coeffs`/`state` slices, and this function is compiled with
        // AVX2/FMA enabled.
        unsafe {
            use std::arch::x86_64::*;

            let coeffs_ptr = self.coeffs.as_ptr() as *const f64;
            let state_ptr = self.state.as_ptr() as *const f64;

            // Use two independent vector accumulators and unroll by 4 pairs
            // per iteration to keep the AVX2 FMA pipeline full on cores with
            // multi-cycle FMA latency.
            let mut acc0 = _mm256_setzero_pd();
            let mut acc1 = _mm256_setzero_pd();
            let mut i = 0;
            while i + 7 < half {
                let c0 = _mm256_loadu_pd(coeffs_ptr.add(i));
                let old0 = _mm256_loadu_pd(state_ptr.add(start + i));
                let new0_raw = _mm256_loadu_pd(state_ptr.add(state_pos - i - 3));
                let new0 = _mm256_permute4x64_pd(new0_raw, 0x1b);
                let pair0 = _mm256_add_pd(old0, new0);
                acc0 = _mm256_fmadd_pd(c0, pair0, acc0);

                let c1 = _mm256_loadu_pd(coeffs_ptr.add(i + 4));
                let old1 = _mm256_loadu_pd(state_ptr.add(start + i + 4));
                let new1_raw = _mm256_loadu_pd(state_ptr.add(state_pos - i - 7));
                let new1 = _mm256_permute4x64_pd(new1_raw, 0x1b);
                let pair1 = _mm256_add_pd(old1, new1);
                acc1 = _mm256_fmadd_pd(c1, pair1, acc1);

                i += 8;
            }
            let mut acc = _mm256_add_pd(acc0, acc1);

            // Vector tail: process the remaining pairs (if any) with a single
            // accumulator so the scalar loop only handles the last odd pair.
            while i + 3 < half {
                let c = _mm256_loadu_pd(coeffs_ptr.add(i));
                let old = _mm256_loadu_pd(state_ptr.add(start + i));
                let new_raw = _mm256_loadu_pd(state_ptr.add(state_pos - i - 3));
                let new = _mm256_permute4x64_pd(new_raw, 0x1b);
                let pair = _mm256_add_pd(old, new);
                acc = _mm256_fmadd_pd(c, pair, acc);
                i += 4;
            }

            let hi = _mm256_extractf128_pd(acc, 1);
            let lo = _mm256_castpd256_pd128(acc);
            let sum128 = _mm_add_pd(lo, hi);
            let sum64 = _mm_hadd_pd(sum128, sum128);
            let mut y = _mm_cvtsd_f64(sum64);

            // Scalar tail for the final pair (kept as `mul_add` for one rounding).
            while i < half {
                let pair = *state_ptr.add(start + i) + *state_ptr.add(state_pos - i);
                y = (*coeffs_ptr.add(i)).mul_add(pair, y);
                i += 1;
            }
            let n_taps = self.coeffs.len();
            if n_taps % 2 == 1 {
                let c = *coeffs_ptr.add(half);
                let s = *state_ptr.add(start + half);
                y = c.mul_add(s, y);
            }
            // Caller guarantees T is f64, so the sizes match.
            std::mem::transmute_copy::<f64, T>(&y)
        }
    }

    /// x86_64 AVX2/FMA block fast path for symmetric FIR filters when `T` is `f64`.
    ///
    /// Processes four output samples in parallel, keeping four independent
    /// vector accumulators so each coefficient broadcast is amortised over four
    /// outputs and the inner loop avoids shuffles.
    ///
    /// # Safety
    /// Must only be called when `T` is `f64` (checked by the caller via
    /// `size_of::<T>() == 8`) and when AVX2/FMA are available on the host.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2,fma")]
    #[inline(never)]
    unsafe fn process_block_symmetric_f64_avx2(&mut self, samples: &mut [T]) {
        // SAFETY: caller verified `T` is `f64`. Pointer casts are therefore
        // valid, all accesses stay within `coeffs`/`state`/`samples`, and this
        // function is compiled with AVX2/FMA enabled.
        unsafe {
            use std::arch::x86_64::*;

            let coeffs_ptr = self.coeffs.as_ptr() as *const f64;
            let state_ptr = self.state.as_mut_ptr() as *mut f64;
            let samples_ptr = samples.as_mut_ptr() as *mut f64;
            let n_taps = self.coeffs.len();
            let half = n_taps / 2;
            let len = samples.len();
            let mut pos = self.state_pos;
            let mut i = 0;

            // Main loop: handle four samples at once while the next four
            // positions all fit in the upper half of the doubled buffer.
            while i + 3 < len && pos + 3 < 2 * n_taps {
                let x = _mm256_loadu_pd(samples_ptr.add(i));

                // Duplicate the newly arrived samples so the upper half stays
                // a contiguous window of the last `n_taps` samples.
                _mm256_storeu_pd(state_ptr.add(pos), x);
                _mm256_storeu_pd(state_ptr.add(pos - n_taps), x);

                let mut acc0 = _mm256_setzero_pd();
                let mut acc1 = _mm256_setzero_pd();
                let mut acc2 = _mm256_setzero_pd();
                let mut acc3 = _mm256_setzero_pd();
                let mut k = 0;
                while k + 3 < half {
                    let c0 = _mm256_set1_pd(*coeffs_ptr.add(k));
                    let old0 = _mm256_loadu_pd(state_ptr.add(pos - n_taps + 1 + k));
                    let new0 = _mm256_loadu_pd(state_ptr.add(pos - k));
                    let pair0 = _mm256_add_pd(old0, new0);
                    acc0 = _mm256_fmadd_pd(c0, pair0, acc0);

                    let c1 = _mm256_set1_pd(*coeffs_ptr.add(k + 1));
                    let old1 = _mm256_loadu_pd(state_ptr.add(pos - n_taps + 2 + k));
                    let new1 = _mm256_loadu_pd(state_ptr.add(pos - k - 1));
                    let pair1 = _mm256_add_pd(old1, new1);
                    acc1 = _mm256_fmadd_pd(c1, pair1, acc1);

                    let c2 = _mm256_set1_pd(*coeffs_ptr.add(k + 2));
                    let old2 = _mm256_loadu_pd(state_ptr.add(pos - n_taps + 3 + k));
                    let new2 = _mm256_loadu_pd(state_ptr.add(pos - k - 2));
                    let pair2 = _mm256_add_pd(old2, new2);
                    acc2 = _mm256_fmadd_pd(c2, pair2, acc2);

                    let c3 = _mm256_set1_pd(*coeffs_ptr.add(k + 3));
                    let old3 = _mm256_loadu_pd(state_ptr.add(pos - n_taps + 4 + k));
                    let new3 = _mm256_loadu_pd(state_ptr.add(pos - k - 3));
                    let pair3 = _mm256_add_pd(old3, new3);
                    acc3 = _mm256_fmadd_pd(c3, pair3, acc3);

                    k += 4;
                }
                let mut acc = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
                while k < half {
                    let c = _mm256_set1_pd(*coeffs_ptr.add(k));
                    let old = _mm256_loadu_pd(state_ptr.add(pos - n_taps + 1 + k));
                    let new = _mm256_loadu_pd(state_ptr.add(pos - k));
                    let pair = _mm256_add_pd(old, new);
                    acc = _mm256_fmadd_pd(c, pair, acc);
                    k += 1;
                }

                if n_taps % 2 == 1 {
                    let c = _mm256_set1_pd(*coeffs_ptr.add(half));
                    let s = _mm256_loadu_pd(state_ptr.add(pos - half));
                    acc = _mm256_fmadd_pd(c, s, acc);
                }

                _mm256_storeu_pd(samples_ptr.add(i), acc);

                pos += 4;
                if pos >= 2 * n_taps {
                    pos -= n_taps;
                }
                i += 4;
            }

            // Scalar tail for any remaining samples and for crossing the
            // circular-buffer wrap boundary one sample at a time.
            while i < len {
                let x = *samples_ptr.add(i);
                *state_ptr.add(pos) = x;
                *state_ptr.add(pos - n_taps) = x;
                let y = self.compute_output_symmetric();
                *samples_ptr.add(i) = std::mem::transmute_copy::<T, f64>(&y);
                pos += 1;
                if pos == 2 * n_taps {
                    pos = n_taps;
                }
                i += 1;
            }

            self.state_pos = pos;
        }
    }

    /// aarch64 NEON block fast path for symmetric FIR filters when `T` is `f64`.
    ///
    /// Linearizes the filter state and input block into a contiguous scratch
    /// buffer once, then computes eight output samples in parallel with four
    /// 128-bit vector accumulators. This avoids per-sample circular-buffer
    /// wrap handling and duplicated state writes while keeping coefficient
    /// lane broadcasts amortised across eight outputs.
    ///
    /// # Safety
    /// Must only be called when `T` is `f64` (checked by the caller via
    /// `size_of::<T>() == 8`) and when NEON is available on the host.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[target_feature(enable = "neon")]
    #[inline(never)]
    unsafe fn process_block_symmetric_f64_neon(&mut self, samples: &mut [T]) {
        // SAFETY: caller verified `T` is `f64`. Pointer casts are valid, all
        // accesses stay within `coeffs`/`state`/`samples`/`scratch`, and this
        // function is compiled with NEON enabled.
        unsafe {
            use std::arch::aarch64::*;

            let n_taps = self.coeffs.len();
            let len = samples.len();
            if len == 0 {
                return;
            }
            let half = n_taps / 2;

            // Build a linear buffer: previous `n_taps - 1` samples followed by
            // the current input block.
            let scratch_len = n_taps - 1 + len;
            self.scratch.resize(scratch_len, T::zero());
            let prev_start = self.state_pos - n_taps + 1;
            self.scratch[..n_taps - 1].copy_from_slice(&self.state[prev_start..self.state_pos]);
            self.scratch[n_taps - 1..].copy_from_slice(samples);

            let coeffs_ptr = self.coeffs.as_ptr() as *const f64;
            let scratch_ptr = self.scratch.as_ptr() as *const f64;
            let out_ptr = samples.as_mut_ptr() as *mut f64;

            let mut j = 0;
            // Main loop: eight outputs per iteration.
            while j + 7 < len {
                let mut acc0 = vdupq_n_f64(0.0);
                let mut acc1 = vdupq_n_f64(0.0);
                let mut acc2 = vdupq_n_f64(0.0);
                let mut acc3 = vdupq_n_f64(0.0);

                let mut i = 0;
                while i + 1 < half {
                    let c = vld1q_f64(coeffs_ptr.add(i));

                    // Coefficient `i` contributes to all eight outputs.
                    let old0 = vld1q_f64_x4(scratch_ptr.add(j + i));
                    let new0 = vld1q_f64_x4(scratch_ptr.add(j + n_taps - 1 - i));
                    let pair00 = vaddq_f64(old0.0, new0.0);
                    let pair01 = vaddq_f64(old0.1, new0.1);
                    let pair02 = vaddq_f64(old0.2, new0.2);
                    let pair03 = vaddq_f64(old0.3, new0.3);
                    acc0 = vfmaq_laneq_f64::<0>(acc0, pair00, c);
                    acc1 = vfmaq_laneq_f64::<0>(acc1, pair01, c);
                    acc2 = vfmaq_laneq_f64::<0>(acc2, pair02, c);
                    acc3 = vfmaq_laneq_f64::<0>(acc3, pair03, c);

                    // Coefficient `i + 1` contributes to all eight outputs.
                    let old1 = vld1q_f64_x4(scratch_ptr.add(j + i + 1));
                    let new1 = vld1q_f64_x4(scratch_ptr.add(j + n_taps - 2 - i));
                    let pair10 = vaddq_f64(old1.0, new1.0);
                    let pair11 = vaddq_f64(old1.1, new1.1);
                    let pair12 = vaddq_f64(old1.2, new1.2);
                    let pair13 = vaddq_f64(old1.3, new1.3);
                    acc0 = vfmaq_laneq_f64::<1>(acc0, pair10, c);
                    acc1 = vfmaq_laneq_f64::<1>(acc1, pair11, c);
                    acc2 = vfmaq_laneq_f64::<1>(acc2, pair12, c);
                    acc3 = vfmaq_laneq_f64::<1>(acc3, pair13, c);

                    i += 2;
                }
                if i < half {
                    let c = vdupq_n_f64(*coeffs_ptr.add(i));
                    let old = vld1q_f64_x4(scratch_ptr.add(j + i));
                    let new = vld1q_f64_x4(scratch_ptr.add(j + n_taps - 1 - i));
                    let pair0 = vaddq_f64(old.0, new.0);
                    let pair1 = vaddq_f64(old.1, new.1);
                    let pair2 = vaddq_f64(old.2, new.2);
                    let pair3 = vaddq_f64(old.3, new.3);
                    acc0 = vfmaq_f64(acc0, c, pair0);
                    acc1 = vfmaq_f64(acc1, c, pair1);
                    acc2 = vfmaq_f64(acc2, c, pair2);
                    acc3 = vfmaq_f64(acc3, c, pair3);
                }

                if n_taps % 2 == 1 {
                    let c = vdupq_n_f64(*coeffs_ptr.add(half));
                    let s = vld1q_f64_x4(scratch_ptr.add(j + half));
                    acc0 = vfmaq_f64(acc0, c, s.0);
                    acc1 = vfmaq_f64(acc1, c, s.1);
                    acc2 = vfmaq_f64(acc2, c, s.2);
                    acc3 = vfmaq_f64(acc3, c, s.3);
                }

                vst1q_f64(out_ptr.add(j), acc0);
                vst1q_f64(out_ptr.add(j + 2), acc1);
                vst1q_f64(out_ptr.add(j + 4), acc2);
                vst1q_f64(out_ptr.add(j + 6), acc3);

                j += 8;
            }

            // Scalar tail for any remaining samples, using the same arithmetic
            // order as the scalar symmetric path.
            while j < len {
                let window = scratch_ptr.add(j);
                let mut y: f64 = 0.0;
                for k in 0..half {
                    let pair = *window.add(k) + *window.add(n_taps - 1 - k);
                    y = (*coeffs_ptr.add(k)).mul_add(pair, y);
                }
                if n_taps % 2 == 1 {
                    y = (*coeffs_ptr.add(half)).mul_add(*window.add(half), y);
                }
                *out_ptr.add(j) = y;
                j += 1;
            }

            // Restore the doubled-buffer invariant for subsequent calls.
            let tail_start = len;
            self.state[1..n_taps]
                .copy_from_slice(&self.scratch[tail_start..tail_start + n_taps - 1]);
            self.state[0] = T::zero();
            self.state_pos = n_taps;
        }
    }

    /// Returns true if the coefficient vector is symmetric (within floating-point
    /// rounding tolerance). The windowed-sinc designs are mathematically symmetric
    /// but can differ by a few ULPs after normalization, so exact equality is too
    /// strict.
    fn coeffs_are_symmetric(coeffs: &[T]) -> bool {
        let n = coeffs.len();
        let eps = T::epsilon();
        for i in 0..n / 2 {
            let a = coeffs[i];
            let b = coeffs[n - 1 - i];
            let scale = a.abs().max(b.abs()).max(T::one());
            if (a - b).abs() > eps * scale {
                return false;
            }
        }
        true
    }

    /// Calculates the filter's magnitude response at a single frequency `f`.
    pub fn result(&self, f: T) -> T {
        let two_pi: T = lit::<T>(2.0) * T::PI();
        let omega = two_pi * f / self.srate;
        let mut real = T::zero();
        let mut imag = T::zero();

        for (n, &coeff) in self.coeffs.iter().enumerate() {
            let phase = -(lit::<T>(n as f64)) * omega;
            real += coeff * phase.cos();
            imag += coeff * phase.sin();
        }

        (real * real + imag * imag).sqrt()
    }

    /// Calculates the filter's response in dB at a single frequency `f`.
    pub fn log_result(&self, f: T) -> T {
        let result = self.result(f);
        if result > T::zero() {
            lit::<T>(20.0) * result.log10()
        } else {
            lit(-200.0)
        }
    }

    /// Vectorized version to compute the SPL response for a vector of frequencies.
    ///
    /// # Performance
    /// This implementation avoids per-tap allocations by using a direct nested loop.
    pub fn np_log_result(&self, freq: &Array1<T>) -> Array1<T> {
        let mut out = Array1::zeros(freq.len());
        self.np_log_result_into(freq, &mut out);
        out
    }

    /// Vectorized SPL response written into a pre-allocated output buffer.
    ///
    /// Performs no allocation: each grid point accumulates its own `(real,
    /// imag)` pair in registers with the same ascending-tap summation order
    /// as [`Fir::np_log_result`], so FIR-bank response loops can evaluate
    /// every filter without allocating in the hot path. See
    /// [`compute_fir_bank_response_into`](crate::compute_fir_bank_response_into).
    ///
    /// # Panics
    /// Panics in debug builds if `out.len() != freq.len()`.
    pub fn np_log_result_into(&self, freq: &Array1<T>, out: &mut Array1<T>) {
        debug_assert_eq!(freq.len(), out.len());
        let two_pi: T = lit::<T>(2.0) * T::PI();
        let omega_base = two_pi / self.srate;
        let min_val: T = lit(1.0e-20);
        let scale: T = lit::<T>(20.0);

        for (i, &f) in freq.iter().enumerate() {
            let mut real = T::zero();
            let mut imag = T::zero();
            for (n, &coeff) in self.coeffs.iter().enumerate() {
                let phase = -(lit::<T>(n as f64)) * f * omega_base;
                real += coeff * phase.cos();
                imag += coeff * phase.sin();
            }
            let mag_sq = real * real + imag * imag;
            out[i] = scale * mag_sq.sqrt().max(min_val).log10();
        }
    }
}

impl<T: FilterFloat> fmt::Display for Fir<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.filter_type {
            FirFilterType::Bandpass | FirFilterType::Bandstop => {
                write!(
                    f,
                    "Type:{},Freq:{:.1}-{:.1},Rate:{:.1},Taps:{},Window:{}",
                    self.filter_type.short_name(),
                    self.freq.to_f64().unwrap_or(0.0),
                    self.freq_upper.unwrap_or(T::zero()).to_f64().unwrap_or(0.0),
                    self.srate.to_f64().unwrap_or(0.0),
                    self.n_taps(),
                    self.window.short_name()
                )
            }
            _ => {
                write!(
                    f,
                    "Type:{},Freq:{:.1},Rate:{:.1},Taps:{},Window:{}",
                    self.filter_type.short_name(),
                    self.freq.to_f64().unwrap_or(0.0),
                    self.srate.to_f64().unwrap_or(0.0),
                    self.n_taps(),
                    self.window.short_name()
                )
            }
        }
    }
}

/// Compute the FIR bank SPL response at given frequencies.
pub fn fir_bank_spl<T: FilterFloat>(freq: &Array1<T>, fir_bank: &FirBank<T>) -> Array1<T> {
    compute_fir_bank_response(freq, fir_bank)
}

/// Compute a FIR bank SPL response into caller-owned reusable buffers.
///
/// `response` and `filter_scratch` must both have the same length as `freq`.
/// Mirrors [`peq_spl_into`](crate::peq_spl_into) for IIR banks: no per-filter
/// allocation in the loop.
pub fn fir_bank_spl_into<T: FilterFloat>(
    freq: &Array1<T>,
    fir_bank: &FirBank<T>,
    response: &mut Array1<T>,
    filter_scratch: &mut Array1<T>,
) {
    compute_fir_bank_response_into(freq, fir_bank, response, filter_scratch)
}

/// Calculate the recommended preamp gain to avoid clipping for a FIR bank.
///
/// This computes the maximum gain across the audible frequency range
/// and returns the negative of that value.
pub fn fir_bank_preamp_gain<T: FilterFloat>(fir_bank: &FirBank<T>) -> T {
    if fir_bank.is_empty() {
        return T::zero();
    }

    // Sample frequencies across the audible range
    let freqs = Array1::logspace(
        lit(10.0),
        lit::<T>(20.0_f64).log10(),
        lit::<T>(20000.0_f64).log10(),
        500,
    );
    let response = fir_bank_spl(&freqs, fir_bank);

    // Find maximum gain
    let max_gain = response
        .iter()
        .copied()
        .fold(T::neg_infinity(), |a, b| a.max(b));

    // Return negative of max gain (to reduce overall level)
    -max_gain
}

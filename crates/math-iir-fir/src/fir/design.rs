use super::window_type::WindowType;
use super::window_type::generate_window;
use crate::traits::{FilterFloat, lit};

/// Designs a lowpass FIR filter using the windowed-sinc method.
pub(super) fn design_fir_lowpass<T: FilterFloat>(
    n_taps: usize,
    cutoff: T,
    srate: T,
    window: WindowType,
    kaiser_beta: T,
) -> Vec<T> {
    // Ensure odd number of taps for symmetry
    let n = if n_taps.is_multiple_of(2) {
        n_taps + 1
    } else {
        n_taps
    };

    let mut h = vec![T::zero(); n];
    let fc = cutoff / srate; // Normalized cutoff frequency
    let m: T = lit::<T>((n - 1) as f64) / lit::<T>(2.0);
    let two_pi: T = lit::<T>(2.0) * T::PI();

    // Generate ideal lowpass sinc function
    for (i, h_val) in h.iter_mut().enumerate() {
        let x = lit::<T>(i as f64) - m;
        if x == T::zero() {
            *h_val = lit::<T>(2.0) * fc;
        } else {
            *h_val = (two_pi * fc * x).sin() / (T::PI() * x);
        }
    }

    // Apply window
    let window_coeffs = generate_window(n, window, kaiser_beta);
    for (i, h_val) in h.iter_mut().enumerate() {
        *h_val *= window_coeffs[i];
    }

    // Normalize to unit gain at DC
    let sum: T = h.iter().copied().sum();
    if sum.abs() > lit(1e-10) {
        for h_val in h.iter_mut() {
            *h_val /= sum;
        }
    }

    h
}

/// Designs a highpass FIR filter using spectral inversion.
pub(super) fn design_fir_highpass<T: FilterFloat>(
    n_taps: usize,
    cutoff: T,
    srate: T,
    window: WindowType,
    kaiser_beta: T,
) -> Vec<T> {
    // Start with lowpass filter
    let mut h = design_fir_lowpass(n_taps, cutoff, srate, window, kaiser_beta);

    // Spectral inversion: negate all coefficients and add 1 to center tap
    let m = h.len() / 2;
    for h_val in h.iter_mut() {
        *h_val = -*h_val;
    }
    h[m] += T::one();

    h
}

/// Designs a bandpass FIR filter.
pub(super) fn design_fir_bandpass<T: FilterFloat>(
    n_taps: usize,
    freq_low: T,
    freq_high: T,
    srate: T,
    window: WindowType,
    kaiser_beta: T,
) -> Vec<T> {
    // Ensure odd number of taps
    let n = if n_taps.is_multiple_of(2) {
        n_taps + 1
    } else {
        n_taps
    };

    let mut h = vec![T::zero(); n];
    let fc_low = freq_low / srate;
    let fc_high = freq_high / srate;
    let m: T = lit::<T>((n - 1) as f64) / lit::<T>(2.0);
    let two_pi: T = lit::<T>(2.0) * T::PI();

    // Generate ideal bandpass filter (difference of two sinc functions)
    for (i, h_val) in h.iter_mut().enumerate() {
        let x = lit::<T>(i as f64) - m;
        if x == T::zero() {
            *h_val = lit::<T>(2.0) * (fc_high - fc_low);
        } else {
            let sinc_high = (two_pi * fc_high * x).sin() / (T::PI() * x);
            let sinc_low = (two_pi * fc_low * x).sin() / (T::PI() * x);
            *h_val = sinc_high - sinc_low;
        }
    }

    // Apply window
    let window_coeffs = generate_window(n, window, kaiser_beta);
    for (i, h_val) in h.iter_mut().enumerate() {
        *h_val *= window_coeffs[i];
    }

    // Normalize at the middle of the requested passband. A bandpass has zero
    // DC gain, so tap-sum normalization (used by a lowpass) is not meaningful.
    let center_freq = (freq_low + freq_high) / lit::<T>(2.0);
    let omega = two_pi * center_freq / srate;
    let mut real = T::zero();
    let mut imag = T::zero();
    for (i, &coefficient) in h.iter().enumerate() {
        let phase = -lit::<T>(i as f64) * omega;
        real += coefficient * phase.cos();
        imag += coefficient * phase.sin();
    }
    let center_gain = (real * real + imag * imag).sqrt();
    if center_gain > lit(1e-10) {
        for coefficient in &mut h {
            *coefficient /= center_gain;
        }
    }

    h
}

/// Designs a bandstop FIR filter using spectral inversion.
pub(super) fn design_fir_bandstop<T: FilterFloat>(
    n_taps: usize,
    freq_low: T,
    freq_high: T,
    srate: T,
    window: WindowType,
    kaiser_beta: T,
) -> Vec<T> {
    // Start with bandpass filter
    let mut h = design_fir_bandpass(n_taps, freq_low, freq_high, srate, window, kaiser_beta);

    // Spectral inversion: negate all coefficients and add 1 to center tap
    let m = h.len() / 2;
    for h_val in h.iter_mut() {
        *h_val = -*h_val;
    }
    h[m] += T::one();

    // Spectral inversion of a finite, windowed bandpass is only approximately
    // unity at DC. Normalize the completed bandstop exactly at that passband.
    let dc_gain: T = h.iter().copied().sum();
    if dc_gain.abs() > lit(1e-10) {
        for coefficient in &mut h {
            *coefficient /= dc_gain;
        }
    }

    h
}

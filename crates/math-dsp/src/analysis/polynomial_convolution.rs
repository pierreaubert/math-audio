//! Matrix-free polynomial convolution for offline nonlinear response identification.
//!
//! The model is `y = sum(h[k] * x.powi(k + 1))`, with causal finite kernels
//! and ordinary linear convolution. The supplied output length may truncate
//! the convolution tail. Its adjoint uses the same truncation. This operator
//! neither fits a model nor establishes identifiability, harmonic support,
//! calibration, or a physical distortion metric. Callers must validate those
//! separately. Construction plans FFTs; repeated applications reuse buffers.

// Rust guideline compliant 2026-02-21
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::Arc;

/// Offline FFT operator and adjoint for causal polynomial convolution.
///
/// Coefficients are order-major: all first-order taps, then second-order taps,
/// through the requested order. Buffers are private and reused. This type is
/// intended for a worker thread, rather than an audio callback.
pub struct PolynomialConvolutionOperator {
    spectra: Vec<Vec<Complex<f64>>>,
    forward_fft: Arc<dyn Fft<f64>>,
    inverse_fft: Arc<dyn Fft<f64>>,
    work: Vec<Complex<f64>>,
    sum: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    coefficients: Vec<f64>,
    support: usize,
    output_len: usize,
    buffer_bytes: usize,
}

impl std::fmt::Debug for PolynomialConvolutionOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolynomialConvolutionOperator")
            .field("orders", &self.spectra.len())
            .field("support", &self.support)
            .field("output_len", &self.output_len)
            .field("fft_len", &self.work.len())
            .finish_non_exhaustive()
    }
}

impl PolynomialConvolutionOperator {
    /// Prepare spectra and reusable buffers within explicit shape limits.
    ///
    /// `max_fft_samples` is checked before FFT planning or buffer allocation.
    /// Orders one through five match the experimental ESS polynomial model.
    /// Input amplitude is retained, rather than normalized independently.
    /// `max_buffer_bytes` bounds this operator's vectors, including temporary
    /// powers and FFT scratch. FFT planner storage and allocator overhead are
    /// excluded; these limits are not a process RSS or wall-time guarantee.
    ///
    /// # Errors
    /// Rejects empty/non-finite input, invalid dimensions, unsupported order,
    /// arithmetic overflow, excessive FFT size, or non-finite input powers.
    pub fn new(
        input: &[f64],
        orders: usize,
        support: usize,
        output_len: usize,
        max_fft_samples: usize,
        max_buffer_bytes: usize,
    ) -> Result<Self, String> {
        if input.is_empty() || input.iter().any(|x| !x.is_finite()) {
            return Err("polynomial input must be nonempty and finite".into());
        }
        // Five is the highest order validated by the ESS diagnostic corpus.
        if !(1..=5).contains(&orders) || support == 0 || output_len == 0 {
            return Err("polynomial orders/support/output length are invalid".into());
        }
        let full_len = input
            .len()
            .checked_add(support - 1)
            .ok_or("polynomial convolution length overflow")?;
        if output_len > full_len {
            return Err("polynomial output extends beyond the linear convolution".into());
        }
        let fft_len = full_len
            .checked_next_power_of_two()
            .ok_or("polynomial FFT length overflow")?;
        if fft_len > max_fft_samples {
            return Err("polynomial FFT sample limit exceeded".into());
        }
        let coefficient_len = orders
            .checked_mul(support)
            .ok_or("polynomial coefficient length overflow")?;
        let checked_bytes = |scratch_len: usize| -> Result<usize, String> {
            let complex_count = fft_len
                .checked_mul(orders + 2)
                .and_then(|count| count.checked_add(scratch_len))
                .ok_or("polynomial buffer length overflow")?;
            let bytes = complex_count
                .checked_mul(std::mem::size_of::<Complex<f64>>())
                .and_then(|bytes| {
                    coefficient_len
                        .checked_add(input.len())
                        .and_then(|count| count.checked_mul(std::mem::size_of::<f64>()))
                        .and_then(|extra| bytes.checked_add(extra))
                })
                .ok_or("polynomial buffer byte count overflow")?;
            if bytes > max_buffer_bytes || bytes > isize::MAX as usize {
                return Err("polynomial buffer byte limit exceeded".into());
            }
            Ok(bytes)
        };
        checked_bytes(0)?;
        let mut planner = FftPlanner::<f64>::new();
        let forward_fft = planner.plan_fft_forward(fft_len);
        let inverse_fft = planner.plan_fft_inverse(fft_len);
        let scratch_len = forward_fft
            .get_inplace_scratch_len()
            .max(inverse_fft.get_inplace_scratch_len());
        let buffer_bytes = checked_bytes(scratch_len)?;
        let mut scratch = vec![Complex::default(); scratch_len];
        let mut spectra = Vec::with_capacity(orders);
        let mut powers = input.to_vec();
        for order in 0..orders {
            let mut spectrum = vec![Complex::default(); fft_len];
            for (value, power) in spectrum.iter_mut().zip(&powers) {
                value.re = *power;
            }
            forward_fft.process_with_scratch(&mut spectrum, &mut scratch);
            if spectrum
                .iter()
                .any(|z| !z.re.is_finite() || !z.im.is_finite())
            {
                return Err("polynomial input spectrum is non-finite".into());
            }
            spectra.push(spectrum);
            if order + 1 < orders {
                for (power, x) in powers.iter_mut().zip(input) {
                    *power *= x;
                    if !power.is_finite() {
                        return Err("polynomial input power is non-finite".into());
                    }
                }
            }
        }
        Ok(Self {
            spectra,
            forward_fft,
            inverse_fft,
            work: vec![Complex::default(); fft_len],
            sum: vec![Complex::default(); fft_len],
            scratch,
            coefficients: vec![0.0; coefficient_len],
            support,
            output_len,
            buffer_bytes,
        })
    }

    /// Return the number of order-major kernel coefficients.
    #[must_use]
    pub fn coefficient_len(&self) -> usize {
        self.coefficients.len()
    }

    /// Return the vector storage bound including construction's temporary input powers.
    #[must_use]
    pub fn buffer_bytes(&self) -> usize {
        self.buffer_bytes
    }

    /// Return the exact number of retained output samples.
    #[must_use]
    pub fn output_len(&self) -> usize {
        self.output_len
    }

    /// Apply the model while preserving the output on an error.
    ///
    /// # Errors
    /// Rejects mismatched dimensions, non-finite coefficients, or overflow.
    pub fn apply(&mut self, coefficients: &[f64], output: &mut [f64]) -> Result<(), String> {
        if coefficients.len() != self.coefficient_len() || output.len() != self.output_len {
            return Err("polynomial forward dimensions mismatch".into());
        }
        if coefficients.iter().any(|x| !x.is_finite()) {
            return Err("polynomial coefficients must be finite".into());
        }
        self.sum.fill(Complex::default());
        for (kernel, spectrum) in coefficients.chunks_exact(self.support).zip(&self.spectra) {
            self.work.fill(Complex::default());
            for (value, coefficient) in self.work.iter_mut().zip(kernel) {
                value.re = *coefficient;
            }
            self.forward_fft
                .process_with_scratch(&mut self.work, &mut self.scratch);
            for ((sum, kernel_bin), input_bin) in self.sum.iter_mut().zip(&self.work).zip(spectrum)
            {
                *sum += *kernel_bin * *input_bin;
            }
        }
        self.inverse_fft
            .process_with_scratch(&mut self.sum, &mut self.scratch);
        let scale = 1.0 / self.work.len() as f64;
        if self.sum[..self.output_len]
            .iter()
            .any(|z| !(z.re * scale).is_finite())
        {
            return Err("polynomial forward result is non-finite".into());
        }
        for (value, sample) in output.iter_mut().zip(&self.sum) {
            *value = sample.re * scale;
        }
        Ok(())
    }

    /// Apply the transpose of the retained-output model, preserving output on errors.
    ///
    /// # Errors
    /// Rejects mismatched dimensions, non-finite samples, or overflow.
    pub fn apply_adjoint(&mut self, samples: &[f64], output: &mut [f64]) -> Result<(), String> {
        if samples.len() != self.output_len || output.len() != self.coefficient_len() {
            return Err("polynomial adjoint dimensions mismatch".into());
        }
        if samples.iter().any(|x| !x.is_finite()) {
            return Err("polynomial adjoint samples must be finite".into());
        }
        self.sum.fill(Complex::default());
        for (value, sample) in self.sum.iter_mut().zip(samples) {
            value.re = *sample;
        }
        self.forward_fft
            .process_with_scratch(&mut self.sum, &mut self.scratch);
        let scale = 1.0 / self.work.len() as f64;
        for (kernel, spectrum) in self
            .coefficients
            .chunks_exact_mut(self.support)
            .zip(&self.spectra)
        {
            for ((value, input_bin), sample_bin) in
                self.work.iter_mut().zip(spectrum).zip(&self.sum)
            {
                *value = input_bin.conj() * *sample_bin;
            }
            self.inverse_fft
                .process_with_scratch(&mut self.work, &mut self.scratch);
            for (value, sample) in kernel.iter_mut().zip(&self.work) {
                *value = sample.re * scale;
            }
        }
        if self.coefficients.iter().any(|x| !x.is_finite()) {
            return Err("polynomial adjoint result is non-finite".into());
        }
        output.copy_from_slice(&self.coefficients);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct(input: &[f64], taps: &[f64], support: usize, len: usize) -> Vec<f64> {
        let mut y = vec![0.0; len];
        for (order, kernel) in taps.chunks_exact(support).enumerate() {
            for (delay, tap) in kernel.iter().enumerate() {
                for (time, x) in input.iter().enumerate() {
                    if time + delay < len {
                        y[time + delay] += tap * x.powi(order as i32 + 1);
                    }
                }
            }
        }
        y
    }

    #[test]
    fn fft_matches_signed_direct_convolution_and_transpose_for_truncated_tails() {
        let x: Vec<f64> = (0..37)
            .map(|i| ((i * 17 % 29) as f64 - 14.0) / 19.0)
            .collect();
        for orders in 1..=5 {
            for support in [1, 7, 43] {
                for len in [1, x.len(), x.len() + support - 1] {
                    let h: Vec<f64> = (0..orders * support)
                        .map(|i| ((i * 13 % 17) as f64 - 8.0) / 23.0)
                        .collect();
                    let v: Vec<f64> = (0..len)
                        .map(|i| ((i * 7 % 19) as f64 - 9.0) / 11.0)
                        .collect();
                    let expected = direct(&x, &h, support, len);
                    let mut operator =
                        PolynomialConvolutionOperator::new(&x, orders, support, len, 128, 1 << 20)
                            .unwrap();
                    let mut actual = vec![0.0; len];
                    operator.apply(&h, &mut actual).unwrap();
                    for (a, b) in actual.iter().zip(&expected) {
                        assert!((a - b).abs() < 2e-13, "{a} != {b}");
                    }
                    let mut transpose = vec![0.0; h.len()];
                    operator.apply_adjoint(&v, &mut transpose).unwrap();
                    // Each transpose component is independently measured using one
                    // time-domain unit kernel, not the FFT implementation.
                    for (index, a) in transpose.iter().enumerate() {
                        let mut unit = vec![0.0; h.len()];
                        unit[index] = 1.0;
                        let b: f64 = direct(&x, &unit, support, len)
                            .iter()
                            .zip(&v)
                            .map(|(a, b)| a * b)
                            .sum();
                        assert!((a - b).abs() < 2e-13, "adjoint {a} != {b}");
                    }
                    let left: f64 = actual.iter().zip(&v).map(|(a, b)| a * b).sum();
                    let right: f64 = h.iter().zip(&transpose).map(|(a, b)| a * b).sum();
                    assert!((left - right).abs() < 2e-12);
                    operator.apply(&h, &mut actual).unwrap();
                    for (a, b) in actual.iter().zip(&expected) {
                        assert!((a - b).abs() < 2e-13);
                    }
                }
            }
        }
    }

    #[test]
    fn invalid_shapes_and_numerics_fail_without_replacing_caller_output() {
        for (input, orders, support, len, cap) in [
            (vec![], 1, 1, 1, 1),
            (vec![f64::NAN], 1, 1, 1, 1),
            (vec![1.0], 6, 1, 1, 1),
            (vec![1.0], 1, 0, 1, 1),
            (vec![1.0], 1, 1, 2, 2),
            (vec![1.0; 8], 1, 2, 8, 8),
            (vec![f64::MAX], 2, 1, 1, 1),
            (vec![1.0], 1, usize::MAX, 1, 8),
        ] {
            assert!(
                PolynomialConvolutionOperator::new(&input, orders, support, len, cap, 1 << 20)
                    .is_err()
            );
        }
        let mut operator =
            PolynomialConvolutionOperator::new(&[1.0, -1.0], 2, 2, 3, 4, 1 << 20).unwrap();
        let bytes = operator.buffer_bytes();
        assert!(PolynomialConvolutionOperator::new(&[1.0, -1.0], 2, 2, 3, 4, bytes - 1).is_err());
        assert!(PolynomialConvolutionOperator::new(&[1.0, -1.0], 2, 2, 3, 4, bytes).is_ok());
        let mut output = vec![7.0; 3];
        assert!(operator.apply(&[f64::NAN; 4], &mut output).is_err());
        assert_eq!(output, vec![7.0; 3]);
        assert!(operator.apply(&[f64::MAX; 4], &mut output).is_err());
        assert_eq!(output, vec![7.0; 3]);
        let mut coefficients = vec![7.0; 4];
        assert!(
            operator
                .apply_adjoint(&[f64::INFINITY; 3], &mut coefficients)
                .is_err()
        );
        assert_eq!(coefficients, vec![7.0; 4]);
        assert!(
            operator
                .apply_adjoint(&[f64::MAX; 3], &mut coefficients)
                .is_err()
        );
        assert_eq!(coefficients, vec![7.0; 4]);
    }
}

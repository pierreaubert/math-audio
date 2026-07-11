//! Fixed-frequency graphic equalizer with learnable per-band gains.

#![allow(
    clippy::uninlined_format_args,
    reason = "format strings are clearer with explicit arguments in error messages"
)]

use ndarray::{ArrayD, IxDyn};

use crate::error::AutodiffError;
use crate::iir::sos_filter::SosFilter;
use crate::module::DiffModule;
use crate::tensor::DiffTensor;

/// Default quality factor for the peaking biquad sections.
pub const DEFAULT_Q: f64 = 1.0 / std::f64::consts::SQRT_2;

/// ISO octave center frequencies from 31.25 Hz to 16 kHz.
const ISO_CENTER_FREQUENCIES: [f64; 10] = [
    31.25, 62.5, 125.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0, 16_000.0,
];

/// Build normalized biquad coefficients for a 0 dB peaking filter at `fc`.
///
/// The returned `(b, a)` are normalized so that `a[0] == 1.0`. The numerator
/// can be scaled by a linear gain to obtain the final graphic-EQ section.
fn peak_coefficients(fc: f64, q: f64, fs: f64) -> ([f64; 3], [f64; 3]) {
    let coeffs =
        math_audio_iir_fir::Biquad::new(math_audio_iir_fir::BiquadFilterType::Peak, fc, fs, q, 0.0)
            .coefficients();
    (
        [coeffs.b0, coeffs.b1, coeffs.b2],
        [1.0, coeffs.a1, coeffs.a2],
    )
}

/// Fixed-frequency bank of peaking biquads at ISO octave center frequencies.
///
/// Each band/channel has a single learnable linear gain. The underlying
/// cascade is a [`SosFilter`] whose coefficients are rebuilt from the gain
/// parameters on every forward/backward pass.
#[derive(Debug, Clone)]
pub struct GraphicEq {
    pub nfft: usize,
    pub fs: f64,
    pub n_bands: usize,
    pub n_channels: usize,
    pub frequencies: Vec<f64>,
    pub alias_decay_db: f64,
    pub param: ArrayD<f64>,
    pub param_grad: ArrayD<f64>,
    inner: SosFilter,
}

impl GraphicEq {
    /// Create a new graphic equalizer.
    ///
    /// `n_bands` selects the first `n_bands` frequencies from the ISO octave
    /// center list. `n_channels` is both the input and output channel count;
    /// each channel is processed independently by the same filter bank.
    ///
    /// # Errors
    ///
    /// Returns an error if `nfft`, `n_bands`, or `n_channels` is zero, or if
    /// `n_bands` exceeds the number of available ISO center frequencies.
    pub fn new(
        nfft: usize,
        fs: f64,
        n_bands: usize,
        n_channels: usize,
        alias_decay_db: f64,
    ) -> Result<Self, AutodiffError> {
        if nfft == 0 {
            return Err(AutodiffError::Message(format!(
                "GraphicEq: nfft must be greater than 0, got {nfft}"
            )));
        }
        if n_bands == 0 {
            return Err(AutodiffError::Message(format!(
                "GraphicEq: n_bands must be greater than 0, got {n_bands}"
            )));
        }
        if n_channels == 0 {
            return Err(AutodiffError::Message(format!(
                "GraphicEq: n_channels must be greater than 0, got {n_channels}"
            )));
        }
        if n_bands > ISO_CENTER_FREQUENCIES.len() {
            return Err(AutodiffError::Message(format!(
                "GraphicEq: n_bands {} exceeds available ISO center frequencies {}",
                n_bands,
                ISO_CENTER_FREQUENCIES.len()
            )));
        }
        if fs <= 0.0 || !fs.is_finite() {
            return Err(AutodiffError::Message(format!(
                "GraphicEq: sample rate must be positive and finite, got {}",
                fs
            )));
        }

        let frequencies: Vec<f64> = ISO_CENTER_FREQUENCIES[..n_bands]
            .iter()
            .map(|&f| f.min(fs * 0.499))
            .collect();

        let param = ArrayD::ones(IxDyn(&[n_bands, n_channels]));
        let param_grad = ArrayD::zeros(IxDyn(&[n_bands, n_channels]));
        let inner = SosFilter::new(nfft, n_bands, n_channels, n_channels, alias_decay_db)?;

        let mut geq = Self {
            nfft,
            fs,
            n_bands,
            n_channels,
            frequencies,
            alias_decay_db,
            param,
            param_grad,
            inner,
        };
        geq.rebuild_inner();
        Ok(geq)
    }

    fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }

    /// Fill an SOS coefficient tensor for a graphic EQ.
    ///
    /// `param` must have shape `(n_bands, 6, n_channels, n_channels)`.
    /// Per-channel diagonal coefficients realize the peaking filters; all
    /// off-diagonal couplings are set to a zero-response section
    /// (`b = [0, 0, 0]`, `a = [1, 0, 0]`) so that the cascade response stays
    /// finite and the output channel matrix remains diagonal.
    fn fill_sos_param(param: &mut ArrayD<f64>, frequencies: &[f64], gains: &ArrayD<f64>, fs: f64) {
        let n_bands = frequencies.len();
        let n_channels = gains.shape()[1];
        param.fill(0.0);
        for band in 0..n_bands {
            for out_ch in 0..n_channels {
                for in_ch in 0..n_channels {
                    param[[band, 3, out_ch, in_ch]] = 1.0;
                }
            }
        }
        for (band, &fc) in frequencies.iter().enumerate() {
            let (b_peak, a) = peak_coefficients(fc, DEFAULT_Q, fs);
            for ch in 0..n_channels {
                let gain = gains[[band, ch]];
                for (tap, &b_tap) in b_peak.iter().enumerate() {
                    param[[band, tap, ch, ch]] = gain * b_tap;
                    param[[band, 3 + tap, ch, ch]] = a[tap];
                }
            }
        }
    }

    /// Rebuild the inner SOS coefficients from the current gain parameters.
    fn rebuild_inner(&mut self) {
        Self::fill_sos_param(
            &mut self.inner.param,
            &self.frequencies,
            &self.param,
            self.fs,
        );
    }

    /// Build a fresh inner SOS filter reflecting the current parameters.
    ///
    /// Used by the immutable `forward` pass.
    fn build_fresh_inner(&self) -> SosFilter {
        let mut inner = self.inner.clone();
        Self::fill_sos_param(&mut inner.param, &self.frequencies, &self.param, self.fs);
        inner
    }

    /// Map the inner SOS coefficient gradients back to per-band gain gradients.
    fn map_inner_grad(&mut self) {
        for (band, &fc) in self.frequencies.iter().enumerate() {
            let (b_peak, _a) = peak_coefficients(fc, DEFAULT_Q, self.fs);
            for ch in 0..self.n_channels {
                let mut accum = 0.0;
                for (tap, &b_tap) in b_peak.iter().enumerate() {
                    accum += self.inner.param_grad[[band, tap, ch, ch]] * b_tap;
                }
                self.param_grad[[band, ch]] += accum;
            }
        }
    }
}

impl DiffModule<f64> for GraphicEq {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "GraphicEq::forward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let n_bins = input_shape[1];
        let n_in = input_shape[2];
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "GraphicEq::forward: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if n_in != self.n_channels {
            return Err(AutodiffError::Message(format!(
                "GraphicEq::forward: expected {} input channels, got {}",
                self.n_channels, n_in
            )));
        }

        let inner = self.build_fresh_inner();
        inner.forward(input)
    }

    fn backward(
        &mut self,
        input: &DiffTensor<f64>,
        output: &DiffTensor<f64>,
        grad_output: &DiffTensor<f64>,
    ) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        let grad_shape = grad_output.data.shape();
        let output_shape = output.data.shape();
        if grad_shape != output_shape {
            return Err(AutodiffError::Message(format!(
                "GraphicEq::backward: grad_output shape {:?} does not match output shape {:?}",
                grad_shape, output_shape
            )));
        }
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "GraphicEq::backward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let n_bins = input_shape[1];
        let n_in = input_shape[2];
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "GraphicEq::backward: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if n_in != self.n_channels {
            return Err(AutodiffError::Message(format!(
                "GraphicEq::backward: expected {} input channels, got {}",
                self.n_channels, n_in
            )));
        }

        self.rebuild_inner();
        self.inner.zero_grad();
        let grad_input = self.inner.backward(input, output, grad_output)?;
        self.map_inner_grad();

        Ok(grad_input)
    }

    fn input_channels(&self) -> usize {
        self.n_channels
    }

    fn output_channels(&self) -> usize {
        self.n_channels
    }

    fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }

    fn parameters(&self) -> Vec<&ArrayD<f64>> {
        vec![&self.param]
    }

    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<f64>> {
        vec![&mut self.param]
    }

    fn gradients(&self) -> Vec<&ArrayD<f64>> {
        vec![&self.param_grad]
    }

    fn zero_grad(&mut self) {
        self.param_grad.fill(0.0);
        self.inner.zero_grad();
    }
}

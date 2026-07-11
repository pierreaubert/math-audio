//! Differentiable parametric equalizer: cascade of peaking or shelving sections.

#![allow(
    clippy::cast_precision_loss,
    reason = "nfft is an audio buffer length that fits exactly in f64 for practical values"
)]
#![allow(
    clippy::similar_names,
    reason = "coefficient derivative names are intentionally paired (db_dparam/da_dparam)"
)]
#![allow(
    clippy::uninlined_format_args,
    reason = "format strings are clearer with explicit arguments in error messages"
)
]

use ndarray::{ArrayD, IxDyn};

use crate::error::AutodiffError;
use crate::iir::sos_filter::SosFilter;
use crate::module::DiffModule;
use crate::tensor::DiffTensor;

/// Minimum allowed quality factor.
const Q_MIN: f64 = 0.1;
/// Maximum allowed quality factor.
const Q_MAX: f64 = 20.0;
/// Minimum allowed gain in dB.
const GAIN_DB_MIN: f64 = -60.0;
/// Maximum allowed gain in dB.
const GAIN_DB_MAX: f64 = 60.0;

/// Parametric EQ band type.
#[derive(Debug, Clone, Copy)]
pub enum PeqBandType {
    /// Peaking (bell) filter.
    Peak,
    /// Low shelving filter.
    Lowshelf,
    /// High shelving filter.
    Highshelf,
}

/// Coefficients and their physical parameter derivatives for a single
/// peaking/shelving section.
#[derive(Debug, Clone, Copy)]
struct SectionCoeffs {
    b: [f64; 3],
    a: [f64; 3],
    /// `db_dparam[tap][param]` w.r.t. physical parameters (`fc`, `Q`, `gain_db`).
    db_dparam: [[f64; 3]; 3],
    /// `da_dparam[tap][param]` w.r.t. physical parameters.
    da_dparam: [[f64; 3]; 3],
}

impl SectionCoeffs {
    fn zeros() -> Self {
        Self {
            b: [0.0; 3],
            a: [0.0; 3],
            db_dparam: [[0.0; 3]; 3],
            da_dparam: [[0.0; 3]; 3],
        }
    }
}

/// Sigmoid activation mapping raw parameters to the `(0, 1)` interval.
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Derivative of [`sigmoid`] expressed as a function of its output.
#[inline]
fn sigmoid_derivative_from_output(s: f64) -> f64 {
    s * (1.0 - s)
}

/// Map a raw cutoff parameter to a physical cutoff frequency in Hz.
fn raw_to_fc(fc_raw: f64, half_fs: f64) -> f64 {
    sigmoid(fc_raw) * half_fs
}

/// Derivative of [`raw_to_fc`] w.r.t. the raw cutoff parameter.
fn raw_to_fc_derivative(fc_raw: f64, half_fs: f64) -> f64 {
    sigmoid_derivative_from_output(sigmoid(fc_raw)) * half_fs
}

/// Map a raw Q parameter to a physical quality factor.
fn raw_to_q(q_raw: f64) -> f64 {
    q_raw.exp().clamp(Q_MIN, Q_MAX)
}

/// Derivative of [`raw_to_q`] w.r.t. the raw Q parameter.
fn raw_to_q_derivative(q_raw: f64) -> f64 {
    let q = q_raw.exp();
    if q <= Q_MIN || q >= Q_MAX {
        0.0
    } else {
        q
    }
}

/// Map a raw gain parameter to a physical gain in dB.
fn raw_to_gain_db(gain_raw: f64) -> f64 {
    gain_raw.clamp(GAIN_DB_MIN, GAIN_DB_MAX)
}

/// Derivative of [`raw_to_gain_db`] w.r.t. the raw gain parameter.
fn raw_to_gain_db_derivative(gain_raw: f64) -> f64 {
    if gain_raw <= GAIN_DB_MIN || gain_raw >= GAIN_DB_MAX {
        0.0
    } else {
        1.0
    }
}

/// Compute normalized RBJ peaking or shelving coefficients.
fn compute_peq_coeffs(
    fc: f64,
    q: f64,
    gain_db: f64,
    fs: f64,
    band_type: PeqBandType,
) -> ([f64; 3], [f64; 3]) {
    let filter_type = match band_type {
        PeqBandType::Peak => math_audio_iir_fir::BiquadFilterType::Peak,
        PeqBandType::Lowshelf => math_audio_iir_fir::BiquadFilterType::Lowshelf,
        PeqBandType::Highshelf => math_audio_iir_fir::BiquadFilterType::Highshelf,
    };
    let coeffs = math_audio_iir_fir::Biquad::new(filter_type, fc, fs, q, gain_db).coefficients();
    (
        [coeffs.b0, coeffs.b1, coeffs.b2],
        [1.0, coeffs.a1, coeffs.a2],
    )
}

/// Compute normalized RBJ peaking or shelving coefficients and their physical
/// parameter derivatives using central finite differences.
fn compute_peq_coeffs_with_grads(
    fc: f64,
    q: f64,
    gain_db: f64,
    fs: f64,
    band_type: PeqBandType,
) -> SectionCoeffs {
    let eps = 1e-5;
    let (b, a) = compute_peq_coeffs(fc, q, gain_db, fs, band_type);
    let mut coeffs = SectionCoeffs::zeros();
    coeffs.b = b;
    coeffs.a = a;

    for p in 0..3 {
        let (fc_plus, q_plus, gain_plus) = match p {
            0 => (fc + eps, q, gain_db),
            1 => (fc, q + eps, gain_db),
            2 => (fc, q, gain_db + eps),
            _ => unreachable!(),
        };
        let (fc_minus, q_minus, gain_minus) = match p {
            0 => (fc - eps, q, gain_db),
            1 => (fc, q - eps, gain_db),
            2 => (fc, q, gain_db - eps),
            _ => unreachable!(),
        };

        let (b_plus, a_plus) = compute_peq_coeffs(fc_plus, q_plus, gain_plus, fs, band_type);
        let (b_minus, a_minus) = compute_peq_coeffs(fc_minus, q_minus, gain_minus, fs, band_type);

        for tap in 0..3 {
            coeffs.db_dparam[tap][p] = (b_plus[tap] - b_minus[tap]) / (2.0 * eps);
            coeffs.da_dparam[tap][p] = (a_plus[tap] - a_minus[tap]) / (2.0 * eps);
        }
    }

    coeffs
}

/// Differentiable parametric equalizer: a cascade of peaking or shelving
/// sections, each with learnable frequency, Q, and gain.
#[derive(Debug, Clone)]
pub struct ParametricEq {
    /// FFT length.
    pub nfft: usize,
    /// Sample rate in Hz.
    pub fs: f64,
    /// Number of cascaded sections.
    pub n_sections: usize,
    /// Number of input/output channels.
    pub n_channels: usize,
    /// Band type for every section.
    pub band_type: PeqBandType,
    /// Anti-aliasing decay in dB.
    pub alias_decay_db: f64,
    /// Raw parameters, shape `(n_sections, 3, n_channels)` for
    /// `[fc_raw, q_raw, gain_db_raw]`.
    pub param: ArrayD<f64>,
    /// Accumulated parameter gradients, same shape as `param`.
    pub param_grad: ArrayD<f64>,
    inner: SosFilter,
}

impl ParametricEq {
    /// Create a new parametric equalizer.
    ///
    /// # Errors
    ///
    /// Returns an error if `nfft`, `n_sections`, or `n_channels` is zero, or if
    /// `fs` is not positive and finite.
    pub fn new(
        nfft: usize,
        fs: f64,
        n_sections: usize,
        n_channels: usize,
        band_type: PeqBandType,
        alias_decay_db: f64,
    ) -> Result<Self, AutodiffError> {
        if nfft == 0 {
            return Err(AutodiffError::Message(format!(
                "ParametricEq: nfft must be greater than 0, got {nfft}"
            )));
        }
        if n_sections == 0 {
            return Err(AutodiffError::Message(format!(
                "ParametricEq: n_sections must be greater than 0, got {n_sections}"
            )));
        }
        if n_channels == 0 {
            return Err(AutodiffError::Message(format!(
                "ParametricEq: n_channels must be greater than 0, got {n_channels}"
            )));
        }
        if fs <= 0.0 || !fs.is_finite() {
            return Err(AutodiffError::Message(format!(
                "ParametricEq: sample rate must be positive and finite, got {fs}"
            )));
        }

        let param = ArrayD::zeros(IxDyn(&[n_sections, 3, n_channels]));
        let param_grad = ArrayD::zeros(IxDyn(&[n_sections, 3, n_channels]));
        let mut inner = SosFilter::new(nfft, n_sections, n_channels, n_channels, alias_decay_db)?;

        // Initialize the cascade as identity: each diagonal section is b=[1,0,0],
        // a=[1,0,0]; off-diagonal couplings are zero-response (b=[0,0,0],
        // a=[1,0,0]).
        inner.param.fill(0.0);
        for section in 0..n_sections {
            for out_ch in 0..n_channels {
                for in_ch in 0..n_channels {
                    inner.param[[section, 3, out_ch, in_ch]] = 1.0;
                }
            }
            for ch in 0..n_channels {
                inner.param[[section, 0, ch, ch]] = 1.0;
            }
        }

        Ok(Self {
            nfft,
            fs,
            n_sections,
            n_channels,
            band_type,
            alias_decay_db,
            param,
            param_grad,
            inner,
        })
    }

    fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }

    /// Fill an SOS coefficient tensor from raw parametric-EQ parameters.
    ///
    /// `sos_param` must have shape `(n_sections, 6, n_channels, n_channels)`.
    /// Per-channel diagonal coefficients realize the peaking/shelving filters;
    /// all off-diagonal couplings are set to a zero-response section
    /// (`b = [0, 0, 0]`, `a = [1, 0, 0]`).
    fn fill_sos_param(
        sos_param: &mut ArrayD<f64>,
        peq_param: &ArrayD<f64>,
        fs: f64,
        band_type: PeqBandType,
    ) {
        let shape = peq_param.shape();
        let n_sections = shape[0];
        let n_channels = shape[2];

        sos_param.fill(0.0);
        for section in 0..n_sections {
            for out_ch in 0..n_channels {
                for in_ch in 0..n_channels {
                    sos_param[[section, 3, out_ch, in_ch]] = 1.0;
                }
            }
        }

        let half_fs = fs / 2.0;
        for section in 0..n_sections {
            for ch in 0..n_channels {
                let fc_raw = peq_param[[section, 0, ch]];
                let q_raw = peq_param[[section, 1, ch]];
                let gain_raw = peq_param[[section, 2, ch]];
                let fc = raw_to_fc(fc_raw, half_fs);
                let q = raw_to_q(q_raw);
                let gain_db = raw_to_gain_db(gain_raw);

                let (b, a) = compute_peq_coeffs(fc, q, gain_db, fs, band_type);
                for tap in 0..3 {
                    sos_param[[section, tap, ch, ch]] = b[tap];
                    sos_param[[section, 3 + tap, ch, ch]] = a[tap];
                }
            }
        }
    }

    /// Rebuild the inner SOS coefficients from the current raw parameters.
    fn rebuild_inner(&mut self) {
        Self::fill_sos_param(&mut self.inner.param, &self.param, self.fs, self.band_type);
    }

    /// Build a fresh inner SOS filter reflecting the current parameters.
    ///
    /// Used by the immutable `forward` pass.
    fn build_fresh_inner(&self) -> SosFilter {
        let mut inner = self.inner.clone();
        Self::fill_sos_param(&mut inner.param, &self.param, self.fs, self.band_type);
        inner
    }
}

impl DiffModule<f64> for ParametricEq {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "ParametricEq::forward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let n_bins = input_shape[1];
        let n_in = input_shape[2];
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "ParametricEq::forward: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if n_in != self.n_channels {
            return Err(AutodiffError::Message(format!(
                "ParametricEq::forward: expected {} input channels, got {}",
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
                "ParametricEq::backward: grad_output shape {:?} does not match output shape {:?}",
                grad_shape, output_shape
            )));
        }
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "ParametricEq::backward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let n_bins = input_shape[1];
        let n_in = input_shape[2];
        if n_bins != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "ParametricEq::backward: expected {} frequency bins, got {}",
                self.n_bins(),
                n_bins
            )));
        }
        if n_in != self.n_channels {
            return Err(AutodiffError::Message(format!(
                "ParametricEq::backward: expected {} input channels, got {}",
                self.n_channels, n_in
            )));
        }

        self.rebuild_inner();
        self.inner.zero_grad();
        let grad_input = self.inner.backward(input, output, grad_output)?;

        let inner_grad = self
            .inner
            .param_grad
            .view()
            .into_shape_with_order((self.n_sections, 6, self.n_channels, self.n_channels))
            .map_err(|e| {
                AutodiffError::Message(format!(
                    "ParametricEq::backward: failed to reshape inner param_grad: {e}"
                ))
            })?;
        let mut param_grad = self
            .param_grad
            .view_mut()
            .into_shape_with_order((self.n_sections, 3, self.n_channels))
            .map_err(|e| {
                AutodiffError::Message(format!(
                    "ParametricEq::backward: failed to reshape param_grad: {e}"
                ))
            })?;

        let half_fs = self.fs / 2.0;
        for section in 0..self.n_sections {
            for ch in 0..self.n_channels {
                let fc_raw = self.param[[section, 0, ch]];
                let q_raw = self.param[[section, 1, ch]];
                let gain_raw = self.param[[section, 2, ch]];
                let fc = raw_to_fc(fc_raw, half_fs);
                let q = raw_to_q(q_raw);
                let gain_db = raw_to_gain_db(gain_raw);
                let coeffs = compute_peq_coeffs_with_grads(fc, q, gain_db, self.fs, self.band_type);

                let dfc_dfc_raw = raw_to_fc_derivative(fc_raw, half_fs);
                let dq_dq_raw = raw_to_q_derivative(q_raw);
                let dgain_db_dgain_raw = raw_to_gain_db_derivative(gain_raw);

                for p in 0..3 {
                    let mut accum = 0.0;
                    for tap in 0..3 {
                        let dl_db = inner_grad[[section, tap, ch, ch]];
                        let dl_da = inner_grad[[section, 3 + tap, ch, ch]];
                        let db_dp = coeffs.db_dparam[tap][p];
                        let da_dp = coeffs.da_dparam[tap][p];
                        let (db_dp_raw, da_dp_raw) = match p {
                            0 => (db_dp * dfc_dfc_raw, da_dp * dfc_dfc_raw),
                            1 => (db_dp * dq_dq_raw, da_dp * dq_dq_raw),
                            2 => (db_dp * dgain_db_dgain_raw, da_dp * dgain_db_dgain_raw),
                            _ => unreachable!(),
                        };
                        accum += dl_db * db_dp_raw + dl_da * da_dp_raw;
                    }
                    param_grad[[section, p, ch]] += accum;
                }
            }
        }

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

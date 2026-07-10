#![allow(clippy::cast_precision_loss, reason = "FFT sizes are audio buffer lengths that fit exactly in f64 for practical values")]

use ndarray::{Array1, ArrayD};
use num_complex::Complex;
use realfft::RealFftPlanner;

use crate::error::AutodiffError;
use crate::module::DiffModule;
use crate::tensor::DiffTensor;

/// Convert an FFT size to `f64` for arithmetic.
#[inline]
const fn nfft_as_f64(nfft: usize) -> f64 {
    nfft as f64
}

/// Real-to-complex FFT differentiable module.
#[derive(Debug, Clone)]
pub struct Fft {
    pub nfft: usize,
}

impl Fft {
    /// Create a new FFT module.
    #[must_use]
    pub const fn new(nfft: usize) -> Self {
        Self { nfft }
    }

    fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }
}

impl DiffModule<f64> for Fft {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(self.nfft);

        let mut input_vec = vec![0.0; self.nfft];
        for (i, sample) in input.data.iter().enumerate().take(self.nfft) {
            input_vec[i] = sample.re;
        }

        let mut spectrum = r2c.make_output_vec();
        r2c.process(&mut input_vec, &mut spectrum)?;

        Ok(DiffTensor::from_array(Array1::from_iter(spectrum)))
    }

    fn backward(
        &mut self,
        _input: &DiffTensor<f64>,
        _output: &DiffTensor<f64>,
        grad_output: &DiffTensor<f64>,
    ) -> Result<DiffTensor<f64>, AutodiffError> {
        let mut planner = RealFftPlanner::<f64>::new();
        let c2r = planner.plan_fft_inverse(self.nfft);

        let mut grad_vec = vec![Complex::new(0.0, 0.0); self.n_bins()];
        for (i, sample) in grad_output.data.iter().enumerate().take(self.n_bins()) {
            grad_vec[i] = *sample;
        }

        let mut grad_input = c2r.make_output_vec();
        c2r.process(&mut grad_vec, &mut grad_input)?;

        let scale = nfft_as_f64(self.nfft);
        Ok(DiffTensor::from_array(Array1::from_iter(
            grad_input.into_iter().map(|r| Complex::new(r / scale, 0.0)),
        )))
    }

    fn input_channels(&self) -> usize {
        1
    }

    fn output_channels(&self) -> usize {
        1
    }

    fn n_bins(&self) -> usize {
        self.n_bins()
    }

    fn parameters(&self) -> Vec<&ArrayD<f64>> {
        vec![]
    }

    fn gradients(&self) -> Vec<&ArrayD<f64>> {
        vec![]
    }

    fn zero_grad(&mut self) {}
}

/// Complex-to-real inverse FFT differentiable module.
#[derive(Debug, Clone)]
pub struct Ifft {
    pub nfft: usize,
}

impl Ifft {
    /// Create a new inverse FFT module.
    #[must_use]
    pub const fn new(nfft: usize) -> Self {
        Self { nfft }
    }

    fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }
}

impl DiffModule<f64> for Ifft {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let mut planner = RealFftPlanner::<f64>::new();
        let c2r = planner.plan_fft_inverse(self.nfft);

        let mut input_vec = vec![Complex::new(0.0, 0.0); self.n_bins()];
        for (i, sample) in input.data.iter().enumerate().take(self.n_bins()) {
            input_vec[i] = *sample;
        }

        let mut output = c2r.make_output_vec();
        c2r.process(&mut input_vec, &mut output)?;

        let scale = nfft_as_f64(self.nfft);
        Ok(DiffTensor::from_array(Array1::from_iter(
            output.into_iter().map(|r| Complex::new(r / scale, 0.0)),
        )))
    }

    fn backward(
        &mut self,
        _input: &DiffTensor<f64>,
        _output: &DiffTensor<f64>,
        grad_output: &DiffTensor<f64>,
    ) -> Result<DiffTensor<f64>, AutodiffError> {
        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(self.nfft);

        let mut grad_vec = vec![0.0; self.nfft];
        for (i, sample) in grad_output.data.iter().enumerate().take(self.nfft) {
            grad_vec[i] = sample.re;
        }

        let mut spectrum = r2c.make_output_vec();
        r2c.process(&mut grad_vec, &mut spectrum)?;

        Ok(DiffTensor::from_array(Array1::from_iter(spectrum)))
    }

    fn input_channels(&self) -> usize {
        1
    }

    fn output_channels(&self) -> usize {
        1
    }

    fn n_bins(&self) -> usize {
        self.n_bins()
    }

    fn parameters(&self) -> Vec<&ArrayD<f64>> {
        vec![]
    }

    fn gradients(&self) -> Vec<&ArrayD<f64>> {
        vec![]
    }

    fn zero_grad(&mut self) {}
}

/// Real-to-complex FFT with an exponential anti-aliasing envelope.
#[derive(Debug, Clone)]
pub struct FftAntiAlias {
    pub nfft: usize,
    pub alias_decay_db: f64,
    pub gamma: f64,
    pub envelope: Vec<f64>,
}

impl FftAntiAlias {
    /// Create a new anti-aliased FFT module.
    ///
    /// The envelope decays by `alias_decay_db` dB across the FFT window.
    #[must_use]
    pub fn new(nfft: usize, alias_decay_db: f64) -> Self {
        let gamma = 10_f64.powf(-alias_decay_db.abs() / (20.0 * nfft_as_f64(nfft)));

        let mut envelope = Vec::with_capacity(nfft);
        let mut value = 1.0;
        for _ in 0..nfft {
            envelope.push(value);
            value *= gamma;
        }

        Self {
            nfft,
            alias_decay_db,
            gamma,
            envelope,
        }
    }

    fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }
}

impl DiffModule<f64> for FftAntiAlias {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(self.nfft);

        let mut input_vec = vec![0.0; self.nfft];
        for (i, sample) in input.data.iter().enumerate().take(self.nfft) {
            input_vec[i] = sample.re * self.envelope[i];
        }

        let mut spectrum = r2c.make_output_vec();
        r2c.process(&mut input_vec, &mut spectrum)?;

        Ok(DiffTensor::from_array(Array1::from_iter(spectrum)))
    }

    fn backward(
        &mut self,
        _input: &DiffTensor<f64>,
        _output: &DiffTensor<f64>,
        grad_output: &DiffTensor<f64>,
    ) -> Result<DiffTensor<f64>, AutodiffError> {
        let mut planner = RealFftPlanner::<f64>::new();
        let c2r = planner.plan_fft_inverse(self.nfft);

        let mut grad_vec = vec![Complex::new(0.0, 0.0); self.n_bins()];
        for (i, sample) in grad_output.data.iter().enumerate().take(self.n_bins()) {
            grad_vec[i] = *sample;
        }

        let mut grad_time = c2r.make_output_vec();
        c2r.process(&mut grad_vec, &mut grad_time)?;

        let scale = nfft_as_f64(self.nfft);
        Ok(DiffTensor::from_array(Array1::from_iter(
            grad_time
                .iter()
                .zip(&self.envelope)
                .map(|(sample, env)| Complex::new(sample * env / scale, 0.0)),
        )))
    }

    fn input_channels(&self) -> usize {
        1
    }

    fn output_channels(&self) -> usize {
        1
    }

    fn n_bins(&self) -> usize {
        self.n_bins()
    }

    fn parameters(&self) -> Vec<&ArrayD<f64>> {
        vec![]
    }

    fn gradients(&self) -> Vec<&ArrayD<f64>> {
        vec![]
    }

    fn zero_grad(&mut self) {}
}

/// Complex-to-real inverse FFT with an exponential anti-aliasing envelope.
#[derive(Debug, Clone)]
pub struct IfftAntiAlias {
    pub nfft: usize,
    pub alias_decay_db: f64,
    pub gamma: f64,
    pub envelope: Vec<f64>,
}

impl IfftAntiAlias {
    /// Create a new anti-aliased inverse FFT module.
    ///
    /// The envelope decays by `alias_decay_db` dB across the FFT window.
    #[must_use]
    pub fn new(nfft: usize, alias_decay_db: f64) -> Self {
        let gamma = 10_f64.powf(-alias_decay_db.abs() / (20.0 * nfft_as_f64(nfft)));

        let mut envelope = Vec::with_capacity(nfft);
        let mut value = 1.0;
        for _ in 0..nfft {
            envelope.push(value);
            value *= gamma;
        }

        Self {
            nfft,
            alias_decay_db,
            gamma,
            envelope,
        }
    }

    fn n_bins(&self) -> usize {
        self.nfft / 2 + 1
    }
}

impl DiffModule<f64> for IfftAntiAlias {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let mut planner = RealFftPlanner::<f64>::new();
        let c2r = planner.plan_fft_inverse(self.nfft);

        let mut input_vec = vec![Complex::new(0.0, 0.0); self.n_bins()];
        for (i, sample) in input.data.iter().enumerate().take(self.n_bins()) {
            input_vec[i] = *sample;
        }

        let mut output = c2r.make_output_vec();
        c2r.process(&mut input_vec, &mut output)?;

        let scale = nfft_as_f64(self.nfft);
        Ok(DiffTensor::from_array(Array1::from_iter(
            output
                .iter()
                .zip(&self.envelope)
                .map(|(sample, env)| Complex::new(sample * env / scale, 0.0)),
        )))
    }

    fn backward(
        &mut self,
        _input: &DiffTensor<f64>,
        _output: &DiffTensor<f64>,
        grad_output: &DiffTensor<f64>,
    ) -> Result<DiffTensor<f64>, AutodiffError> {
        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(self.nfft);

        let mut grad_vec = vec![0.0; self.nfft];
        for (i, sample) in grad_output.data.iter().enumerate().take(self.nfft) {
            grad_vec[i] = sample.re * self.envelope[i];
        }

        let mut spectrum = r2c.make_output_vec();
        r2c.process(&mut grad_vec, &mut spectrum)?;

        Ok(DiffTensor::from_array(Array1::from_iter(spectrum)))
    }

    fn input_channels(&self) -> usize {
        1
    }

    fn output_channels(&self) -> usize {
        1
    }

    fn n_bins(&self) -> usize {
        self.n_bins()
    }

    fn parameters(&self) -> Vec<&ArrayD<f64>> {
        vec![]
    }

    fn gradients(&self) -> Vec<&ArrayD<f64>> {
        vec![]
    }

    fn zero_grad(&mut self) {}
}

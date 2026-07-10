//! System composition modules.

use ndarray::ArrayD;
use num_complex::Complex;

use crate::error::AutodiffError;
use crate::module::DiffModule;
use crate::tensor::DiffTensor;

/// Sequential composition of differentiable modules.
pub struct Series {
    modules: Vec<Box<dyn DiffModule<f64>>>,
    nfft: usize,
    input_channels: usize,
    output_channels: usize,
}

impl std::fmt::Debug for Series {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Series")
            .field("nfft", &self.nfft)
            .field("input_channels", &self.input_channels)
            .field("output_channels", &self.output_channels)
            .field("module_count", &self.modules.len())
            .finish_non_exhaustive()
    }
}

impl Series {
    /// Create a new series from a vector of modules.
    ///
    /// Verifies that all modules share the same number of frequency bins and
    /// that adjacent modules have compatible channel counts.
    ///
    /// # Errors
    ///
    /// Returns an error if the module list is empty or if compatibility checks
    /// fail.
    pub fn new(modules: Vec<Box<dyn DiffModule<f64>>>) -> Result<Self, AutodiffError> {
        if modules.is_empty() {
            return Err(AutodiffError::Message(
                "Series: must contain at least one module".to_string(),
            ));
        }
        let nfft = modules[0].n_bins();
        let input_channels = modules[0].input_channels();
        let mut output_channels = modules[0].output_channels();
        for (i, module) in modules.iter().enumerate().skip(1) {
            if module.n_bins() != nfft {
                return Err(AutodiffError::Message(format!(
                    "Series: module {} has {} bins, expected {}",
                    i,
                    module.n_bins(),
                    nfft
                )));
            }
            if module.input_channels() != output_channels {
                return Err(AutodiffError::Message(format!(
                    "Series: module {} expects {} input channels, previous module outputs {}",
                    i,
                    module.input_channels(),
                    output_channels
                )));
            }
            output_channels = module.output_channels();
        }
        Ok(Self {
            modules,
            nfft,
            input_channels,
            output_channels,
        })
    }

    /// Return the contained modules.
    #[must_use]
    pub fn modules(&self) -> &[Box<dyn DiffModule<f64>>] {
        &self.modules
    }
}

impl DiffModule<f64> for Series {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let mut x = input.clone();
        for module in &self.modules {
            x = module.forward(&x)?;
        }
        Ok(x)
    }

    fn backward(
        &mut self,
        input: &DiffTensor<f64>,
        _output: &DiffTensor<f64>,
        grad_output: &DiffTensor<f64>,
    ) -> Result<DiffTensor<f64>, AutodiffError> {
        // Recompute forward pass to capture intermediate inputs/outputs.
        let mut intermediates = Vec::with_capacity(self.modules.len());
        let mut x = input.clone();
        for module in &self.modules {
            x = module.forward(&x)?;
            intermediates.push(x.clone());
        }

        // Backpropagate in reverse order.
        let mut grad = grad_output.clone();
        for (i, module) in self.modules.iter_mut().enumerate().rev() {
            let in_i = if i == 0 { input } else { &intermediates[i - 1] };
            let out_i = &intermediates[i];
            grad = module.backward(in_i, out_i, &grad)?;
        }
        Ok(grad)
    }

    fn input_channels(&self) -> usize {
        self.input_channels
    }

    fn output_channels(&self) -> usize {
        self.output_channels
    }

    fn n_bins(&self) -> usize {
        self.nfft
    }

    fn parameters(&self) -> Vec<&ArrayD<f64>> {
        self.modules
            .iter()
            .flat_map(|module| module.parameters())
            .collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<f64>> {
        self.modules
            .iter_mut()
            .flat_map(|module| module.parameters_mut())
            .collect()
    }

    fn gradients(&self) -> Vec<&ArrayD<f64>> {
        self.modules
            .iter()
            .flat_map(|module| module.gradients())
            .collect()
    }

    fn zero_grad(&mut self) {
        for module in &mut self.modules {
            module.zero_grad();
        }
    }
}

/// Shell composition: input layer, core, and output layer.
pub struct Shell {
    /// Input transformation layer.
    pub input_layer: Box<dyn DiffModule<f64>>,
    /// Differentiable core whose parameters are optimized.
    pub core: Box<dyn DiffModule<f64>>,
    /// Output transformation layer.
    pub output_layer: Box<dyn DiffModule<f64>>,
}

impl std::fmt::Debug for Shell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shell")
            .field("input_layer", &"<dyn DiffModule>")
            .field("core", &"<dyn DiffModule>")
            .field("output_layer", &"<dyn DiffModule>")
            .finish()
    }
}

impl Shell {
    /// Create a new shell.
    ///
    /// # Errors
    ///
    /// Returns an error if channel or frequency-bin dimensions are incompatible.
    pub fn new(
        input_layer: Box<dyn DiffModule<f64>>,
        core: Box<dyn DiffModule<f64>>,
        output_layer: Box<dyn DiffModule<f64>>,
    ) -> Result<Self, AutodiffError> {
        if input_layer.n_bins() != core.n_bins() {
            return Err(AutodiffError::Message(format!(
                "Shell: input_layer has {} bins, core has {}",
                input_layer.n_bins(),
                core.n_bins()
            )));
        }
        if core.n_bins() != output_layer.n_bins() {
            return Err(AutodiffError::Message(format!(
                "Shell: core has {} bins, output_layer has {}",
                core.n_bins(),
                output_layer.n_bins()
            )));
        }
        if input_layer.output_channels() != core.input_channels() {
            return Err(AutodiffError::Message(format!(
                "Shell: input_layer outputs {} channels, core expects {}",
                input_layer.output_channels(),
                core.input_channels()
            )));
        }
        if core.output_channels() != output_layer.input_channels() {
            return Err(AutodiffError::Message(format!(
                "Shell: core outputs {} channels, output_layer expects {}",
                core.output_channels(),
                output_layer.input_channels()
            )));
        }
        Ok(Self {
            input_layer,
            core,
            output_layer,
        })
    }

    /// Forward pass: `input_layer -> core -> output_layer`.
    ///
    /// # Errors
    ///
    /// Returns an error if any sub-module operation fails.
    pub fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let x1 = self.input_layer.forward(input)?;
        let x2 = self.core.forward(&x1)?;
        self.output_layer.forward(&x2)
    }

    /// Backward pass through the shell.
    ///
    /// Recomputes forward intermediates internally and backpropagates through
    /// `output_layer`, `core`, and `input_layer` in that order.
    ///
    /// # Errors
    ///
    /// Returns an error if any sub-module operation fails.
    pub fn backward(
        &mut self,
        input: &DiffTensor<f64>,
        _output: &DiffTensor<f64>,
        grad_output: &DiffTensor<f64>,
    ) -> Result<DiffTensor<f64>, AutodiffError> {
        let x1 = self.input_layer.forward(input)?;
        let x2 = self.core.forward(&x1)?;
        let x3 = self.output_layer.forward(&x2)?;

        let grad_x2 = self.output_layer.backward(&x2, &x3, grad_output)?;
        let grad_x1 = self.core.backward(&x1, &x2, &grad_x2)?;
        self.input_layer.backward(input, &x1, &grad_x1)
    }

    /// Zero gradients in all parameter-bearing sub-modules.
    pub fn zero_grad(&mut self) {
        self.input_layer.zero_grad();
        self.core.zero_grad();
        self.output_layer.zero_grad();
    }

    /// Return parameter tensors from all parameter-bearing sub-modules.
    #[must_use]
    pub fn parameters(&self) -> Vec<&ArrayD<f64>> {
        let mut params = Vec::new();
        params.extend(self.input_layer.parameters());
        params.extend(self.core.parameters());
        params.extend(self.output_layer.parameters());
        params
    }

    /// Return mutable parameter tensors from all parameter-bearing sub-modules.
    #[must_use]
    pub fn parameters_mut(&mut self) -> Vec<&mut ArrayD<f64>> {
        let mut params = Vec::new();
        params.extend(self.input_layer.parameters_mut());
        params.extend(self.core.parameters_mut());
        params.extend(self.output_layer.parameters_mut());
        params
    }

    /// Return gradient tensors from all parameter-bearing sub-modules.
    #[must_use]
    pub fn gradients(&self) -> Vec<&ArrayD<f64>> {
        let mut grads = Vec::new();
        grads.extend(self.input_layer.gradients());
        grads.extend(self.core.gradients());
        grads.extend(self.output_layer.gradients());
        grads
    }

    /// Compute the complex frequency response of the core module.
    ///
    /// Builds a unit-impulse spectrum for each input channel, runs only the
    /// core (ignoring input/output layers), and returns the response with
    /// shape `(n_bins, n_out, n_in)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the core forward pass fails.
    pub fn get_freq_response(&self) -> Result<DiffTensor<f64>, AutodiffError> {
        let n_in = self.core.input_channels();
        let n_out = self.core.output_channels();
        let n_bins = self.core.n_bins();

        // Build an identity spectrum: input[in_ch, f, in_ch] = 1 for all f.
        let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[n_in, n_bins, n_in]));
        for in_ch in 0..n_in {
            for f in 0..n_bins {
                input_data[[in_ch, f, in_ch]] = Complex::new(1.0, 0.0);
            }
        }
        let input = DiffTensor::from_array(input_data);
        let output = self.core.forward(&input)?;
        let output_shape = output.data.shape();
        if output_shape != [n_in, n_bins, n_out] {
            return Err(AutodiffError::Message(format!(
                "Shell::get_freq_response: unexpected core output shape {:?}, expected {:?}",
                output_shape,
                [n_in, n_bins, n_out]
            )));
        }

        // Permute (n_in, n_bins, n_out) -> (n_bins, n_out, n_in).
        let mut response = ArrayD::zeros(ndarray::IxDyn(&[n_bins, n_out, n_in]));
        for in_ch in 0..n_in {
            for f in 0..n_bins {
                for out_ch in 0..n_out {
                    response[[f, out_ch, in_ch]] = output.data[[in_ch, f, out_ch]];
                }
            }
        }

        Ok(DiffTensor::from_array(response))
    }
}

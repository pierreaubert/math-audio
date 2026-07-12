//! Closed-loop recursion module.

#![allow(
    clippy::uninlined_format_args,
    reason = "format strings are clearer with explicit arguments in error messages"
)]
#![allow(
    clippy::similar_names,
    reason = "ff/fb suffixes denote feedforward/feedback quantities"
)]

use nalgebra::DMatrix;
use ndarray::{Array2, Array3, ArrayD, Axis, IxDyn};
use num_complex::Complex;
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::{Arc, Mutex},
};

use crate::error::AutodiffError;
use crate::module::{DiffModule, validate_spectral_gradient_shape};
use crate::tensor::DiffTensor;

/// Extract the transfer matrix of a submodule as `(n_bins, n_out, n_in)`.
fn module_response(
    module: &dyn DiffModule<f64>,
    identity: &DiffTensor<f64>,
) -> Result<Array3<Complex<f64>>, AutodiffError> {
    let nb = module.n_bins();
    let identity_shape = identity.data.shape();
    if identity_shape.len() != 3 || identity_shape[1] != nb {
        return Err(AutodiffError::Message(format!(
            "Recursion: identity spectrum shape {:?} incompatible with module bins {nb}",
            identity_shape
        )));
    }
    let n_in = identity_shape[0];
    let output = module.forward(identity)?;
    let out_shape = output.data.shape();
    if out_shape.len() != 3 || out_shape[0] != n_in || out_shape[1] != nb {
        return Err(AutodiffError::Message(format!(
            "Recursion: submodule response has unexpected shape {:?}",
            out_shape
        )));
    }
    let n_out = out_shape[2];
    let mut h = Array3::zeros((nb, n_out, n_in));
    for i in 0..n_in {
        for f in 0..nb {
            for o in 0..n_out {
                h[[f, o, i]] = output.data[[i, f, o]];
            }
        }
    }
    Ok(h)
}

fn ndarray2_to_dmatrix(mat: &Array2<Complex<f64>>) -> DMatrix<Complex<f64>> {
    let data: Vec<Complex<f64>> = mat.iter().copied().collect();
    DMatrix::from_row_slice(mat.nrows(), mat.ncols(), &data)
}

fn dmatrix_to_ndarray2(mat: &DMatrix<Complex<f64>>) -> Array2<Complex<f64>> {
    let mut out = Array2::zeros((mat.nrows(), mat.ncols()));
    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            out[[i, j]] = mat[(i, j)];
        }
    }
    out
}

fn invert_complex_matrix(
    mat: &Array2<Complex<f64>>,
    bin: usize,
) -> Result<Array2<Complex<f64>>, AutodiffError> {
    let dm = ndarray2_to_dmatrix(mat);
    let inv = dm.try_inverse().ok_or_else(|| {
        AutodiffError::Message(format!(
            "Recursion: failed to invert (I - H_fb) at frequency bin {bin}"
        ))
    })?;
    Ok(dmatrix_to_ndarray2(&inv))
}

/// Compute `out = a @ b` into the pre-allocated `out` buffer.
#[allow(clippy::many_single_char_names)]
fn matmul_into(a: &Array2<Complex<f64>>, b: &Array2<Complex<f64>>, out: &mut Array2<Complex<f64>>) {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();
    assert_eq!(b.nrows(), k, "matmul_into: incompatible inner dimensions");
    assert_eq!(out.dim(), (m, n), "matmul_into: incompatible output shape");
    out.fill(Complex::new(0.0, 0.0));
    for i in 0..m {
        for l in 0..k {
            let a_il = a[[i, l]];
            if a_il == Complex::new(0.0, 0.0) {
                continue;
            }
            for j in 0..n {
                out[[i, j]] += a_il * b[[l, j]];
            }
        }
    }
}

/// Compute the conjugate transpose `out = a^H` into the pre-allocated `out` buffer.
fn conj_transpose_into(src: &Array2<Complex<f64>>, dst: &mut Array2<Complex<f64>>) {
    let (m, n) = (src.nrows(), src.ncols());
    assert_eq!(
        dst.dim(),
        (n, m),
        "conj_transpose_into: incompatible output shape"
    );
    for i in 0..m {
        for j in 0..n {
            dst[[j, i]] = src[[i, j]].conj();
        }
    }
}

/// Fill an `(n, nb, n)` tensor with an identity spectrum.
fn fill_identity_spectrum(n: usize, nb: usize, tensor: &mut DiffTensor<f64>) {
    tensor.data.fill(Complex::new(0.0, 0.0));
    for i in 0..n {
        for f in 0..nb {
            tensor.data[[i, f, i]] = Complex::new(1.0, 0.0);
        }
    }
}

type ClosedLoopResponse = (
    Array3<Complex<f64>>,
    Array3<Complex<f64>>,
    Array3<Complex<f64>>,
    Array3<Complex<f64>>,
);

/// Closed-loop MIMO composition `y = (I - H_fb)^-1 @ H_ff @ x`.
pub struct Recursion {
    pub feedforward: Box<dyn DiffModule<f64>>,
    pub feedback: Box<dyn DiffModule<f64>>,
    n_bins: usize,
    response_cache: Mutex<Option<(u64, Arc<ClosedLoopResponse>)>>,
    // Reusable backward buffers to avoid per-call heap allocations.
    identity_ff: DiffTensor<f64>,
    identity_fb: DiffTensor<f64>,
    h_ff_response: DiffTensor<f64>,
    h_fb_response: DiffTensor<f64>,
    grad_ff: DiffTensor<f64>,
    grad_fb: DiffTensor<f64>,
    dl_dh_closed: Array3<Complex<f64>>,
    grad_input: ArrayD<Complex<f64>>,
    // 2-D per-bin work buffers.
    a_buf: Array2<Complex<f64>>,
    a_h_buf: Array2<Complex<f64>>,
    h_ff_f_buf: Array2<Complex<f64>>,
    h_ff_h_buf: Array2<Complex<f64>>,
    dl_dh_closed_f_buf: Array2<Complex<f64>>,
    dl_dh_ff_bin_buf: Array2<Complex<f64>>,
    work_buf: Array2<Complex<f64>>,
    work2_buf: Array2<Complex<f64>>,
}

impl std::fmt::Debug for Recursion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Recursion")
            .field("n_bins", &self.n_bins)
            .field("feedforward", &"<dyn DiffModule>")
            .field("feedback", &"<dyn DiffModule>")
            .field("response_cache", &"<cached closed-loop response>")
            .field("backward_buffers", &"<reused>")
            .finish_non_exhaustive()
    }
}

impl Recursion {
    /// Create a new closed-loop recursion module.
    ///
    /// # Errors
    ///
    /// Returns an error if the feedforward and feedback modules have incompatible
    /// frequency-bin counts or channel dimensions.
    pub fn new(
        feedforward: Box<dyn DiffModule<f64>>,
        feedback: Box<dyn DiffModule<f64>>,
    ) -> Result<Self, AutodiffError> {
        if feedforward.n_bins() != feedback.n_bins() {
            return Err(AutodiffError::Message(format!(
                "Recursion: feedforward has {} bins, feedback has {}",
                feedforward.n_bins(),
                feedback.n_bins()
            )));
        }
        if feedforward.output_channels() != feedback.input_channels() {
            return Err(AutodiffError::Message(format!(
                "Recursion: feedforward outputs {}, feedback expects {}",
                feedforward.output_channels(),
                feedback.input_channels()
            )));
        }
        if feedback.output_channels() != feedforward.output_channels() {
            return Err(AutodiffError::Message(format!(
                "Recursion: feedback outputs {}, feedforward outputs {}",
                feedback.output_channels(),
                feedforward.output_channels()
            )));
        }
        let n_bins = feedforward.n_bins();
        let n_in = feedforward.input_channels();
        let n_out = feedforward.output_channels();

        let mut identity_ff = DiffTensor::zeros(IxDyn(&[n_in, n_bins, n_in]));
        fill_identity_spectrum(n_in, n_bins, &mut identity_ff);
        let mut identity_fb = DiffTensor::zeros(IxDyn(&[n_out, n_bins, n_out]));
        fill_identity_spectrum(n_out, n_bins, &mut identity_fb);

        Ok(Self {
            n_bins,
            feedforward,
            feedback,
            response_cache: Mutex::new(None),
            identity_ff,
            identity_fb,
            h_ff_response: DiffTensor::zeros(IxDyn(&[n_in, n_bins, n_out])),
            h_fb_response: DiffTensor::zeros(IxDyn(&[n_out, n_bins, n_out])),
            grad_ff: DiffTensor::zeros(IxDyn(&[n_in, n_bins, n_out])),
            grad_fb: DiffTensor::zeros(IxDyn(&[n_out, n_bins, n_out])),
            dl_dh_closed: Array3::zeros((n_bins, n_out, n_in)),
            grad_input: ArrayD::zeros(IxDyn(&[0, 0, 0])),
            a_buf: Array2::zeros((n_out, n_out)),
            a_h_buf: Array2::zeros((n_out, n_out)),
            h_ff_f_buf: Array2::zeros((n_out, n_in)),
            h_ff_h_buf: Array2::zeros((n_in, n_out)),
            dl_dh_closed_f_buf: Array2::zeros((n_out, n_in)),
            dl_dh_ff_bin_buf: Array2::zeros((n_out, n_in)),
            work_buf: Array2::zeros((n_out, n_out)),
            work2_buf: Array2::zeros((n_out, n_out)),
        })
    }

    fn n_bins(&self) -> usize {
        self.n_bins
    }

    fn response_fingerprint(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.n_bins.hash(&mut hasher);
        for parameter in self
            .feedforward
            .parameters()
            .into_iter()
            .chain(self.feedback.parameters())
        {
            parameter.shape().hash(&mut hasher);
            for &value in parameter {
                value.to_bits().hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    fn cached_closed_loop_response(&self) -> Result<Arc<ClosedLoopResponse>, AutodiffError> {
        let fingerprint = self.response_fingerprint();
        {
            let cache = self
                .response_cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some((cached_fingerprint, response)) = cache.as_ref()
                && *cached_fingerprint == fingerprint
            {
                return Ok(Arc::clone(response));
            }
        }

        let response = Arc::new(self.closed_loop_response()?);
        let mut cache = self
            .response_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *cache = Some((fingerprint, Arc::clone(&response)));
        Ok(response)
    }

    #[allow(clippy::type_complexity)]
    fn closed_loop_response(&self) -> Result<ClosedLoopResponse, AutodiffError> {
        let n_in = self.feedforward.input_channels();
        let n_out = self.feedforward.output_channels();
        let nb = self.n_bins();

        let h_ff = module_response(self.feedforward.as_ref(), &self.identity_ff)?; // (nb, n_out, n_in)
        let h_fb = module_response(self.feedback.as_ref(), &self.identity_fb)?; // (nb, n_out, n_out)

        let mut h_closed = Array3::zeros((nb, n_out, n_in));
        let mut a_arr = Array3::zeros((nb, n_out, n_out));

        for f in 0..nb {
            let i_min_h_fb = {
                let mut m = Array2::zeros((n_out, n_out));
                for r in 0..n_out {
                    for c in 0..n_out {
                        m[[r, c]] = if r == c {
                            Complex::new(1.0, 0.0)
                        } else {
                            Complex::new(0.0, 0.0)
                        };
                        m[[r, c]] -= h_fb[[f, r, c]];
                    }
                }
                m
            };
            let a = invert_complex_matrix(&i_min_h_fb, f)?;
            for r in 0..n_out {
                for c in 0..n_out {
                    a_arr[[f, r, c]] = a[[r, c]];
                }
            }
            for o in 0..n_out {
                for i in 0..n_in {
                    let mut sum = Complex::new(0.0, 0.0);
                    for k in 0..n_out {
                        sum += a[[o, k]] * h_ff[[f, k, i]];
                    }
                    h_closed[[f, o, i]] = sum;
                }
            }
        }

        Ok((h_closed, h_ff, h_fb, a_arr))
    }
}

impl DiffModule<f64> for Recursion {
    fn forward(&self, input: &DiffTensor<f64>) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        if input_shape.len() < 3 {
            return Err(AutodiffError::Message(format!(
                "Recursion::forward: input must have at least 3 dimensions, got {:?}",
                input_shape
            )));
        }
        let nb = input_shape[1];
        let n_in = input_shape[2];
        if nb != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "Recursion::forward: expected {} bins, got {}",
                self.n_bins(),
                nb
            )));
        }
        if n_in != self.feedforward.input_channels() {
            return Err(AutodiffError::Message(format!(
                "Recursion::forward: expected {} input channels, got {}",
                self.feedforward.input_channels(),
                n_in
            )));
        }

        let response = self.cached_closed_loop_response()?;
        let h_closed = &response.0;
        let n_out = self.feedforward.output_channels();
        let mut output_shape = input_shape.to_vec();
        output_shape[2] = n_out;
        let mut output = ArrayD::zeros(IxDyn(&output_shape));

        if input_shape.len() == 3
            && let Some(input_data) = input.data.as_slice()
        {
            let batch = input_shape[0];
            let output_data = output
                .as_slice_mut()
                .expect("new recursion output must be contiguous");
            for batch_index in 0..batch {
                for f in 0..nb {
                    for output_channel in 0..n_out {
                        let mut sum = Complex::default();
                        for input_channel in 0..n_in {
                            let input_index = (batch_index * nb + f) * n_in + input_channel;
                            sum += input_data[input_index]
                                * h_closed[[f, output_channel, input_channel]];
                        }
                        let output_index = (batch_index * nb + f) * n_out + output_channel;
                        output_data[output_index] = sum;
                    }
                }
            }
        } else {
            for o in 0..n_out {
                for i in 0..n_in {
                    for f in 0..nb {
                        let h = h_closed[[f, o, i]];
                        let input_slice = input.data.index_axis(Axis(1), f);
                        let input_bin = input_slice.index_axis(Axis(1), i);
                        let mut output_slice = output.index_axis_mut(Axis(1), f);
                        let mut output_bin = output_slice.index_axis_mut(Axis(1), o);
                        for (destination, &source) in output_bin.iter_mut().zip(input_bin.iter()) {
                            *destination += source * h;
                        }
                    }
                }
            }
        }

        Ok(DiffTensor::from_array(output))
    }

    #[allow(clippy::too_many_lines)]
    fn backward(
        &mut self,
        input: &DiffTensor<f64>,
        _output: &DiffTensor<f64>,
        grad_output: &DiffTensor<f64>,
    ) -> Result<DiffTensor<f64>, AutodiffError> {
        let input_shape = input.data.shape();
        let grad_shape = grad_output.data.shape();
        let n_out = self.feedforward.output_channels();
        validate_spectral_gradient_shape("Recursion::backward", input_shape, grad_shape, n_out)?;
        let nb = input_shape[1];
        let n_in = input_shape[2];
        if nb != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "Recursion::backward: expected {} bins, got {}",
                self.n_bins(),
                nb
            )));
        }
        if n_in != self.feedforward.input_channels() {
            return Err(AutodiffError::Message(format!(
                "Recursion::backward: expected {} input channels, got {n_in}",
                self.feedforward.input_channels()
            )));
        }

        let response = self.cached_closed_loop_response()?;
        let (h_closed, h_ff, h_fb, a_arr) = response.as_ref();

        // Reusable buffers are sized for the module's fixed channel/bin counts.
        // If the input shape differs from the cached buffer, re-allocate.
        if self.grad_input.shape() == input_shape {
            self.grad_input.fill(Complex::new(0.0, 0.0));
        } else {
            self.grad_input = ArrayD::zeros(IxDyn(input_shape));
        }
        self.dl_dh_closed.fill(Complex::new(0.0, 0.0));
        self.grad_ff.data.fill(Complex::new(0.0, 0.0));
        self.grad_fb.data.fill(Complex::new(0.0, 0.0));
        self.h_ff_response.data.fill(Complex::new(0.0, 0.0));
        self.h_fb_response.data.fill(Complex::new(0.0, 0.0));

        // dL/dH_closed[f, o, i] = sum_b grad_output[b, f, o] * conj(input[b, f, i])
        if input_shape.len() == 3
            && let Some(grad_output_data) = grad_output.data.as_slice()
            && let Some(input_data) = input.data.as_slice()
        {
            let batch = input_shape[0];
            for f in 0..nb {
                for o in 0..n_out {
                    for i in 0..n_in {
                        let mut sum = Complex::default();
                        for batch_index in 0..batch {
                            let grad_index = (batch_index * nb + f) * n_out + o;
                            let input_index = (batch_index * nb + f) * n_in + i;
                            sum += grad_output_data[grad_index] * input_data[input_index].conj();
                        }
                        self.dl_dh_closed[[f, o, i]] = sum;
                    }
                }
            }
        } else {
            for f in 0..nb {
                for o in 0..n_out {
                    for i in 0..n_in {
                        let grad_slice = grad_output.data.index_axis(Axis(1), f);
                        let grad_bin = grad_slice.index_axis(Axis(1), o);
                        let input_slice = input.data.index_axis(Axis(1), f);
                        let input_bin = input_slice.index_axis(Axis(1), i);
                        self.dl_dh_closed[[f, o, i]] = grad_bin
                            .iter()
                            .zip(input_bin.iter())
                            .map(|(g, x)| g * x.conj())
                            .sum::<Complex<f64>>();
                    }
                }
            }
        }

        // Build per-bin dL/dH_ff and dL/dH_fb, populating response/gradient
        // tensors for the feedforward and feedback backward calls in-place.
        for f in 0..nb {
            // Fill response tensors for this bin.
            for i in 0..n_in {
                for o in 0..n_out {
                    self.h_ff_response.data[[i, f, o]] = h_ff[[f, o, i]];
                }
            }
            for i in 0..n_out {
                for o in 0..n_out {
                    self.h_fb_response.data[[i, f, o]] = h_fb[[f, o, i]];
                }
            }

            // Copy this bin's matrices into reusable 2-D buffers.
            for r in 0..n_out {
                for c in 0..n_out {
                    self.a_buf[[r, c]] = a_arr[[f, r, c]];
                }
            }
            conj_transpose_into(&self.a_buf, &mut self.a_h_buf);
            for r in 0..n_out {
                for c in 0..n_in {
                    self.h_ff_f_buf[[r, c]] = h_ff[[f, r, c]];
                }
            }
            conj_transpose_into(&self.h_ff_f_buf, &mut self.h_ff_h_buf);
            for r in 0..n_out {
                for c in 0..n_in {
                    self.dl_dh_closed_f_buf[[r, c]] = self.dl_dh_closed[[f, r, c]];
                }
            }

            // dL/dH_ff[f] = A^H @ dL/dH_closed[f]
            matmul_into(
                &self.a_h_buf,
                &self.dl_dh_closed_f_buf,
                &mut self.dl_dh_ff_bin_buf,
            );
            for o in 0..n_out {
                for i in 0..n_in {
                    self.grad_ff.data[[i, f, o]] = self.dl_dh_ff_bin_buf[[o, i]];
                }
            }

            // dL/dH_fb[f] = A^H @ dL/dH_closed[f] @ H_ff^H @ A^H
            matmul_into(&self.a_h_buf, &self.dl_dh_closed_f_buf, &mut self.work_buf);
            matmul_into(&self.work_buf, &self.h_ff_h_buf, &mut self.work2_buf);
            matmul_into(&self.work2_buf, &self.a_h_buf, &mut self.work_buf);
            for r in 0..n_out {
                for c in 0..n_out {
                    self.grad_fb.data[[c, f, r]] = self.work_buf[[r, c]];
                }
            }
        }

        // Backward through feedforward submodule.
        let _ = self
            .feedforward
            .backward(&self.identity_ff, &self.h_ff_response, &self.grad_ff)?;

        // Backward through feedback submodule.
        let _ = self
            .feedback
            .backward(&self.identity_fb, &self.h_fb_response, &self.grad_fb)?;

        // dL/dinput[b, f, i] = sum_o conj(H_closed[f, o, i]) * grad_output[b, f, o]
        if input_shape.len() == 3
            && let Some(grad_output_data) = grad_output.data.as_slice()
        {
            let batch = input_shape[0];
            let grad_input_data = self
                .grad_input
                .as_slice_mut()
                .expect("recursion gradient buffer must be contiguous");
            for batch_index in 0..batch {
                for f in 0..nb {
                    for input_channel in 0..n_in {
                        let mut sum = Complex::default();
                        for output_channel in 0..n_out {
                            let output_index = (batch_index * nb + f) * n_out + output_channel;
                            sum += grad_output_data[output_index]
                                * h_closed[[f, output_channel, input_channel]].conj();
                        }
                        let input_index = (batch_index * nb + f) * n_in + input_channel;
                        grad_input_data[input_index] = sum;
                    }
                }
            }
        } else {
            for i in 0..n_in {
                for o in 0..n_out {
                    for f in 0..nb {
                        let h_conj = h_closed[[f, o, i]].conj();
                        let grad_slice = grad_output.data.index_axis(Axis(1), f);
                        let grad_bin = grad_slice.index_axis(Axis(1), o);
                        let mut input_grad_slice = self.grad_input.index_axis_mut(Axis(1), f);
                        let mut input_grad_bin = input_grad_slice.index_axis_mut(Axis(1), i);
                        for (destination, &gradient) in
                            input_grad_bin.iter_mut().zip(grad_bin.iter())
                        {
                            *destination += gradient * h_conj;
                        }
                    }
                }
            }
        }

        Ok(DiffTensor::from_array(self.grad_input.clone()))
    }

    fn input_channels(&self) -> usize {
        self.feedforward.input_channels()
    }
    fn output_channels(&self) -> usize {
        self.feedforward.output_channels()
    }
    fn n_bins(&self) -> usize {
        self.n_bins()
    }
    fn parameters(&self) -> Vec<&ArrayD<f64>> {
        let mut p = Vec::new();
        p.extend(self.feedforward.parameters());
        p.extend(self.feedback.parameters());
        p
    }
    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<f64>> {
        let mut p = Vec::new();
        p.extend(self.feedforward.parameters_mut());
        p.extend(self.feedback.parameters_mut());
        p
    }
    fn gradients(&self) -> Vec<&ArrayD<f64>> {
        let mut g = Vec::new();
        g.extend(self.feedforward.gradients());
        g.extend(self.feedback.gradients());
        g
    }
    fn zero_grad(&mut self) {
        self.feedforward.zero_grad();
        self.feedback.zero_grad();
    }
}

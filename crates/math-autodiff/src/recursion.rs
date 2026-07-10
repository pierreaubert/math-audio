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

use crate::error::AutodiffError;
use crate::module::DiffModule;
use crate::tensor::DiffTensor;

/// Build an identity spectrum of shape `(n_in, n_bins, n_in)`.
fn identity_spectrum(n_in: usize, n_bins: usize) -> DiffTensor<f64> {
    let mut data = ArrayD::zeros(IxDyn(&[n_in, n_bins, n_in]));
    for i in 0..n_in {
        for f in 0..n_bins {
            data[[i, f, i]] = Complex::new(1.0, 0.0);
        }
    }
    DiffTensor::from_array(data)
}

/// Extract the transfer matrix of a submodule as `(n_bins, n_out, n_in)`.
fn module_response(
    module: &dyn DiffModule<f64>,
    n_in: usize,
) -> Result<Array3<Complex<f64>>, AutodiffError> {
    let nb = module.n_bins();
    let identity = identity_spectrum(n_in, nb);
    let output = module.forward(&identity)?;
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
}

impl std::fmt::Debug for Recursion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Recursion")
            .field("n_bins", &self.n_bins)
            .field("feedforward", &"<dyn DiffModule>")
            .field("feedback", &"<dyn DiffModule>")
            .finish()
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
        Ok(Self {
            n_bins: feedforward.n_bins(),
            feedforward,
            feedback,
        })
    }

    fn n_bins(&self) -> usize {
        self.n_bins
    }

    #[allow(clippy::type_complexity)]
    fn closed_loop_response(&self) -> Result<ClosedLoopResponse, AutodiffError> {
        let n_in = self.feedforward.input_channels();
        let n_out = self.feedforward.output_channels();
        let nb = self.n_bins();

        let h_ff = module_response(self.feedforward.as_ref(), n_in)?; // (nb, n_out, n_in)
        let h_fb = module_response(self.feedback.as_ref(), n_out)?; // (nb, n_out, n_out)

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

        let (h_closed, _, _, _) = self.closed_loop_response()?;
        let n_out = self.feedforward.output_channels();
        let mut output_shape = input_shape.to_vec();
        output_shape[2] = n_out;
        let mut output = ArrayD::zeros(IxDyn(&output_shape));

        for o in 0..n_out {
            for i in 0..n_in {
                for f in 0..nb {
                    let h = h_closed[[f, o, i]];
                    let input_slice = input.data.index_axis(Axis(1), f);
                    let input_bin = input_slice.index_axis(Axis(1), i);
                    let mut output_slice = output.index_axis_mut(Axis(1), f);
                    let mut output_bin = output_slice.index_axis_mut(Axis(1), o);
                    output_bin += &input_bin.mapv(|x| x * h);
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
        if input_shape.len() < 3 || grad_shape.len() < 3 {
            return Err(AutodiffError::Message(
                "Recursion::backward: input and grad_output must have at least 3 dimensions"
                    .to_string(),
            ));
        }
        let nb = input_shape[1];
        let n_in = input_shape[2];
        let n_out = self.feedforward.output_channels();
        if nb != self.n_bins() {
            return Err(AutodiffError::Message(format!(
                "Recursion::backward: expected {} bins, got {}",
                self.n_bins(),
                nb
            )));
        }
        if grad_shape[2] != n_out {
            return Err(AutodiffError::Message(format!(
                "Recursion::backward: expected {} output channels, got {}",
                n_out, grad_shape[2]
            )));
        }

        let (h_closed, h_ff, h_fb, a_arr) = self.closed_loop_response()?;

        // dL/dH_closed[f, o, i] = sum_b grad_output[b, f, o] * conj(input[b, f, i])
        let mut dl_dh_closed = Array3::<Complex<f64>>::zeros((nb, n_out, n_in));
        for f in 0..nb {
            for o in 0..n_out {
                for i in 0..n_in {
                    let grad_slice = grad_output.data.index_axis(Axis(1), f);
                    let grad_bin = grad_slice.index_axis(Axis(1), o);
                    let input_slice = input.data.index_axis(Axis(1), f);
                    let input_bin = input_slice.index_axis(Axis(1), i);
                    dl_dh_closed[[f, o, i]] = grad_bin
                        .iter()
                        .zip(input_bin.iter())
                        .map(|(g, x)| g * x.conj())
                        .sum::<Complex<f64>>();
                }
            }
        }

        // Build per-bin dL/dH_ff and dL/dH_fb.
        let mut dl_dh_ff = Array3::<Complex<f64>>::zeros((nb, n_out, n_in));
        let mut dl_dh_fb = Array3::<Complex<f64>>::zeros((nb, n_out, n_out));

        for f in 0..nb {
            let a = {
                let mut m = Array2::zeros((n_out, n_out));
                for r in 0..n_out {
                    for c in 0..n_out {
                        m[[r, c]] = a_arr[[f, r, c]];
                    }
                }
                m
            };
            let a_conj = a.mapv(|x| x.conj());
            let a_h = a_conj.t();
            let h_ff_f = {
                let mut m = Array2::zeros((n_out, n_in));
                for r in 0..n_out {
                    for c in 0..n_in {
                        m[[r, c]] = h_ff[[f, r, c]];
                    }
                }
                m
            };
            let dl_dh_closed_f = {
                let mut m = Array2::zeros((n_out, n_in));
                for r in 0..n_out {
                    for c in 0..n_in {
                        m[[r, c]] = dl_dh_closed[[f, r, c]];
                    }
                }
                m
            };

            // dL/dH_ff[f] = A^H @ dL/dH_closed[f]
            let dl_dh_ff_bin = a_h.dot(&dl_dh_closed_f);
            for o in 0..n_out {
                for i in 0..n_in {
                    dl_dh_ff[[f, o, i]] = dl_dh_ff_bin[[o, i]];
                }
            }

            // dL/dH_fb[f] = A^H @ dL/dH_closed[f] @ H_ff^H @ A^H
            let dl_dh_fb_bin = a_h
                .dot(&dl_dh_closed_f)
                .dot(&h_ff_f.mapv(|x| x.conj()).t())
                .dot(&a_h);
            for r in 0..n_out {
                for c in 0..n_out {
                    dl_dh_fb[[f, r, c]] = dl_dh_fb_bin[[r, c]];
                }
            }
        }

        // Backward through feedforward submodule.
        {
            let identity_in = identity_spectrum(n_in, nb);
            let h_ff_response = {
                let mut out = ArrayD::zeros(IxDyn(&[n_in, nb, n_out]));
                for i in 0..n_in {
                    for f in 0..nb {
                        for o in 0..n_out {
                            out[[i, f, o]] = h_ff[[f, o, i]];
                        }
                    }
                }
                DiffTensor::from_array(out)
            };
            let mut grad_ff = ArrayD::zeros(IxDyn(&[n_in, nb, n_out]));
            for i in 0..n_in {
                for f in 0..nb {
                    for o in 0..n_out {
                        grad_ff[[i, f, o]] = dl_dh_ff[[f, o, i]];
                    }
                }
            }
            let _ = self.feedforward.backward(
                &identity_in,
                &h_ff_response,
                &DiffTensor::from_array(grad_ff),
            )?;
        }

        // Backward through feedback submodule.
        {
            let identity_in = identity_spectrum(n_out, nb);
            let h_fb_response = {
                let mut out = ArrayD::zeros(IxDyn(&[n_out, nb, n_out]));
                for i in 0..n_out {
                    for f in 0..nb {
                        for o in 0..n_out {
                            out[[i, f, o]] = h_fb[[f, o, i]];
                        }
                    }
                }
                DiffTensor::from_array(out)
            };
            let mut grad_fb = ArrayD::zeros(IxDyn(&[n_out, nb, n_out]));
            for i in 0..n_out {
                for f in 0..nb {
                    for o in 0..n_out {
                        grad_fb[[i, f, o]] = dl_dh_fb[[f, o, i]];
                    }
                }
            }
            let _ = self.feedback.backward(
                &identity_in,
                &h_fb_response,
                &DiffTensor::from_array(grad_fb),
            )?;
        }

        // dL/dinput[b, f, i] = sum_o conj(H_closed[f, o, i]) * grad_output[b, f, o]
        let mut grad_input = ArrayD::zeros(IxDyn(input_shape));
        for i in 0..n_in {
            for o in 0..n_out {
                for f in 0..nb {
                    let h_conj = h_closed[[f, o, i]].conj();
                    let grad_slice = grad_output.data.index_axis(Axis(1), f);
                    let grad_bin = grad_slice.index_axis(Axis(1), o);
                    let mut input_grad_slice = grad_input.index_axis_mut(Axis(1), f);
                    let mut input_grad_bin = input_grad_slice.index_axis_mut(Axis(1), i);
                    input_grad_bin += &grad_bin.mapv(|g| g * h_conj);
                }
            }
        }

        Ok(DiffTensor::from_array(grad_input))
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

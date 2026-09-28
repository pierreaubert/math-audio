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
use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex};

use crate::error::AutodiffError;
use crate::module::{
    DiffModule, Scalar, fconst, fnv1a_init, fnv1a_step, to_f64, validate_spectral_gradient_shape,
};
use crate::tensor::DiffTensor;

/// Extract the transfer matrix of a submodule as `(n_bins, n_out, n_in)`.
fn module_response<T: Scalar + 'static>(
    module: &dyn DiffModule<T>,
    identity: &DiffTensor<T>,
) -> Result<Array3<Complex<T>>, AutodiffError> {
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

fn invert_complex_matrix<T: Scalar>(
    mat: &Array2<Complex<T>>,
    bin: usize,
) -> Result<Array2<Complex<T>>, AutodiffError> {
    // The inverse runs in f64 (nalgebra implements ComplexField only for
    // concrete f32/f64) and is cast once each way.
    let mat64 = mat
        .to_owned()
        .mapv(|c| Complex::new(to_f64(c.re), to_f64(c.im)));
    let dm = ndarray2_to_dmatrix(&mat64);
    let inv = dm.try_inverse().ok_or_else(|| {
        AutodiffError::Message(format!(
            "Recursion: failed to invert (I - H_fb) at frequency bin {bin}"
        ))
    })?;
    let arr64 = dmatrix_to_ndarray2(&inv);
    Ok(arr64
        .to_owned()
        .mapv(|c| Complex::new(fconst::<T>(c.re), fconst::<T>(c.im))))
}

/// Compute `out = a @ b` into the pre-allocated `out` buffer.
#[allow(clippy::many_single_char_names)]
fn matmul_into<T: Scalar>(
    a: &Array2<Complex<T>>,
    b: &Array2<Complex<T>>,
    out: &mut Array2<Complex<T>>,
) {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();
    assert_eq!(b.nrows(), k, "matmul_into: incompatible inner dimensions");
    assert_eq!(out.dim(), (m, n), "matmul_into: incompatible output shape");
    out.fill(Complex::new(T::zero(), T::zero()));
    for i in 0..m {
        for l in 0..k {
            let a_il = a[[i, l]];
            if a_il == Complex::new(T::zero(), T::zero()) {
                continue;
            }
            for j in 0..n {
                out[[i, j]] += a_il * b[[l, j]];
            }
        }
    }
}

/// Compute the conjugate transpose `out = a^H` into the pre-allocated `out` buffer.
fn conj_transpose_into<T: Scalar>(src: &Array2<Complex<T>>, dst: &mut Array2<Complex<T>>) {
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

/// Maximum channel count served by [`fused_backward_bins`].
///
/// Wider systems fall back to the reusable-buffer path in
/// [`Recursion::backward`]; the bound keeps the per-bin stack temporaries
/// small (two 16x16 complex matrices).
const MAX_STACK_CHANNELS: usize = 16;

/// Contiguous slice views for [`fused_backward_bins`].
///
/// Source tensors use `(bins, rows, cols)` C-order; destination tensors use
/// `(cols, bins, rows)` C-order, matching the layouts built by
/// [`Recursion::backward`].
struct FusedBinViews<'a, T> {
    h_ff: &'a [Complex<T>],
    h_fb: &'a [Complex<T>],
    a: &'a [Complex<T>],
    dl_dh_closed: &'a [Complex<T>],
    h_ff_response: &'a mut [Complex<T>],
    h_fb_response: &'a mut [Complex<T>],
    grad_ff: &'a mut [Complex<T>],
    grad_fb: &'a mut [Complex<T>],
}

/// Fused per-bin backward solve over contiguous slices.
///
/// Computes `dL/dH_ff[f] = A^H @ G` once per bin and reuses it for
/// `dL/dH_fb[f] = (A^H @ G) @ H_ff^H @ A^H`, scattering both gradients and
/// the response blocks directly into their destination tensors. Inner
/// accumulation order matches [`matmul_into`], so results agree with the
/// buffered path. Each iteration touches only its own disjoint slice regions
/// plus loop-local temporaries, keeping bins independent.
#[allow(
    clippy::many_single_char_names,
    reason = "matrix-index names match the surrounding recursion math"
)]
fn fused_backward_bins<T: Scalar>(
    views: FusedBinViews<'_, T>,
    nb: usize,
    n_in: usize,
    n_out: usize,
) {
    debug_assert!(n_out <= MAX_STACK_CHANNELS && n_in <= MAX_STACK_CHANNELS);
    let zero = Complex::new(T::zero(), T::zero());
    let FusedBinViews {
        h_ff,
        h_fb,
        a,
        dl_dh_closed,
        h_ff_response,
        h_fb_response,
        grad_ff,
        grad_fb,
    } = views;
    for f in 0..nb {
        let a_base = f * n_out * n_out;
        let hff_base = f * n_out * n_in;
        // Loop-local uninitialized scratch: only `[..n_out][..n_in]` is
        // written below, and only that region is read back, so no per-bin
        // zeroing is needed and iterations share no state.
        let mut t1: [[MaybeUninit<Complex<T>>; MAX_STACK_CHANNELS]; MAX_STACK_CHANNELS] =
            [[MaybeUninit::uninit(); MAX_STACK_CHANNELS]; MAX_STACK_CHANNELS];
        for i in 0..n_in {
            let row = (i * nb + f) * n_out;
            for o in 0..n_out {
                h_ff_response[row + o] = h_ff[hff_base + o * n_in + i];
                let mut sum = zero;
                for k in 0..n_out {
                    sum += a[a_base + k * n_out + o].conj() * dl_dh_closed[hff_base + k * n_in + i];
                }
                t1[o][i].write(sum);
                grad_ff[row + o] = sum;
            }
        }
        // T2 = T1 @ H_ff^H (n_out x n_out) on the stack.
        let mut t2: [[MaybeUninit<Complex<T>>; MAX_STACK_CHANNELS]; MAX_STACK_CHANNELS] =
            [[MaybeUninit::uninit(); MAX_STACK_CHANNELS]; MAX_STACK_CHANNELS];
        for r in 0..n_out {
            for k in 0..n_out {
                let mut sum = zero;
                for j in 0..n_in {
                    // SAFETY: `t1[r][j]` was written for all `r < n_out`,
                    // `j < n_in` in the loop above.
                    let t = unsafe { t1[r][j].assume_init() };
                    sum += t * h_ff[hff_base + k * n_in + j].conj();
                }
                t2[r][k].write(sum);
            }
        }
        // T3 = T2 @ A^H scattered to grad_fb alongside the H_fb response copy.
        let hfb_base = f * n_out * n_out;
        for c in 0..n_out {
            let row = (c * nb + f) * n_out;
            for r in 0..n_out {
                h_fb_response[row + r] = h_fb[hfb_base + r * n_out + c];
                let mut sum = zero;
                for k in 0..n_out {
                    // SAFETY: `t2[r][k]` was written for all `r < n_out`,
                    // `k < n_out` in the loop above.
                    let t = unsafe { t2[r][k].assume_init() };
                    sum += t * a[a_base + c * n_out + k].conj();
                }
                grad_fb[row + r] = sum;
            }
        }
    }
}

/// Fill an `(n, nb, n)` tensor with an identity spectrum.
fn fill_identity_spectrum<T: Scalar>(n: usize, nb: usize, tensor: &mut DiffTensor<T>) {
    tensor.data.fill(Complex::new(T::zero(), T::zero()));
    for i in 0..n {
        for f in 0..nb {
            tensor.data[[i, f, i]] = Complex::new(T::one(), T::zero());
        }
    }
}

type ClosedLoopResponse<T> = (
    Array3<Complex<T>>,
    Array3<Complex<T>>,
    Array3<Complex<T>>,
    Array3<Complex<T>>,
);

/// Closed-loop MIMO composition `y = (I - H_fb)^-1 @ H_ff @ x`.
pub struct Recursion<T: 'static = f64> {
    pub feedforward: Box<dyn DiffModule<T>>,
    pub feedback: Box<dyn DiffModule<T>>,
    n_bins: usize,
    response_cache: Mutex<Option<(u64, Arc<ClosedLoopResponse<T>>)>>,
    // Reusable backward buffers to avoid per-call heap allocations.
    identity_ff: DiffTensor<T>,
    identity_fb: DiffTensor<T>,
    h_ff_response: DiffTensor<T>,
    h_fb_response: DiffTensor<T>,
    grad_ff: DiffTensor<T>,
    grad_fb: DiffTensor<T>,
    dl_dh_closed: Array3<Complex<T>>,
    grad_input: ArrayD<Complex<T>>,
    // 2-D per-bin work buffers.
    a_buf: Array2<Complex<T>>,
    a_h_buf: Array2<Complex<T>>,
    h_ff_f_buf: Array2<Complex<T>>,
    h_ff_h_buf: Array2<Complex<T>>,
    dl_dh_closed_f_buf: Array2<Complex<T>>,
    dl_dh_ff_bin_buf: Array2<Complex<T>>,
    work_buf: Array2<Complex<T>>,
    work2_buf: Array2<Complex<T>>,
}

impl<T> std::fmt::Debug for Recursion<T> {
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

impl<T: Scalar> Recursion<T> {
    /// Create a new closed-loop recursion module.
    ///
    /// # Errors
    ///
    /// Returns an error if the feedforward and feedback modules have incompatible
    /// frequency-bin counts or channel dimensions.
    pub fn new(
        feedforward: Box<dyn DiffModule<T>>,
        feedback: Box<dyn DiffModule<T>>,
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
        let mut hash = fnv1a_step(fnv1a_init(), self.n_bins as u64);
        for parameter in self
            .feedforward
            .parameters()
            .into_iter()
            .chain(self.feedback.parameters())
        {
            for &dim in parameter.shape() {
                hash = fnv1a_step(hash, dim as u64);
            }
            if let Some(slice) = parameter.as_slice() {
                for &value in slice {
                    hash = fnv1a_step(hash, value.hash_bits());
                }
            } else {
                for &value in parameter {
                    hash = fnv1a_step(hash, value.hash_bits());
                }
            }
        }
        hash
    }

    fn cached_closed_loop_response(&self) -> Result<Arc<ClosedLoopResponse<T>>, AutodiffError> {
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
    fn closed_loop_response(&self) -> Result<ClosedLoopResponse<T>, AutodiffError> {
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
                            Complex::new(T::one(), T::zero())
                        } else {
                            Complex::new(T::zero(), T::zero())
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
                    let mut sum = Complex::new(T::zero(), T::zero());
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

impl<T: Scalar> DiffModule<T> for Recursion<T> {
    fn forward(&self, input: &DiffTensor<T>) -> Result<DiffTensor<T>, AutodiffError> {
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
                        let mut sum = Complex::new(T::zero(), T::zero());
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
        input: &DiffTensor<T>,
        _output: &DiffTensor<T>,
        grad_output: &DiffTensor<T>,
    ) -> Result<DiffTensor<T>, AutodiffError> {
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
        // (Scratch tensors need no fill: every element is overwritten below.
        // Only the accumulating non-contiguous grad_input path re-zeroes.)
        if self.grad_input.shape() != input_shape {
            self.grad_input = ArrayD::zeros(IxDyn(input_shape));
        }

        // dL/dH_closed[f, o, i] = sum_b grad_output[b, f, o] * conj(input[b, f, i])
        if input_shape.len() == 3
            && let Some(grad_output_data) = grad_output.data.as_slice()
            && let Some(input_data) = input.data.as_slice()
            && let Some(dl_dh_closed) = self.dl_dh_closed.as_slice_mut()
        {
            let batch = input_shape[0];
            for f in 0..nb {
                for o in 0..n_out {
                    for i in 0..n_in {
                        let mut sum = Complex::new(T::zero(), T::zero());
                        for batch_index in 0..batch {
                            let grad_index = (batch_index * nb + f) * n_out + o;
                            let input_index = (batch_index * nb + f) * n_in + i;
                            sum += grad_output_data[grad_index] * input_data[input_index].conj();
                        }
                        dl_dh_closed[(f * n_out + o) * n_in + i] = sum;
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
                            .sum::<Complex<T>>();
                    }
                }
            }
        }

        // Build per-bin dL/dH_ff and dL/dH_fb, populating response/gradient
        // tensors for the feedforward and feedback backward calls in-place.
        // Fast path: fused kernel over contiguous slices. Fallback: buffered
        // 2-D path for wide channel counts or non-contiguous storage.
        let small = n_out <= MAX_STACK_CHANNELS && n_in <= MAX_STACK_CHANNELS;
        let views = small
            .then(|| {
                Some(FusedBinViews {
                    h_ff: h_ff.as_slice()?,
                    h_fb: h_fb.as_slice()?,
                    a: a_arr.as_slice()?,
                    dl_dh_closed: self.dl_dh_closed.as_slice()?,
                    h_ff_response: self.h_ff_response.data.as_slice_mut()?,
                    h_fb_response: self.h_fb_response.data.as_slice_mut()?,
                    grad_ff: self.grad_ff.data.as_slice_mut()?,
                    grad_fb: self.grad_fb.data.as_slice_mut()?,
                })
            })
            .flatten();
        if let Some(views) = views {
            fused_backward_bins(views, nb, n_in, n_out);
        } else {
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
        }

        // Backward through feedforward submodule (dLoss/dInput is discarded,
        // so only parameter gradients are accumulated).
        self.feedforward.backward_params_only(
            &self.identity_ff,
            &self.h_ff_response,
            &self.grad_ff,
        )?;

        // Backward through feedback submodule.
        self.feedback.backward_params_only(
            &self.identity_fb,
            &self.h_fb_response,
            &self.grad_fb,
        )?;

        // dL/dinput[b, f, i] = sum_o conj(H_closed[f, o, i]) * grad_output[b, f, o]
        if input_shape.len() == 3
            && let Some(grad_output_data) = grad_output.data.as_slice()
            && let Some(h_closed_data) = h_closed.as_slice()
        {
            let batch = input_shape[0];
            let grad_input_data = self
                .grad_input
                .as_slice_mut()
                .expect("recursion gradient buffer must be contiguous");
            for batch_index in 0..batch {
                for f in 0..nb {
                    for input_channel in 0..n_in {
                        let mut sum = Complex::new(T::zero(), T::zero());
                        for output_channel in 0..n_out {
                            let output_index = (batch_index * nb + f) * n_out + output_channel;
                            let h_index = (f * n_out + output_channel) * n_in + input_channel;
                            sum += grad_output_data[output_index] * h_closed_data[h_index].conj();
                        }
                        let input_index = (batch_index * nb + f) * n_in + input_channel;
                        grad_input_data[input_index] = sum;
                    }
                }
            }
        } else {
            // Accumulating path: re-zero the reused buffer first.
            self.grad_input.fill(Complex::new(T::zero(), T::zero()));
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

        let grad_input = std::mem::replace(&mut self.grad_input, ArrayD::zeros(IxDyn(&[])));
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
    fn parameters(&self) -> Vec<&ArrayD<T>> {
        let mut p = Vec::new();
        p.extend(self.feedforward.parameters());
        p.extend(self.feedback.parameters());
        p
    }
    fn parameters_mut(&mut self) -> Vec<&mut ArrayD<T>> {
        let mut p = Vec::new();
        p.extend(self.feedforward.parameters_mut());
        p.extend(self.feedback.parameters_mut());
        p
    }
    fn gradients(&self) -> Vec<&ArrayD<T>> {
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

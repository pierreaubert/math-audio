//! Chroma feature extraction.
//!
//! Ported from bliss-audio chroma.rs — already pure Rust, no aubio dependency.
//! Computes 13 interval/triad features from the chromagram.

use super::utils::{hz_to_octs_inplace, normalize, stft};
use crate::stft::generate_hann_window;
use ndarray::{Array, Array1, Array2, Axis, Zip, arr2, s};
use oxiblas_ndarray::blas::dot_view;
use realfft::{RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use std::collections::HashMap;
use std::sync::Arc;

const WINDOW_SIZE: usize = 8192;
const MAX_VALUE: f32 = 1.0;
const MIN_VALUE: f32 = 0.0;
const MAX_L2_INTERVAL: f32 = 0.25;
const MAX_L2_TRIAD: f32 = 0.025;
const MAX_TRIAD_INTERVAL_RATIO: f32 = std::f32::consts::FRAC_PI_2;

/// Error type for chroma analysis.
#[derive(Debug, Clone)]
pub struct ChromaError(pub String);

impl std::fmt::Display for ChromaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "chroma error: {}", self.0)
    }
}

impl std::error::Error for ChromaError {}

/// Reusable chroma feature extractor.
///
/// Caches the Hann window, FFT plan, STFT scratch buffers, and the chroma
/// filter matrix so repeated analysis on the same window size does not
/// re-allocate or re-compute them on every call.
///
/// The cached filter is stored in transposed form `(n_bins, n_chroma)` as
/// `f32` so the bin-major matmul loop reads half the data and accumulates in
/// the faster single-precision path without per-call conversions.
pub struct ChromaFeatureExtractor {
    window_size: usize,
    hop_length: usize,
    hann_window: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    fft_input: Vec<f32>,
    fft_output: Vec<Complex<f32>>,
    spectrum_buf: Option<Array2<f64>>,
    padded_buf: Vec<f32>,
    filter_cache: HashMap<(u32, usize, u32, i64), Array2<f32>>,
    chroma_buf: Vec<f32>,
}

impl ChromaFeatureExtractor {
    /// Create a new extractor with the default chroma window size (8192 samples).
    pub fn new() -> Self {
        Self::with_window_size(WINDOW_SIZE, 2205)
    }

    /// Create a new extractor with a custom window and hop size.
    pub fn with_window_size(window_size: usize, hop_length: usize) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(window_size);
        let hann_window = generate_hann_window(window_size);
        let n_bins = window_size / 2 + 1;
        Self {
            window_size,
            hop_length,
            hann_window,
            fft,
            fft_input: vec![0.0; window_size],
            fft_output: vec![Complex::new(0.0, 0.0); n_bins],
            spectrum_buf: None,
            padded_buf: Vec::new(),
            filter_cache: HashMap::new(),
            chroma_buf: Vec::new(),
        }
    }

    /// Reset internal state (currently a no-op; buffers are overwritten each analysis).
    pub fn reset(&mut self) {}

    /// Compute the STFT magnitude spectrum for the configured window/hop,
    /// reusing internal buffers.
    fn compute_stft(&mut self, signal: &[f32]) -> Array2<f64> {
        let window_length = self.window_size;
        let hop_length = self.hop_length;
        let n_frames = (signal.len() as f32 / hop_length as f32).ceil() as usize;
        let n_bins = window_length / 2 + 1;

        let mut spectrum = match self.spectrum_buf.take() {
            Some(mut s) => {
                if s.dim() == (n_bins, n_frames) {
                    s.fill(0.0);
                    s
                } else {
                    Array2::zeros((n_bins, n_frames))
                }
            }
            None => Array2::zeros((n_bins, n_frames)),
        };

        // Reuse the reflect-pad buffer instead of allocating each call.
        let pad = window_length / 2;
        let padded_len = signal.len() + 2 * pad;
        self.padded_buf.resize(padded_len, 0.0);
        for i in 0..pad {
            self.padded_buf[i] = signal[pad - i];
        }
        self.padded_buf[pad..pad + signal.len()].copy_from_slice(signal);
        let suffix_start = pad + signal.len();
        for i in 0..pad {
            self.padded_buf[suffix_start + i] = signal[signal.len() - 2 - i];
        }

        for (window, mut stft_col) in self.padded_buf
            .windows(window_length)
            .step_by(hop_length)
            .zip(spectrum.axis_iter_mut(Axis(1)))
        {
            for i in 0..window_length {
                self.fft_input[i] = window[i] * self.hann_window[i];
            }
            self.fft
                .process(&mut self.fft_input, &mut self.fft_output)
                .expect("real FFT forward failed");
            for i in 0..n_bins {
                let x = self.fft_output[i];
                // Store squared magnitude; chroma_stft_with_filter expects it,
                // and pip_track works on squared values with an adjusted threshold.
                stft_col[i] = (x.re * x.re + x.im * x.im) as f64;
            }
        }

        spectrum
    }

    /// Compute 13 chroma interval features from the full song samples.
    ///
    /// Returns a Vec of 13 normalized features (6 interval classes + 4 triads + 2 L2 norms + 1 ratio).
    pub fn compute(&mut self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>, ChromaError> {
        let n_chroma = 12u32;

        let mut spectrum = self.compute_stft(samples);
        let tuning = estimate_tuning(sample_rate, &spectrum, self.window_size, 0.01, n_chroma)?;
        let cache_key = (sample_rate, self.window_size, n_chroma, Self::quantize_tuning(tuning));

        let filter = match self.filter_cache.get(&cache_key) {
            Some(f) => f,
            None => {
                let f = chroma_filter(sample_rate, self.window_size, n_chroma, tuning)?;
                // Store in transposed, contiguous, single-precision form for the
                // cache-friendly f32 matmul fast path.
                let f_t = f.t().mapv(|x| x as f32);
                self.filter_cache.insert(cache_key, f_t);
                self.filter_cache.get(&cache_key).unwrap()
            }
        };

        let n_frames = spectrum.shape()[1];
        self.chroma_buf.resize(n_chroma as usize * n_frames, 0.0);
        let chroma = chroma_stft_with_filter(filter, &mut spectrum, &mut self.chroma_buf)?;
        self.spectrum_buf = Some(spectrum);

        let mut raw_features = chroma_interval_features(chroma)?;

        let (mut interval_class, mut interval_class_mode) =
            raw_features.view_mut().split_at(Axis(0), 6);

        let l2_norm_interval_class = dot_view(&interval_class.view(), &interval_class.view()).sqrt();
        let l2_norm_interval_class_mode =
            dot_view(&interval_class_mode.view(), &interval_class_mode.view()).sqrt();

        if l2_norm_interval_class > 0. {
            interval_class /= l2_norm_interval_class;
        }
        if l2_norm_interval_class_mode > 0. {
            interval_class_mode /= l2_norm_interval_class_mode;
        }

        let mut features: Vec<f32> = raw_features
            .mapv_into_any(|x| normalize(x as f32, MIN_VALUE, MAX_VALUE))
            .to_vec();

        let normalized_l2_norm_interval_class =
            (2. * (l2_norm_interval_class as f32 - 0.) / (MAX_L2_INTERVAL - 0.) - 1.).min(1.);
        features.push(normalized_l2_norm_interval_class);

        let normalized_l2_norm_interval_class_mode =
            (2. * (l2_norm_interval_class_mode as f32 - 0.) / (MAX_L2_TRIAD - 0.) - 1.).min(1.);
        features.push(normalized_l2_norm_interval_class_mode);

        let angle = (20. * l2_norm_interval_class_mode).atan2(l2_norm_interval_class + 1e-12_f64);
        let normalized_ratio = 2. * (angle as f32 - 0.) / (MAX_TRIAD_INTERVAL_RATIO - 0.) - 1.;
        features.push(normalized_ratio);

        Ok(features)
    }

    fn quantize_tuning(tuning: f64) -> i64 {
        // Tuning is in the range [-0.5, 0.5) bins; quantize to 1e-4 bins.
        (tuning * 10_000.0).round() as i64
    }
}

impl Default for ChromaFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute chroma STFT using a pre-computed, transposed, single-precision
/// filter matrix.
///
/// `filter` must have shape `(n_bins, n_chroma)` (the transpose of the
/// canonical `(n_chroma, n_bins)` filter). `spectrum` must contain squared
/// magnitudes (as produced by `ChromaFeatureExtractor::compute_stft`).
/// `chroma_buf` is a reusable scratch buffer large enough to hold
/// `n_chroma * n_frames` `f32` values.
fn chroma_stft_with_filter(
    filter: &Array2<f32>,
    spectrum: &mut Array2<f64>,
    chroma_buf: &mut [f32],
) -> Result<Array2<f64>, ChromaError> {
    chroma_matmul(filter, spectrum, chroma_buf);

    let n_chroma = filter.shape()[1];
    let n_frames = spectrum.shape()[1];

    // Normalize columns while still in f32, then convert to f64 for the
    // downstream pipeline. This keeps the normalization pass in the faster
    // precision and avoids allocating a second copy of the chromagram.
    let mut sums = vec![0.0_f32; n_frames];
    for r in 0..n_chroma {
        let row = &chroma_buf[r * n_frames..(r + 1) * n_frames];
        for j in 0..n_frames {
            sums[j] += row[j].abs();
        }
    }
    for sum in &mut sums {
        if *sum < f32::MIN_POSITIVE {
            *sum = 1.0;
        }
    }
    for r in 0..n_chroma {
        let row = &mut chroma_buf[r * n_frames..(r + 1) * n_frames];
        for j in 0..n_frames {
            row[j] /= sums[j];
        }
    }

    let mut raw_chroma = Array2::<f64>::zeros((n_chroma, n_frames));
    if let Some(out) = raw_chroma.as_slice_memory_order_mut() {
        for (dst, &src) in out.iter_mut().zip(chroma_buf.iter()) {
            *dst = f64::from(src);
        }
    } else {
        for ((r, j), dst) in raw_chroma.indexed_iter_mut() {
            *dst = f64::from(chroma_buf[r * n_frames + j]);
        }
    }

    Ok(raw_chroma)
}

/// Cache-friendly dense multiply for the small chroma filter shape.
///
/// `filter` is passed in already-transposed form with shape `(n_bins, n_chroma)`
/// and contiguous rows. `spectrum` has shape `(n_bins, n_frames)`. The result
/// is written into `chroma_buf` as `n_chroma * n_frames` `f32` values. The
/// bin-major loop reads both operands contiguously and accumulates into a small
/// result that stays in L1 cache.
fn chroma_matmul(filter: &Array2<f32>, spectrum: &Array2<f64>, chroma_buf: &mut [f32]) {
    let n_bins = filter.shape()[0];
    let n_chroma = filter.shape()[1];
    let n_frames = spectrum.shape()[1];
    let out_len = n_chroma * n_frames;

    chroma_buf[..out_len].fill(0.0);

    // Fast path: flat slices for contiguous memory-order access.
    if let (Some(f), Some(s)) = (
        filter.as_slice_memory_order(),
        spectrum.as_slice_memory_order(),
    ) {
        for i in 0..n_bins {
            let f_row = &f[i * n_chroma..(i + 1) * n_chroma];
            let s_row = &s[i * n_frames..(i + 1) * n_frames];
            for (r, &fc) in f_row.iter().enumerate() {
                if fc == 0.0 {
                    continue;
                }
                let out_row = &mut chroma_buf[r * n_frames..(r + 1) * n_frames];
                for j in 0..n_frames {
                    out_row[j] += fc * (s_row[j] as f32);
                }
            }
        }
    } else {
        // Fallback for non-contiguous inputs (e.g. the standalone
        // `compute_chroma_features` path whose spectrum is permuted).
        for i in 0..n_bins {
            let f_row = filter.row(i);
            let s_row = spectrum.row(i);
            for (r, &fc) in f_row.iter().enumerate() {
                let out_row = &mut chroma_buf[r * n_frames..(r + 1) * n_frames];
                for j in 0..n_frames {
                    out_row[j] += fc * (s_row[j] as f32);
                }
            }
        }
    }
}

/// Compute 13 chroma interval features from the full song samples.
///
/// Returns a Vec of 13 normalized features (6 interval classes + 4 triads + 2 L2 norms + 1 ratio).
pub fn compute_chroma_features(samples: &[f32], sample_rate: u32) -> Result<Vec<f32>, ChromaError> {
    let n_chroma = 12u32;

    let mut spectrum = stft(samples, WINDOW_SIZE, 2205);
    let tuning = estimate_tuning(sample_rate, &spectrum, WINDOW_SIZE, 0.01, 12)?;
    let chroma = chroma_stft(sample_rate, &mut spectrum, WINDOW_SIZE, n_chroma, tuning)?;

    let mut raw_features = chroma_interval_features(chroma)?;

    let (mut interval_class, mut interval_class_mode) =
        raw_features.view_mut().split_at(Axis(0), 6);

    let l2_norm_interval_class = dot_view(&interval_class.view(), &interval_class.view()).sqrt();
    let l2_norm_interval_class_mode =
        dot_view(&interval_class_mode.view(), &interval_class_mode.view()).sqrt();

    if l2_norm_interval_class > 0. {
        interval_class /= l2_norm_interval_class;
    }
    if l2_norm_interval_class_mode > 0. {
        interval_class_mode /= l2_norm_interval_class_mode;
    }

    let mut features: Vec<f32> = raw_features
        .mapv_into_any(|x| normalize(x as f32, MIN_VALUE, MAX_VALUE))
        .to_vec();

    let normalized_l2_norm_interval_class =
        (2. * (l2_norm_interval_class as f32 - 0.) / (MAX_L2_INTERVAL - 0.) - 1.).min(1.);
    features.push(normalized_l2_norm_interval_class);

    let normalized_l2_norm_interval_class_mode =
        (2. * (l2_norm_interval_class_mode as f32 - 0.) / (MAX_L2_TRIAD - 0.) - 1.).min(1.);
    features.push(normalized_l2_norm_interval_class_mode);

    let angle = (20. * l2_norm_interval_class_mode).atan2(l2_norm_interval_class + 1e-12_f64);
    let normalized_ratio = 2. * (angle as f32 - 0.) / (MAX_TRIAD_INTERVAL_RATIO - 0.) - 1.;
    features.push(normalized_ratio);

    Ok(features)
}

fn chroma_interval_features(mut chroma: Array2<f64>) -> Result<Array1<f64>, ChromaError> {
    // Apply the exponential transform and normalize each frame in-place.
    chroma.mapv_inplace(|x| (x * 15.).exp());
    for mut column in chroma.columns_mut() {
        let mut sum = column.iter().map(|&x| x.abs()).sum();
        if sum < 0.0001 {
            sum = 1.;
        }
        column /= sum;
    }

    let templates = arr2(&[
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]);
    let interval_feature_matrix = extract_interval_features(&chroma, &templates);
    interval_feature_matrix.mean_axis(Axis(1)).ok_or_else(|| {
        ChromaError("Tried to run chroma on empty array. Need at least one sample.".to_string())
    })
}

/// 12-bit masks for the 10 interval templates (columns of the template matrix).
const TEMPLATE_MASKS: [u16; 10] = [
    0xFFF, // all bins
    0x001, // bin 0
    0x002, // bin 1
    0x184, // bins 2, 7, 8
    0x248, // bins 3, 6, 9
    0x010, // bin 4
    0x120, // bins 5, 8
    0x0C0, // bins 6, 7
    0x200, // bin 9
    0x000, // none
];

fn extract_interval_features(chroma: &Array2<f64>, _templates: &Array2<i32>) -> Array2<f64> {
    let n_frames = chroma.shape()[1];
    let mut result = Array2::<f64>::zeros((TEMPLATE_MASKS.len(), n_frames));
    for (t_idx, &mask) in TEMPLATE_MASKS.iter().enumerate() {
        for shift in 0..12 {
            let rotated = ((mask << shift) | (mask >> (12 - shift))) & 0xFFF;
            for frame in 0..n_frames {
                let mut prod = 1.0_f64;
                let mut bits = rotated;
                while bits != 0 {
                    let bin = bits.trailing_zeros() as usize;
                    prod *= chroma[[bin, frame]];
                    bits &= bits - 1;
                }
                result[[t_idx, frame]] += prod;
            }
        }
    }
    result
}

#[cfg(test)]
fn normalize_feature_sequence(feature: &Array2<f64>) -> Array2<f64> {
    let mut normalized_sequence = feature.to_owned();
    for mut column in normalized_sequence.columns_mut() {
        let mut sum = column.mapv(|x| x.abs()).sum();
        if sum < 0.0001 {
            sum = 1.;
        }
        column /= sum;
    }
    normalized_sequence
}

fn chroma_filter(
    sample_rate: u32,
    n_fft: usize,
    n_chroma: u32,
    tuning: f64,
) -> Result<Array2<f64>, ChromaError> {
    let ctroct = 5.0;
    let octwidth = 2.;
    let n_chroma_float = f64::from(n_chroma);
    let n_chroma2 = (n_chroma_float / 2.0).round() as u32;
    let n_chroma2_float = f64::from(n_chroma2);

    let frequencies = Array::linspace(0., f64::from(sample_rate), n_fft + 1);

    let mut freq_bins = frequencies;
    hz_to_octs_inplace(&mut freq_bins, tuning, n_chroma);
    freq_bins.mapv_inplace(|x| x * n_chroma_float);
    freq_bins[0] = freq_bins[1] - 1.5 * n_chroma_float;

    let mut binwidth_bins = Array::ones(freq_bins.raw_dim());
    binwidth_bins.slice_mut(s![0..freq_bins.len() - 1]).assign(
        &(&freq_bins.slice(s![1..]) - &freq_bins.slice(s![..-1]))
            .mapv(|x| if x <= 1. { 1. } else { x }),
    );

    let mut d: Array2<f64> = Array::zeros((n_chroma as usize, freq_bins.len()));
    for (idx, mut row) in d.rows_mut().into_iter().enumerate() {
        row.fill(idx as f64);
    }
    d = -d + &freq_bins;

    d.mapv_inplace(|x| {
        (x + n_chroma2_float + 10. * n_chroma_float) % n_chroma_float - n_chroma2_float
    });
    d = d / binwidth_bins;
    d.mapv_inplace(|x| (-0.5 * (2. * x) * (2. * x)).exp());

    let mut wts = d;
    for mut col in wts.columns_mut() {
        let mut sum = col.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if sum < f64::MIN_POSITIVE {
            sum = 1.;
        }
        col /= sum;
    }

    freq_bins.mapv_inplace(|x| (-0.5 * ((x / n_chroma_float - ctroct) / octwidth).powi(2)).exp());
    wts *= &freq_bins;

    // np.roll by -3
    let mut b = Array2::zeros(wts.dim());
    b.slice_mut(s![-3.., ..]).assign(&wts.slice(s![..3, ..]));
    b.slice_mut(s![..-3, ..]).assign(&wts.slice(s![3.., ..]));

    wts = b;
    let non_aliased = 1 + n_fft / 2;
    Ok(wts.slice_move(s![.., ..non_aliased]))
}

fn pip_track(
    sample_rate: u32,
    spectrum: &Array2<f64>,
    n_fft: usize,
) -> Result<(Vec<f64>, Vec<f64>), ChromaError> {
    let sample_rate_float = f64::from(sample_rate);
    let fmin = 150.0_f64;
    let fmax = 4000.0_f64.min(sample_rate_float / 2.0);
    // Spectrum contains squared magnitudes; keep the same effective threshold
    // in the squared domain: (0.1 * max_magnitude)^2 = 0.01 * max_sq.
    let threshold = 0.01;

    let fft_freqs = Array::linspace(0., sample_rate_float / 2., 1 + n_fft / 2);

    let length = spectrum.len_of(Axis(0));

    let freq_mask: Vec<bool> = fft_freqs
        .iter()
        .map(|&f| (fmin <= f) && (f < fmax))
        .collect();

    let ref_value = spectrum.map_axis(Axis(0), |x| {
        let first: f64 = *x.first().expect("empty spectrum axis");
        x.fold(first, |acc, &elem| if acc > elem { acc } else { elem }) * threshold
    });

    let taken_columns = freq_mask.iter().filter(|&&x| x).count();
    let mut pitches = Vec::with_capacity(taken_columns * length);
    let mut mags = Vec::with_capacity(taken_columns * length);

    // Issue #5: guard against empty freq mask (e.g. very low sample rates
    // where the first FFT bin is already above fmax).
    let beginning = match freq_mask.iter().position(|&b| b) {
        Some(b) => b,
        None => return Ok((pitches, mags)),
    };
    let end = match freq_mask.iter().rposition(|&b| b) {
        Some(e) => e,
        None => return Ok((pitches, mags)),
    };

    let zipped = Zip::indexed(spectrum.slice(s![beginning..end - 3, ..]))
        .and(spectrum.slice(s![beginning + 1..end - 2, ..]))
        .and(spectrum.slice(s![beginning + 2..end - 1, ..]));

    zipped.for_each(|(i, j), &before_elem, &elem, &after_elem| {
        if elem > ref_value[j] && after_elem <= elem && before_elem < elem {
            let avg = 0.5 * (after_elem - before_elem);
            let mut shift = 2. * elem - after_elem - before_elem;
            if shift.abs() < f64::MIN_POSITIVE {
                shift += 1.;
            }
            shift = avg / shift;
            pitches.push(((i + beginning + 1) as f64 + shift) * sample_rate_float / n_fft as f64);
            // `elem` is a squared magnitude; return the linear magnitude for tuning.
            mags.push((elem + 0.5 * avg * shift).sqrt());
        }
    });

    Ok((pitches, mags))
}

fn pitch_tuning(
    frequencies: &mut Array1<f64>,
    resolution: f64,
    bins_per_octave: u32,
) -> Result<f64, ChromaError> {
    if frequencies.is_empty() {
        return Ok(0.0);
    }
    hz_to_octs_inplace(frequencies, 0.0, 12);
    frequencies.mapv_inplace(|x| f64::from(bins_per_octave) * x % 1.0);
    frequencies.mapv_inplace(|x| if x >= 0.5 { x - 1. } else { x });

    let indexes = ((frequencies.to_owned() - -0.5) / resolution).mapv(|x| x as usize);
    let mut counts: Array1<usize> = Array::zeros(((0.5 - -0.5) / resolution) as usize);
    for &idx in indexes.iter() {
        if idx < counts.len() {
            counts[idx] += 1;
        }
    }
    let max_index = counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, v)| *v)
        .map(|(i, _)| i)
        .ok_or_else(|| ChromaError("empty counts in pitch_tuning".to_string()))?;

    Ok((-50. + (100. * resolution * max_index as f64)) / 100.)
}

fn estimate_tuning(
    sample_rate: u32,
    spectrum: &Array2<f64>,
    n_fft: usize,
    resolution: f64,
    bins_per_octave: u32,
) -> Result<f64, ChromaError> {
    let (pitch, mag) = pip_track(sample_rate, spectrum, n_fft)?;

    let (filtered_pitch, filtered_mag): (Vec<f64>, Vec<f64>) = pitch
        .iter()
        .zip(&mag)
        .filter(|&(&p, _)| p > 0.)
        .map(|(x, y)| (*x, *y))
        .unzip();

    if filtered_pitch.is_empty() {
        return Ok(0.);
    }

    // Compute median of magnitudes
    let mut sorted_mags = filtered_mag.clone();
    sorted_mags.sort_by(|a, b| a.total_cmp(b));
    let threshold = if sorted_mags.len() % 2 == 0 {
        (sorted_mags[sorted_mags.len() / 2 - 1] + sorted_mags[sorted_mags.len() / 2]) / 2.0
    } else {
        sorted_mags[sorted_mags.len() / 2]
    };

    let mut pitch_arr: Array1<f64> = filtered_pitch
        .iter()
        .zip(&filtered_mag)
        .filter_map(|(&p, &m)| if m >= threshold { Some(p) } else { None })
        .collect::<Vec<f64>>()
        .into();

    pitch_tuning(&mut pitch_arr, resolution, bins_per_octave)
}

fn chroma_stft(
    sample_rate: u32,
    spectrum: &mut Array2<f64>,
    n_fft: usize,
    n_chroma: u32,
    tuning: f64,
) -> Result<Array2<f64>, ChromaError> {
    // Standalone path receives linear magnitudes from `stft`; square them to
    // match the input contract of `chroma_stft_with_filter`.
    spectrum.mapv_inplace(|x| x * x);
    let filter = chroma_filter(sample_rate, n_fft, n_chroma, tuning)?;
    // Transpose to the contiguous `(n_bins, n_chroma)` layout and convert to
    // the single-precision form expected by the matmul fast path.
    let filter_t = filter.t().mapv(|x| x as f32);
    let n_frames = spectrum.shape()[1];
    let mut chroma_buf = vec![0.0_f32; n_chroma as usize * n_frames];
    chroma_stft_with_filter(&filter_t, spectrum, &mut chroma_buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chroma_features_length() {
        // Generate a simple tone
        let sr = 22050u32;
        let duration = 5.0;
        let n = (sr as f32 * duration) as usize;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();

        let features = compute_chroma_features(&signal, sr).unwrap();
        assert_eq!(features.len(), 13);
    }

    #[test]
    fn test_normalize_feature_sequence_basic() {
        let array = arr2(&[[0.1, 0.3, 0.4, 0.], [1.1, 0.53, 1.01, 0.]]);
        let expected = arr2(&[
            [0.08333333, 0.36144578, 0.28368794, 0.],
            [0.91666667, 0.63855422, 0.71631206, 0.],
        ]);

        let normalized = normalize_feature_sequence(&array);

        for (expected, actual) in normalized.iter().zip(expected.iter()) {
            assert!((expected - actual).abs() < 1e-6);
        }
    }

    #[test]
    fn test_chroma_empty_freq_mask_low_sr() {
        // Issue #5: very low sample rates can produce an empty freq mask
        // because Nyquist is below fmin (150 Hz).  This should not panic.
        let sr = 200u32; // Nyquist = 100 Hz < fmin = 150 Hz
        // Need at least WINDOW_SIZE (8192) + 1 samples for reflect_pad.
        let signal = vec![0.5f32; 8200];
        let result = compute_chroma_features(&signal, sr);
        assert!(
            result.is_ok(),
            "low-SR chroma should not error: {:?}",
            result
        );
    }
}

#[test]
fn test_pitch_tuning_empty_returns_zero() {
    let mut freqs = ndarray::Array1::zeros((0,));
    let tuning = pitch_tuning(&mut freqs, 0.01, 12).unwrap();
    assert_eq!(tuning, 0.0);
}

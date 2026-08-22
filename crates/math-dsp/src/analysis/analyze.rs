use super::compute::compute_fft;
use super::compute::compute_group_delay;
use super::compute::compute_rt60_clarity_spectra;
use super::compute::compute_single_fft_spectrum_internal;
use super::compute::compute_spectrogram;
use super::compute::compute_welch_spectrum_into;
use super::compute::with_welch_buffers;
use super::estimate::estimate_lag_with_confidence;
use super::interpolate::interpolate_log;
use super::interpolate::interpolate_log_phase;
use super::load::load_wav_mono;
use super::load::load_wav_mono_with_rate;
use super::measurement::analyze_ess_recording;
use super::misc::generate_log_frequencies;
use super::misc::next_power_of_two;
use super::misc::wav_next_power_of_two;
use super::plan::plan_fft_inverse;
use super::plan::plan_real_fft_forward;
use super::types::AnalysisResult;
use super::types::EssAnalysisResult;
use super::types::WavAnalysisOutput;
use super::types::WindowType;
use super::wav_analysis_config::WavAnalysisConfig;
use rustfft::num_complex::Complex;
use std::f32::consts::PI;
use std::path::Path;

/// Analyze a buffer of audio samples and return frequency response
///
/// # Arguments
/// * `samples` - Mono audio samples (f32, -1.0 to 1.0)
/// * `sample_rate` - Sample rate in Hz
/// * `config` - Analysis configuration
///
/// # Returns
/// Analysis result with frequency, magnitude, and phase data
pub fn analyze_wav_buffer(
    samples: &[f32],
    sample_rate: u32,
    config: &WavAnalysisConfig,
) -> Result<WavAnalysisOutput, String> {
    if samples.is_empty() {
        return Err("Signal is empty".to_string());
    }

    // Determine FFT size
    let fft_size = if config.single_fft {
        config
            .fft_size
            .unwrap_or_else(|| wav_next_power_of_two(samples.len()))
    } else {
        config.fft_size.unwrap_or(16384)
    };

    // Generate logarithmically spaced frequency points
    let log_freqs = generate_log_frequencies(config.num_points, config.min_freq, config.max_freq);

    // Compute spectrum and interpolate magnitude/phase at log frequencies.
    // For the Welch path we write into thread-local reusable buffers so that
    // the full-size per-bin vectors are not allocated on every call.
    let num_bins = fft_size / 2;
    let (mut interp_mag, interp_phase) = if config.single_fft {
        let (freqs, magnitudes_db, phases_deg) =
            compute_single_fft_spectrum_internal(samples, sample_rate, fft_size, config.no_window)?;
        (
            interpolate_log(&freqs, &magnitudes_db, &log_freqs),
            interpolate_log_phase(&freqs, &phases_deg, &log_freqs),
        )
    } else {
        let fft = plan_real_fft_forward(fft_size);
        with_welch_buffers(
            fft_size,
            &fft,
            |bufs| -> Result<(Vec<f32>, Vec<f32>), String> {
                compute_welch_spectrum_into(
                    samples,
                    sample_rate,
                    fft_size,
                    config.overlap,
                    &bufs.hann_window,
                    &bufs.scaled_window,
                    &fft,
                    &mut bufs.real_buffer,
                    &mut bufs.spectrum,
                    &mut bufs.real_scratch,
                    &mut bufs.magnitude_sum,
                    &mut bufs.phase_real_sum,
                    &mut bufs.phase_imag_sum,
                    &mut bufs.freqs,
                    &mut bufs.magnitudes_db,
                    &mut bufs.phases_deg,
                )?;
                let freqs = &bufs.freqs[..num_bins];
                let magnitudes_db = &bufs.magnitudes_db[..num_bins];
                let phases_deg = &bufs.phases_deg[..num_bins];
                Ok((
                    interpolate_log(freqs, magnitudes_db, &log_freqs),
                    interpolate_log_phase(freqs, phases_deg, &log_freqs),
                ))
            },
        )?
    };

    // Apply pink compensation if requested (for log sweeps)
    // Pink compensation is a correction for the raw magnitude of a log
    // sweep. Guard it to the single-FFT, rectangular-window mode selected by
    // `for_log_sweep`; applying the same slope to a stationary/Welch analysis
    // would silently manufacture a frequency tilt.
    if config.pink_compensation && config.single_fft && config.no_window {
        let ref_freq = 1000.0;
        for (i, freq) in log_freqs.iter().enumerate() {
            if *freq > 0.0 {
                let correction = 10.0 * (freq / ref_freq).log10();
                interp_mag[i] += correction;
            }
        }
    }

    if !config.subwoofer {
        if let Some(room_slope_db) = config.room_slope_db {
            apply_room_slope(
                &mut interp_mag,
                &log_freqs,
                config.min_freq,
                config.max_freq,
                room_slope_db,
            );
        }
    }

    Ok(WavAnalysisOutput {
        frequencies: log_freqs,
        magnitude_db: interp_mag,
        phase_deg: interp_phase,
    })
}

/// Analyze a WAV file and return frequency response
///
/// # Arguments
/// * `path` - Path to WAV file
/// * `config` - Analysis configuration
///
/// # Returns
/// Analysis result with frequency, magnitude, and phase data
pub fn analyze_wav_file(
    path: &Path,
    config: &WavAnalysisConfig,
) -> Result<WavAnalysisOutput, String> {
    let (samples, sample_rate) = load_wav_mono_with_rate(path)?;
    analyze_wav_buffer(&samples, sample_rate, config)
}

/// Add a log-frequency response tilt to a magnitude curve.
///
/// `slope_db` is the desired change between `min_freq` and `max_freq`:
/// `-10.0` adds 0 dB at the lower limit and -10 dB at the upper limit.
pub(super) fn apply_room_slope(
    magnitudes_db: &mut [f32],
    frequencies: &[f32],
    min_freq: f32,
    max_freq: f32,
    slope_db: f32,
) {
    if magnitudes_db.len() != frequencies.len()
        || !slope_db.is_finite()
        || !min_freq.is_finite()
        || !max_freq.is_finite()
        || min_freq <= 0.0
        || max_freq <= min_freq
    {
        return;
    }

    let log_span = (max_freq / min_freq).ln();
    for (magnitude, &frequency) in magnitudes_db.iter_mut().zip(frequencies) {
        if frequency.is_finite() && frequency > 0.0 {
            let position = ((frequency / min_freq).ln() / log_span).clamp(0.0, 1.0);
            *magnitude += slope_db * position;
        }
    }
}

/// Time-align recorded and reference signals given an estimated lag.
pub(crate) fn align_signals<'a>(
    lag: isize,
    reference: &'a [f32],
    recorded: &'a [f32],
) -> Result<(&'a [f32], &'a [f32]), String> {
    if lag >= 0 {
        let lag_usize = lag as usize;
        if lag_usize >= recorded.len() {
            return Err("Lag is larger than recorded signal length".to_string());
        }
        Ok((reference, &recorded[lag_usize..]))
    } else {
        let lag_usize = lag.checked_abs().ok_or("lag overflow")? as usize;
        if lag_usize >= reference.len() {
            return Err("Negative lag is larger than reference signal length".to_string());
        }
        Ok((&reference[lag_usize..], recorded))
    }
}

/// Adapt the canonical ESS result to the legacy, log-spaced `AnalysisResult`
/// surface. Keeping this conversion here means the recording entry point no
/// longer has a second ESS deconvolution implementation while preserving the
/// metrics and CSV shape expected by existing callers.
fn analysis_result_from_ess(ess: EssAnalysisResult, sample_rate: u32) -> AnalysisResult {
    let frequencies = generate_log_frequencies(2000, 20.0, 20_000.0);
    let magnitudes_db: Vec<f32> = ess
        .frequency_response
        .iter()
        .map(|value| {
            let magnitude = value.norm();
            if magnitude > 1e-10 {
                20.0 * magnitude.log10()
            } else {
                -200.0
            }
        })
        .collect();
    let phases_deg: Vec<f32> = ess
        .frequency_response
        .iter()
        .map(|value| value.arg().to_degrees())
        .collect();
    let spl_db = interpolate_log(&ess.frequencies, &magnitudes_db, &frequencies);
    let phase_deg = interpolate_log_phase(&ess.frequencies, &phases_deg, &frequencies);

    let mut impulse_response = ess.impulse_response;
    let peak_idx = impulse_response
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
        .map(|(index, _)| index)
        .unwrap_or(0);
    let pre_ring_samples = (0.005 * sample_rate as f32) as usize;
    let shift_amount = peak_idx.saturating_sub(pre_ring_samples);
    if shift_amount > 0 {
        impulse_response.rotate_left(shift_amount);
    }

    let impulse_time_ms: Vec<f32> = (0..impulse_response.len())
        .map(|index| index as f32 / sample_rate as f32 * 1000.0)
        .collect();
    let thd_percent = interpolate_log(&ess.frequencies, &ess.thd_percent, &frequencies);
    let harmonic_distortion_db = ess
        .harmonic_distortion_db
        .iter()
        .map(|curve| interpolate_log(&ess.frequencies, curve, &frequencies))
        .collect();
    let excess_group_delay_ms = compute_group_delay(&frequencies, &phase_deg);
    let (rt60_ms, clarity_c50_db, clarity_c80_db) =
        compute_rt60_clarity_spectra(&impulse_response, sample_rate as f32, &frequencies);
    let (spectrogram_db, _, _) =
        compute_spectrogram(&impulse_response, sample_rate as f32, 512, 128);

    AnalysisResult {
        frequencies,
        spl_db,
        phase_deg,
        estimated_lag_samples: ess.lag.lag_samples,
        impulse_response,
        impulse_time_ms,
        excess_group_delay_ms,
        thd_percent,
        harmonic_distortion_db,
        rt60_ms,
        clarity_c50_db,
        clarity_c80_db,
        spectrogram_db,
    }
}

/// Analyze a recorded WAV file against a reference signal
///
/// # Arguments
/// * `recorded_path` - Path to the recorded WAV file
/// * `reference_signal` - Reference signal (should match the signal used for playback)
/// * `sample_rate` - Sample rate in Hz
/// * `sweep_range` - Optional (start_freq, end_freq) if the signal is a log sweep
///
/// # Returns
/// Analysis result with frequency, SPL, and phase data
pub fn analyze_recording(
    recorded_path: &Path,
    reference_signal: &[f32],
    sample_rate: u32,
    sweep_range: Option<(f32, f32)>,
) -> Result<AnalysisResult, String> {
    // Load recorded WAV
    log::debug!("[FFT Analysis] Loading recorded file: {:?}", recorded_path);
    let recorded = load_wav_mono(recorded_path)?;
    log::debug!(
        "[FFT Analysis] Loaded {} samples from recording",
        recorded.len()
    );
    log::debug!(
        "[FFT Analysis] Reference has {} samples",
        reference_signal.len()
    );

    if recorded.is_empty() {
        return Err("Recorded signal is empty!".to_string());
    }
    if reference_signal.is_empty() {
        return Err("Reference signal is empty!".to_string());
    }

    if let Some(sweep_range) = sweep_range {
        let ess = analyze_ess_recording(&recorded, reference_signal, sample_rate, sweep_range)?;
        return Ok(analysis_result_from_ess(ess, sample_rate));
    }

    // Don't truncate yet - we need full signals for lag estimation
    let recorded = &recorded[..];
    let reference = reference_signal;

    // Debug: Check signal statistics (guarded to skip O(n) computation when disabled)
    if log::log_enabled!(log::Level::Debug) {
        let ref_max = reference
            .iter()
            .map(|&x| x.abs())
            .fold(0.0_f32, |a, b| a.max(b));
        let rec_max = recorded
            .iter()
            .map(|&x| x.abs())
            .fold(0.0_f32, |a, b| a.max(b));
        let ref_rms =
            (reference.iter().map(|&x| x * x).sum::<f32>() / reference.len() as f32).sqrt();
        let rec_rms = (recorded.iter().map(|&x| x * x).sum::<f32>() / recorded.len() as f32).sqrt();

        log::debug!(
            "[FFT Analysis] Reference: max={:.4}, RMS={:.4}",
            ref_max,
            ref_rms
        );
        log::debug!(
            "[FFT Analysis] Recorded:  max={:.4}, RMS={:.4}",
            rec_max,
            rec_rms
        );
        log::debug!(
            "[FFT Analysis] First 5 reference samples: {:?}",
            &reference[..5.min(reference.len())]
        );
        log::debug!(
            "[FFT Analysis] First 5 recorded samples:  {:?}",
            &recorded[..5.min(recorded.len())]
        );

        let check_len = reference.len().min(recorded.len());
        let mut identical_count = 0;
        for (r, c) in reference[..check_len]
            .iter()
            .zip(recorded[..check_len].iter())
        {
            if (r - c).abs() < 1e-6 {
                identical_count += 1;
            }
        }
        log::debug!(
            "[FFT Analysis] Identical samples: {}/{} ({:.1}%)",
            identical_count,
            check_len,
            identical_count as f32 * 100.0 / check_len as f32
        );
    }

    // Estimate lag using cross-correlation
    let lag_estimate = estimate_lag_with_confidence(reference, recorded)?;
    let lag = lag_estimate.lag_samples;

    log::debug!(
        "[FFT Analysis] Estimated lag: {} samples ({:.2} ms)",
        lag,
        lag as f32 * 1000.0 / sample_rate as f32
    );

    // Time-align the signals before FFT
    let (aligned_ref, aligned_rec) = align_signals(lag, reference, recorded)?;

    log::debug!(
        "[FFT Analysis] Aligned lengths: ref={}, rec={} (tail included)",
        aligned_ref.len(),
        aligned_rec.len()
    );

    // Compute FFT size to include the longer of the two (usually rec with tail)
    let fft_size = next_power_of_two(aligned_ref.len().max(aligned_rec.len()));

    let ref_spectrum = compute_fft(aligned_ref, fft_size, WindowType::Tukey(0.1))?;
    let rec_spectrum = compute_fft(aligned_rec, fft_size, WindowType::Tukey(0.1))?;

    // Generate 2000 log-spaced frequency points between 20 Hz and 20 kHz
    let num_output_points = 2000;
    let log_start = 20.0_f32.ln();
    let log_end = 20000.0_f32.ln();

    let mut frequencies = Vec::with_capacity(num_output_points);
    let mut spl_db = Vec::with_capacity(num_output_points);
    let mut phase_deg = Vec::with_capacity(num_output_points);

    let freq_resolution = sample_rate as f32 / fft_size as f32;
    let num_bins = fft_size / 2; // Single-sided spectrum

    // Compute regularization threshold relative to the peak reference energy.
    // Bins where the reference has very little energy (e.g., disconnected speaker
    // with a misaligned sweep) produce unreliable transfer functions — division by
    // near-zero gives spurious high-dB peaks. We skip bins where the reference
    // energy is more than 60 dB below the peak.
    let ref_peak_mag_sq = ref_spectrum[..num_bins.min(ref_spectrum.len())]
        .iter()
        .map(|c| c.norm_sqr())
        .fold(0.0_f32, |a, b| a.max(b));
    // 60 dB below peak = 10^(-6) in power
    let ref_regularization_threshold = (ref_peak_mag_sq * 1e-6).max(f32::MIN_POSITIVE);

    // Apply 1/24 octave smoothing for each target frequency
    let mut skipped_count = 0;
    // Loop-invariant 1/24 octave bandwidth factors: f * 2^(+/- 1/48).
    let octave_fraction = 1.0 / 48.0;
    let octave_factor_lower = 2.0_f32.powf(-octave_fraction);
    let octave_factor_upper = 2.0_f32.powf(octave_fraction);
    for i in 0..num_output_points {
        // Log-spaced target frequency
        let target_freq =
            (log_start + (log_end - log_start) * i as f32 / (num_output_points - 1) as f32).exp();

        // 1/24 octave bandwidth: +/- 1/48 octave around target frequency
        // Lower and upper frequency bounds: f * 2^(+/- 1/48)
        let freq_lower = target_freq * octave_factor_lower;
        let freq_upper = target_freq * octave_factor_upper;

        // Find FFT bins within this frequency range
        let bin_lower = ((freq_lower / freq_resolution).floor() as usize).max(1);
        let bin_upper = ((freq_upper / freq_resolution).ceil() as usize).min(num_bins);

        if bin_lower > bin_upper || bin_upper >= ref_spectrum.len() {
            if skipped_count < 5 {
                log::debug!(
                    "[FFT Analysis] Skipping freq {:.1} Hz: bin_lower={}, bin_upper={}, ref_spectrum.len()={}",
                    target_freq,
                    bin_lower,
                    bin_upper,
                    ref_spectrum.len()
                );
            }
            skipped_count += 1;
            // Output noise-floor placeholder so all channels produce the same
            // number of frequency points (prevents ndarray shape mismatches).
            frequencies.push(target_freq);
            spl_db.push(-200.0);
            phase_deg.push(0.0);
            continue;
        }

        // Average transfer function magnitude and phase across bins in the smoothing range
        let mut sum_magnitude = 0.0;
        let mut sum_sin = 0.0; // For circular averaging of phase
        let mut sum_cos = 0.0;
        let mut bin_count = 0;

        for k in bin_lower..=bin_upper {
            if k >= ref_spectrum.len() {
                break;
            }

            // Compute transfer function: H(f) = recorded / reference
            // This gives the system response (for loopback, should be ~1.0 or 0 dB)
            // Skip bins where the reference energy is too low (>60 dB below peak):
            // dividing by near-zero produces unreliable, spuriously high values
            // (e.g., disconnected speaker where the recording is just noise).
            let ref_mag_sq = ref_spectrum[k].norm_sqr();
            if ref_mag_sq <= ref_regularization_threshold {
                continue;
            }
            let transfer_function = rec_spectrum[k] / ref_spectrum[k];
            let magnitude = transfer_function.norm();

            // Phase from cross-spectrum (signals are already time-aligned)
            let cross_spectrum = ref_spectrum[k].conj() * rec_spectrum[k];
            let phase_rad = cross_spectrum.arg();

            // Accumulate for averaging
            sum_magnitude += magnitude;
            sum_sin += phase_rad.sin();
            sum_cos += phase_rad.cos();
            bin_count += 1;
        }

        // When no valid bins contribute (reference energy too low at this frequency,
        // e.g., LFE sweep above 500 Hz), output a noise-floor value instead of skipping.
        // Skipping would produce fewer output points than other channels, causing
        // ndarray shape mismatches when curves are combined downstream.
        let (avg_magnitude, db) = if bin_count == 0 {
            (0.0, -200.0)
        } else {
            let avg = sum_magnitude / bin_count as f32;
            (avg, 20.0 * avg.max(1e-10).log10())
        };

        if frequencies.len() < 5 {
            log::debug!(
                "[FFT Analysis] freq={:.1} Hz: avg_magnitude={:.6}, dB={:.2}",
                target_freq,
                avg_magnitude,
                db
            );
        }

        // Average phase using circular mean
        let avg_phase_rad = sum_sin.atan2(sum_cos);
        let phase = avg_phase_rad * 180.0 / PI;

        frequencies.push(target_freq);
        spl_db.push(db);
        phase_deg.push(phase);
    }

    log::debug!(
        "[FFT Analysis] Generated {} frequency points for CSV output",
        frequencies.len()
    );
    log::debug!(
        "[FFT Analysis] Skipped {} frequency points (out of {})",
        skipped_count,
        num_output_points
    );

    if log::log_enabled!(log::Level::Debug) && !spl_db.is_empty() {
        let min_spl = spl_db.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_spl = spl_db.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        log::debug!(
            "[FFT Analysis] SPL range: {:.2} dB to {:.2} dB",
            min_spl,
            max_spl
        );
    }

    // --- Compute Impulse Response ---
    // H(f) = Recorded(f) / Reference(f)
    let mut transfer_function = vec![Complex::new(0.0, 0.0); fft_size];
    for k in 0..fft_size {
        // Use the same relative Tikhonov regularisation as the smoothed FR
        // path. An absolute floor lets out-of-band reference noise dominate
        // the IR, RT60, and THD calculations.
        let reference_bin = ref_spectrum[k];
        let ref_mag_sq = reference_bin.norm_sqr();
        transfer_function[k] =
            rec_spectrum[k] * reference_bin.conj() / (ref_mag_sq + ref_regularization_threshold);
    }

    // IFFT to get Impulse Response
    let ifft = plan_fft_inverse(fft_size);
    ifft.process(&mut transfer_function);

    // Normalize and take real part (input was real, so output should be real-ish)
    // Scale by 1.0/N is done by IFFT? rustfft typically does NOT scale.
    // Standard IFFT definition: sum(X[k] * exp(...)) / N?
    // RustFFT inverse is unnormalized sum. So we divide by N.
    let norm = 1.0 / fft_size as f32;
    let mut impulse_response: Vec<f32> = transfer_function.iter().map(|c| c.re * norm).collect();

    // Find the peak and shift the IR so the peak is near the beginning
    // This is necessary because the IFFT result has the peak at an arbitrary position
    // due to the phase of the transfer function (system latency)
    let peak_idx = impulse_response
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Shift the IR so peak is at a small offset (e.g., 5ms for pre-ringing visibility)
    let pre_ring_samples = (0.005 * sample_rate as f32) as usize; // 5ms pre-ring buffer
    let shift_amount = peak_idx.saturating_sub(pre_ring_samples);

    if shift_amount > 0 {
        impulse_response.rotate_left(shift_amount);
        log::info!(
            "[FFT Analysis] IR peak was at index {}, shifted by {} samples to put peak near beginning",
            peak_idx,
            shift_amount
        );
    }

    // Generate time vector for IR (0 to duration)
    let _ir_duration_sec = fft_size as f32 / sample_rate as f32;
    let impulse_time_ms: Vec<f32> = (0..fft_size)
        .map(|i| i as f32 / sample_rate as f32 * 1000.0)
        .collect();

    // --- THD is only computed by the canonical ESS path ---
    // `sweep_range = Some(..)` early-returns through `analyze_ess_recording`
    // above, so this legacy transfer-function path always runs with
    // `sweep_range = None` and reports zero THD: without sweep timing there
    // is no Farina reference for separating harmonics in the IR.
    let thd_percent = vec![0.0; frequencies.len()];
    let harmonic_distortion_db = Vec::new();

    // --- Compute Excess Group Delay ---
    let excess_group_delay_ms = compute_group_delay(&frequencies, &phase_deg);

    // --- Compute Acoustic Metrics ---
    // Debug: Log impulse response stats
    let ir_max = impulse_response.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let ir_len = impulse_response.len();
    log::info!(
        "[Analysis] Impulse response: len={}, max_abs={:.6}, sample_rate={}",
        ir_len,
        ir_max,
        sample_rate
    );

    let (rt60_ms, clarity_c50_db, clarity_c80_db) =
        compute_rt60_clarity_spectra(&impulse_response, sample_rate as f32, &frequencies);

    // Debug: Log computed metrics
    if !rt60_ms.is_empty() {
        let rt60_min = rt60_ms.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let rt60_max = rt60_ms.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        log::info!(
            "[Analysis] RT60 range: {:.1} - {:.1} ms",
            rt60_min,
            rt60_max
        );
    }
    if !clarity_c50_db.is_empty() {
        let c50_min = clarity_c50_db.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let c50_max = clarity_c50_db
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        log::info!(
            "[Analysis] Clarity C50 range: {:.1} - {:.1} dB",
            c50_min,
            c50_max
        );
    }

    // Compute Spectrogram
    let (spectrogram_db, _, _) =
        compute_spectrogram(&impulse_response, sample_rate as f32, 512, 128);

    Ok(AnalysisResult {
        frequencies,
        spl_db,
        phase_deg,
        estimated_lag_samples: lag,
        impulse_response,
        impulse_time_ms,
        excess_group_delay_ms,
        thd_percent,
        harmonic_distortion_db,
        rt60_ms,
        clarity_c50_db,
        clarity_c80_db,
        spectrogram_db,
    })
}

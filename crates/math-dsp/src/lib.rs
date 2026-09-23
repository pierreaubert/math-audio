//! DSP utilities for audio signal processing
//!
//! This crate provides:
//! - **Signal generation**: Test signals (tones, sweeps, noise)
//! - **Signal analysis**: FFT-based frequency analysis, microphone compensation
//! - **Acoustic metrics**: RT60, clarity (C50/C80), THD, spectrogram
//!
//! # Example
//!
//! ```rust
//! use math_audio_dsp::{signals, analysis};
//!
//! // Generate a 1 kHz tone
//! let signal = signals::gen_tone(1000.0, 0.5, 48000, 1.0);
//!
//! // Analyze a WAV file
//! let config = analysis::WavAnalysisConfig::default();
//! // let result = analysis::analyze_wav_buffer(&signal, 48000, &config);
//! ```

pub mod analysis;
pub mod audio_features;
pub mod binaural_loudness;
pub mod binaural_matrix;
pub mod capture_array;
pub mod capture_resample;
pub mod capture_tdoa;
pub mod ebur128;
pub mod esprit;
pub mod fast_math;
pub mod fdn;
pub mod fdw;
pub mod instantaneous_frequency;
pub mod psychoacoustics;
pub mod replaygain;
pub mod response;
pub mod rir_early_late;
pub mod rir_waterfall;
pub mod rir_wavelet;
pub mod rtpghi;
pub mod signals;
pub mod simd;
pub mod stft;
pub mod tonal_transient;
pub mod waveform;

// DSP building blocks (moved from sotf-host)
pub mod adaa;
pub mod auto_makeup;
pub mod channel_linking;
pub mod dc_blocker;
pub mod delta_monitor;
pub mod detector;
pub mod dynamics_core;
pub mod envelope;
pub mod envelope_follower;
pub mod lookahead;
pub mod smoothing;
pub mod true_peak;

// Re-export commonly used types
pub use analysis::{
    AnalysisResult, AveragedEssResponse, AveragedResponse, ClockDriftEstimate,
    CompensationOutOfRange, CrossCorrelationEnvelopeResult, EssAnalysisResult, LagEstimate,
    MeasurementQualityConfig, MeasurementQualityReport, MicrophoneCompensation, WavAnalysisConfig,
    WavAnalysisOutput, WindowedFrequencyResponse, analyze_ess_recording,
    analyze_log_sweep_recording, analyze_recording, analyze_wav_buffer, analyze_wav_file,
    assess_measurement_quality, assess_measurement_quality_from_silence, average_complex_responses,
    average_deconvolved_sweeps, average_ess_recordings, compute_average_response,
    compute_clarity_spectrum, compute_group_delay, compute_h1_transfer_response,
    compute_impulse_response_from_fr, compute_rt60_broadband, compute_rt60_spectrum,
    compute_spectrogram, compute_windowed_fr, correct_clock_drift, cross_correlate_envelope,
    deconvolve_mls, deconvolve_mls_to_ir, deconvolve_sweep, detect_clipping,
    effective_sweep_duration_seconds, estimate_clock_drift, estimate_lag_with_confidence,
    extract_log_sweep_harmonic_impulse_responses, find_db_point, fit_linear_phase_delay_seconds,
    read_analysis_csv, smooth_response_f32, smooth_response_f64, write_analysis_csv,
    write_wav_analysis_csv,
};

pub use fdw::{FdwAnalysis, FdwConfig, analyze_impulse_response_fdw};

pub use binaural_loudness::{
    BinauralChannel, BinauralDownmix, BinauralLoudness, BinauralLoudnessResult, SurroundLayout,
    measure_binaural, measure_binaural_from_surround,
};

pub use binaural_matrix::{
    MatrixInverseBin, TransferMatrixBin, align_ir_to_reference_peak, condition_number,
    deconvolve_sweep_to_ir, direct_peak_sample, direct_peak_windowed_half_spectrum,
    direct_windowed_half_spectrum, fdw_complex_half_spectrum, half_spectrum_to_fir,
    position_errors, solve_minimax_regularized_inverse_bin, solve_regularized_inverse_bin,
    solve_weighted_regularized_inverse_bin, suppress_log_sweep_harmonic_residues,
};

pub use signals::{
    add_silence_padding, apply_fade_in, apply_fade_out, clip, frames_for, gen_allpass_probe,
    gen_dirac, gen_log_sweep, gen_m_noise, gen_m_noise_seeded, gen_mls, gen_narrowband_probe,
    gen_pink_noise, gen_pink_noise_seeded, gen_tone, gen_two_tone, gen_white_noise,
    gen_white_noise_seeded, interleave_per_channel, mono_to_stereo, prepare_signal_for_playback,
    prepare_signal_for_playback_channels, replicate_mono, try_gen_log_sweep,
};

pub use capture_array::{
    DEFAULT_DOA_BAND_HI_HZ, DEFAULT_DOA_BAND_LO_HZ, DoaEstimate, MicArray, PairDelay,
    SPEED_OF_SOUND_M_S, calibrate_geometry_scale, check_geometry, delay_and_sum_power,
    estimate_doa_ls, pairwise_tdoas_vs_first, tdoa_residuals,
};
pub use capture_resample::{
    RESAMPLE_KAISER_BETA, RESAMPLE_PASSBAND_HZ, RESAMPLE_PHASES, RESAMPLE_TAPS,
    resample_to_common_clock,
};
pub use capture_tdoa::{
    ClockSkew, MAX_PLAUSIBLE_SKEW_PPM, MIN_TDOA_CONFIDENCE_DB, TIMING_CHIRP_HI_HZ,
    TIMING_CHIRP_LO_HZ, TdoaConfig, TdoaEstimate, TdoaWeighting, estimate_chirp_tdoa,
    estimate_clock_skew, post_correction_uncertainty_us, validate_drift_series,
};
pub use replaygain::{
    ReplayGainAnalyzer, ReplayGainInfo, ReplayGainTrackData, compute_album_gain,
    compute_album_gain_pooled,
};
pub use response::{biquad_complex_response, fir_complex_response, lr4_crossover_response};
pub use waveform::{WAVEFORM_SAMPLES, compute_waveform};

use thiserror::Error;

/// Errors that can occur during autodiff module operations.
#[derive(Debug, Error)]
pub enum AutodiffError {
    /// An error from the underlying real FFT implementation.
    #[error("FFT error: {0}")]
    Fft(#[from] realfft::FftError),

    /// A generic error with a descriptive message.
    #[error("{0}")]
    Message(String),
}

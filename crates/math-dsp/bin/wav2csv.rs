use clap::Parser;
use math_audio_dsp::analysis::{WavAnalysisConfig, analyze_wav_file, write_wav_analysis_csv};
use std::fs;
use std::path::PathBuf;

/// Convert WAV file to frequency/SPL/phase CSV
#[derive(Parser)]
#[command(name = "wav2csv")]
#[command(about = "Analyze WAV file and output frequency/SPL/phase CSV")]
#[command(
    long_about = "Analyze WAV files and output frequency response as CSV.\n\n\
For stationary signals (music, noise): Use default Welch's method\n\
For log sweeps: Use --single-fft --pink-compensation --no-window\n\
For impulse responses: Use --single-fft"
)]
struct Cli {
    /// Input WAV file or directory containing WAV files
    input: PathBuf,

    /// Output CSV file, or output directory when INPUT is a directory
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Number of frequency points (default: 2000)
    #[arg(short, long, default_value = "2000")]
    pub num_points: usize,

    /// Minimum frequency in Hz (default: 20)
    #[arg(long, default_value = "20.0")]
    min_freq: f32,

    /// Maximum frequency in Hz (default: 20000)
    #[arg(long, default_value = "20000.0")]
    max_freq: f32,

    /// FFT size (default: 16384)
    #[arg(long)]
    fft_size: Option<usize>,

    /// Window overlap ratio (0.0-1.0, default: 0.5)
    #[arg(long, default_value = "0.5")]
    overlap: f32,

    /// Use single FFT instead of Welch's method (better for sweeps and impulse responses)
    #[arg(long)]
    single_fft: bool,

    /// Apply pink compensation (-3dB/octave) for log sweeps
    #[arg(long)]
    pink_compensation: bool,

    /// Use rectangular window (no windowing) instead of Hann
    #[arg(long)]
    no_window: bool,
}

fn main() {
    let cli = Cli::parse();

    if let Err(e) = run(cli) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), String> {
    // Build configuration from CLI arguments
    let config = WavAnalysisConfig {
        num_points: cli.num_points,
        min_freq: cli.min_freq,
        max_freq: cli.max_freq,
        fft_size: cli.fft_size,
        overlap: cli.overlap,
        single_fft: cli.single_fft,
        pink_compensation: cli.pink_compensation,
        no_window: cli.no_window,
    };

    let input_metadata = fs::metadata(&cli.input)
        .map_err(|e| format!("Failed to inspect input {:?}: {}", cli.input, e))?;

    if input_metadata.is_dir() {
        return run_directory(&cli.input, cli.output.as_deref(), &config);
    }

    run_file(&cli.input, cli.output.as_deref(), &config)
}

fn run_file(
    input: &std::path::Path,
    output: Option<&std::path::Path>,
    config: &WavAnalysisConfig,
) -> Result<(), String> {
    println!("Loading WAV file: {:?}", input);

    let result = analyze_wav_file(input, config)?;

    println!(
        "Analyzed {} frequency points from {:.1} Hz to {:.1} Hz",
        result.frequencies.len(),
        result.frequencies.first().unwrap_or(&0.0),
        result.frequencies.last().unwrap_or(&0.0)
    );

    // Determine output path
    let output_path = output.map(PathBuf::from).unwrap_or_else(|| {
        let mut path = input.to_owned();
        path.set_extension("csv");
        path
    });

    // Write CSV
    println!("Writing CSV to: {:?}", output_path);
    write_wav_analysis_csv(&result, &output_path)?;

    println!("Done!");
    Ok(())
}

fn run_directory(
    input_dir: &std::path::Path,
    output: Option<&std::path::Path>,
    config: &WavAnalysisConfig,
) -> Result<(), String> {
    let mut inputs = fs::read_dir(input_dir)
        .map_err(|e| format!("Failed to read input directory {:?}: {}", input_dir, e))?
        .map(|entry| {
            entry
                .map(|entry| entry.path())
                .map_err(|e| format!("Failed to read directory entry: {}", e))
        })
        .collect::<Result<Vec<_>, _>>()?;

    inputs.retain(|path| {
        path.is_file()
            && path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
    });
    inputs.sort();

    if inputs.is_empty() {
        return Err(format!("No WAV files found in {:?}", input_dir));
    }

    let output_dir = output.unwrap_or(input_dir);
    if output_dir.exists() && !output_dir.is_dir() {
        return Err(format!(
            "Output path must be a directory when input is a directory: {:?}",
            output_dir
        ));
    }
    if !output_dir.exists() {
        fs::create_dir_all(output_dir)
            .map_err(|e| format!("Failed to create output directory {:?}: {}", output_dir, e))?;
    }

    println!(
        "Converting {} WAV files from {:?} to {:?}",
        inputs.len(),
        input_dir,
        output_dir
    );

    for input in &inputs {
        let file_stem = input
            .file_stem()
            .ok_or_else(|| format!("Cannot determine filename for {:?}", input))?;
        let output_path = output_dir.join(file_stem).with_extension("csv");
        run_file(input, Some(&output_path), config)?;
    }

    println!("Converted {} WAV files.", inputs.len());
    Ok(())
}

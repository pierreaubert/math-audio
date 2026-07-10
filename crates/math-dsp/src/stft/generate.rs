/// Generate a Hann window of the given size.
/// Uses N (not N-1) divisor for perfect COLA with 50% overlap.
pub fn generate_hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - ((2.0 * std::f32::consts::PI * i as f32) / size as f32).cos()))
        .collect()
}

/// Generate a symmetric Hann window of the given size.
/// Uses N-1 divisor — suitable for spectral analysis (zero at endpoints).
pub fn generate_hann_window_symmetric(size: usize) -> Vec<f32> {
    if size <= 1 {
        return vec![1.0; size];
    }
    let n_minus_1 = (size as f32) - 1.0;
    (0..size)
        .map(|i| 0.5 * (1.0 - ((2.0 * std::f32::consts::PI * i as f32) / n_minus_1).cos()))
        .collect()
}

/// Generate a sqrt(Hann) window for WOLA (Weighted Overlap-Add) processing.
/// When used as both analysis and synthesis window, the product is Hann,
/// which has perfect COLA at 50% overlap.
pub fn generate_sqrt_hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let hann = 0.5 * (1.0 - ((2.0 * std::f32::consts::PI * i as f32) / size as f32).cos());
            hann.sqrt()
        })
        .collect()
}

/// Design a dual-window pair satisfying the COLA (Constant Overlap-Add) condition.
///
/// # Arguments
/// * `analysis_size` - Analysis window length (long, e.g. 1024)
/// * `synthesis_size` - Synthesis window length (short, e.g. 256)
/// * `hop_size` - Hop size in samples
///
/// # Returns
/// (analysis_window, synthesis_window) pair
pub fn design_dual_windows(
    analysis_size: usize,
    synthesis_size: usize,
    hop_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert!(analysis_size > 0, "analysis_size must be positive");
    assert!(synthesis_size > 1, "synthesis_size must be at least two");
    assert!(
        synthesis_size <= analysis_size,
        "synthesis_size must not exceed analysis_size"
    );
    assert!(hop_size > 0, "hop_size must be positive");
    // Analysis window: Hann
    let w_a = generate_hann_window(analysis_size);

    // Synthesis window: truncated Hann centered in the analysis window,
    // normalized to satisfy COLA
    let offset = (analysis_size - synthesis_size) / 2;

    // Start with a Hann window of synthesis_size
    let w_s_raw = generate_hann_window(synthesis_size);

    // Compute the COLA sum: Σ_k w_a(n - k*hop) * w_s(n - k*hop)
    // across all hop-shifted positions. We need this to be constant.
    // Normalize w_s so the sum equals 1.
    let mut cola_sum = vec![0.0f32; hop_size];
    for (syn_idx, &synthesis_value) in w_s_raw.iter().enumerate() {
        let ana_idx = offset + syn_idx;
        cola_sum[ana_idx % hop_size] += w_a[ana_idx] * synthesis_value;
    }

    assert!(
        cola_sum.iter().all(|&sum| sum > 1e-10),
        "hop_size leaves at least one COLA phase uncovered"
    );

    let mut w_s = vec![0.0f32; analysis_size];
    for (i, &synthesis_value) in w_s_raw.iter().enumerate() {
        let ana_idx = offset + i;
        // Each hop residue is an independent pointwise COLA equation. Scaling
        // all synthesis samples in that residue by its own overlap sum makes
        // every equation equal one, rather than only normalizing their mean.
        w_s[ana_idx] = synthesis_value / cola_sum[ana_idx % hop_size];
    }

    (w_a, w_s)
}

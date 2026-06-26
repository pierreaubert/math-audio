use super::misc::estimate_noise_cutoff;
use super::misc::linear_fit_indexed;
use crate::config::SsirConfig;
use crate::detection::find_direct_sound_toa;

/// Reusable scratch buffers for Schroeder decay-curve construction.
///
/// [`DecayCurve::from_rir_with_workspace`] writes the intermediate energy
/// integrals into [`Self::energy`] and then moves [`Self::samples`] into the
/// returned curve, leaving the caller's workspace ready for the next RIR.
#[derive(Debug, Default)]
pub struct DecayWorkspace {
    energy: Vec<f64>,
    samples: Vec<f64>,
}

impl DecayWorkspace {
    /// Create a new, empty workspace.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Schroeder backward-integrated decay curve of a RIR, expressed in dB
/// relative to the curve's peak.
///
/// `samples[n] = 10·log10( ∫_n^{cutoff} h²(τ) dτ / ∫_0^{cutoff} h²(τ) dτ )`
///
/// Truncation at `cutoff` (the estimated noise-floor crossover) avoids the
/// "lift" that a never-decaying integrated noise tail would otherwise add
/// to the curve — see Chu (1978) and Lundeby et al. (1995).
#[derive(Debug, Clone)]
pub struct DecayCurve {
    /// Sample-by-sample Schroeder decay in dB (0 dB at the start, decreasing).
    pub samples: Vec<f64>,
    /// Sample rate the curve was computed at.
    pub sample_rate: f64,
    /// Index (within `samples`) at which the underlying RIR was truncated
    /// before backward-integration. Below this sample the curve is dominated
    /// by noise and should not be used for slope fitting.
    pub noise_cutoff_sample: usize,
}

impl DecayCurve {
    /// Compute the Schroeder decay curve from a RIR while reusing scratch
    /// buffers in `ws`.
    ///
    /// `start_sample` is the index of the direct sound; integration starts
    /// from there. `noise_cutoff_sample` (absolute index in `rir`) lets the
    /// caller override the auto-detected noise truncation; if `None`,
    /// [`estimate_noise_cutoff`] is used.
    ///
    /// The `ws.samples` buffer is moved into the returned [`DecayCurve`];
    /// `ws.energy` retains its capacity for the next call.
    pub fn from_rir_with_workspace(
        rir: &[f32],
        sample_rate: f64,
        start_sample: usize,
        noise_cutoff_sample: Option<usize>,
        ws: &mut DecayWorkspace,
    ) -> Self {
        if rir.is_empty() || start_sample >= rir.len() {
            return Self {
                samples: Vec::new(),
                sample_rate,
                noise_cutoff_sample: 0,
            };
        }

        let cutoff_abs =
            noise_cutoff_sample.unwrap_or_else(|| estimate_noise_cutoff(rir, start_sample));
        let cutoff_abs = cutoff_abs.min(rir.len());
        let cutoff_rel = cutoff_abs.saturating_sub(start_sample);

        // Square h(n) and backward-integrate from `cutoff_abs` down to
        // `start_sample`. We work in f64 throughout — for a 200 ms IR at
        // 48 kHz this is < 10k accumulations; precision matters at the
        // −35 dB tail.
        let n = cutoff_rel;
        if n == 0 {
            return Self {
                samples: Vec::new(),
                sample_rate,
                noise_cutoff_sample: 0,
            };
        }

        ws.energy.resize(n, 0.0);
        ws.samples.resize(n, 0.0);

        let mut acc = 0.0_f64;
        // Walk from end → start, summing h². After this loop `ws.energy[i]`
        // = ∫_{start_sample+i}^{cutoff_abs} h²(τ) dτ.
        for i in (0..n).rev() {
            let s = f64::from(rir[start_sample + i]);
            acc += s * s;
            ws.energy[i] = acc;
        }

        let total = ws.energy[0];
        if total <= 0.0 || !total.is_finite() {
            return Self {
                samples: Vec::new(),
                sample_rate,
                noise_cutoff_sample: cutoff_abs,
            };
        }

        // Convert to dB relative to the peak. The smallest representable
        // value past the tail still produces a finite dB (clamped at
        // −300 dB) so consumers don't have to handle `-∞`.
        let inv_total = 1.0 / total;
        for i in 0..n {
            let r = ws.energy[i] * inv_total;
            ws.samples[i] = if r <= 0.0 { -300.0 } else { 10.0 * r.log10() };
        }

        Self {
            samples: std::mem::take(&mut ws.samples),
            sample_rate,
            noise_cutoff_sample: cutoff_abs,
        }
    }

    /// Compute the Schroeder decay curve from a RIR.
    ///
    /// This is a convenience wrapper around
    /// [`Self::from_rir_with_workspace`] that allocates a temporary
    /// [`DecayWorkspace`].
    pub fn from_rir(
        rir: &[f32],
        sample_rate: f64,
        start_sample: usize,
        noise_cutoff_sample: Option<usize>,
    ) -> Self {
        let mut ws = DecayWorkspace::new();
        Self::from_rir_with_workspace(rir, sample_rate, start_sample, noise_cutoff_sample, &mut ws)
    }

    /// First sample index at which the curve is `≤ threshold_db`. Returns
    /// `None` if the curve never reaches the threshold.
    pub fn first_crossing(&self, threshold_db: f64) -> Option<usize> {
        self.samples.iter().position(|&v| v <= threshold_db)
    }

    /// Least-squares fit of the decay between two dB thresholds.
    ///
    /// Returns `(slope_db_per_s, intercept_db, r_squared)`. `None` if either
    /// threshold is never reached or fewer than two samples lie in the band.
    pub fn fit_db_range(&self, upper_db: f64, lower_db: f64) -> Option<(f64, f64, f64)> {
        debug_assert!(upper_db > lower_db);
        let i_upper = self.first_crossing(upper_db)?;
        let i_lower = self.first_crossing(lower_db)?;
        if i_lower <= i_upper + 1 {
            return None;
        }

        // x is time in seconds (sample index / sample_rate); y is decay dB.
        let dt = 1.0 / self.sample_rate;
        let ys = &self.samples[i_upper..=i_lower];
        linear_fit_indexed(ys, i_upper, dt)
    }
}

/// Compute a direct-sound-anchored Schroeder decay curve for a RIR.
///
/// This is a convenience wrapper around [`DecayCurve::from_rir`]. It detects
/// the direct sound with the SSIR detector, falls back to sample 0 when no
/// direct sound can be identified, and uses the automatic noise-tail cutoff.
pub fn schroeder_curve(rir: &[f32], sample_rate: f64) -> DecayCurve {
    if rir.is_empty() || sample_rate <= 0.0 {
        return DecayCurve {
            samples: Vec::new(),
            sample_rate,
            noise_cutoff_sample: 0,
        };
    }

    let cfg = SsirConfig::new(sample_rate);
    let start = find_direct_sound_toa(rir, &cfg).unwrap_or(0);
    DecayCurve::from_rir(rir, sample_rate, start, None)
}

/// Compute ISO 3382 single-band metrics on a broadband RIR.
///
/// This is a convenience wrapper around
/// [`analyze_iso3382_with_workspace`](crate::metrics::analyze_iso3382_with_workspace)
/// that allocates a temporary [`BandAnalysisContext`](crate::bands::BandAnalysisContext).
/// For per-band analysis, bandpass the RIR with one of the helpers in
/// [`crate::bands`] and call this function on the filtered signal.
pub fn analyze_iso3382(rir: &[f32], sample_rate: f64) -> super::Iso3382Metrics {
    let mut ctx = crate::bands::BandAnalysisContext::new();
    super::analyze_iso3382_with_workspace(rir, sample_rate, &mut ctx)
}

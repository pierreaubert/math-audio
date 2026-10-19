use super::misc::estimate_noise_cutoff;
use super::misc::linear_fit_indexed;
use crate::config::SsirConfig;
use crate::detection::find_direct_sound_toa;

/// Reusable scratch buffers for Schroeder decay-curve construction.
///
/// [`Self::compute`] retains both intermediate integrals and curve samples.
/// Its borrowed result avoids allocating an owned output for each band.
#[derive(Debug, Default)]
pub struct DecayWorkspace {
    energy: Vec<f64>,
    curve: DecayCurve,
}

impl DecayWorkspace {
    /// Create a new, empty workspace.
    pub fn new() -> Self {
        Self::default()
    }
    /// Build a borrowed curve without relinquishing reusable output storage.
    ///
    /// Equal or shorter records allocate nothing after the first call.
    /// Empty, silent, invalid-rate, or non-finite-energy input yields an empty curve.
    pub fn compute(
        &mut self,
        rir: &[f32],
        sample_rate: f64,
        start_sample: usize,
        noise_cutoff_sample: Option<usize>,
    ) -> &DecayCurve {
        self.curve.samples.clear();
        self.curve.sample_rate = sample_rate;
        self.curve.noise_cutoff_sample = 0;
        if rir.is_empty()
            || start_sample >= rir.len()
            || !sample_rate.is_finite()
            || sample_rate <= 0.0
        {
            return &self.curve;
        }
        let cutoff = noise_cutoff_sample
            .unwrap_or_else(|| estimate_noise_cutoff(rir, start_sample))
            .min(rir.len());
        self.curve.noise_cutoff_sample = cutoff;
        let count = cutoff.saturating_sub(start_sample);
        if count == 0 {
            return &self.curve;
        }
        self.energy.resize(count, 0.0);
        let mut total = 0.0_f64;
        for i in (0..count).rev() {
            total += f64::from(rir[start_sample + i]).powi(2);
            self.energy[i] = total;
        }
        if total <= 0.0 || !total.is_finite() {
            return &self.curve;
        }
        self.curve.samples.resize(count, 0.0);
        for (output, &energy) in self.curve.samples.iter_mut().zip(&self.energy) {
            let fraction = energy / total;
            *output = if fraction <= 0.0 {
                -300.0
            } else {
                10.0 * fraction.log10()
            };
        }
        &self.curve
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
#[derive(Debug, Clone, Default)]
pub struct DecayCurve {
    /// Sample-by-sample Schroeder decay in dB (0 dB at the start, decreasing).
    pub samples: Vec<f64>,
    /// Sample rate the curve was computed at.
    pub sample_rate: f64,
    /// Absolute input-RIR index at which the underlying signal was truncated
    /// before backward-integration. Below this sample the curve is dominated
    /// by noise and should not be used for slope fitting.
    pub noise_cutoff_sample: usize,
}

impl DecayCurve {
    /// Compute an owned decay curve while reusing intermediate workspace.
    ///
    /// `start_sample` anchors the direct arrival. An optional absolute cutoff
    /// overrides automatic tail truncation. The returned curve takes the output
    /// storage; [`DecayWorkspace::compute`] instead retains and lends that storage.
    pub fn from_rir_with_workspace(
        rir: &[f32],
        sample_rate: f64,
        start_sample: usize,
        noise_cutoff_sample: Option<usize>,
        ws: &mut DecayWorkspace,
    ) -> Self {
        ws.compute(rir, sample_rate, start_sample, noise_cutoff_sample);
        std::mem::take(&mut ws.curve)
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
        ws.compute(rir, sample_rate, start_sample, noise_cutoff_sample);
        ws.curve
    }

    /// First sample index at which the curve is `≤ threshold_db`. Returns
    /// `None` if the curve never reaches the threshold.
    pub fn first_crossing(&self, threshold_db: f64) -> Option<usize> {
        self.samples.iter().position(|&v| v <= threshold_db)
    }

    /// Least-squares fit of the decay between two dB thresholds.
    ///
    /// Returns `(slope_db_per_s, intercept_db, r_squared)`. `None` if either
    /// threshold is never reached, fewer than two samples lie in the band,
    /// or the truncated tail lacks 10 dB of energy headroom below the fit.
    /// A high fit r² alone does not establish sufficient decay dynamic range.
    pub fn fit_db_range(&self, upper_db: f64, lower_db: f64) -> Option<(f64, f64, f64)> {
        if !upper_db.is_finite()
            || !lower_db.is_finite()
            || upper_db <= lower_db
            || !self.sample_rate.is_finite()
            || self.sample_rate <= 0.0
        {
            return None;
        }
        let i_upper = self.first_crossing(upper_db)?;
        let i_lower = self.first_crossing(lower_db)?;
        if i_lower <= i_upper + 1 {
            return None;
        }

        // x is time in seconds (sample index / sample_rate); y is decay dB.
        let dt = 1.0 / self.sample_rate;
        let ys = &self.samples[i_upper..=i_lower];
        let fit = linear_fit_indexed(ys, i_upper, dt)?;
        if fit.0 >= 0.0 {
            return None;
        }

        // Estimate the omitted energy using the terminal mean-square power
        // and the provisional decay's energy time constant. The Schroeder
        // ordinate at n-k is exactly the last k samples' normalized energy.
        // A physical 5 ms averaging interval avoids sample-rate dependence.
        let tail_len = ((0.005 * self.sample_rate).ceil() as usize).clamp(1, self.samples.len());
        let tail_energy_fraction =
            10.0_f64.powf(self.samples[self.samples.len() - tail_len] / 10.0);
        let energy_decay_rate = -fit.0 * std::f64::consts::LN_10 / 10.0;
        let omitted_fraction =
            tail_energy_fraction * self.sample_rate / (tail_len as f64 * energy_decay_rate);
        // At least 10 dB headroom limits the estimated unobserved tail to
        // one tenth of the energy at the lowest fitting ordinate. Reject
        // insufficient records rather than treating their forced final drop
        // as an observed decay. This is a quality gate, not tail compensation.
        if !omitted_fraction.is_finite() || 10.0_f64.powf(lower_db / 10.0) < 10.0 * omitted_fraction
        {
            return None;
        }
        Some(fit)
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

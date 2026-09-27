//! FMV/TMB tone-stack component model (model ID 8).
//!
//! The passive treble/mid/bass network of Yeh & Smith (DAFx-06 Fig. 1),
//! solved exactly: resistive nodal analysis plus a descriptor-to-explicit
//! state-space reduction over the three capacitor voltages, discretized
//! with the bilinear transform (exact causal input absorption). Coefficient
//! math runs in f64 at prepare/control time; the per-sample loop is a
//! 3-state f32 filter. See `references/component-references.md` for the
//! frozen topology, value sets, and pre-registered tolerances.
//!
//! Documented v1 choices: ideal source with unloaded output exactly as in
//! Yeh's SPICE-verified analysis; bass sweeps linearly (Yeh's log taper is
//! deferred); pot wiper segments clamp to a 1 ohm contact minimum.

use crate::chain::DcBlocker;
use crate::effects::{DefectConfig, DefectState};
use crate::level::db_to_gain;
use crate::process::{
    AnalogError, AnalogProcessor, ControlSmoother, ProcessSpec, checked_block_len, finite_output,
    flush_denormal, sanitize_sample, validate_finite_range,
};

const CONTROL_SMOOTHING_MS: f32 = 10.0;
const MAX_DEFECT_DELAY_SAMPLES: usize = 64;
/// Physical wiper/contact minimum for a pot segment, in ohms.
const WIPER_MINIMUM_OHMS: f64 = 1.0;
/// Rebuild threshold on smoothed knobs; static knobs never rebuild.
const REBUILD_EPSILON: f32 = 1e-6;
/// Reference frequency in Hz for the noon level normalization.
const REFERENCE_HZ: f64 = 1_000.0;
/// Knob triple used for the level reference (all knobs at noon).
const REFERENCE_KNOBS: (f64, f64, f64) = (0.5, 0.5, 0.5);

/// Published component value set for the TMB network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToneStackValues {
    /// '59 schematic values (the set Yeh SPICE-verified): 56k slope,
    /// 250 pF treble cap, 0.02 uF bass+mid caps.
    #[default]
    Schematic59,
    /// Production values (shipped units + reissues): 100k slope,
    /// 0.1 uF bass cap, otherwise schematic.
    Production,
}

impl ToneStackValues {
    /// `(slope_ohms, treble_f, bass_f, mid_f, treble_ohms, bass_ohms, mid_ohms)`.
    fn components(self) -> (f64, f64, f64, f64, f64, f64, f64) {
        match self {
            Self::Schematic59 => (56e3, 250e-12, 20e-9, 20e-9, 250e3, 1e6, 25e3),
            Self::Production => (100e3, 250e-12, 100e-9, 20e-9, 250e3, 1e6, 25e3),
        }
    }
}

/// Discrete 3-state filter for one knob setting (f32 realtime coefficients).
#[derive(Debug, Clone, Copy)]
struct DiscreteFilter {
    ad: [[f32; 3]; 3],
    bd: [f32; 3],
    cd: [f32; 3],
    dd: f32,
}

/// Invert a square matrix in place (Gauss-Jordan with partial pivot).
/// Returns `None` when a pivot is numerically zero (unreachable for the
/// connected TMB network, but the build path never panics).
// elimination subscripting needs row/column indices by construction.
#[allow(clippy::needless_range_loop)]
fn invert<const N: usize>(matrix: &mut [[f64; N]; N]) -> Option<[[f64; N]; N]> {
    let mut inverse = [[0.0; N]; N];
    for (index, row) in inverse.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    for column in 0..N {
        let mut pivot = column;
        for row in column + 1..N {
            if matrix[row][column].abs() > matrix[pivot][column].abs() {
                pivot = row;
            }
        }
        if matrix[pivot][column].abs() < 1e-300 {
            return None;
        }
        matrix.swap(column, pivot);
        inverse.swap(column, pivot);
        let divisor = matrix[column][column];
        for k in column..N {
            matrix[column][k] /= divisor;
        }
        for k in 0..N {
            inverse[column][k] /= divisor;
        }
        for row in 0..N {
            if row != column {
                let factor = matrix[row][column];
                for k in column..N {
                    let pivot_entry = matrix[column][k];
                    matrix[row][k] -= factor * pivot_entry;
                }
                for k in 0..N {
                    let pivot_entry = inverse[column][k];
                    inverse[row][k] -= factor * pivot_entry;
                }
            }
        }
    }
    Some(inverse)
}

/// Stamp a resistor between two nodes into a conductance matrix.
fn stamp(matrix: &mut [[f64; 6]; 6], i: usize, j: usize, ohms: f64) {
    let conductance = 1.0 / ohms.max(WIPER_MINIMUM_OHMS);
    matrix[i][i] += conductance;
    matrix[j][j] += conductance;
    matrix[i][j] -= conductance;
    matrix[j][i] -= conductance;
}

/// Stamp a resistor to ground into a conductance matrix.
fn shunt(matrix: &mut [[f64; 6]; 6], i: usize, ohms: f64) {
    matrix[i][i] += 1.0 / ohms.max(WIPER_MINIMUM_OHMS);
}

/// Build the discrete filter for one knob setting.
///
/// Nodes (unknowns): S slope, T treble-top, W wiper/output, X
/// treble-bottom, Y mid-top, Z mid-wiper. The input is an ideal 1 V source
/// (Yeh's loading-independent analysis); the output is unloaded.
// State-space assembly subscripting needs node/state indices by construction.
#[allow(clippy::needless_range_loop)]
fn build_filter(
    treble: f64,
    mid: f64,
    bass: f64,
    sample_rate: f64,
    values: ToneStackValues,
) -> Option<DiscreteFilter> {
    const S: usize = 0;
    const T: usize = 1;
    const W: usize = 2;
    const X: usize = 3;
    const Y: usize = 4;
    const Z: usize = 5;
    let (slope, c_treble, c_bass, c_mid, r_treble, r_bass, r_mid) = values.components();
    // 6x6 resistive conductance matrix + source vector (input = 1 V).
    let mut conductance = [[0.0; 6]; 6];
    let mut source = [0.0; 6];
    conductance[S][S] += 1.0 / slope;
    source[S] += 1.0 / slope;
    stamp(&mut conductance, T, W, r_treble * (1.0 - treble));
    stamp(&mut conductance, W, X, r_treble * treble);
    stamp(&mut conductance, X, Y, r_bass * bass);
    stamp(&mut conductance, Y, Z, r_mid * (1.0 - mid));
    shunt(&mut conductance, Z, r_mid * mid);
    // Descriptor reduction: E1 dx/dt + x = B1 u with x the three capacitor
    // voltages (C1: IN-T, C2: S-X, C3: S-Z; IN is the known source).
    let caps = [(T, None), (S, Some(X)), (S, Some(Z))];
    let mut inverse_g = conductance;
    let g_inv = invert(&mut inverse_g)?;
    // Selector (states from node voltages) and injection (cap currents).
    let mut selector = [[0.0; 6]; 3];
    let mut injection = [[0.0; 3]; 6];
    let mut source_gain = [0.0; 3];
    for (k, (i, j)) in caps.iter().enumerate() {
        match j {
            // Terminal pair among the unknowns: x = v[i] - v[j].
            Some(other) => {
                selector[k][*i] += 1.0;
                selector[k][*other] -= 1.0;
                injection[*i][k] += 1.0;
                injection[*other][k] -= 1.0;
            }
            // Terminal pair (known input, unknown): x = u - v[i].
            None => {
                selector[k][*i] -= 1.0;
                source_gain[k] += 1.0;
                injection[*i][k] -= 1.0;
            }
        }
    }
    // E1 = S G^-1 M C, B1 = S G^-1 b + s_u.
    let cap_values = [c_treble, c_bass, c_mid];
    let mut e1 = [[0.0; 3]; 3];
    let mut b1 = [0.0; 3];
    for k in 0..3 {
        // Row k of S G^-1.
        let mut row = [0.0; 6];
        for (n, entry) in row.iter_mut().enumerate() {
            let mut sum = 0.0;
            for m in 0..6 {
                sum += selector[k][m] * g_inv[m][n];
            }
            *entry = sum;
        }
        let mut gain = source_gain[k];
        for n in 0..6 {
            gain += row[n] * source[n];
        }
        b1[k] = gain;
        for j in 0..3 {
            let mut sum = 0.0;
            for n in 0..6 {
                sum += row[n] * injection[n][j];
            }
            e1[k][j] = sum * cap_values[j];
        }
    }
    let mut e1_work = e1;
    let e1_inv = invert(&mut e1_work)?;
    // Continuous state-space: A = -E1^-1, B = E1^-1 B1.
    let mut mat_a = [[0.0; 3]; 3];
    let mut vec_b = [0.0; 3];
    for i in 0..3 {
        for j in 0..3 {
            mat_a[i][j] = -e1_inv[i][j];
        }
        let mut sum = 0.0;
        for j in 0..3 {
            sum += e1_inv[i][j] * b1[j];
        }
        vec_b[i] = sum;
    }
    // Output row: Cc = w' G^-1 M C E1^-1, D = w' G^-1 b - Cc B1.
    let mut gov_row = [0.0; 6];
    gov_row.copy_from_slice(&g_inv[W]);
    let mut tmp = [0.0; 3];
    for j in 0..3 {
        let mut sum = 0.0;
        for n in 0..6 {
            sum += gov_row[n] * injection[n][j];
        }
        tmp[j] = sum * cap_values[j];
    }
    let mut vec_c = [0.0; 3];
    for i in 0..3 {
        let mut sum = 0.0;
        for j in 0..3 {
            sum += tmp[j] * e1_inv[j][i];
        }
        vec_c[i] = sum;
    }
    let mut feedthrough = 0.0;
    for n in 0..6 {
        feedthrough += gov_row[n] * source[n];
    }
    for j in 0..3 {
        feedthrough -= vec_c[j] * b1[j];
    }
    // Bilinear discretization with exact causal input absorption:
    // Ad = M (I + A T/2), Bt = M B T, Bd = (Ad + I)/2 Bt,
    // Dd = D + Cc Bt / 2, with M = (I - A T/2)^-1.
    let sample_period = 1.0 / sample_rate;
    let mut work = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let identity = f64::from(i == j);
            work[i][j] = identity - mat_a[i][j] * sample_period / 2.0;
        }
    }
    let m_mat = invert(&mut work)?;
    let mut mat_ad = [[0.0; 3]; 3];
    let mut vec_bt = [0.0; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                let identity = f64::from(k == j);
                sum += m_mat[i][k] * (identity + mat_a[k][j] * sample_period / 2.0);
            }
            mat_ad[i][j] = sum;
        }
        let mut sum = 0.0;
        for k in 0..3 {
            sum += m_mat[i][k] * vec_b[k];
        }
        vec_bt[i] = sum * sample_period;
    }
    let mut vec_bd = [0.0; 3];
    let mut scalar = 0.0;
    for i in 0..3 {
        let mut sum = 0.0;
        for k in 0..3 {
            let identity = f64::from(i == k);
            sum += (mat_ad[i][k] + identity) / 2.0 * vec_bt[k];
        }
        vec_bd[i] = sum;
        scalar += vec_c[i] * vec_bt[i];
    }
    let dd = feedthrough + scalar / 2.0;
    let mut filter = DiscreteFilter {
        ad: [[0.0; 3]; 3],
        bd: [0.0; 3],
        cd: [0.0; 3],
        dd: dd as f32,
    };
    for i in 0..3 {
        for j in 0..3 {
            filter.ad[i][j] = mat_ad[i][j] as f32;
        }
        filter.bd[i] = vec_bd[i] as f32;
        filter.cd[i] = vec_c[i] as f32;
    }
    if !filter.dd.is_finite()
        || filter.ad.iter().flatten().any(|v| !v.is_finite())
        || filter.bd.iter().any(|v| !v.is_finite())
        || filter.cd.iter().any(|v| !v.is_finite())
    {
        return None;
    }
    Some(filter)
}

/// Discrete frequency response at one frequency (f64, for the level
/// reference; 3x3 complex solve by Cramer's rule).
fn discrete_response(filter: &DiscreteFilter, frequency_hz: f64, sample_rate: f64) -> (f64, f64) {
    let theta = 2.0 * std::f64::consts::PI * frequency_hz / sample_rate;
    let (sin, cos) = theta.sin_cos();
    // Solve (zI - Ad) w = Bd with z = cos + j sin.
    let mut a_re = [[0.0; 3]; 3];
    let mut a_im = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            a_re[i][j] = f64::from(i == j) * cos - f64::from(filter.ad[i][j]);
            a_im[i][j] = f64::from(i == j) * sin;
        }
    }
    let det = det3(a_re, a_im);
    let mut num_re = 0.0;
    let mut num_im = 0.0;
    // w = M^-1 Bd; accumulate Cd w + Dd via adjugate columns.
    for col in 0..3 {
        let mut m_re = a_re;
        let mut m_im = a_im;
        for row in 0..3 {
            m_re[row][col] = f64::from(filter.bd[row]);
            m_im[row][col] = 0.0;
        }
        let (d_re, d_im) = det3(m_re, m_im);
        // w_col = det(M_col) / det(M); accumulate Cd[col] * w_col.
        let denom = det.0 * det.0 + det.1 * det.1;
        let w_re = (d_re * det.0 + d_im * det.1) / denom;
        let w_im = (d_im * det.0 - d_re * det.1) / denom;
        num_re += f64::from(filter.cd[col]) * w_re;
        num_im += f64::from(filter.cd[col]) * w_im;
    }
    (num_re + f64::from(filter.dd), num_im)
}

/// Determinant of a 3x3 complex matrix (Sarrus' rule).
fn det3(re: [[f64; 3]; 3], im: [[f64; 3]; 3]) -> (f64, f64) {
    let mut out_re = 0.0;
    let mut out_im = 0.0;
    for start in 0..3 {
        // Forward diagonal.
        let mut f_re = 1.0;
        let mut f_im = 0.0;
        // Backward diagonal.
        let mut b_re = 1.0;
        let mut b_im = 0.0;
        for offset in 0..3 {
            let (e_re, e_im) = (
                re[offset][(start + offset) % 3],
                im[offset][(start + offset) % 3],
            );
            (f_re, f_im) = (f_re * e_re - f_im * e_im, f_re * e_im + f_im * e_re);
            let (g_re, g_im) = (
                re[offset][(start + 3 - offset) % 3],
                im[offset][(start + 3 - offset) % 3],
            );
            (b_re, b_im) = (b_re * g_re - b_im * g_im, b_re * g_im + b_im * g_re);
        }
        out_re += f_re - b_re;
        out_im += f_im - b_im;
    }
    (out_re, out_im)
}

/// Level-reference gain: |H(1 kHz)| of the discrete noon filter.
fn noon_reference_gain(values: ToneStackValues, sample_rate: f64) -> Option<f64> {
    let (t, m, l) = REFERENCE_KNOBS;
    let filter = build_filter(t, m, l, sample_rate, values)?;
    let (re, im) = discrete_response(&filter, REFERENCE_HZ, sample_rate);
    let gain = (re * re + im * im).sqrt();
    if gain.is_finite() && gain > 0.0 {
        Some(gain)
    } else {
        None
    }
}

/// A bounded FMV/TMB tone-stack filter with three smoothed knobs.
#[derive(Debug)]
pub struct ToneStackModel {
    treble: ControlSmoother,
    mid: ControlSmoother,
    bass: ControlSmoother,
    output_gain_db: ControlSmoother,
    amount: ControlSmoother,
    mix: ControlSmoother,
    values: ToneStackValues,
    defects: DefectConfig,
    spec: Option<ProcessSpec>,
    channels: Vec<ToneStackChannelState>,
}

impl Default for ToneStackModel {
    fn default() -> Self {
        Self::new()
    }
}

impl ToneStackModel {
    pub fn new() -> Self {
        let sample_rate = 48_000.0;
        Self {
            treble: ControlSmoother::new(0.5, CONTROL_SMOOTHING_MS, sample_rate),
            mid: ControlSmoother::new(0.5, CONTROL_SMOOTHING_MS, sample_rate),
            bass: ControlSmoother::new(0.5, CONTROL_SMOOTHING_MS, sample_rate),
            output_gain_db: ControlSmoother::new(0.0, CONTROL_SMOOTHING_MS, sample_rate),
            amount: ControlSmoother::new(1.0, CONTROL_SMOOTHING_MS, sample_rate),
            mix: ControlSmoother::new(1.0, CONTROL_SMOOTHING_MS, sample_rate),
            values: ToneStackValues::default(),
            defects: DefectConfig::default(),
            spec: None,
            channels: Vec::new(),
        }
    }

    pub fn treble(&self) -> f32 {
        self.treble.target()
    }

    pub fn mid(&self) -> f32 {
        self.mid.target()
    }

    pub fn bass(&self) -> f32 {
        self.bass.target()
    }

    pub fn output_gain_db(&self) -> f32 {
        self.output_gain_db.target()
    }

    pub fn amount(&self) -> f32 {
        self.amount.target()
    }

    pub fn mix(&self) -> f32 {
        self.mix.target()
    }

    pub fn values(&self) -> ToneStackValues {
        self.values
    }

    pub fn defects(&self) -> DefectConfig {
        self.defects
    }

    pub fn set_treble(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("treble", value, 0.0, 1.0)?;
        self.treble.set_target(value);
        Ok(())
    }

    pub fn set_mid(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("mid", value, 0.0, 1.0)?;
        self.mid.set_target(value);
        Ok(())
    }

    pub fn set_bass(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("bass", value, 0.0, 1.0)?;
        self.bass.set_target(value);
        Ok(())
    }

    pub fn set_output_gain_db(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("output_gain_db", value, -60.0, 24.0)?;
        self.output_gain_db.set_target(value);
        Ok(())
    }

    pub fn set_amount(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("amount", value, 0.0, 1.0)?;
        if value == 0.0 {
            self.amount.set_immediate(value);
        } else {
            self.amount.set_target(value);
        }
        Ok(())
    }

    pub fn set_mix(&mut self, value: f32) -> Result<(), AnalogError> {
        validate_finite_range("mix", value, 0.0, 1.0)?;
        if value == 0.0 {
            self.mix.set_immediate(value);
        } else {
            self.mix.set_target(value);
        }
        Ok(())
    }

    /// Select the component value set. Structural: rebuilds the prepared
    /// filters and level reference immediately (no smoothing).
    pub fn set_values(&mut self, values: ToneStackValues) {
        self.values = values;
        if let Some(spec) = self.spec {
            for channel in &mut self.channels {
                channel.rebuild(spec.sample_rate, values);
            }
        }
    }

    pub fn set_defects(&mut self, defects: DefectConfig) -> Result<(), AnalogError> {
        defects
            .validate(MAX_DEFECT_DELAY_SAMPLES)
            .map_err(|message| AnalogError::InvalidDefectConfig(message.to_string()))?;
        self.defects = defects;
        if let Some(spec) = self.spec {
            self.prepare(spec)?;
        } else {
            for channel in &mut self.channels {
                channel.reset();
            }
        }
        Ok(())
    }

    fn process_frame(&mut self, frame: &mut [f32], channels: usize) {
        let treble = self.treble.advance();
        let mid = self.mid.advance();
        let bass = self.bass.advance();
        let output_gain = db_to_gain(self.output_gain_db.advance());
        let amount = self.amount.advance();
        let mix = self.mix.advance();
        let input_sum = frame
            .iter()
            .take(channels)
            .map(|sample| sanitize_sample(*sample))
            .sum::<f32>();
        for (channel, sample) in frame.iter_mut().enumerate().take(channels) {
            let dry = sanitize_sample(*sample);
            let crosstalk_input = if channels > 1 {
                (input_sum - dry) / (channels - 1) as f32
            } else {
                0.0
            };
            let shaped = self.channels[channel].process(
                dry,
                treble,
                mid,
                bass,
                output_gain,
                crosstalk_input,
            );
            let effected = dry + amount * (shaped - dry);
            *sample = finite_output(dry + mix * (effected - dry));
        }
    }
}

impl AnalogProcessor for ToneStackModel {
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), AnalogError> {
        spec.validate()?;
        let channels = (0..spec.channels)
            .map(|_| ToneStackChannelState::new(spec.sample_rate, self.values, self.defects))
            .collect();
        self.treble.reconfigure(spec.sample_rate);
        self.mid.reconfigure(spec.sample_rate);
        self.output_gain_db.reconfigure(spec.sample_rate);
        self.amount.reconfigure(spec.sample_rate);
        self.mix.reconfigure(spec.sample_rate);
        self.bass.reconfigure(spec.sample_rate);
        self.channels = channels;
        self.spec = Some(spec);
        self.reset();
        Ok(())
    }

    fn reset(&mut self) {
        self.treble.reset();
        self.mid.reset();
        self.bass.reset();
        self.output_gain_db.reset();
        self.amount.reset();
        self.mix.reset();
        let (values, spec) = (self.values, self.spec);
        for channel in &mut self.channels {
            channel.reset();
            // Rebuild at the reset (target) knobs so reset output matches a
            // fresh prepare bit-exactly.
            if let Some(spec) = spec {
                channel.retune(
                    self.treble.target(),
                    self.mid.target(),
                    self.bass.target(),
                    spec.sample_rate,
                    values,
                    true,
                );
            }
        }
    }

    fn process_interleaved(
        &mut self,
        samples: &mut [f32],
        frames: usize,
    ) -> Result<(), AnalogError> {
        let spec = self.spec.ok_or(AnalogError::NotPrepared)?;
        checked_block_len(spec, frames, samples.len())?;
        for frame in samples.chunks_exact_mut(spec.channels).take(frames) {
            self.process_frame(frame, spec.channels);
        }
        Ok(())
    }

    fn latency_samples(&self) -> usize {
        0
    }
}

#[derive(Debug)]
struct ToneStackChannelState {
    filter: DiscreteFilter,
    built_knobs: (f32, f32, f32),
    output_scale: f32,
    values: ToneStackValues,
    sample_rate: f32,
    state: [f32; 3],
    defects: DefectState,
    dc_blocker: DcBlocker,
}

impl ToneStackChannelState {
    fn new(sample_rate: f32, values: ToneStackValues, defects: DefectConfig) -> Self {
        let mut channel = Self {
            filter: DiscreteFilter {
                ad: [[0.0; 3]; 3],
                bd: [0.0; 3],
                cd: [0.0; 3],
                dd: 0.0,
            },
            built_knobs: (0.5, 0.5, 0.5),
            output_scale: 1.0,
            values,
            sample_rate,
            state: [0.0; 3],
            defects: DefectState::new(defects, sample_rate, MAX_DEFECT_DELAY_SAMPLES),
            dc_blocker: DcBlocker::new(sample_rate),
        };
        channel.rebuild(sample_rate, values);
        channel
    }

    /// Rebuild the noon level reference and retune at noon. The build path
    /// is infallible in practice; a failed build keeps the previous filter
    /// (deterministic, bounded) instead of panicking.
    fn rebuild(&mut self, sample_rate: f32, values: ToneStackValues) {
        self.sample_rate = sample_rate;
        self.values = values;
        if let Some(gain) = noon_reference_gain(values, f64::from(sample_rate)) {
            self.output_scale = (1.0 / gain) as f32;
        }
        self.retune(0.5, 0.5, 0.5, sample_rate, values, true);
    }

    /// Retune the filter when smoothed knobs moved (or `force`).
    fn retune(
        &mut self,
        treble: f32,
        mid: f32,
        bass: f32,
        sample_rate: f32,
        values: ToneStackValues,
        force: bool,
    ) {
        let moved = force
            || (treble - self.built_knobs.0).abs() > REBUILD_EPSILON
            || (mid - self.built_knobs.1).abs() > REBUILD_EPSILON
            || (bass - self.built_knobs.2).abs() > REBUILD_EPSILON;
        if !moved {
            return;
        }
        if let Some(filter) = build_filter(
            f64::from(treble),
            f64::from(mid),
            f64::from(bass),
            f64::from(sample_rate),
            values,
        ) {
            self.filter = filter;
            self.built_knobs = (treble, mid, bass);
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn process(
        &mut self,
        input: f32,
        treble: f32,
        mid: f32,
        bass: f32,
        output_gain: f32,
        crosstalk_input: f32,
    ) -> f32 {
        self.retune(treble, mid, bass, self.sample_rate, self.values, false);
        let [x0, x1, x2] = self.state;
        let output = self.filter.cd[0] * x0
            + self.filter.cd[1] * x1
            + self.filter.cd[2] * x2
            + self.filter.dd * input;
        let mut next = [0.0; 3];
        for (row, slot) in self.filter.ad.iter().zip(next.iter_mut()) {
            *slot = row[0] * x0 + row[1] * x1 + row[2] * x2;
        }
        for (slot, gain) in next.iter_mut().zip(self.filter.bd.iter()) {
            *slot = flush_denormal(*slot + gain * input);
        }
        self.state = next;
        let scaled = output * self.output_scale;
        let effected = self.defects.process(scaled, crosstalk_input);
        finite_output(self.dc_blocker.process(effected) * output_gain)
    }

    fn reset(&mut self) {
        self.state = [0.0; 3];
        self.defects.reset();
        self.dc_blocker.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AnalogModel;
    use crate::analysis::measure_harmonics;
    use std::f32::consts::TAU;

    /// Yeh & Smith DAFx-06 symbolic transfer-function magnitude in dB.
    ///
    /// Independent transcription of the paper's closed-form `H(s)` (verified
    /// by Yeh against SPICE at t=m=l=0.5): `C1` treble cap, `C2` bass cap,
    /// `C3` mid cap, `R1` treble pot, `R2` bass pot, `R3` mid pot, `R4`
    /// slope resistor, `t`/`m`/`l` the three controls.
    // Eleven arguments mirror the paper's eleven symbols one-to-one so the
    // transcription stays auditable; bundling them would obscure it.
    #[allow(clippy::too_many_arguments)]
    fn yeh_oracle_db(
        c1: f64,
        c2: f64,
        c3: f64,
        r1: f64,
        r2: f64,
        r3: f64,
        r4: f64,
        t: f64,
        m: f64,
        l: f64,
        frequency_hz: f64,
    ) -> f64 {
        let b1 = t * c1 * r1 + m * c3 * r3 + l * (c1 * r2 + c2 * r2) + (c1 * r3 + c2 * r3);
        let b2 = t * (c1 * c2 * r1 * r4 + c1 * c3 * r1 * r4)
            - m * m * (c1 * c3 * r3 * r3 + c2 * c3 * r3 * r3)
            + m * (c1 * c3 * r1 * r3 + c1 * c3 * r3 * r3 + c2 * c3 * r3 * r3)
            + l * (c1 * c2 * r1 * r2 + c1 * c2 * r2 * r4 + c1 * c3 * r2 * r4)
            + l * m * (c1 * c3 * r2 * r3 + c2 * c3 * r2 * r3)
            + (c1 * c2 * r1 * r3 + c1 * c2 * r3 * r4 + c1 * c3 * r3 * r4);
        let b3 = l * m * (c1 * c2 * c3 * r1 * r2 * r3 + c1 * c2 * c3 * r2 * r3 * r4)
            - m * m * (c1 * c2 * c3 * r1 * r3 * r3 + c1 * c2 * c3 * r3 * r3 * r4)
            + m * (c1 * c2 * c3 * r1 * r3 * r3 + c1 * c2 * c3 * r3 * r3 * r4)
            + t * c1 * c2 * c3 * r1 * r3 * r4
            - t * m * c1 * c2 * c3 * r1 * r3 * r4
            + t * l * c1 * c2 * c3 * r1 * r2 * r4;
        let a1 = (c1 * r1 + c1 * r3 + c2 * r3 + c2 * r4 + c3 * r4)
            + m * c3 * r3
            + l * (c1 * r2 + c2 * r2);
        let a2 = m
            * (c1 * c3 * r1 * r3 - c2 * c3 * r3 * r4 + c1 * c3 * r3 * r3 + c2 * c3 * r3 * r3)
            + l * m * (c1 * c3 * r2 * r3 + c2 * c3 * r2 * r3)
            - m * m * (c1 * c3 * r3 * r3 + c2 * c3 * r3 * r3)
            + l * (c1 * c2 * r2 * r4 + c1 * c2 * r1 * r2 + c1 * c3 * r2 * r4 + c2 * c3 * r2 * r4)
            + (c1 * c2 * r1 * r4
                + c1 * c3 * r1 * r4
                + c1 * c2 * r3 * r4
                + c1 * c2 * r1 * r3
                + c1 * c3 * r3 * r4
                + c2 * c3 * r3 * r4);
        let a3 = l * m * (c1 * c2 * c3 * r1 * r2 * r3 + c1 * c2 * c3 * r2 * r3 * r4)
            - m * m * (c1 * c2 * c3 * r1 * r3 * r3 + c1 * c2 * c3 * r3 * r3 * r4)
            + m * (c1 * c2 * c3 * r3 * r3 * r4 + c1 * c2 * c3 * r1 * r3 * r3
                - c1 * c2 * c3 * r1 * r3 * r4)
            + l * c1 * c2 * c3 * r1 * r2 * r4
            + c1 * c2 * c3 * r1 * r3 * r4;
        // H(jw): num = b1 s + b2 s^2 + b3 s^3, den = 1 + a1 s + a2 s^2 + a3 s^3.
        let w = 2.0 * std::f64::consts::PI * frequency_hz;
        let num_re = -b2 * w * w;
        let num_im = b1 * w - b3 * w * w * w;
        let den_re = 1.0 - a2 * w * w;
        let den_im = a1 * w - a3 * w * w * w;
        let mag_sq = (num_re * num_re + num_im * num_im) / (den_re * den_re + den_im * den_im);
        10.0 * mag_sq.log10()
    }

    /// Crate-standard final DC-blocker magnitude in dB (exact LTI response).
    fn dc_blocker_db(frequency_hz: f64, sample_rate: f64) -> f64 {
        let theta = 2.0 * std::f64::consts::PI * frequency_hz / sample_rate;
        let c = (-2.0 * std::f64::consts::PI * 20.0 / sample_rate).exp();
        let num = 2.0 - 2.0 * theta.cos();
        let den = 1.0 + c * c - 2.0 * c * theta.cos();
        10.0 * (num / den).log10()
    }

    /// Nearest coherent-bin frequency for clean settled-sine measurement.
    fn coherent(frequency_hz: f32, sample_rate: f32, record: usize) -> f32 {
        let bin = (frequency_hz * record as f32 / sample_rate).round();
        bin * sample_rate / record as f32
    }

    /// Measure the fundamental magnitude in dBFS of a settled sine through a
    /// freshly prepared model. Knobs are set before prepare so smoothers
    /// start settled; the record covers settling plus a measured tail.
    fn measured_db(
        values: ToneStackValues,
        treble: f32,
        mid: f32,
        bass: f32,
        sample_rate: f32,
        target_hz: f32,
        amplitude: f32,
    ) -> f32 {
        const SETTLE: usize = 14_400;
        const MEASURE: usize = 4_800;
        let frequency = coherent(target_hz, sample_rate, MEASURE);
        let mut model = ToneStackModel::new();
        model.set_values(values);
        model.set_treble(treble).unwrap();
        model.set_mid(mid).unwrap();
        model.set_bass(bass).unwrap();
        model
            .prepare(ProcessSpec::new(sample_rate, 1, SETTLE + MEASURE))
            .unwrap();
        let mut samples: Vec<f32> = (0..SETTLE + MEASURE)
            .map(|index| amplitude * (TAU * frequency * index as f32 / sample_rate).sin())
            .collect();
        model
            .process_interleaved(&mut samples, SETTLE + MEASURE)
            .unwrap();
        let tail = &samples[SETTLE..];
        let report = measure_harmonics(tail, sample_rate, frequency, 2).unwrap();
        let fundamental = report.component(1).unwrap().amplitude;
        20.0 * (fundamental / amplitude).log10()
    }

    #[test]
    fn tonestack_model_is_finite_and_resettable() {
        let mut model = ToneStackModel::new();
        model.prepare(ProcessSpec::new(48_000.0, 2, 128)).unwrap();
        let mut buffer = vec![16.0_f32; 256];
        model.process_interleaved(&mut buffer, 128).unwrap();
        assert!(buffer.iter().all(|sample| sample.is_finite()));
        model.reset();
        let mut actual = [0.25_f32, -0.25];
        model.process_interleaved(&mut actual, 1).unwrap();
        assert!(actual.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn tonestack_model_exposes_a_distinct_append_only_id() {
        let model = AnalogModel::ToneStack(ToneStackModel::new());
        assert_eq!(model.model_id(), AnalogModel::TONE_STACK_ID);
        assert!(matches!(
            AnalogModel::from_id(AnalogModel::TONE_STACK_ID),
            Ok(AnalogModel::ToneStack(_))
        ));
    }

    #[test]
    fn response_shape_matches_the_yeh_symbolic_oracle() {
        // End-to-end: settled sines through the realtime path vs the
        // SPICE-verified symbolic H(s). The comparison removes the fixed
        // noon level reference and the exact DC-blocker response, so the
        // residual spread is derivation + discretization error only.
        let knob_sets = [
            (0.5, 0.5, 0.5),
            (0.9, 0.1, 0.8),
            (0.1, 0.9, 0.2),
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 0.0),
        ];
        let frequencies = [
            50.0, 100.0, 200.0, 500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0, 15_000.0,
        ];
        for values in [ToneStackValues::Schematic59, ToneStackValues::Production] {
            let (slope, c_t, c_b, c_m, r_t, r_b, r_m) = values.components();
            for (treble, mid, bass) in knob_sets {
                let mut deltas = Vec::new();
                for frequency in frequencies {
                    // Oracle in Yeh's (C1,C2,C3,R1,R2,R3,R4) notation: treble
                    // cap, bass cap, mid cap, treble, bass, mid pots, slope.
                    let oracle = yeh_oracle_db(
                        c_t,
                        c_b,
                        c_m,
                        r_t,
                        r_b,
                        r_m,
                        slope,
                        f64::from(treble),
                        f64::from(mid),
                        f64::from(bass),
                        f64::from(frequency),
                    );
                    let measured = f64::from(measured_db(
                        values, treble, mid, bass, 48_000.0, frequency, 0.5,
                    ));
                    let coherent_f = f64::from(coherent(frequency, 48_000.0, 4_800));
                    let expected = oracle + dc_blocker_db(coherent_f, 48_000.0);
                    deltas.push(measured - expected);
                }
                let spread = deltas.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b))
                    - deltas.iter().fold(f64::INFINITY, |a, b| a.min(*b));
                assert!(
                    spread <= 0.25,
                    "{values:?} knobs ({treble}, {mid}, {bass}): shape spread {spread:.3} dB over {deltas:.2?}"
                );
            }
        }
    }

    #[test]
    fn noon_reference_level_is_unity_at_1khz_for_both_value_sets() {
        for values in [ToneStackValues::Schematic59, ToneStackValues::Production] {
            let measured = measured_db(values, 0.5, 0.5, 0.5, 48_000.0, 1_000.0, 0.5);
            assert!(
                measured.abs() < 0.1,
                "{values:?}: noon 1 kHz reads {measured:.3} dB, expected 0"
            );
        }
    }

    #[test]
    fn noon_curve_shows_the_published_mid_scoop() {
        let at_100 = measured_db(
            ToneStackValues::Schematic59,
            0.5,
            0.5,
            0.5,
            48_000.0,
            100.0,
            0.5,
        );
        let mut dip = f32::INFINITY;
        for frequency in [400.0, 500.0, 700.0, 1_000.0] {
            let level = measured_db(
                ToneStackValues::Schematic59,
                0.5,
                0.5,
                0.5,
                48_000.0,
                frequency,
                0.5,
            );
            dip = dip.min(level);
        }
        assert!(
            at_100 - dip >= 6.0,
            "noon scoop too shallow: 100 Hz at {at_100:.2} dB, dip at {dip:.2} dB"
        );
    }

    #[test]
    fn knobs_sweep_their_bands_monotonically_with_range() {
        let bass: Vec<f32> = [0.0, 0.25, 0.5, 0.75, 1.0]
            .iter()
            .map(|bass| {
                measured_db(
                    ToneStackValues::Schematic59,
                    0.5,
                    0.5,
                    *bass,
                    48_000.0,
                    40.0,
                    0.5,
                )
            })
            .collect();
        assert!(
            bass.windows(2).all(|pair| pair[1] > pair[0]),
            "bass sweep not increasing: {bass:.2?}"
        );
        assert!(
            bass[4] - bass[0] >= 10.0,
            "bass range too small: {bass:.2?}"
        );
        let mid: Vec<f32> = [0.0, 0.25, 0.5, 0.75, 1.0]
            .iter()
            .map(|mid| {
                measured_db(
                    ToneStackValues::Schematic59,
                    0.5,
                    *mid,
                    0.5,
                    48_000.0,
                    500.0,
                    0.5,
                )
            })
            .collect();
        assert!(
            mid.windows(2).all(|pair| pair[1] > pair[0]),
            "mid sweep not increasing: {mid:.2?}"
        );
        assert!(mid[4] - mid[0] >= 4.0, "mid range too small: {mid:.2?}");
        let treble: Vec<f32> = [0.0, 0.25, 0.5, 0.75, 1.0]
            .iter()
            .map(|treble| {
                measured_db(
                    ToneStackValues::Schematic59,
                    *treble,
                    0.5,
                    0.5,
                    48_000.0,
                    8_000.0,
                    0.5,
                )
            })
            .collect();
        assert!(
            treble.windows(2).all(|pair| pair[1] > pair[0]),
            "treble sweep not increasing: {treble:.2?}"
        );
        assert!(
            treble[4] - treble[0] >= 8.0,
            "treble range too small: {treble:.2?}"
        );
    }

    /// Real 3x3 inverse by adjugate (test-only DC-gain probe).
    // Adjugate subscripting needs row/column indices by construction.
    #[allow(clippy::needless_range_loop)]
    fn invert3(matrix: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
        let det = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
        if det.abs() < 1e-300 {
            return None;
        }
        let cofactor = |i: usize, j: usize| {
            let rows = [0, 1, 2]
                .into_iter()
                .filter(|r| *r != i)
                .collect::<Vec<_>>();
            let cols = [0, 1, 2]
                .into_iter()
                .filter(|c| *c != j)
                .collect::<Vec<_>>();
            let minor = matrix[rows[0]][cols[0]] * matrix[rows[1]][cols[1]]
                - matrix[rows[0]][cols[1]] * matrix[rows[1]][cols[0]];
            if (i + j).is_multiple_of(2) {
                minor
            } else {
                -minor
            }
        };
        let mut inverse = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                inverse[j][i] = cofactor(i, j) / det;
            }
        }
        Some(inverse)
    }

    /// H(z=1) = Cd (I - Ad)^-1 Bd + Dd straight from built f32 matrices.
    // Subscripting needs row/column indices by construction.
    #[allow(clippy::needless_range_loop)]
    fn dc_gain_db(filter: &DiscreteFilter) -> f64 {
        let mut one_minus_ad = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                one_minus_ad[i][j] = f64::from(i == j) - f64::from(filter.ad[i][j]);
            }
        }
        let inverse = invert3(one_minus_ad).expect("nonsingular");
        let mut gain = f64::from(filter.dd);
        for i in 0..3 {
            let mut w = 0.0;
            for j in 0..3 {
                w += inverse[i][j] * f64::from(filter.bd[j]);
            }
            gain += f64::from(filter.cd[i]) * w;
        }
        20.0 * gain.abs().log10()
    }

    #[test]
    fn dc_gain_is_zero_for_both_value_sets() {
        // The stack blocks DC by construction; probe the built matrices.
        for values in [ToneStackValues::Schematic59, ToneStackValues::Production] {
            let filter = build_filter(0.5, 0.5, 0.5, 48_000.0, values).expect("builds");
            let gain_db = dc_gain_db(&filter);
            assert!(
                gain_db < -90.0,
                "{values:?}: DC gain {gain_db:.1} dB, expected below -90 dB"
            );
        }
    }

    #[test]
    fn builds_at_all_knob_extremes_and_both_value_sets() {
        for values in [ToneStackValues::Schematic59, ToneStackValues::Production] {
            for corner in 0..8_u8 {
                let treble = f64::from((corner >> 2) & 1);
                let mid = f64::from((corner >> 1) & 1);
                let bass = f64::from(corner & 1);
                let filter = build_filter(treble, mid, bass, 48_000.0, values)
                    .expect("extreme knobs must build");
                assert!(filter.dd.is_finite());
                let mut model = ToneStackModel::new();
                model.set_values(values);
                model.set_treble(treble as f32).unwrap();
                model.set_mid(mid as f32).unwrap();
                model.set_bass(bass as f32).unwrap();
                model.prepare(ProcessSpec::new(48_000.0, 1, 256)).unwrap();
                let mut block = vec![0.5_f32; 256];
                model.process_interleaved(&mut block, 256).unwrap();
                assert!(block.iter().all(|sample| sample.is_finite()));
            }
        }
    }

    #[test]
    fn sample_rates_agree_below_18khz() {
        for frequency in [1_000.0, 10_000.0] {
            let at_48 = measured_db(
                ToneStackValues::Schematic59,
                0.5,
                0.5,
                0.5,
                48_000.0,
                frequency,
                0.5,
            );
            let at_96 = measured_db(
                ToneStackValues::Schematic59,
                0.5,
                0.5,
                0.5,
                96_000.0,
                frequency,
                0.5,
            );
            assert!(
                (at_48 - at_96).abs() <= 0.5,
                "{frequency} Hz: 48k reads {at_48:.3} dB, 96k reads {at_96:.3} dB"
            );
        }
    }

    #[test]
    fn knob_and_value_setters_validate() {
        let mut model = ToneStackModel::new();
        assert!(model.set_treble(-0.1).is_err());
        assert!(model.set_mid(1.1).is_err());
        assert!(model.set_bass(f32::NAN).is_err());
        model.set_treble(0.8).unwrap();
        model.set_mid(0.2).unwrap();
        model.set_bass(0.9).unwrap();
        assert_eq!(model.values(), ToneStackValues::Schematic59);
        model.prepare(ProcessSpec::new(48_000.0, 1, 64)).unwrap();
        model.set_values(ToneStackValues::Production);
        assert_eq!(model.values(), ToneStackValues::Production);
        let mut block = vec![0.25_f32; 64];
        model.process_interleaved(&mut block, 64).unwrap();
        assert!(block.iter().all(|sample| sample.is_finite()));
    }
}

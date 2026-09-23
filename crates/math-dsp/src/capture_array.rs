//! C5–C6 compact-array direction finding and geometry helpers.
//!
//! Operates on drift-corrected channels (C3) plus known mic geometry.
//! Far-field plane-wave assumption throughout: valid when the source
//! distance exceeds ~10× the array aperture.

use crate::capture_tdoa::{TdoaConfig, estimate_chirp_tdoa};

/// Speed of sound used for delay↔distance conversion (m/s at ~20 °C dry
/// air; override via explicit scaling if calibrated otherwise).
pub const SPEED_OF_SOUND_M_S: f64 = 343.0;
/// Default DOA band (Hz): midrange, where magnitude-only UMIK-class
/// calibration still leaves phase usable. Above ~3 kHz per-unit phase is
/// uncalibrated (HF degradation); below ~300 Hz compact apertures resolve
/// nothing (see aperture-vs-wavelength notes on [`estimate_doa_ls`]).
pub const DEFAULT_DOA_BAND_LO_HZ: f64 = 300.0;
/// See [`DEFAULT_DOA_BAND_LO_HZ`].
pub const DEFAULT_DOA_BAND_HI_HZ: f64 = 3000.0;

/// Microphone positions in metres, sample rate in Hz.
#[derive(Debug, Clone)]
pub struct MicArray {
    positions: Vec<[f64; 3]>,
    sample_rate_hz: f64,
}

impl MicArray {
    /// Positions in metres; all finite, at least one mic, rate positive.
    pub fn new(positions: Vec<[f64; 3]>, sample_rate_hz: f64) -> Result<Self, String> {
        if positions.is_empty() {
            return Err("MicArray: need at least one microphone".to_string());
        }
        if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
            return Err("MicArray: sample rate must be positive".to_string());
        }
        if positions.iter().any(|p| p.iter().any(|c| !c.is_finite())) {
            return Err("MicArray: positions must be finite".to_string());
        }
        Ok(Self {
            positions,
            sample_rate_hz,
        })
    }

    /// Number of microphones.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// True when the array holds no microphones (never after [`MicArray::new`]).
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Position of microphone `i` in metres.
    pub fn position(&self, i: usize) -> Result<[f64; 3], String> {
        self.positions
            .get(i)
            .copied()
            .ok_or_else(|| format!("MicArray: mic index {i} out of range"))
    }

    /// Full inter-mic distance matrix in metres.
    pub fn inter_mic_distances(&self) -> Vec<Vec<f64>> {
        self.positions
            .iter()
            .map(|a| {
                self.positions
                    .iter()
                    .map(|b| {
                        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2))
                            .sqrt()
                    })
                    .collect()
            })
            .collect()
    }

    /// Largest inter-mic distance in metres (array aperture).
    pub fn aperture(&self) -> f64 {
        self.inter_mic_distances()
            .iter()
            .flat_map(|row| row.iter())
            .fold(0.0f64, |m, &v| m.max(v))
    }

    /// True when the array is flat to `tol_m` in z (elevation sign of a
    /// DOA estimate is then ambiguous — see [`DoaEstimate`]).
    pub fn is_coplanar(&self, tol_m: f64) -> bool {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in &self.positions {
            lo = lo.min(p[2]);
            hi = hi.max(p[2]);
        }
        hi - lo <= tol_m
    }
}

/// One pairwise TDOA measurement: delay of mic `b` relative to mic `a`,
/// in (fractional) samples. Positive = `b` lags `a`.
#[derive(Debug, Clone, Copy)]
pub struct PairDelay {
    /// First mic index.
    pub a: usize,
    /// Second mic index.
    pub b: usize,
    /// Delay of `b` relative to `a`, in samples.
    pub delay_samples: f64,
}

/// Predicted delay of `b` relative to `a` for a plane wave from unit
/// direction `dir` (pointing *toward* the source), in samples.
///
/// Sign: wavefronts travel in `−dir`, so arrival time is
/// `t(p) = t₀ − (p·dir)/c` — the mic closer to the source (larger `p·dir`)
/// leads. Positive return = `b` lags `a`, matching [`PairDelay`] and the
/// C1 convention (recording lags reference).
fn predicted_delay(array: &MicArray, dir: &[f64; 3], a: usize, b: usize) -> Result<f64, String> {
    let pa = array.position(a)?;
    let pb = array.position(b)?;
    let dot = (pb[0] - pa[0]) * dir[0] + (pb[1] - pa[1]) * dir[1] + (pb[2] - pa[2]) * dir[2];
    Ok(-dot / SPEED_OF_SOUND_M_S * array.sample_rate_hz)
}

fn check_dir(dir: &[f64; 3]) -> Result<(), String> {
    if dir.iter().any(|c| !c.is_finite()) {
        return Err("direction must be finite".to_string());
    }
    let norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    if (norm - 1.0).abs() > 1e-6 {
        return Err(format!("direction must be unit (norm {norm})"));
    }
    Ok(())
}

/// C6: residuals (measured − predicted) for each pair, in samples.
pub fn tdoa_residuals(
    array: &MicArray,
    dir: &[f64; 3],
    pairs: &[PairDelay],
) -> Result<Vec<f64>, String> {
    check_dir(dir)?;
    pairs
        .iter()
        .map(|pair| {
            if !pair.delay_samples.is_finite() {
                return Err("pair delay must be finite".to_string());
            }
            Ok(pair.delay_samples - predicted_delay(array, dir, pair.a, pair.b)?)
        })
        .collect()
}

/// C6: self-consistency of measured direct-sound TDOAs against declared
/// geometry. Returns the worst residual in samples when every pair agrees
/// within `tol_samples`; fails loudly (`Err` naming the worst pair and its
/// residual) on mis-measured tape.
pub fn check_geometry(
    array: &MicArray,
    dir: &[f64; 3],
    pairs: &[PairDelay],
    tol_samples: f64,
) -> Result<f64, String> {
    if pairs.is_empty() {
        return Err("check_geometry: need at least one pair".to_string());
    }
    let residuals = tdoa_residuals(array, dir, pairs)?;
    let (worst_pair, worst) = pairs
        .iter()
        .zip(residuals.iter())
        .max_by(|a, b| {
            a.1.abs()
                .partial_cmp(&b.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(pair, resid)| (pair, resid.abs()))
        .unwrap_or((&pairs[0], f64::INFINITY));
    if worst <= tol_samples {
        Ok(worst)
    } else {
        Err(format!(
            "geometry inconsistent: pair ({},{}) residual {worst:.3} samples exceeds tol {tol_samples} — re-measure tape",
            worst_pair.a, worst_pair.b
        ))
    }
}

/// C6: scale-only self-calibration from a known source direction.
///
/// Returns the scale factor `s` (multiply all positions by `s`) minimising
/// Σ(s·pred − meas)² plus the RMS residual in samples. Corrects
/// tape-measure scale errors; rotation, shear, or per-mic blunders still
/// fail [`check_geometry`] loudly — full 3-D self-calibration from
/// unknown sources is out of scope.
pub fn calibrate_geometry_scale(
    array: &MicArray,
    dir: &[f64; 3],
    pairs: &[PairDelay],
) -> Result<(f64, f64), String> {
    check_dir(dir)?;
    if pairs.is_empty() {
        return Err("calibrate_geometry_scale: need at least one pair".to_string());
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for pair in pairs {
        if !pair.delay_samples.is_finite() {
            return Err("pair delay must be finite".to_string());
        }
        let pred = predicted_delay(array, dir, pair.a, pair.b)?;
        num += pred * pair.delay_samples;
        den += pred * pred;
    }
    if den <= 0.0 {
        return Err("calibrate_geometry_scale: degenerate geometry (coincident mics?)".to_string());
    }
    let scale = num / den;
    let rms = (pairs
        .iter()
        .map(|pair| {
            let pred = predicted_delay(array, dir, pair.a, pair.b).unwrap_or(0.0);
            (scale * pred - pair.delay_samples).powi(2)
        })
        .sum::<f64>()
        / pairs.len() as f64)
        .sqrt();
    Ok((scale, rms))
}

/// Direction-of-arrival estimate for one arrival.
#[derive(Debug, Clone, Copy)]
pub struct DoaEstimate {
    /// Unit direction *toward* the source (arrival direction reversed).
    pub direction: [f64; 3],
    /// Azimuth in degrees, atan2(y, x), (−180, 180].
    pub azimuth_deg: f64,
    /// Elevation above the horizontal plane in degrees, [−90, 90].
    pub elevation_deg: f64,
    /// RMS TDOA residual in samples (fit quality).
    pub rms_residual_samples: f64,
    /// Residual-based confidence in [0, 1] (`1/(1+rms)`).
    pub confidence: f64,
    /// True for planar geometry: directions mirrored across the array plane
    /// fit equally well. The returned direction is only one representative.
    pub elevation_ambiguous: bool,
}

/// Solve in the measured geometric subspace without inventing a third baseline.
fn solve_direction(a: &[[f64; 3]; 3], b: &[f64; 3]) -> Result<([f64; 3], bool), String> {
    if a.iter().flatten().chain(b).any(|value| !value.is_finite()) {
        return Err("estimate_doa_ls: non-finite normal equations".into());
    }
    let matrix = nalgebra::Matrix3::from_fn(|row, col| a[row][col]);
    let eigen = matrix
        .try_symmetric_eigen(f64::EPSILON, 128)
        .ok_or("estimate_doa_ls: geometry decomposition did not converge")?;
    let largest = eigen.eigenvalues.max();
    if !largest.is_finite() || largest <= 0.0 {
        return Err("estimate_doa_ls: coincident geometry".into());
    }
    // Numerical rank only: callers must additionally consider survey errors
    // relative to the smallest baseline. This is not an acoustic confidence.
    let tolerance = largest * 1e-10;
    let target = nalgebra::Vector3::from_column_slice(b);
    let mut solution = nalgebra::Vector3::zeros();
    let mut normal = nalgebra::Vector3::zeros();
    let mut rank = 0;
    for index in 0..3 {
        let axis = eigen.eigenvectors.column(index);
        let value = eigen.eigenvalues[index];
        if value > tolerance {
            solution += axis * (axis.dot(&target) / value);
            rank += 1;
        } else {
            normal = axis.into_owned();
        }
    }
    if rank < 2 {
        return Err("estimate_doa_ls: collinear geometry cannot identify a direction".into());
    }
    let planar = rank == 2;
    if planar {
        let squared = solution.norm_squared();
        if !squared.is_finite() || squared > 1.0 + 1e-8 {
            return Err(
                "estimate_doa_ls: planar delays are incompatible with a unit direction".into(),
            );
        }
        // Choose one deterministic representative of the mirror pair. Never
        // interpret this choice as resolving which side of the array is real.
        let axis = (0..3)
            .max_by(|&a, &b| normal[a].abs().total_cmp(&normal[b].abs()))
            .ok_or("estimate_doa_ls: missing plane normal")?;
        if normal[axis] < 0.0 {
            normal = -normal;
        }
        solution += normal * (1.0 - squared).max(0.0).sqrt();
    }
    let norm = solution.norm();
    if !norm.is_finite() || norm <= 0.0 {
        return Err("estimate_doa_ls: degenerate solution".into());
    }
    solution /= norm;
    Ok(([solution[0], solution[1], solution[2]], planar))
}

/// C5: direction-of-arrival by TDOA multilateration (least squares).
///
/// Minimises Σ(τ_ab − (p_a − p_b)·u/c·sr)² over unconstrained `u`, then
/// normalises. Chosen over delay-and-sum because it is closed-form,
/// exact in free field, and its residual directly feeds confidence;
/// [`delay_and_sum_power`] remains available for scoring candidate
/// directions (e.g. picked reflections).
///
/// Assumptions (beside the function, as required): far-field plane wave;
/// geometry known to millimetre level (see [`check_geometry`]); mic phase
/// response matched across the analysis band — with magnitude-only
/// calibration keep to [`DEFAULT_DOA_BAND_LO_HZ`]–[`DEFAULT_DOA_BAND_HI_HZ`].
/// Aperture-vs-wavelength: below ~λ/2 aperture the array is directionally
/// flat (0.12 m aperture → useful above ~1.4 kHz); above ~λ/1 spacing,
/// spatial aliasing mirrors the estimate — the caller must reject unsupported bands. Residual-based
/// `confidence` alone cannot detect aliasing or poor angular resolution.
pub fn estimate_doa_ls(array: &MicArray, pairs: &[PairDelay]) -> Result<DoaEstimate, String> {
    if array.len() < 3 {
        return Err("estimate_doa_ls: need at least 3 microphones".to_string());
    }
    if pairs.is_empty() {
        return Err("estimate_doa_ls: need at least one pair".to_string());
    }
    // Normal equations for u. Rows are negated baselines because measured
    // delays follow t(p) = t₀ − (p·d)/c (see `predicted_delay`); solving
    // for u then yields the direction *toward* the source.
    let k = array.sample_rate_hz / SPEED_OF_SOUND_M_S;
    let mut ata = [[0.0; 3]; 3];
    let mut atb = [0.0; 3];
    for pair in pairs {
        if !pair.delay_samples.is_finite() {
            return Err("pair delay must be finite".to_string());
        }
        let pa = array.position(pair.a)?;
        let pb = array.position(pair.b)?;
        let row = [
            (pa[0] - pb[0]) * k,
            (pa[1] - pb[1]) * k,
            (pa[2] - pb[2]) * k,
        ];
        for i in 0..3 {
            atb[i] += row[i] * pair.delay_samples;
            for j in 0..3 {
                ata[i][j] += row[i] * row[j];
            }
        }
    }
    let (direction, elevation_ambiguous) = solve_direction(&ata, &atb)?;
    let mut rms = 0.0;
    for pair in pairs {
        let resid = pair.delay_samples - predicted_delay(array, &direction, pair.a, pair.b)?;
        rms += resid * resid;
    }
    rms = (rms / pairs.len() as f64).sqrt();
    Ok(DoaEstimate {
        direction,
        azimuth_deg: direction[1].atan2(direction[0]).to_degrees(),
        elevation_deg: (direction[2].clamp(-1.0, 1.0)).asin().to_degrees(),
        rms_residual_samples: rms,
        confidence: 1.0 / (1.0 + rms),
        elevation_ambiguous,
    })
}

/// C5 scoring helper: delay-and-sum steered-response power for drift-
/// corrected `channels` at candidate direction `dir`.
///
/// Fractional steering by linear interpolation (scoring only — estimation
/// itself is multilateration, see [`estimate_doa_ls`]). Returns mean
/// squared steered output; larger = more energy from `dir`.
pub fn delay_and_sum_power(
    channels: &[Vec<f32>],
    array: &MicArray,
    dir: &[f64; 3],
) -> Result<f64, String> {
    check_dir(dir)?;
    if channels.len() != array.len() {
        return Err("delay_and_sum_power: channel count must match mic count".to_string());
    }
    if channels.iter().any(|ch| ch.is_empty()) {
        return Err("delay_and_sum_power: channels must be non-empty".to_string());
    }
    let n = channels.iter().map(|ch| ch.len()).min().unwrap_or(0);
    let mut acc = 0.0f64;
    for i in 0..n {
        let mut sum = 0.0;
        for (mic, ch) in channels.iter().enumerate() {
            // Steer: advance mic `mic` by its plane-wave delay vs mic 0.
            let d = predicted_delay(array, dir, 0, mic)?;
            let t = i as f64 + d;
            let i0 = t.floor() as i64;
            let frac = (t - t.floor()) as f32;
            let at = |j: i64| {
                if j < 0 || j >= ch.len() as i64 {
                    0.0
                } else {
                    ch[j as usize]
                }
            };
            sum += at(i0) * (1.0 - frac) + at(i0 + 1) * frac;
        }
        acc += (sum as f64 / channels.len() as f64).powi(2);
    }
    Ok(acc / n as f64)
}

/// Pairwise C1 TDOAs of `channels[1..]` relative to `channels[0]`.
/// `None` entries mark pairs whose C1 estimate is invalid — never
/// fabricated zeros.
pub fn pairwise_tdoas_vs_first(
    channels: &[Vec<f32>],
    config: &TdoaConfig,
) -> Vec<Option<PairDelay>> {
    if channels.is_empty() {
        return Vec::new();
    }
    channels[1..]
        .iter()
        .enumerate()
        .map(|(k, ch)| {
            let est = estimate_chirp_tdoa(&channels[0], ch, config);
            est.valid.then_some(PairDelay {
                a: 0,
                b: k + 1,
                delay_samples: est.offset_samples,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signals::gen_log_sweep;

    /// Tetrahedron-ish compact array, 0.12 m aperture.
    fn test_array() -> MicArray {
        MicArray::new(
            vec![
                [0.0, 0.0, 0.0],
                [0.12, 0.0, 0.0],
                [0.0, 0.12, 0.0],
                [0.0, 0.0, 0.12],
            ],
            48_000.0,
        )
        .expect("test array")
    }

    /// Exact free-field pair delays (all pairs) for a plane wave from `dir`.
    fn exact_pairs(array: &MicArray, dir: &[f64; 3]) -> Vec<PairDelay> {
        let mut pairs = Vec::new();
        for a in 0..array.len() {
            for b in (a + 1)..array.len() {
                pairs.push(PairDelay {
                    a,
                    b,
                    delay_samples: predicted_delay(array, dir, a, b).expect("indices"),
                });
            }
        }
        pairs
    }

    fn dir_from_az_el(az_deg: f64, el_deg: f64) -> [f64; 3] {
        let (az, el) = (az_deg.to_radians(), el_deg.to_radians());
        [el.cos() * az.cos(), el.cos() * az.sin(), el.sin()]
    }

    #[test]
    fn planar_array_preserves_mirror_ambiguity_in_any_orientation() {
        for positions in [
            vec![[0.0, 0.0, 0.0], [0.12, 0.0, 0.0], [0.0, 0.12, 0.0]],
            vec![
                [0.0, 0.0, 0.0],
                [0.12, 0.0, 0.0],
                [0.0, 0.12, 0.12],
                [0.12, 0.12, 0.12],
            ],
        ] {
            let array = MicArray::new(positions, 48_000.0).unwrap();
            let source = dir_from_az_el(30.0, 60.0);
            let pairs = exact_pairs(&array, &source);
            let estimate = estimate_doa_ls(&array, &pairs).unwrap();
            assert!(estimate.elevation_ambiguous);
            assert!(estimate.rms_residual_samples < 1e-9);
            assert!((estimate.direction.iter().map(|x| x * x).sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(
                tdoa_residuals(&array, &estimate.direction, &pairs)
                    .unwrap()
                    .iter()
                    .all(|residual| residual.abs() < 1e-9)
            );
        }
    }

    #[test]
    fn planar_broadside_is_ambiguous_and_impossible_delays_are_rejected() {
        let array = MicArray::new(
            vec![[0.0, 0.0, 0.0], [0.12, 0.0, 0.0], [0.0, 0.12, 0.0]],
            48_000.0,
        )
        .unwrap();
        let pairs = exact_pairs(&array, &[0.0, 0.0, -1.0]);
        let estimate = estimate_doa_ls(&array, &pairs).unwrap();
        assert!(estimate.elevation_ambiguous);
        assert!((estimate.direction[2].abs() - 1.0).abs() < 1e-12);
        let impossible: Vec<_> = pairs
            .into_iter()
            .map(|mut pair| {
                pair.delay_samples = 1000.0;
                pair
            })
            .collect();
        assert!(estimate_doa_ls(&array, &impossible).is_err());
    }

    #[test]
    fn doa_recovers_known_direction() {
        let array = test_array();
        let dir = dir_from_az_el(30.0, 10.0);
        let pairs = exact_pairs(&array, &dir);
        let est = estimate_doa_ls(&array, &pairs).expect("ls solves");
        assert!(
            (est.azimuth_deg - 30.0).abs() < 0.5,
            "az {}",
            est.azimuth_deg
        );
        assert!(
            (est.elevation_deg - 10.0).abs() < 0.5,
            "el {}",
            est.elevation_deg
        );
        assert!(est.rms_residual_samples < 1e-9);
        assert!((est.confidence - 1.0).abs() < 1e-9);
        assert!(!est.elevation_ambiguous);
    }

    #[test]
    fn doa_recovers_known_direction_across_sample_rates() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [0.12, 0.0, 0.0],
            [0.0, 0.12, 0.0],
            [0.0, 0.0, 0.12],
        ];
        for sr in [6_000.0, 12_000.0, 44_100.0, 48_000.0, 88_200.0, 96_000.0] {
            let array = MicArray::new(positions.clone(), sr).expect("test array");
            let dir = dir_from_az_el(30.0, 10.0);
            let pairs = exact_pairs(&array, &dir);
            let est = estimate_doa_ls(&array, &pairs).expect("ls solves");
            assert!(
                (est.azimuth_deg - 30.0).abs() < 0.5,
                "az {} at {sr} Hz",
                est.azimuth_deg
            );
            assert!(
                (est.elevation_deg - 10.0).abs() < 0.5,
                "el {} at {sr} Hz",
                est.elevation_deg
            );
            assert!(est.rms_residual_samples < 1e-9);
        }
    }

    #[test]
    fn doa_tolerates_measurement_noise() {
        let array = test_array();
        let dir = dir_from_az_el(-45.0, 25.0);
        // Deterministic ±0.03-sample perturbations (no RNG in tests).
        let pairs: Vec<PairDelay> = exact_pairs(&array, &dir)
            .into_iter()
            .enumerate()
            .map(|(k, mut p)| {
                p.delay_samples += 0.03 * ((k * 37 % 11) as f64 / 10.0 * 2.0 - 1.0);
                p
            })
            .collect();
        let est = estimate_doa_ls(&array, &pairs).expect("ls solves");
        assert!(
            (est.azimuth_deg - -45.0).abs() < 2.0,
            "az {}",
            est.azimuth_deg
        );
        assert!(
            (est.elevation_deg - 25.0).abs() < 2.0,
            "el {}",
            est.elevation_deg
        );
        assert!(est.confidence < 1.0 && est.confidence > 0.9);
    }

    #[test]
    fn doa_rejects_degenerate_inputs() {
        let two = MicArray::new(vec![[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], 48_000.0).expect("two");
        let pairs = vec![PairDelay {
            a: 0,
            b: 1,
            delay_samples: 1.0,
        }];
        assert!(estimate_doa_ls(&two, &pairs).is_err());
        let array = test_array();
        assert!(estimate_doa_ls(&array, &[]).is_err());
        // Collinear array: singular system.
        let line = MicArray::new(
            vec![[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
            48_000.0,
        )
        .expect("line");
        let line_pairs = vec![
            PairDelay {
                a: 0,
                b: 1,
                delay_samples: 1.0,
            },
            PairDelay {
                a: 1,
                b: 2,
                delay_samples: 1.0,
            },
        ];
        assert!(estimate_doa_ls(&line, &line_pairs).is_err());
    }

    #[test]
    fn geometry_check_passes_exact_and_fails_loudly_on_tape_error() {
        let array = test_array();
        let dir = dir_from_az_el(30.0, 10.0);
        let pairs = exact_pairs(&array, &dir);
        assert!(check_geometry(&array, &dir, &pairs, 0.01).is_ok());
        // 5 cm tape error on mic 1's x position.
        let bad = MicArray::new(
            vec![
                [0.0, 0.0, 0.0],
                [0.17, 0.0, 0.0],
                [0.0, 0.12, 0.0],
                [0.0, 0.0, 0.12],
            ],
            48_000.0,
        )
        .expect("bad array");
        let err = check_geometry(&bad, &dir, &pairs, 0.5).expect_err("must fail loudly");
        assert!(err.contains("re-measure tape"), "{err}");
    }

    #[test]
    fn scale_calibration_recovers_tape_scale() {
        let array = test_array();
        let dir = dir_from_az_el(30.0, 10.0);
        // Measurements from a 3% oversized array.
        let big = MicArray::new(
            array
                .positions
                .iter()
                .map(|p| [p[0] * 1.03, p[1] * 1.03, p[2] * 1.03])
                .collect(),
            48_000.0,
        )
        .expect("big array");
        let pairs = exact_pairs(&big, &dir);
        let (scale, rms) = calibrate_geometry_scale(&array, &dir, &pairs).expect("calibrates");
        assert!((scale - 1.03).abs() < 1e-9, "scale {scale}");
        assert!(rms < 1e-9);
    }

    #[test]
    fn pairwise_tdoa_end_to_end_with_integer_delay() {
        // Oracle-free: channel 1 is channel 0 shifted by exactly 5 samples.
        let ch0 = gen_log_sweep(500.0, 12_000.0, 0.9, 48_000, 0.5);
        let mut ch1 = vec![0.0; ch0.len()];
        ch1[5..].copy_from_slice(&ch0[..ch0.len() - 5]);
        let channels = vec![ch0, ch1];
        let config = TdoaConfig::default();
        let pairs = pairwise_tdoas_vs_first(&channels, &config);
        assert_eq!(pairs.len(), 1);
        let pair = pairs[0].expect("valid pair");
        assert_eq!((pair.a, pair.b), (0, 1));
        assert!(
            (pair.delay_samples - 5.0).abs() < 0.05,
            "got {}",
            pair.delay_samples
        );
        // Pure noise second channel: None, never a fabricated zero.
        let noise = crate::signals::gen_white_noise_seeded(0.5, 48_000, 0.5, 3);
        let channels = vec![channels[0].clone(), noise];
        assert!(pairwise_tdoas_vs_first(&channels, &config)[0].is_none());
    }

    #[test]
    fn delay_and_sum_scores_true_direction_highest() {
        // Four mics 0.1 m apart on x (0.1/343·48000 ≈ 14 samples apart);
        // impulse arrives from −x so mic m lags by 14·m samples.
        let array = MicArray::new(
            vec![
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.3, 0.0, 0.0],
            ],
            48_000.0,
        )
        .expect("line array");
        let channels: Vec<Vec<f32>> = (0..4)
            .map(|m| {
                let mut ch = vec![0.0f32; 1000];
                ch[100 + 14 * m] = 1.0;
                ch
            })
            .collect();
        let true_dir = [-1.0, 0.0, 0.0];
        let wrong_dir = [1.0, 0.0, 0.0];
        let good = delay_and_sum_power(&channels, &array, &true_dir).expect("scores");
        let bad = delay_and_sum_power(&channels, &array, &wrong_dir).expect("scores");
        assert!(good > 3.0 * bad, "good {good} bad {bad}");
    }
}

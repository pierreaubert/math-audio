//! Faithful Kautz room correction: dry-plus-bank transfer with checked APIs.
//!
//! The legacy [`KautzFilter`](super::kautz_filter::KautzFilter) is a bank-only
//! processor (`y = Σ gain·φ(x)`) whose `optimize_gains` fits normalized basis
//! magnitudes. That fit does not implement the playback transfer. The real
//! playback chain (see `kautz_runtime.rs` in the EQ plugin) is dry-plus-bank:
//!
//! ```text
//! H(f) = 1 + Σ_k g_k · B_k(f)
//! ```
//!
//! where `B_k` is the ordered allpass-coupled Kautz basis of section `k` and
//! `g_k` is a signed, dimensionless linear coefficient — not dB, not
//! peak-normalized. [`KautzCorrection`] implements exactly this transfer with
//! the same ordered basis as sample processing, and documents the bank-only
//! versus correction response at every boundary ([`bank_response`] vs
//! [`correction_response`]).
//!
//! Legacy bank-only processing is untouched and stays compatible: nothing
//! here adds unity to the existing API silently.
//!
//! [`bank_response`]: KautzCorrection::bank_response
//! [`correction_response`]: KautzCorrection::correction_response

use super::kautz_section::KautzSection;
use crate::traits::FilterFloat;
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use thiserror::Error;

/// Minimum supported stimulus-clock sample rate in Hz.
pub const MIN_SUPPORTED_RATE_HZ: f64 = 2_000.0;
/// Maximum supported sample rate in Hz.
pub const MAX_SUPPORTED_RATE_HZ: f64 = 768_000.0;
/// Minimum Q factor (mirrors the section's internal clamp boundary, but this
/// API refuses instead of clamping so fits stay exactly reproducible).
pub const MIN_Q_FACTOR: f64 = 0.1;
/// Largest stable pole radius accepted without refusal.
pub const MAX_POLE_RADIUS: f64 = 0.9999;
/// Canonical serialization format tag.
pub const KAUTZ_CORRECTION_FORMAT: &str = "kautz-correction-v1";

/// Structured refusal for every checked Kautz correction API.
///
/// Fallible constructors and setters return `Err` before touching any live
/// state, so a failure never leaves a partially changed usable bank.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum KautzError {
    /// The section bank is empty; at least one (freq, Q) pair is required.
    #[error("Kautz bank is empty (need at least one section)")]
    EmptyBank,
    /// Array lengths disagree.
    #[error("{what}: expected {expected} items, got {got}")]
    LengthMismatch {
        /// Which arrays disagree.
        what: &'static str,
        /// Required length.
        expected: usize,
        /// Provided length.
        got: usize,
    },
    /// A value that must be finite is not.
    #[error("{what} is not finite at index {index}: {value}")]
    NonFinite {
        /// Which value is bad.
        what: &'static str,
        /// Position in the input array (`usize::MAX` for scalars).
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// A frequency grid is not strictly increasing.
    #[error("frequency grid is not strictly increasing at index {index}")]
    UnorderedGrid {
        /// First index violating strict increase.
        index: usize,
    },
    /// Sample rate outside the supported range.
    #[error("unsupported sample rate: {rate} Hz (supported {min}..={max} Hz)")]
    UnsupportedRate {
        /// The rejected rate.
        rate: f64,
        /// Supported minimum.
        min: f64,
        /// Supported maximum.
        max: f64,
    },
    /// A pole sits at or above Nyquist.
    #[error("pole {freq} Hz at index {index} is not sub-Nyquist (Nyquist {nyquist} Hz)")]
    PoleAboveNyquist {
        /// The rejected pole frequency.
        freq: f64,
        /// Nyquist of the target rate.
        nyquist: f64,
        /// Position in the mode list.
        index: usize,
    },
    /// Q factor below the reproducible minimum.
    #[error("Q factor {q} at index {index} must be finite and >= {minimum}")]
    NonPositiveQ {
        /// The rejected Q.
        q: f64,
        /// Position in the mode list.
        index: usize,
        /// Minimum accepted Q.
        minimum: f64,
    },
    /// Pole radius too close to the unit circle for stable evaluation.
    #[error("pole radius {radius} at index {index} exceeds stable maximum {maximum}")]
    UnstableRadius {
        /// The rejected radius.
        radius: f64,
        /// Position in the mode list.
        index: usize,
        /// Largest accepted radius.
        maximum: f64,
    },
    /// A required work budget is zero.
    #[error("zero work budget for {what}")]
    ZeroBudget {
        /// Which budget is zero.
        what: &'static str,
    },
    /// Bounds or constraints are mutually incompatible.
    #[error("incompatible bounds: {reason}")]
    IncompatibleBounds {
        /// Why the bounds cannot be satisfied together.
        reason: String,
    },
    /// Model configuration this implementation does not support.
    #[error("unsupported model configuration: {reason}")]
    UnsupportedModel {
        /// What is unsupported.
        reason: String,
    },
}

/// Fallible result for the checked Kautz correction API.
pub type KautzResult<T> = Result<T, KautzError>;

/// Pole radius for `(freq_hz, q)` at `srate`, before any clamping.
fn pole_radius(freq_hz: f64, q: f64, srate: f64) -> f64 {
    (-PI * freq_hz / (q * srate)).exp()
}

fn check_rate(srate: f64) -> KautzResult<()> {
    if !srate.is_finite() {
        return Err(KautzError::NonFinite {
            what: "sample rate",
            index: usize::MAX,
            value: srate,
        });
    }
    if !(MIN_SUPPORTED_RATE_HZ..=MAX_SUPPORTED_RATE_HZ).contains(&srate) {
        return Err(KautzError::UnsupportedRate {
            rate: srate,
            min: MIN_SUPPORTED_RATE_HZ,
            max: MAX_SUPPORTED_RATE_HZ,
        });
    }
    Ok(())
}

fn check_mode(freq_hz: f64, q: f64, srate: f64, index: usize) -> KautzResult<()> {
    if !freq_hz.is_finite() || !q.is_finite() {
        return Err(KautzError::NonFinite {
            what: "mode (freq, Q)",
            index,
            value: if freq_hz.is_finite() { q } else { freq_hz },
        });
    }
    if freq_hz <= 0.0 || freq_hz >= srate / 2.0 {
        return Err(KautzError::PoleAboveNyquist {
            freq: freq_hz,
            nyquist: srate / 2.0,
            index,
        });
    }
    if q < MIN_Q_FACTOR {
        return Err(KautzError::NonPositiveQ {
            q,
            index,
            minimum: MIN_Q_FACTOR,
        });
    }
    let radius = pole_radius(freq_hz, q, srate);
    if !radius.is_finite() || radius > MAX_POLE_RADIUS {
        return Err(KautzError::UnstableRadius {
            radius,
            index,
            maximum: MAX_POLE_RADIUS,
        });
    }
    Ok(())
}

/// Dry-plus-bank Kautz room correction.
///
/// Sample processing realizes `y[n] = x[n] + Σ_k (strength · gain_k) · φ_k(x[n])`
/// with the ordered allpass-coupled basis, so the streamed transfer equals
/// [`correction_response`](Self::correction_response). All sample/block
/// processing is bounded and allocation-free; per-channel state isolation
/// comes from using one instance per channel.
///
/// Gains are signed dimensionless linear coefficients. A strength scalar
/// `s` gives `H_s = 1 + s·(H_full − 1)` without refitting.
pub struct KautzCorrection<T: FilterFloat = f64> {
    sections: Vec<KautzSection<T>>,
    /// Rate-independent section identity: (pole frequency Hz, Q).
    modes: Vec<(f64, f64)>,
    srate: f64,
    gains: Vec<T>,
    /// Cached `strength · gains` for the allocation-free sample path.
    effective: Vec<T>,
    strength: f64,
    bypass: bool,
}

impl<T: FilterFloat> KautzCorrection<T> {
    /// Build a correction bank from `(frequency_hz, q_factor)` room modes.
    ///
    /// Gains start at zero (unity correction), strength at 1, bypass off.
    /// Every mode is validated before anything is stored; invalid input
    /// returns `Err` and constructs nothing.
    pub fn new(modes: &[(f64, f64)], sample_rate: f64) -> KautzResult<Self> {
        check_rate(sample_rate)?;
        if modes.is_empty() {
            return Err(KautzError::EmptyBank);
        }
        for (index, &(freq, q)) in modes.iter().enumerate() {
            check_mode(freq, q, sample_rate, index)?;
        }
        let zero = T::zero();
        let sections = modes
            .iter()
            .map(|&(freq, q)| {
                KautzSection::new(
                    T::from_f64(freq).unwrap_or(zero),
                    T::from_f64(q).unwrap_or(zero),
                    zero,
                    T::from_f64(sample_rate).unwrap_or(zero),
                )
            })
            .collect::<Vec<_>>();
        // Validated finite above, representable by construction: rate and
        // modes sit far inside f32 range, so conversion cannot fail.
        let n = sections.len();
        Ok(Self {
            sections,
            modes: modes.to_vec(),
            srate: sample_rate,
            gains: vec![zero; n],
            effective: vec![zero; n],
            strength: 1.0,
            bypass: false,
        })
    }

    /// Number of sections (never zero: [`new`](Self::new) refuses empties).
    pub fn len(&self) -> usize {
        self.sections.len()
    }

    /// Always false; kept so generic code can probe emptiness.
    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }

    /// Stimulus-clock sample rate in Hz.
    pub fn sample_rate(&self) -> f64 {
        self.srate
    }

    /// Section identity as `(pole frequency Hz, Q)` pairs, in order.
    pub fn modes(&self) -> &[(f64, f64)] {
        &self.modes
    }

    /// Current signed linear gains (without strength scaling).
    pub fn gains(&self) -> Vec<f64> {
        self.gains
            .iter()
            .map(|&g| g.to_f64().unwrap_or(f64::NAN))
            .collect()
    }

    /// Current strength scalar.
    pub fn strength(&self) -> f64 {
        self.strength
    }

    /// Whether bypass is engaged (output equals input; states keep tracking).
    pub fn is_bypass(&self) -> bool {
        self.bypass
    }

    /// Replace all gains atomically: length and finiteness are checked first,
    /// so a refusal leaves existing gains untouched.
    pub fn set_gains(&mut self, gains: &[f64]) -> KautzResult<()> {
        if gains.len() != self.sections.len() {
            return Err(KautzError::LengthMismatch {
                what: "gains vs sections",
                expected: self.sections.len(),
                got: gains.len(),
            });
        }
        let mut converted = Vec::with_capacity(gains.len());
        for (index, &g) in gains.iter().enumerate() {
            if !g.is_finite() {
                return Err(KautzError::NonFinite {
                    what: "gain",
                    index,
                    value: g,
                });
            }
            let t = T::from_f64(g).unwrap_or(T::zero());
            if !t.is_finite() {
                return Err(KautzError::NonFinite {
                    what: "gain (not representable at this float width)",
                    index,
                    value: g,
                });
            }
            converted.push(t);
        }
        self.gains = converted;
        self.refresh_effective();
        Ok(())
    }

    /// Set the strength scalar `s` giving `H_s = 1 + s·(H_full − 1)`.
    ///
    /// Any finite value is accepted (0 = unity, 1 = full correction);
    /// refusal on non-finite input leaves the previous strength in place.
    pub fn set_strength(&mut self, strength: f64) -> KautzResult<()> {
        if !strength.is_finite() {
            return Err(KautzError::NonFinite {
                what: "strength",
                index: usize::MAX,
                value: strength,
            });
        }
        self.strength = strength;
        self.refresh_effective();
        Ok(())
    }

    /// Engage or release bypass. Bypassed processing outputs the dry input
    /// while section states keep tracking, so toggling is click-free.
    /// [`reset`](Self::reset) still zeroes the tracked state.
    pub fn set_bypass(&mut self, bypass: bool) {
        self.bypass = bypass;
    }

    fn refresh_effective(&mut self) {
        let s = T::from_f64(self.strength).unwrap_or(T::zero());
        for ((eff, g), section) in self
            .effective
            .iter_mut()
            .zip(self.gains.iter())
            .zip(self.sections.iter_mut())
        {
            *eff = s * *g;
            // Mirror for readers of the legacy section layout; the
            // canonical store is `gains` (raw) plus `effective` (scaled).
            section.gain = *g;
        }
    }

    /// Process one sample: dry plus the weighted ordered basis chain.
    ///
    /// Bounded and allocation-free. With bypass engaged the output equals
    /// the input (states keep tracking underneath).
    #[inline]
    pub fn process(&mut self, sample: T) -> T {
        let mut chain = sample;
        let mut wet = T::zero();
        for (section, &gain) in self.sections.iter_mut().zip(self.effective.iter()) {
            let (basis, ap_out) = section.process_section(chain);
            wet += gain * basis;
            chain = ap_out;
        }
        if self.bypass { sample } else { sample + wet }
    }

    /// Process a block in place, sample by sample. Any length is accepted,
    /// including empty and partial trailing blocks.
    pub fn process_block(&mut self, buf: &mut [T]) {
        for sample in buf.iter_mut() {
            *sample = self.process(*sample);
        }
    }

    /// Zero all section states. Gains, strength, modes and rate are kept.
    pub fn reset(&mut self) {
        for section in self.sections.iter_mut() {
            section.reset();
        }
    }

    /// Rebuild the bank at a new sample rate, keeping modes, gains and
    /// strength. Poles are revalidated at the new rate *before* any live
    /// state is replaced: refusal leaves the running instance untouched,
    /// and success resets state (old states belong to the old rate).
    pub fn set_sample_rate(&mut self, sample_rate: f64) -> KautzResult<()> {
        check_rate(sample_rate)?;
        for (index, &(freq, q)) in self.modes.iter().enumerate() {
            check_mode(freq, q, sample_rate, index)?;
        }
        let zero = T::zero();
        let sections = self
            .modes
            .iter()
            .zip(self.gains.iter())
            .map(|(&(freq, q), &gain)| {
                let mut section = KautzSection::new(
                    T::from_f64(freq).unwrap_or(zero),
                    T::from_f64(q).unwrap_or(zero),
                    gain,
                    T::from_f64(sample_rate).unwrap_or(zero),
                );
                section.reset();
                section
            })
            .collect::<Vec<_>>();
        self.sections = sections;
        self.srate = sample_rate;
        Ok(())
    }

    /// Serialize to the versioned plugin-bank spec (canonical field names).
    pub fn to_spec(&self) -> KautzBankSpec {
        KautzBankSpec {
            format: KAUTZ_CORRECTION_FORMAT.to_string(),
            sample_rate_hz: self.srate,
            strength: self.strength,
            bypass: self.bypass,
            sections: self
                .modes
                .iter()
                .zip(self.gains())
                .map(|(&(freq_hz, q_factor), gain)| KautzSectionSpec {
                    freq_hz,
                    q_factor,
                    gain,
                })
                .collect(),
        }
    }

    /// Rebuild from a plugin-bank spec. The spec's own rate is adopted after
    /// the same validation as [`new`](Self::new); unknown formats and unit
    /// violations are refused before anything is constructed.
    pub fn from_spec(spec: &KautzBankSpec) -> KautzResult<Self> {
        if spec.format != KAUTZ_CORRECTION_FORMAT {
            return Err(KautzError::UnsupportedModel {
                reason: format!(
                    "unknown bank format '{}' (this build reads '{}')",
                    spec.format, KAUTZ_CORRECTION_FORMAT
                ),
            });
        }
        let modes: Vec<(f64, f64)> = spec
            .sections
            .iter()
            .map(|s| (s.freq_hz, s.q_factor))
            .collect();
        let mut bank = Self::new(&modes, spec.sample_rate_hz)?;
        let gains: Vec<f64> = spec.sections.iter().map(|s| s.gain).collect();
        bank.set_gains(&gains)?;
        bank.set_strength(spec.strength)?;
        bank.set_bypass(spec.bypass);
        Ok(bank)
    }
}

impl KautzCorrection<f64> {
    /// Bank-only complex response `Σ_k g_k·B_k(f)` with strength applied.
    ///
    /// This is the legacy processing path *without* dry: useful for
    /// diagnostics, never the playback transfer.
    pub fn bank_response(&self, freq_hz: f64) -> Complex<f64> {
        let mut total = Complex::new(0.0, 0.0);
        let mut chain = Complex::new(1.0, 0.0);
        for (section, &gain) in self.sections.iter().zip(self.gains.iter()) {
            let basis = section.basis_response(freq_hz, self.srate, chain);
            total += basis * (self.strength * gain);
            chain *= section.allpass_response(freq_hz, self.srate);
        }
        total
    }

    /// Declared playback transfer `H(f) = 1 + bank(f)`: the response the
    /// streamed dry-plus-bank processing realizes.
    pub fn correction_response(&self, freq_hz: f64) -> Complex<f64> {
        Complex::new(1.0, 0.0) + self.bank_response(freq_hz)
    }

    /// Playback magnitude in dB, `20·log10|H(f)|`.
    pub fn correction_db(&self, freq_hz: f64) -> f64 {
        let mag = self.correction_response(freq_hz).norm().max(1e-12);
        20.0 * mag.log10()
    }
}

/// One serializable section: pole identity plus its signed linear gain.
///
/// Gains are dimensionless linear coefficients. There is deliberately no dB
/// alias: a dB-valued gain would silently change the transfer, so legacy
/// exporters must convert to linear before writing this field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KautzSectionSpec {
    /// Pole frequency in Hz.
    #[serde(alias = "freq", alias = "frequency", alias = "f0")]
    pub freq_hz: f64,
    /// Pole Q factor.
    #[serde(alias = "q", alias = "Q", alias = "quality")]
    pub q_factor: f64,
    /// Signed linear gain (dimensionless).
    #[serde(alias = "weight")]
    pub gain: f64,
}

/// Versioned plugin-bank spec for the dry-plus-bank correction.
///
/// Carries everything needed to rebuild an identical instance at any
/// supported rate: pole identities are rate-independent `(freq, Q)` pairs,
/// so a rate change rebuilds radii without touching gains or strength.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KautzBankSpec {
    /// Format tag; must equal [`KAUTZ_CORRECTION_FORMAT`].
    pub format: String,
    /// Rate the radii below were built for (Hz).
    #[serde(alias = "sample_rate")]
    pub sample_rate_hz: f64,
    /// Strength scalar (defaults to full correction when absent).
    #[serde(default = "default_strength")]
    pub strength: f64,
    /// Bypass flag.
    #[serde(default)]
    pub bypass: bool,
    /// Ordered sections; order matters (allpass chain + zero-weight
    /// sections must round-trip untouched).
    pub sections: Vec<KautzSectionSpec>,
}

fn default_strength() -> f64 {
    1.0
}

/// Parse a JSON plugin-bank spec, mapping syntax failures to a structured
/// model refusal instead of a serialization error.
pub fn parse_bank_spec(json: &str) -> KautzResult<KautzBankSpec> {
    serde_json::from_str(json).map_err(|err| KautzError::UnsupportedModel {
        reason: format!("bank spec is not valid {KAUTZ_CORRECTION_FORMAT} JSON: {err}"),
    })
}

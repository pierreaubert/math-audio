// ============================================================================
// ADAA — Antiderivative Anti-Aliasing for Nonlinear Processors
// ============================================================================
//
// Implements 1st-order and 2nd-order ADAA as described in:
//   Parker et al., "Reducing the Aliasing of Nonlinear Waveshaping Using
//   Continuous-Time Convolution" (DAFx-16)
//
// The core idea: instead of evaluating f(x) directly (which aliases),
// compute (AD1(x[n]) - AD1(x[n-1])) / (x[n] - x[n-1]) where AD1 is the
// antiderivative of f. This effectively applies a moving-average anti-alias
// filter across the nonlinearity.
//
// HARD RULES:
// - No allocations in process()
// - f64 intermediates for numerical stability in the division
// - Fallback to f(x_mid) when consecutive samples are near-identical

const EPSILON: f64 = 1e-5;

// ============================================================================
// First-order ADAA
// ============================================================================

/// First-order Antiderivative Anti-Aliasing processor.
///
/// Wraps a memoryless nonlinearity `f(x)` and its first antiderivative `AD1(x)`.
/// Produces significantly less aliasing than naive evaluation, at minimal CPU cost.
#[derive(Debug, Clone, Copy)]
pub struct Adaa1 {
    /// The nonlinearity f(x)
    f: fn(f64) -> f64,
    /// First antiderivative AD1(x) = integral of f(x)
    ad1: fn(f64) -> f64,
    /// Previous input sample
    x_prev: f64,
}

impl Adaa1 {
    /// Create a new first-order ADAA processor.
    pub fn new(f: fn(f64) -> f64, ad1: fn(f64) -> f64) -> Self {
        Self {
            f,
            ad1,
            x_prev: 0.0,
        }
    }

    /// Process one sample.
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let x = x as f64;
        let x_prev = self.x_prev;
        self.x_prev = x;

        let diff = x - x_prev;
        if diff.abs() < EPSILON {
            // Near-identical consecutive samples: fallback to f(midpoint)
            (self.f)((x + x_prev) * 0.5) as f32
        } else {
            // ADAA1: (AD1(x) - AD1(x_prev)) / (x - x_prev)
            (((self.ad1)(x) - (self.ad1)(x_prev)) / diff) as f32
        }
    }

    /// Process a block of samples in-place.
    #[inline]
    pub fn process_block(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process(*sample);
        }
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.x_prev = 0.0;
    }
}

// ============================================================================
// Second-order ADAA
// ============================================================================

/// Second-order Antiderivative Anti-Aliasing processor.
///
/// Uses the second antiderivative for even better alias suppression,
/// at the cost of one additional sample of latency and slight HF rolloff.
#[derive(Debug, Clone, Copy)]
pub struct Adaa2 {
    /// The nonlinearity f(x)
    f: fn(f64) -> f64,
    /// First antiderivative
    ad1: fn(f64) -> f64,
    /// Second antiderivative AD2(x) = integral of AD1(x)
    ad2: fn(f64) -> f64,
    /// Two previous samples
    x_prev1: f64,
    x_prev2: f64,
    /// Previous D1 value for the second-order finite difference
    d1_prev: f64,
}

impl Adaa2 {
    pub fn new(f: fn(f64) -> f64, ad1: fn(f64) -> f64, ad2: fn(f64) -> f64) -> Self {
        Self {
            f,
            ad1,
            ad2,
            x_prev1: 0.0,
            x_prev2: 0.0,
            d1_prev: 0.0,
        }
    }

    #[inline]
    fn compute_d1(&self, x: f64, x_prev: f64) -> f64 {
        let diff = x - x_prev;
        if diff.abs() < EPSILON {
            (self.ad1)((x + x_prev) * 0.5)
        } else {
            ((self.ad2)(x) - (self.ad2)(x_prev)) / diff
        }
    }

    /// Process one sample. Introduces one sample of latency (see the struct
    /// docs: ADAA2's centered second difference reproduces x[n-1] for a
    /// linear input).
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let x = x as f64;
        let x_prev1 = self.x_prev1;

        let d1 = self.compute_d1(x, x_prev1);

        let diff = (x - self.x_prev2) * 0.5;
        let result = if diff.abs() < EPSILON {
            (self.f)((x + x_prev1 + self.x_prev2) / 3.0)
        } else {
            (d1 - self.d1_prev) / diff
        };

        self.x_prev2 = self.x_prev1;
        self.x_prev1 = x;
        self.d1_prev = d1;

        result as f32
    }

    pub fn process_block(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process(*sample);
        }
    }

    pub fn reset(&mut self) {
        self.x_prev1 = 0.0;
        self.x_prev2 = 0.0;
        self.d1_prev = 0.0;
    }
}

// ============================================================================
// Pre-built ADAA processors for common saturation functions
// ============================================================================

// --- tanh ---
fn tanh_f(x: f64) -> f64 {
    x.tanh()
}
fn tanh_ad1(x: f64) -> f64 {
    // AD1(tanh(x)) = ln(cosh(x))
    // For numerical stability: ln(cosh(x)) = |x| + ln(1 + e^(-2|x|)) - ln(2)
    let abs_x = x.abs();
    abs_x + (-2.0 * abs_x).exp().ln_1p() - std::f64::consts::LN_2
}
fn tanh_ad2(x: f64) -> f64 {
    // AD2(tanh(x)) = integral of ln(cosh(x)).  tanh is odd, so its
    // second antiderivative can be chosen odd as well.  Evaluating the
    // positive branch avoids exp() overflow for the large negative inputs
    // that a high-drive realtime model may legitimately receive.
    let sign = x.signum();
    let magnitude = x.abs();
    let z = (-2.0 * magnitude).exp();
    let li2 = dilog_neg(z);
    sign * (0.5 * (magnitude * magnitude + li2) - std::f64::consts::LN_2 * magnitude
        + std::f64::consts::PI * std::f64::consts::PI / 24.0)
}

/// Approximate Li_2(-z) for z >= 0 using a rapidly converging series.
fn dilog_neg(z: f64) -> f64 {
    if z < 1e-15 {
        return 0.0;
    }
    if (1.0 - z).abs() < 1e-12 {
        return -std::f64::consts::PI * std::f64::consts::PI / 12.0;
    }
    if z <= 1.0 {
        // The direct series sum_{k=1}^{inf} (-z)^k / k^2 converges only
        // linearly (ratio z) and is useless as z -> 1. Use the Landen
        // identity instead:
        //   Li_2(-z) = -Li_2(w) - 0.5*ln(1+z)^2,  w = z / (1 + z)
        // w lies in (0, 1/2], so the series in w converges geometrically
        // with ratio <= 1/2 regardless of how close z is to 1.
        let w = z / (1.0 + z);
        let ln1pz = z.ln_1p();
        let mut sum = 0.0;
        let mut w_pow = 1.0;
        for k in 1..=64 {
            w_pow *= w;
            let term = w_pow / (k * k) as f64;
            sum += term;
            if term < 1e-17 * w {
                break;
            }
        }
        -sum - 0.5 * ln1pz * ln1pz
    } else {
        // For z > 1, use identity: Li_2(-z) = -Li_2(-1/z) - pi^2/6 - 0.5*ln(z)^2
        let ln_z = z.ln();
        -dilog_neg(1.0 / z) - std::f64::consts::PI * std::f64::consts::PI / 6.0 - 0.5 * ln_z * ln_z
    }
}

// --- soft clip: x / (1 + |x|) ---
fn softclip_f(x: f64) -> f64 {
    x / (1.0 + x.abs())
}
fn softclip_ad1(x: f64) -> f64 {
    // integral of x/(1+|x|) dx
    // For x >= 0: integral of x/(1+x) = x - ln(1+x) + C
    // For x < 0: integral of x/(1-x) = -x - ln(1-x) + C  (i.e., -(|x| - ln(1+|x|)))
    // Combined with continuity at x=0 (both give 0):
    // AD1(x) = |x| - ln(1 + |x|)    (always positive, symmetric)
    let abs_x = x.abs();
    abs_x - (1.0 + abs_x).ln()
}
fn softclip_ad2(x: f64) -> f64 {
    // AD2 = integral of AD1. Since f(x) is odd, AD1 is even, AD2 must be odd.
    // For x >= 0: integral of (x - ln(1+x)) = x^2/2 - (1+x)*ln(1+x) + x + C
    // With C chosen so AD2(0) = 0: C = 0 (since 0 - 1*ln(1) + 0 = 0).
    let abs_x = x.abs();
    let one_plus = 1.0 + abs_x;
    let magnitude = 0.5 * abs_x * abs_x - one_plus * one_plus.ln() + abs_x;
    if x >= 0.0 { magnitude } else { -magnitude }
}

// --- hard clip: clamp to [-1, 1] ---
fn hardclip_f(x: f64) -> f64 {
    x.clamp(-1.0, 1.0)
}
fn hardclip_ad1(x: f64) -> f64 {
    // integral of clamp(x, -1, 1)
    // For |x| <= 1: x^2/2
    // For x > 1:  x - 0.5  (value at x=1: 0.5, derivative = 1)
    // For x < -1: -x - 0.5 (value at x=-1: 0.5, derivative = -1)
    // Note: AD1 is even-symmetric for an odd function
    if (-1.0..=1.0).contains(&x) {
        x * x * 0.5
    } else if x > 1.0 {
        x - 0.5
    } else {
        -x - 0.5
    }
}
fn hardclip_ad2(x: f64) -> f64 {
    // integral of hardclip_ad1(x). Since f is odd, AD1 is even, AD2 must be odd.
    // For |x| <= 1: x^3/6  (odd)
    // For x > 1: x^2/2 - x/2 + 1/6
    // For x < -1: -(|x|^2/2 - |x|/2 + 1/6)  (odd symmetry)
    if (-1.0..=1.0).contains(&x) {
        x * x * x / 6.0
    } else if x > 1.0 {
        x * x * 0.5 - 0.5 * x + 1.0 / 6.0
    } else {
        // x < -1: negate the positive formula evaluated at |x|
        let abs_x = x.abs();
        -(abs_x * abs_x * 0.5 - 0.5 * abs_x + 1.0 / 6.0)
    }
}

/// Create a first-order ADAA processor for `tanh(x)` (soft saturation).
pub fn adaa1_tanh() -> Adaa1 {
    Adaa1::new(tanh_f, tanh_ad1)
}

/// Create a first-order ADAA processor for `x/(1+|x|)` (soft clip).
pub fn adaa1_softclip() -> Adaa1 {
    Adaa1::new(softclip_f, softclip_ad1)
}

/// Create a first-order ADAA processor for `clamp(x, -1, 1)` (hard clip).
pub fn adaa1_hardclip() -> Adaa1 {
    Adaa1::new(hardclip_f, hardclip_ad1)
}

/// Create a second-order ADAA processor for `tanh(x)`.
pub fn adaa2_tanh() -> Adaa2 {
    Adaa2::new(tanh_f, tanh_ad1, tanh_ad2)
}

/// Create a second-order ADAA processor for `x/(1+|x|)`.
pub fn adaa2_softclip() -> Adaa2 {
    Adaa2::new(softclip_f, softclip_ad1, softclip_ad2)
}

/// Create a second-order ADAA processor for `clamp(x, -1, 1)`.
pub fn adaa2_hardclip() -> Adaa2 {
    Adaa2::new(hardclip_f, hardclip_ad1, hardclip_ad2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tanh_ad1_identity() {
        // Verify AD1(tanh) = ln(cosh(x)) at several points
        for &x in &[0.0_f64, 0.5, 1.0, 2.0, -1.0, -3.0] {
            let expected = x.cosh().ln();
            let actual = tanh_ad1(x);
            assert!(
                (actual - expected).abs() < 1e-10,
                "tanh_ad1({x}): expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn test_softclip_ad1_identity() {
        // Verify via numerical differentiation: d/dx AD1(x) ≈ |f(x)|
        // AD1 is even (|x| - ln(1+|x|)), so d/dx AD1 = sign(x) * f(|x|)
        // which equals f(x) since f(x) = x/(1+|x|) is odd.
        // Wait — let's verify: AD1(x) = |x| - ln(1+|x|)
        // d/dx AD1 for x > 0: 1 - 1/(1+x) = x/(1+x) = f(x) ✓
        // d/dx AD1 for x < 0: -1 + 1/(1+|x|) = -|x|/(1+|x|) = x/(1+|x|) = f(x) ✓
        let h = 1e-7;
        for &x in &[0.1, 0.5, 1.0, 2.0, -0.5, -2.0] {
            let numerical_derivative = (softclip_ad1(x + h) - softclip_ad1(x - h)) / (2.0 * h);
            let actual = softclip_f(x);
            assert!(
                (numerical_derivative - actual).abs() < 1e-4,
                "softclip AD1 derivative mismatch at x={x}: d/dx AD1={numerical_derivative}, f(x)={actual}"
            );
        }
    }

    #[test]
    fn test_adaa1_tanh_basic() {
        let mut adaa = adaa1_tanh();
        // Process a ramp — should produce tanh-like output without aliasing
        let mut outputs = Vec::new();
        for i in 0..100 {
            let x = (i as f32 - 50.0) / 25.0; // -2.0 to +2.0
            outputs.push(adaa.process(x));
        }
        // Output should be bounded by [-1, 1]
        for &y in &outputs {
            assert!(y.abs() <= 1.01, "Output out of bounds: {y}");
        }
    }

    #[test]
    fn test_adaa1_reduces_aliasing() {
        // High-drive 15 kHz sine through tanh: odd harmonics (45k, 75k, 105k,
        // ...) exceed Nyquist (24 kHz) and fold back into the audible band as
        // alias products (3 kHz, 9 kHz, ...). ADAA1 must strongly suppress
        // those below-fundamental alias products compared to naive evaluation.
        use rustfft::FftPlanner;
        use rustfft::num_complex::Complex;

        let sr = 48000.0;
        let n = 4096;
        // 15000 Hz = bin 1280 exactly, so the fundamental does not leak.
        let freq = 15000.0;
        let drive = 5.0; // heavy drive creates harmonics that alias

        // Naive
        let naive_output: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (drive * (2.0 * std::f64::consts::PI * freq * t).sin()).tanh() as f32
            })
            .collect();

        // ADAA1
        let mut adaa = adaa1_tanh();
        let adaa_output: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                let x = (drive * (2.0 * std::f64::consts::PI * freq * t).sin()) as f32;
                adaa.process(x)
            })
            .collect();

        // FFT both and compare energy in the alias band: everything below the
        // fundamental (bins 1..1280, i.e. ~11.7 Hz .. 15 kHz). With a clean
        // anti-aliased tanh this band only contains aliased harmonic energy.
        let fft = FftPlanner::new().plan_fft_forward(n);
        let alias_band_energy = |signal: &[f32]| -> f64 {
            let mut buf: Vec<Complex<f64>> = signal
                .iter()
                .map(|&s| Complex::new(s as f64, 0.0))
                .collect();
            fft.process(&mut buf);
            let fundamental_bin = 1280; // 15000 Hz
            buf[1..fundamental_bin].iter().map(|c| c.norm_sqr()).sum()
        };
        let naive_alias = alias_band_energy(&naive_output);
        let adaa_alias = alias_band_energy(&adaa_output);

        assert!(naive_alias > 0.0, "naive tanh must produce alias energy");
        assert!(
            adaa_alias < naive_alias * 0.2,
            "ADAA1 should suppress alias-band energy by at least 5x: naive={naive_alias}, adaa={adaa_alias}"
        );
    }

    #[test]
    fn test_adaa1_reset() {
        let mut adaa = adaa1_tanh();
        adaa.process(1.0);
        adaa.reset();
        // After reset, processing 0 should give near-0
        let out = adaa.process(0.0);
        assert!(out.abs() < 0.01);
    }

    #[test]
    fn test_adaa2_tanh_bounded() {
        let mut adaa = adaa2_tanh();
        for i in 0..200 {
            let x = (i as f32 - 100.0) / 30.0;
            let y = adaa.process(x);
            assert!(y.abs() < 2.0, "ADAA2 output unbounded: {y} at x={x}");
        }
    }

    #[test]
    fn test_hardclip_adaa1() {
        let mut adaa = adaa1_hardclip();
        // Ramp through clipping region — skip first sample (transient from x_prev=0)
        let mut outputs = Vec::new();
        for i in 0..100 {
            let x = (i as f32 - 50.0) / 25.0;
            outputs.push(adaa.process(x));
        }
        // After the initial transient, outputs should be bounded
        for (i, &y) in outputs.iter().enumerate().skip(5) {
            assert!(
                y.abs() <= 1.5,
                "Hard clip ADAA output too large at i={i}: {y}"
            );
        }
    }

    #[test]
    fn test_consecutive_identical_samples() {
        let mut adaa = adaa1_tanh();
        // Feeding the same value repeatedly should not produce NaN or infinity
        for _ in 0..100 {
            let y = adaa.process(0.5);
            assert!(y.is_finite(), "Non-finite output: {y}");
        }
    }

    #[test]
    fn test_adaa2_fallback_uses_three_sample_centroid() {
        fn f(x: f64) -> f64 {
            x
        }
        fn ad1(x: f64) -> f64 {
            0.5 * x * x
        }
        fn ad2(x: f64) -> f64 {
            x * x * x / 6.0
        }

        let mut adaa = Adaa2::new(f, ad1, ad2);
        adaa.x_prev1 = 2.0;
        adaa.x_prev2 = 0.0;

        let y = adaa.process(0.0);
        assert!(
            (y - (2.0_f32 / 3.0)).abs() < 1e-6,
            "ADAA2 fallback must evaluate the three-sample centroid, got {y}"
        );
    }

    #[test]
    fn test_dilog_neg_basic() {
        // Li_2(0) = 0
        assert!((dilog_neg(0.0)).abs() < 1e-15);
        // Li_2(-1) = -pi^2/12
        let expected = -std::f64::consts::PI * std::f64::consts::PI / 12.0;
        let actual = dilog_neg(1.0);
        assert!(
            (actual - expected).abs() < 1e-4,
            "Li_2(-1): expected {expected}, got {actual}"
        );

        let near_one = dilog_neg(1.0 - 1e-13);
        assert!(
            (near_one - expected).abs() < 1e-12,
            "Li_2 near -1 should use the stable reference value"
        );
    }

    #[test]
    fn test_dilog_neg_very_small() {
        assert!(dilog_neg(1e-20).abs() < 1e-15);
    }

    #[test]
    fn test_dilog_neg_greater_than_one() {
        let z = 2.0;
        let val = dilog_neg(z);
        // Reference: Li_2(-2) ≈ -1.436746
        assert!((val + 1.436746).abs() < 1e-4);
    }

    #[test]
    fn test_dilog_neg_large_z() {
        let z = 1e6;
        let val = dilog_neg(z);
        let approx = -0.5 * (z.ln() * z.ln()) - std::f64::consts::PI * std::f64::consts::PI / 6.0;
        assert!(
            (val - approx).abs() < 0.1,
            "large z approximation failed: {val} vs {approx}"
        );
    }

    #[test]
    fn test_dilog_neg_identity_relation() {
        // Li_2(-z) + Li_2(-1/z) = -pi^2/6 - 0.5*ln(z)^2
        for z in [1.5, 2.0, 10.0, 100.0] {
            let lhs = dilog_neg(z) + dilog_neg(1.0 / z);
            let rhs = -std::f64::consts::PI * std::f64::consts::PI / 6.0 - 0.5 * z.ln() * z.ln();
            assert!(
                (lhs - rhs).abs() < 1e-6,
                "Identity failed at z={z}: lhs={lhs}, rhs={rhs}"
            );
        }
    }

    #[test]
    fn test_dilog_neg_near_one() {
        let expected = -std::f64::consts::PI * std::f64::consts::PI / 12.0;
        // Use 1e-13 so it triggers the (1-z) < 1e-12 fast path.
        let val = dilog_neg(1.0 - 1e-13);
        assert!((val - expected).abs() < 1e-8);
    }

    #[test]
    fn test_dilog_neg_approaching_one() {
        // z just below 1 is where the truncated series loses accuracy.
        // Reference: Li_2(-z) is continuous at z=1 with Li_2(-1) = -pi^2/12.
        let limit = -std::f64::consts::PI * std::f64::consts::PI / 12.0;
        for &delta in &[1e-2_f64, 1e-4, 1e-6, 1e-8] {
            let z = 1.0 - delta;
            let val = dilog_neg(z);
            // |Li_2'(-z)| = |ln(1+z)/z| <= ln(2) near z=1, so the value must
            // stay within ~ln(2)*delta of the limit.
            let tol = 2.0 * std::f64::consts::LN_2 * delta + 1e-12;
            assert!(
                (val - limit).abs() < tol,
                "Li_2(-{z}): expected ~{limit}, got {val} (tol {tol})"
            );
        }
    }

    #[test]
    fn test_tanh_ad2_derivative_matches_lncosh() {
        // d/dx AD2(tanh) = AD1(tanh) = ln(cosh(x)). The dilog series must be
        // accurate enough that this holds even for tiny x (z = e^{-2x} -> 1).
        for &x in &[1e-4_f64, 1e-3, 0.01, 0.05, 0.5, 2.0] {
            let h = 1e-5_f64;
            let numerical = (tanh_ad2(x + h) - tanh_ad2(x - h)) / (2.0 * h);
            let expected = x.cosh().ln();
            assert!(
                (numerical - expected).abs() < 1e-9,
                "tanh_ad2'({x}): expected {expected}, got {numerical}"
            );
        }
    }

    #[test]
    fn test_adaa2_tanh_quiet_sine_matches_naive() {
        // -60 dBFS sine: tanh is nearly linear here, so ADAA2 output must
        // closely match naive tanh evaluation (RMS ratio ~ 1).
        let sr = 48000.0;
        let amp = 1e-3_f64; // -60 dBFS
        let n = 8192;
        let mut adaa = adaa2_tanh();
        let mut adaa_sq = 0.0_f64;
        let mut naive_sq = 0.0_f64;
        for i in 0..n {
            let x = amp * (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / sr).sin();
            let y = adaa.process(x as f32) as f64;
            let naive = x.tanh();
            adaa_sq += y * y;
            naive_sq += naive * naive;
        }
        let ratio = (adaa_sq / naive_sq).sqrt();
        assert!(
            (ratio - 1.0).abs() < 0.01,
            "ADAA2/naive RMS ratio on quiet sine should be ~1.0, got {ratio}"
        );
    }
}

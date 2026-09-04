/// Least-squares linear fit `y = slope·x + intercept`. Returns
/// `(slope, intercept, r²)`. `None` if `n < 2` or `Var(x) = 0`.
#[cfg(test)]
pub(super) fn linear_fit(xs: &[f64], ys: &[f64]) -> Option<(f64, f64, f64)> {
    linear_fit_impl(xs, ys)
}

/// Linear fit where `x` is `(i_start + i) * dt` — avoids allocating the `x` vector.
///
/// The time base is centred at the window mean (`u = x - x0`) for the
/// accumulation so that `n·Σu² - (Σu)²` does not suffer catastrophic
/// cancellation when `i_start·dt` is large compared with the window
/// duration (e.g. late T30 windows in long RIRs). The returned
/// `(slope, intercept, r²)` is expressed in the original `x` coordinates,
/// so callers are unaffected.
pub(super) fn linear_fit_indexed(ys: &[f64], i_start: usize, dt: f64) -> Option<(f64, f64, f64)> {
    let n = ys.len();
    if n < 2 {
        return None;
    }
    let n_f = n as f64;
    // Window centre in index units; `u` is the centred time in seconds.
    let mean_idx = i_start as f64 + (n_f - 1.0) * 0.5;
    let mut su = 0.0_f64;
    let mut suu = 0.0_f64;
    let mut sy = 0.0_f64;
    let mut suy = 0.0_f64;
    let mut syy = 0.0_f64;
    for (i, &y) in ys.iter().enumerate() {
        let u = ((i_start + i) as f64 - mean_idx) * dt;
        su += u;
        suu += u * u;
        sy += y;
        suy += u * y;
        syy += y * y;
    }

    let denom = n_f * suu - su * su;
    if denom.abs() < f64::EPSILON {
        return None;
    }
    let slope = (n_f * suy - su * sy) / denom;
    // Intercept at the window centre, shifted back to the `x = 0` origin.
    let b0 = (sy - slope * su) / n_f;
    let intercept = b0 - slope * mean_idx * dt;

    let ss_tot = syy - sy * sy / n_f;
    let ss_res: f64 = ys
        .iter()
        .enumerate()
        .map(|(i, &y)| {
            let u = ((i_start + i) as f64 - mean_idx) * dt;
            let r = y - (slope * u + b0);
            r * r
        })
        .sum();
    let r2 = if ss_tot.abs() < f64::EPSILON {
        1.0
    } else {
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    };
    Some((slope, intercept, r2))
}

#[cfg(test)]
fn linear_fit_impl(xs: &[f64], ys: &[f64]) -> Option<(f64, f64, f64)> {
    let n = xs.len();
    if n < 2 || ys.len() != n {
        return None;
    }
    let n_f = n as f64;
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
    let syy: f64 = ys.iter().map(|y| y * y).sum();

    let denom = n_f * sxx - sx * sx;
    if denom.abs() < f64::EPSILON {
        return None;
    }
    let slope = (n_f * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n_f;

    let ss_tot = syy - sy * sy / n_f;
    let ss_res: f64 = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| {
            let pred = slope * x + intercept;
            let r = y - pred;
            r * r
        })
        .sum();
    let r2 = if ss_tot.abs() < f64::EPSILON {
        1.0
    } else {
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    };
    Some((slope, intercept, r2))
}

/// Estimate the sample index at which the RIR drops into the noise floor.
///
/// Lundeby-style iterative estimator (Lundeby et al. 1995) seeded with
/// Chu's method: the noise floor is the mean of `h²` over the last 10 %
/// of the signal, and the cutoff is the first sample at which a running
/// mean of `h²` drops to within 10 dB of that floor. The noise estimate
/// is then refined by re-estimating it from the last 10 % *past the
/// current cutoff* and re-crossing, iterating to convergence (≤ 1 %
/// change, at most 6 rounds). This corrects the seed when the fixed
/// last-10 % tail still contains decaying signal (short RIRs), while
/// converging in one round for stationary noise tails.
///
/// A pre-fit SNR gate rejects unusable decays up front: when the peak
/// `h²` before the tail clears the noise floor by less than 15 dB, the
/// function returns `start_sample` (empty usable region) so downstream
/// Schroeder fits yield NaN instead of fitting noise.
pub fn estimate_noise_cutoff(rir: &[f32], start_sample: usize) -> usize {
    if rir.is_empty() || start_sample >= rir.len() {
        return rir.len();
    }
    let n = rir.len();
    let tail_start = start_sample + ((n - start_sample) * 9) / 10;
    if tail_start >= n {
        return n;
    }

    fn mean_sq(rir: &[f32], range: std::ops::Range<usize>) -> f64 {
        let len = range.len() as f64;
        if len <= 0.0 {
            return f64::NAN;
        }
        let mut acc = 0.0_f64;
        for &s in &rir[range] {
            let v = s as f64;
            acc += v * v;
        }
        acc / len
    }

    // Mean squared value of the last 10 % is the seed noise estimate.
    let mut noise_e = mean_sq(rir, tail_start..n);

    // 5 ms running mean of h². Sample-rate-independent: just use a
    // proportional window (5 % of the signal length, clamped).
    let win = ((n - start_sample) / 20).clamp(32, 4096);
    if win == 0 || win >= n - start_sample {
        return n;
    }

    // First window whose running mean drops below 10 dB above `noise_e`.
    // Stops at `tail_start` — beyond that we're inside the noise tail by
    // definition. A non-positive floor disables the crossing (a running
    // mean can never drop below zero), reproducing the Chu fallback.
    let crossing = |noise_e: f64| -> usize {
        if !(noise_e > 0.0) || !noise_e.is_finite() {
            return tail_start;
        }
        let threshold = noise_e * 10.0;
        let mut win_sum = 0.0_f64;
        for &s in &rir[start_sample..start_sample + win] {
            let v = s as f64;
            win_sum += v * v;
        }
        let inv = win as f64;
        let limit = tail_start.min(n - win);
        for i in (start_sample + win)..limit {
            if win_sum / inv < threshold {
                return i;
            }
            let drop = rir[i - win] as f64;
            let add = rir[i] as f64;
            win_sum += add * add - drop * drop;
        }
        tail_start
    };

    if noise_e > 0.0 && noise_e.is_finite() {
        // Pre-fit SNR gate: the peak usable energy must clear the floor
        // by 15 dB, else there is no fittable decay range above noise.
        let mut peak = 0.0_f64;
        for &s in &rir[start_sample..tail_start] {
            let e = (s as f64) * (s as f64);
            if e > peak {
                peak = e;
            }
        }
        let snr_db = 10.0 * (peak / noise_e).log10();
        if !(snr_db >= 15.0) {
            return start_sample;
        }

        // Lundeby refinement: re-estimate the floor past the current
        // cutoff and re-cross until the floor estimate settles.
        let mut cutoff = crossing(noise_e);
        for _ in 0..6 {
            if cutoff >= n || n - cutoff < 64 {
                return cutoff;
            }
            let past_tail = cutoff + ((n - cutoff) * 9) / 10;
            if past_tail >= n {
                return cutoff;
            }
            let refined = mean_sq(rir, past_tail..n);
            if !(refined > 0.0) || !refined.is_finite() {
                return cutoff;
            }
            if (refined - noise_e).abs() <= 0.01 * noise_e {
                return crossing(refined);
            }
            noise_e = refined;
            cutoff = crossing(noise_e);
        }
        return cutoff;
    }

    crossing(noise_e)
}

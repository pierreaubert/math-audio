use rand::{Rng, RngExt};

/// Stack-friendly variant that writes `count` distinct indices from
/// `0..pool_size` excluding `exclude` into a caller-supplied buffer.
///
/// The buffer must be large enough for the requested `count`; if it is smaller,
/// only the leading `out.len()` entries are filled.
///
/// Returns the number of indices actually written.
pub(crate) fn distinct_indices_into<R: Rng + ?Sized>(
    exclude: usize,
    count: usize,
    pool_size: usize,
    rng: &mut R,
    out: &mut [usize],
) -> usize {
    distinct_indices_with_excludes_into(std::slice::from_ref(&exclude), count, pool_size, rng, out)
}

/// Stack-friendly variant that writes `count` distinct indices from
/// `0..pool_size` excluding every index in `excludes` into a caller-supplied
/// buffer.
///
/// If `pool_size - excludes.len()` is smaller than `count`, only the available
/// distinct indices are written.
///
/// Returns the number of indices actually written.
pub(crate) fn distinct_indices_with_excludes_into<R: Rng + ?Sized>(
    excludes: &[usize],
    count: usize,
    pool_size: usize,
    rng: &mut R,
    out: &mut [usize],
) -> usize {
    let max_available = pool_size.saturating_sub(excludes.len());
    let target = count.min(max_available).min(out.len());
    let mut selected = 0usize;
    'outer: while selected < target {
        let idx = rng.random_range(0..pool_size);
        if excludes.contains(&idx) {
            continue;
        }
        if out[..selected].contains(&idx) {
            continue 'outer;
        }
        out[selected] = idx;
        selected += 1;
    }
    selected
}

use rand::{Rng, RngExt};

/// Return `count` distinct indices from `0..pool_size` excluding `exclude`.
///
/// This uses a small linear scan instead of a `HashSet`, which is faster for
/// the tiny `count` values typical of DE mutation strategies (2–5) and avoids
/// a per-mutant HashSet allocation.
pub(crate) fn distinct_indices<R: Rng + ?Sized>(
    exclude: usize,
    count: usize,
    pool_size: usize,
    rng: &mut R,
) -> Vec<usize> {
    distinct_indices_with_excludes(std::slice::from_ref(&exclude), count, pool_size, rng)
}

/// Return `count` distinct indices from `0..pool_size` excluding every index
/// in `excludes`.
///
/// If `pool_size - excludes.len()` is smaller than `count`, the function
/// returns as many distinct indices as possible after applying the exclusions.
pub(crate) fn distinct_indices_with_excludes<R: Rng + ?Sized>(
    excludes: &[usize],
    count: usize,
    pool_size: usize,
    rng: &mut R,
) -> Vec<usize> {
    let max_available = pool_size.saturating_sub(excludes.len());
    let target = count.min(max_available);
    let mut selected = Vec::with_capacity(target);
    while selected.len() < target {
        let idx = rng.random_range(0..pool_size);
        if !excludes.contains(&idx) && !selected.contains(&idx) {
            selected.push(idx);
        }
    }
    selected
}

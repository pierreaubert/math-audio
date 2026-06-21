use ndarray::{Array1, Array2, Zip};
use rand::Rng;

use crate::distinct_indices::distinct_indices;

pub(crate) fn mutant_current_to_best1_into<R: Rng + ?Sized>(
    out: &mut Array1<f64>,
    i: usize,
    pop: &Array2<f64>,
    best_idx: usize,
    f: f64,
    rng: &mut R,
) {
    let idxs = distinct_indices(i, 2, pop.nrows(), rng);
    let r0 = idxs[0];
    let r1 = idxs[1];

    Zip::from(&mut *out)
        .and(pop.row(i))
        .and(pop.row(best_idx))
        .and(pop.row(r0))
        .and(pop.row(r1))
        .for_each(|o, &curr, &best, &x0, &x1| *o = curr + f * (best - curr + x0 - x1));
}

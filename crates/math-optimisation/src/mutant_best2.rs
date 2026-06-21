use ndarray::{Array1, Array2, Zip};
use rand::Rng;

use crate::distinct_indices::distinct_indices;

pub(crate) fn mutant_best2_into<R: Rng + ?Sized>(
    out: &mut Array1<f64>,
    i: usize,
    pop: &Array2<f64>,
    best_idx: usize,
    f: f64,
    rng: &mut R,
) {
    let idxs = distinct_indices(i, 4, pop.nrows(), rng);
    let r0 = idxs[0];
    let r1 = idxs[1];
    let r2 = idxs[2];
    let r3 = idxs[3];

    Zip::from(&mut *out)
        .and(pop.row(best_idx))
        .and(pop.row(r0))
        .and(pop.row(r1))
        .and(pop.row(r2))
        .and(pop.row(r3))
        .for_each(|o, &best, &x0, &x1, &x2, &x3| {
            *o = best + f * (x0 + x1 - x2 - x3);
        });
}

use ndarray::{Array1, Array2, Zip};
use rand::Rng;

use crate::distinct_indices::distinct_indices;

pub(crate) fn mutant_rand_to_best1_into<R: Rng + ?Sized>(
    out: &mut Array1<f64>,
    i: usize,
    pop: &Array2<f64>,
    best_idx: usize,
    f: f64,
    rng: &mut R,
) {
    // x_r0 + F * (x_best - x_r0 + x_r1 - x_r2)
    let idxs = distinct_indices(i, 3, pop.nrows(), rng);
    let r0 = idxs[0];
    let r1 = idxs[1];
    let r2 = idxs[2];

    Zip::from(&mut *out)
        .and(pop.row(r0))
        .and(pop.row(best_idx))
        .and(pop.row(r1))
        .and(pop.row(r2))
        .for_each(|o, &x_r0, &best, &x1, &x2| {
            *o = x_r0 + f * (best - x_r0 + x1 - x2);
        });
}

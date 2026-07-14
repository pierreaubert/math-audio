use ndarray::{Array1, Array2};
use rand::Rng;

use crate::distinct_indices::distinct_indices_into;

pub(crate) fn mutant_rand1_into<R: Rng + ?Sized>(
    out: &mut Array1<f64>,
    i: usize,
    pop: &Array2<f64>,
    f: f64,
    rng: &mut R,
) {
    let mut idxs = [0usize; 3];
    distinct_indices_into(i, 3, pop.nrows(), rng, &mut idxs);
    let r0 = idxs[0];
    let r1 = idxs[1];
    let r2 = idxs[2];

    let out_slice = out.as_slice_mut().expect("contiguous");
    let row0 = pop.row(r0);
    let x0 = row0.as_slice().expect("contiguous row");
    let row1 = pop.row(r1);
    let x1 = row1.as_slice().expect("contiguous row");
    let row2 = pop.row(r2);
    let x2 = row2.as_slice().expect("contiguous row");
    for j in 0..out_slice.len() {
        out_slice[j] = x0[j] + f * (x1[j] - x2[j]);
    }
}

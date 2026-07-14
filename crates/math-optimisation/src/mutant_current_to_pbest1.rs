//! Current-to-pbest/1 mutation strategy
//!
//! This is the key mutation strategy used by SHADE, L-SHADE and their variants.
//! It blends the current individual with one of the top-p% best individuals,
//! plus a difference vector that can include archive solutions.

use ndarray::{Array1, Array2};
use rand::{Rng, RngExt};

use crate::distinct_indices::{
    distinct_indices_into, distinct_indices_with_excludes_into,
};
use crate::external_archive::ExternalArchive;

/// Current-to-pbest/1 mutation with optional archive
///
/// v_i = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
///
/// Where:
/// - x_pbest is randomly selected from top p% of population
/// - x_r1 is a random individual from population
/// - x_r2 can be from population OR from external archive
#[allow(clippy::too_many_arguments)]
pub(crate) fn mutant_current_to_pbest1_into<R: Rng + ?Sized>(
    out: &mut Array1<f64>,
    i: usize,
    pop: &Array2<f64>,
    sorted_indices: &[usize],
    p_best_size: usize,
    archive: Option<&ExternalArchive>,
    f: f64,
    rng: &mut R,
) {
    let npop = pop.nrows();

    let pbest_idx = if p_best_size >= npop {
        sorted_indices[0]
    } else {
        sorted_indices[rng.random_range(0..p_best_size)]
    };

    let mut r1_buf = [0usize; 1];
    distinct_indices_into(i, 1, npop, rng, &mut r1_buf);
    let r1 = r1_buf[0];

    // Choose r2 from the archive with 50% probability, otherwise from the
    // population, excluding i and r1.
    let out_slice = out.as_slice_mut().expect("contiguous");
    let curr_row = pop.row(i);
    let curr = curr_row.as_slice().expect("contiguous row");
    let pbest_row = pop.row(pbest_idx);
    let pbest = pbest_row.as_slice().expect("contiguous row");
    let x1_row = pop.row(r1);
    let x1 = x1_row.as_slice().expect("contiguous row");

    // Choose r2 from the archive with 50% probability, otherwise from the
    // population, excluding i and r1.
    if let Some(arch) = archive
        && !arch.is_empty()
        && rng.random::<f64>() < 0.5
        && let Some(sol) = arch.random_select(rng)
    {
        let x2 = sol.as_slice().expect("contiguous archive row");
        for j in 0..out_slice.len() {
            out_slice[j] =
                curr[j] + f * (pbest[j] - curr[j]) + f * (x1[j] - x2[j]);
        }
        return;
    }

    let r2_idx = {
        let mut available = [0usize; 1];
        let n_found =
            distinct_indices_with_excludes_into(&[i, r1], 1, npop, rng, &mut available);
        if n_found == 0 {
            r1
        } else {
            available[0]
        }
    };

    let x2_row = pop.row(r2_idx);
    let x2 = x2_row.as_slice().expect("contiguous row");
    for j in 0..out_slice.len() {
        out_slice[j] = curr[j] + f * (pbest[j] - curr[j]) + f * (x1[j] - x2[j]);
    }
}

/// Compute p_best_size from p parameter (0 < p <= 1)
/// p=0.1 means top 10% of population
#[inline]
pub(crate) fn compute_pbest_size(p: f64, npop: usize) -> usize {
    ((p * npop as f64).ceil() as usize).max(1).min(npop)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_pbest_size() {
        assert_eq!(compute_pbest_size(0.1, 100), 10);
        assert_eq!(compute_pbest_size(0.1, 50), 5);
        assert_eq!(compute_pbest_size(0.1, 10), 1);
        assert_eq!(compute_pbest_size(0.5, 10), 5);
    }

    #[test]
    fn test_mutant_basic() {
        let pop = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let sorted_indices = vec![3, 2, 1, 0];

        let mut rng = rand::rng();
        let mut mutant = Array1::zeros(pop.ncols());
        mutant_current_to_pbest1_into(
            &mut mutant,
            0,
            &pop,
            &sorted_indices,
            2,
            None,
            0.5,
            &mut rng,
        );

        assert_eq!(mutant.len(), 2);
    }
}

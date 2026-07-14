use ndarray::{Array1, Array2};
use rand::{Rng, RngExt};

use crate::distinct_indices::distinct_indices_with_excludes_into;
use crate::mutant_rand1::mutant_rand1_into;

/// Adaptive mutation based on Self-Adaptive Mutation (SAM) from the paper.
/// Uses linearly decreasing weight w to select from top individuals.
pub(crate) fn mutant_adaptive_into<R: Rng + ?Sized>(
    out: &mut Array1<f64>,
    i: usize,
    pop: &Array2<f64>,
    sorted_indices: &[usize],
    w: f64,
    f: f64,
    rng: &mut R,
) {
    // Calculate w% of population size for adaptive selection
    let w_size = ((w * pop.nrows() as f64) as usize)
        .max(1)
        .min(pop.nrows() - 1);

    // Select gr_better from top w% individuals randomly
    let top_indices = &sorted_indices[0..w_size];
    let gr_better_idx = top_indices[rng.random_range(0..w_size)];

    // Need two distinct random indices different from i and gr_better_idx
    let mut available = [0usize; 2];
    let n_found = distinct_indices_with_excludes_into(
        &[i, gr_better_idx],
        2,
        pop.nrows(),
        rng,
        &mut available,
    );

    if n_found < 2 {
        // Fallback to standard rand1 if not enough individuals
        mutant_rand1_into(out, i, pop, f, rng);
        return;
    }

    let r1 = available[0];
    let r2 = available[1];

    // Adaptive mutation: x_i + F * (x_gr_better - x_i + x_r1 - x_r2)
    // This is the SAM approach from equation (18) in the paper
    let out_slice = out.as_slice_mut().expect("contiguous");
    let curr_row = pop.row(i);
    let curr = curr_row.as_slice().expect("contiguous row");
    let gr_row = pop.row(gr_better_idx);
    let gr = gr_row.as_slice().expect("contiguous row");
    let row1 = pop.row(r1);
    let x1 = row1.as_slice().expect("contiguous row");
    let row2 = pop.row(r2);
    let x2 = row2.as_slice().expect("contiguous row");
    for j in 0..out_slice.len() {
        out_slice[j] = curr[j] + f * (gr[j] - curr[j] + x1[j] - x2[j]);
    }
}

/// Tests for adaptive differential evolution strategies
#[cfg(test)]
mod tests {
    use crate::{AdaptiveConfig, DEConfigBuilder, Mutation, Strategy, differential_evolution};
    use math_audio_test_functions::quadratic;

    #[test]
    fn test_adaptive_basic() {
        // Test basic adaptive DE functionality
        let bounds = [(-5.0, 5.0), (-5.0, 5.0)];

        // Configure adaptive DE with SAM approach
        let adaptive_config = AdaptiveConfig {
            adaptive_mutation: true,
            wls_enabled: false, // Start with mutation only
            w_max: 0.9,
            w_min: 0.1,
            ..AdaptiveConfig::default()
        };

        let config = DEConfigBuilder::new()
            .seed(42)
            .maxiter(100)
            .popsize(30)
            .strategy(Strategy::AdaptiveBin)
            .mutation(Mutation::Adaptive { initial_f: 0.8 })
            .adaptive(adaptive_config)
            .build()
            .expect("popsize must be >= 4");

        let result = differential_evolution(&quadratic, &bounds, config)
            .expect("optimization should succeed");

        // Should converge to global minimum at (0, 0)
        assert!(
            result.fun < 1e-3,
            "Adaptive DE should converge: f={}",
            result.fun
        );

        // Check that solution is close to expected optimum
        for &xi in result.x.iter() {
            assert!(
                xi.abs() < 0.5,
                "Solution component should be close to 0: {}",
                xi
            );
        }
    }

    #[test]
    fn test_adaptive_with_wls() {
        // Test adaptive DE with Wrapper Local Search
        let bounds = [(-5.0, 5.0), (-5.0, 5.0)];

        let adaptive_config = AdaptiveConfig {
            adaptive_mutation: true,
            wls_enabled: true,
            wls_prob: 0.2, // Apply WLS to 20% of population
            wls_scale: 0.1,
            ..AdaptiveConfig::default()
        };

        let config = DEConfigBuilder::new()
            .seed(123)
            .maxiter(200)
            .popsize(40)
            .strategy(Strategy::AdaptiveExp)
            .adaptive(adaptive_config)
            .build()
            .expect("popsize must be >= 4");

        let result = differential_evolution(&quadratic, &bounds, config)
            .expect("optimization should succeed");

        // Should converge even better with WLS
        assert!(
            result.fun < 1e-4,
            "Adaptive DE with WLS should converge well: f={}",
            result.fun
        );

        for &xi in result.x.iter() {
            assert!(
                xi.abs() < 0.2,
                "Solution should be very close to 0 with WLS: {}",
                xi
            );
        }
    }
}

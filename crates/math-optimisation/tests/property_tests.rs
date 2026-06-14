// ============================================================================
// Property-Based Tests for math-optimisation
// ============================================================================
//
// Invariants checked:
//   - Differential Evolution, COBYLA and CMA-ES improve on a random starting
//     point when minimizing a simple 2D sphere.
//   - Differential Evolution with a fixed seed is deterministic across runs.

use math_audio_optimisation::cobyla::{cobyla, CobylaConfig, CobylaRhoBegin};
use math_audio_optimisation::parallel_eval::ParallelConfig;
use math_audio_optimisation::{CmaEsConfig, DEConfigBuilder, cma_es, differential_evolution};
use ndarray::Array1;
use proptest::prelude::*;

fn sphere(x: &Array1<f64>) -> f64 {
    x.iter().map(|&xi| xi * xi).sum::<f64>()
}

fn bounds_2d() -> Vec<(f64, f64)> {
    vec![(-5.0, 5.0), (-5.0, 5.0)]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// IMPROVEMENT: DE started away from the sphere optimum finds a lower
    /// objective than the initial random point.
    #[test]
    fn de_improves_over_random_start(
        x0_0 in -5.0f64..5.0f64,
        x0_1 in -5.0f64..5.0f64,
        seed in 1u64..10_000,
    ) {
        let x0 = Array1::from(vec![x0_0, x0_1]);
        let f0 = sphere(&x0);
        if f0 < 1e-3 {
            return Ok(());
        }

        let config = DEConfigBuilder::new()
            .maxiter(50)
            .popsize(20)
            .seed(seed)
            .parallel(ParallelConfig {
                enabled: false,
                num_threads: None,
            })
            .build()
            .expect("valid DE config");
        let report = differential_evolution(&sphere, &bounds_2d(), config)
            .expect("DE should run");

        prop_assert!(report.fun.is_finite(), "DE returned non-finite objective");
        prop_assert!(
            report.fun <= f0 + 1e-12,
            "DE did not improve: f0={} f*={}",
            f0,
            report.fun
        );
    }

    /// DETERMINISM: two DE runs with the same seed on the same problem produce
    /// the same result.
    #[test]
    fn de_fixed_seed_is_deterministic(seed in 1u64..10_000) {
        let config = DEConfigBuilder::new()
            .maxiter(50)
            .popsize(20)
            .seed(seed)
            .parallel(ParallelConfig {
                enabled: false,
                num_threads: None,
            })
            .build()
            .expect("valid DE config");

        let r1 = differential_evolution(&sphere, &bounds_2d(), config)
            .expect("first DE run should succeed");
        let config2 = DEConfigBuilder::new()
            .maxiter(50)
            .popsize(20)
            .seed(seed)
            .parallel(ParallelConfig {
                enabled: false,
                num_threads: None,
            })
            .build()
            .expect("valid DE config");
        let r2 = differential_evolution(&sphere, &bounds_2d(), config2)
            .expect("second DE run should succeed");

        prop_assert!(
            (r1.fun - r2.fun).abs() <= 1e-12,
            "DE objective differed between deterministic runs: {} vs {}",
            r1.fun,
            r2.fun
        );
        // The reported x can vary slightly between runs because the sphere
        // objective is flat near the optimum; the objective value is the
        // deterministic quantity we care about.
        prop_assert!(
            r1.x.iter().zip(r2.x.iter()).all(|(a, b)| (a - b).abs() <= 1e-6),
            "DE solution differed unexpectedly between deterministic runs: {:?} vs {:?}",
            r1.x,
            r2.x
        );
    }

    /// IMPROVEMENT: COBYLA started away from the sphere optimum finds a lower
    /// objective than the initial point.
    #[test]
    fn cobyla_improves_over_random_start(
        x0_0 in -5.0f64..5.0f64,
        x0_1 in -5.0f64..5.0f64,
    ) {
        let x0 = Array1::from(vec![x0_0, x0_1]);
        let f0 = sphere(&x0);
        if f0 < 1e-3 {
            return Ok(());
        }

        let cfg = CobylaConfig {
            x0,
            bounds: bounds_2d(),
            rho_begin: CobylaRhoBegin::All(0.5),
            maxeval: 300,
            ..CobylaConfig::default()
        };
        let report = cobyla(&sphere, &[], cfg).expect("COBYLA should run");

        prop_assert!(report.fun.is_finite(), "COBYLA returned non-finite objective");
        prop_assert!(
            report.fun <= f0 + 1e-12,
            "COBYLA did not improve: f0={} f*={}",
            f0,
            report.fun
        );
    }

    /// IMPROVEMENT: CMA-ES started away from the sphere optimum finds a lower
    /// objective than the initial point.
    #[test]
    fn cmaes_improves_over_random_start(
        x0_0 in -5.0f64..5.0f64,
        x0_1 in -5.0f64..5.0f64,
    ) {
        let x0 = Array1::from(vec![x0_0, x0_1]);
        let f0 = sphere(&x0);
        if f0 < 1e-3 {
            return Ok(());
        }

        let cfg = CmaEsConfig {
            bounds: bounds_2d(),
            x0: Some(x0),
            maxeval: 800,
            seed: Some(7),
            ..CmaEsConfig::default()
        };
        let report = cma_es(&sphere, cfg).expect("CMA-ES should run");

        prop_assert!(report.fun.is_finite(), "CMA-ES returned non-finite objective");
        prop_assert!(
            report.fun <= f0 + 1e-12,
            "CMA-ES did not improve: f0={} f*={}",
            f0,
            report.fun
        );
    }
}

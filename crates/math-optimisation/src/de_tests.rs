use crate::{
    CallbackAction, DEConfigBuilder, DifferentialEvolution, LShadeConfig, PolishConfig, Strategy,
};
use ndarray::{Array1, array};
use rand::SeedableRng;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(test)]
mod strategy_tests {
    use super::*;

    #[test]
    fn test_best1_binomial_convergence() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();

        let config = DEConfigBuilder::new()
            .seed(42)
            .maxiter(200)
            .popsize(20)
            .strategy(Strategy::Best1Bin)
            .build()
            .expect("popsize must be >= 4");

        let mut de =
            DifferentialEvolution::new(&sphere, array![-5.0f64, -5.0], array![5.0f64, 5.0])
                .unwrap();
        *de.config_mut() = config;
        let report = de.solve();

        assert!(
            report.fun < 1.0,
            "Should converge near origin: f={}",
            report.fun
        );
    }

    #[test]
    fn test_rand1_exponential_convergence() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();

        let config = DEConfigBuilder::new()
            .seed(123)
            .maxiter(300)
            .popsize(30)
            .strategy(Strategy::Rand1Exp)
            .recombination(0.5)
            .build()
            .expect("popsize must be >= 4");

        let mut de =
            DifferentialEvolution::new(&sphere, array![-5.0f64, -5.0], array![5.0f64, 5.0])
                .unwrap();
        *de.config_mut() = config;
        let report = de.solve();

        assert!(report.fun < 1.0, "Should converge: f={}", report.fun);
    }

    #[test]
    fn test_rand2_binomial_convergence() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();

        let config = DEConfigBuilder::new()
            .seed(456)
            .maxiter(300)
            .popsize(30)
            .strategy(Strategy::Rand2Bin)
            .build()
            .expect("popsize must be >= 4");

        let mut de =
            DifferentialEvolution::new(&sphere, array![-5.0f64, -5.0], array![5.0f64, 5.0])
                .unwrap();
        *de.config_mut() = config;
        let report = de.solve();

        assert!(report.fun < 1.0, "Should converge: f={}", report.fun);
    }

    #[test]
    fn test_current_to_best_convergence() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();

        let config = DEConfigBuilder::new()
            .seed(789)
            .maxiter(200)
            .popsize(25)
            .strategy(Strategy::CurrentToBest1Bin)
            .build()
            .expect("popsize must be >= 4");

        let mut de =
            DifferentialEvolution::new(&sphere, array![-5.0f64, -5.0], array![5.0f64, 5.0])
                .unwrap();
        *de.config_mut() = config;
        let report = de.solve();

        assert!(report.fun < 1.0, "Should converge: f={}", report.fun);
    }

    #[test]
    fn test_best2_convergence() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();

        let config = DEConfigBuilder::new()
            .seed(321)
            .maxiter(300)
            .popsize(30)
            .strategy(Strategy::Best2Bin)
            .build()
            .expect("popsize must be >= 4");

        let mut de =
            DifferentialEvolution::new(&sphere, array![-5.0f64, -5.0], array![5.0f64, 5.0])
                .unwrap();
        *de.config_mut() = config;
        let report = de.solve();

        assert!(report.fun < 1.0, "Should converge: f={}", report.fun);
    }
}

#[cfg(test)]
mod crossover_tests {
    use super::*;
    use crate::crossover_binomial::binomial_crossover;
    use crate::crossover_exponential::exponential_crossover;

    #[test]
    fn test_binomial_crossover_preserves_dimensions() {
        let target = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let mutant = array![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let trial = binomial_crossover(&target, &mutant, 0.5, &mut rng);

        assert_eq!(trial.len(), target.len());
    }

    #[test]
    fn test_exponential_crossover_preserves_dimensions() {
        let target = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let mutant = array![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let trial = exponential_crossover(&target, &mutant, 0.5, &mut rng);

        assert_eq!(trial.len(), target.len());
    }
}

#[cfg(test)]
mod initialization_tests {
    use super::*;
    use crate::init_latin_hypercube::init_latin_hypercube;
    use crate::init_random::init_random;

    #[test]
    fn test_latin_hypercube_dimensions() {
        let lower = array![0.0, 0.0];
        let upper = array![10.0, 10.0];
        let is_free = vec![true, true];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let pop = init_latin_hypercube(2, 20, &lower, &upper, &is_free, &mut rng);

        assert_eq!(pop.nrows(), 20);
        assert_eq!(pop.ncols(), 2);
    }

    #[test]
    fn test_latin_hypercube_bounds() {
        let lower = array![0.0, 0.0];
        let upper = array![10.0, 10.0];
        let is_free = vec![true, true];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let pop = init_latin_hypercube(2, 20, &lower, &upper, &is_free, &mut rng);

        for row in pop.rows() {
            assert!(row[0] >= 0.0 && row[0] <= 10.0);
            assert!(row[1] >= 0.0 && row[1] <= 10.0);
        }
    }

    #[test]
    fn test_random_initialization_dimensions() {
        let lower = array![0.0, 0.0];
        let upper = array![10.0, 10.0];
        let is_free = vec![true, true];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let pop = init_random(2, 20, &lower, &upper, &is_free, &mut rng);

        assert_eq!(pop.nrows(), 20);
        assert_eq!(pop.ncols(), 2);
    }
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_single_dimension() {
        let sphere = |x: &Array1<f64>| x[0] * x[0];

        let config = DEConfigBuilder::new()
            .seed(42)
            .maxiter(100)
            .popsize(10)
            .build()
            .expect("popsize must be >= 4");

        let mut de = DifferentialEvolution::new(&sphere, array![-5.0f64], array![5.0f64]).unwrap();
        *de.config_mut() = config;
        let report = de.solve();

        assert!(report.fun < 1.0, "Should find minimum near 0");
    }

    #[test]
    fn test_fixed_variables() {
        let sphere = |x: &Array1<f64>| x[1] * x[1];

        let config = DEConfigBuilder::new()
            .seed(42)
            .maxiter(50)
            .popsize(10)
            .build()
            .expect("popsize must be >= 4");

        let mut de =
            DifferentialEvolution::new(&sphere, array![-5.0f64, 3.0f64], array![5.0f64, 3.0f64])
                .unwrap();
        *de.config_mut() = config;
        let report = de.solve();

        assert!((report.x[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_deterministic_with_seed() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();

        let config1 = DEConfigBuilder::new()
            .seed(42)
            .maxiter(50)
            .popsize(15)
            .build()
            .expect("popsize must be >= 4");

        let mut de1 =
            DifferentialEvolution::new(&sphere, array![-5.0f64, -5.0], array![5.0f64, 5.0])
                .unwrap();
        *de1.config_mut() = config1;
        let report1 = de1.solve();

        let config2 = DEConfigBuilder::new()
            .seed(42)
            .maxiter(50)
            .popsize(15)
            .build()
            .expect("popsize must be >= 4");

        let mut de2 =
            DifferentialEvolution::new(&sphere, array![-5.0f64, -5.0], array![5.0f64, 5.0])
                .unwrap();
        *de2.config_mut() = config2;
        let report2 = de2.solve();

        // With the same seed, results should be very similar (though not necessarily bitwise identical
        // due to potential floating-point non-associativity in parallel operations)
        let diff0 = (report1.x[0] - report2.x[0]).abs();
        let diff1 = (report1.x[1] - report2.x[1]).abs();

        assert!(
            diff0 < 1e-6,
            "x[0] should be nearly deterministic with same seed: diff = {}",
            diff0
        );
        assert!(
            diff1 < 1e-6,
            "x[1] should be nearly deterministic with same seed: diff = {}",
            diff1
        );
    }
}

#[cfg(test)]
mod callback_tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_callback_stop_early() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let config = DEConfigBuilder::new()
            .seed(42)
            .maxiter(1000)
            .popsize(10)
            .tol(0.0)
            .atol(0.0)
            .callback(Box::new(move |inter| {
                call_count_clone.fetch_add(1, Ordering::SeqCst);
                eprintln!("Callback called at iter {}", inter.iter);
                if inter.iter >= 5 {
                    CallbackAction::Stop
                } else {
                    CallbackAction::Continue
                }
            }))
            .build()
            .expect("popsize must be >= 4");

        let mut de =
            DifferentialEvolution::new(&sphere, array![-5.0f64, -5.0], array![5.0f64, 5.0])
                .unwrap();
        *de.config_mut() = config;
        let report = de.solve();

        let final_count = call_count.load(Ordering::SeqCst);
        eprintln!("Final call_count: {}", final_count);
        eprintln!("Report nit: {}", report.nit);
        assert_eq!(final_count, 5, "Callback should be called exactly 5 times");
        assert_eq!(report.nit, 5, "Should stop after 5 iterations");
    }
}

#[cfg(test)]
mod seeded_legacy_checkpoint_baseline {
    use crate::{DEConfigBuilder, DifferentialEvolution, Strategy};
    use ndarray::{Array1, array};
    use rand::{RngExt, SeedableRng};

    #[test]
    fn seeded_uninterrupted_report_matches_pre_refactor_baseline() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&value| value * value).sum::<f64>();
        let config = DEConfigBuilder::new()
            .seed(0x4a08_2026)
            .maxiter(4)
            .popsize(5)
            .tol(0.0)
            .atol(0.0)
            .strategy(Strategy::Rand1Bin)
            .build()
            .expect("valid seeded baseline configuration");
        let mut solver = DifferentialEvolution::new(&sphere, array![-3.0, -2.0], array![3.0, 2.0])
            .expect("valid bounds");
        *solver.config_mut() = config;
        let report = solver.solve();
        assert_eq!(
            report.x,
            array![0.048_270_894_771_279_33, -0.214_449_842_999_039_46]
        );
        assert_eq!(report.fun, 0.048_318_814_444_332_59);
        assert_eq!(report.nit, 4);
        assert_eq!(report.nfev, 50);
        assert!(!report.success);
        assert_eq!(report.message, "Maximum iterations reached: 4");
        assert_eq!(
            report.population,
            array![
                [0.285_359_702_964_253_06, -0.297_019_750_655_381_1],
                [-0.318_630_764_613_520_63, 1.326_282_387_666_183_1],
                [-0.530_063_409_192_849_6, -0.091_340_490_313_691_54],
                [0.048_270_894_771_279_33, -0.214_449_842_999_039_46],
                [-0.333_702_743_445_064_9, -0.015_427_931_469_863_126],
                [0.794_524_913_155_446, 0.379_038_718_613_149_4],
                [-1.151_411_554_770_454_2, 0.232_404_091_211_391_47],
                [1.236_553_375_211_931_5, -1.243_997_765_955_381_8],
                [0.988_202_568_105_722_9, 0.313_152_526_718_456_4],
                [0.164_672_333_454_577_2, -0.437_498_196_409_211_3],
            ]
        );
        assert_eq!(
            report.population_energies,
            array![
                0.169_650_892_355_231_5,
                1.860_550_535_991_708_3,
                0.289_310_302_935_891_8,
                0.048_318_814_444_332_59,
                0.111_595_542_052_201_6,
                0.774_940_187_832_567_2,
                1.379_760_230_070_707_5,
                3.076_594_691_450_001,
                1.074_608_820_600_899_5,
                0.218_521_649_266_688_3,
            ]
        );
    }

    #[test]
    fn portable_chacha12_stream_matches_rand_std_rng_seeded_stream() {
        let mut legacy = rand::rngs::StdRng::seed_from_u64(0x4a08_2026);
        let mut portable = chacha20::ChaCha12Rng::seed_from_u64(0x4a08_2026);
        for _ in 0..2_048 {
            assert_eq!(legacy.random::<u64>(), portable.random::<u64>());
            assert_eq!(
                legacy.random_range(0_u32..127),
                portable.random_range(0_u32..127)
            );
        }
    }
}

#[cfg(test)]
mod config_validation_tests {
    use super::*;

    #[test]
    fn test_popsize_too_small() {
        let result = DEConfigBuilder::new().popsize(3).build();

        assert!(result.is_err());
    }

    #[test]
    fn test_popsize_minimum() {
        let result = DEConfigBuilder::new().popsize(4).build();

        assert!(result.is_ok());
    }

    #[test]
    fn test_lshade_population_reduction_wired() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();

        let lshade = LShadeConfig {
            np_init: 18,
            np_final: 4,
            p: 0.11,
            arc_rate: 2.1,
            memory_size: 6,
        };

        let config = DEConfigBuilder::new()
            .seed(42)
            .maxiter(50)
            .strategy(Strategy::LShadeBin)
            .lshade(lshade)
            .build()
            .expect("popsize must be >= 4");

        let mut de =
            DifferentialEvolution::new(&sphere, array![-5.0f64, -5.0], array![5.0f64, 5.0])
                .unwrap();
        *de.config_mut() = config;
        let report = de.solve();

        // With 2 free dimensions, L-SHADE initial NP = 18*2 = 36.
        // After 50 generations the population should have been reduced well
        // below the initial size.
        assert!(
            report.population.nrows() < 20,
            "L-SHADE should reduce population below 20, got {}",
            report.population.nrows()
        );
    }

    #[test]
    fn test_nan_objective_does_not_corrupt_selection() {
        // A single NaN evaluation in the population must not derail the
        // optimizer: argmin should skip it and the run should still converge.
        use std::sync::atomic::AtomicUsize;
        let call_count = AtomicUsize::new(0);
        let f = |x: &Array1<f64>| {
            let c = call_count.fetch_add(1, Ordering::SeqCst);
            if c == 4 {
                f64::NAN
            } else {
                x.iter().map(|&xi| xi * xi).sum::<f64>()
            }
        };

        let config = DEConfigBuilder::new()
            .seed(42)
            .maxiter(100)
            .popsize(10)
            .build()
            .expect("popsize must be >= 4");

        let mut de =
            DifferentialEvolution::new(&f, array![-5.0f64, -5.0], array![5.0f64, 5.0]).unwrap();
        *de.config_mut() = config;
        let report = de.solve();

        assert!(
            report.fun.is_finite(),
            "best fitness must be finite, got {}",
            report.fun
        );
        assert!(
            report.fun < 1.0,
            "should converge despite one NaN eval: f={}",
            report.fun
        );
    }
}

#[cfg(test)]
mod polish_tests {
    use super::*;

    #[test]
    fn test_polish_improves_solution() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();

        let config = DEConfigBuilder::new()
            .seed(42)
            .maxiter(20)
            .popsize(10)
            .polish(PolishConfig {
                enabled: true,
                maxeval: 100,
            })
            .build()
            .expect("popsize must be >= 4");

        let mut de =
            DifferentialEvolution::new(&sphere, array![-5.0f64, -5.0], array![5.0f64, 5.0])
                .unwrap();
        *de.config_mut() = config;
        let report = de.solve();

        assert!(
            report.fun < 10.0,
            "Polish should improve solution: f={}",
            report.fun
        );
    }
}

#[cfg(test)]
mod differential_evolution_function_tests {
    use super::*;
    use crate::differential_evolution;

    #[test]
    fn test_convenience_function_sphere() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let config = DEConfigBuilder::new()
            .seed(42)
            .maxiter(200)
            .popsize(20)
            .build()
            .expect("popsize must be >= 4");

        let report = differential_evolution(&sphere, &[(-5.0, 5.0), (-5.0, 5.0)], config)
            .expect("DE should succeed");
        assert!(
            report.fun < 1.0,
            "Should converge near origin: f={}",
            report.fun
        );
    }

    #[test]
    fn test_convenience_function_invalid_bounds() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let config = DEConfigBuilder::new().seed(42).maxiter(10).build().unwrap();
        let result = differential_evolution(&sphere, &[(5.0, -5.0)], config);
        assert!(result.is_err(), "Inverted bounds should error");
    }

    #[test]
    fn test_convenience_function_empty_bounds() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let config = DEConfigBuilder::new().seed(42).maxiter(10).build().unwrap();
        let report = differential_evolution(&sphere, &[], config)
            .expect("0-D optimization should trivially succeed");
        assert!(report.success);
        assert_eq!(report.nit, 0);
        assert_eq!(report.nfev, 1);
    }
}

#[cfg(test)]
mod strategy_coverage_tests {
    use super::*;

    fn test_strategy_converges(strategy: Strategy) {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let config = DEConfigBuilder::new()
            .seed(42)
            .maxiter(300)
            .popsize(30)
            .strategy(strategy)
            .build()
            .expect("popsize must be >= 4");

        let mut de =
            DifferentialEvolution::new(&sphere, array![-5.0f64, -5.0], array![5.0f64, 5.0])
                .unwrap();
        *de.config_mut() = config;
        let report = de.solve();
        assert!(
            report.fun < 1.0,
            "Strategy {:?} should converge: f={}",
            strategy,
            report.fun
        );
    }

    #[test]
    fn test_best1_exp() {
        test_strategy_converges(Strategy::Best1Exp);
    }
    #[test]
    fn test_rand1_exp() {
        test_strategy_converges(Strategy::Rand1Exp);
    }
    #[test]
    fn test_rand2_exp() {
        test_strategy_converges(Strategy::Rand2Exp);
    }
    #[test]
    fn test_current_to_best1_exp() {
        test_strategy_converges(Strategy::CurrentToBest1Exp);
    }
    #[test]
    fn test_best2_exp() {
        test_strategy_converges(Strategy::Best2Exp);
    }
    #[test]
    fn test_rand_to_best1_exp() {
        test_strategy_converges(Strategy::RandToBest1Exp);
    }
    #[test]
    fn test_adaptive_bin() {
        test_strategy_converges(Strategy::AdaptiveBin);
    }
    #[test]
    fn test_adaptive_exp() {
        test_strategy_converges(Strategy::AdaptiveExp);
    }
    #[test]
    fn test_lshade_exp() {
        test_strategy_converges(Strategy::LShadeExp);
    }
}

#[cfg(test)]
mod constraint_and_penalty_tests {
    use super::*;

    #[test]
    fn test_inequality_penalty() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let config = DEConfigBuilder::new()
            .seed(42)
            .maxiter(200)
            .popsize(20)
            .add_penalty_ineq(|x| x[0] + 1.0, 1e3) // x0 >= -1
            .build()
            .expect("popsize must be >= 4");

        let mut de =
            DifferentialEvolution::new(&sphere, array![-5.0f64, -5.0], array![5.0f64, 5.0])
                .unwrap();
        *de.config_mut() = config;
        let report = de.solve();
        assert!(
            report.fun < 1.0,
            "Should converge with inequality penalty: f={}",
            report.fun
        );
    }

    #[test]
    fn test_equality_penalty() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let config = DEConfigBuilder::new()
            .seed(42)
            .maxiter(200)
            .popsize(20)
            .add_penalty_eq(|x| x[0] - 1.0, 1e3) // x0 == 1
            .build()
            .expect("popsize must be >= 4");

        let mut de =
            DifferentialEvolution::new(&sphere, array![-5.0f64, -5.0], array![5.0f64, 5.0])
                .unwrap();
        *de.config_mut() = config;
        let report = de.solve();
        // With strong equality penalty, x0 should be pulled toward 1
        assert!(
            (report.x[0] - 1.0).abs() < 0.5,
            "x0 should be near 1: {}",
            report.x[0]
        );
    }

    #[test]
    fn test_linear_penalty() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let a = ndarray::Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();
        let config = DEConfigBuilder::new()
            .seed(42)
            .maxiter(200)
            .popsize(20)
            .linear_penalty(crate::LinearPenalty {
                a,
                lb: array![-1.0],
                ub: array![1.0],
                weight: 1e3,
            })
            .build()
            .expect("popsize must be >= 4");

        let mut de =
            DifferentialEvolution::new(&sphere, array![-5.0f64, -5.0], array![5.0f64, 5.0])
                .unwrap();
        *de.config_mut() = config;
        let report = de.solve();
        assert!(
            report.fun < 10.0,
            "Should converge with linear penalty: f={}",
            report.fun
        );
    }
}

#[cfg(test)]
mod integrality_and_wls_tests {
    use super::*;

    #[test]
    fn test_integrality_mask() {
        let f = |x: &Array1<f64>| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);
        let config = DEConfigBuilder::new()
            .seed(42)
            .maxiter(100)
            .popsize(10)
            .integrality(vec![true, false])
            .build()
            .expect("popsize must be >= 4");

        let mut de =
            DifferentialEvolution::new(&f, array![0.0f64, 0.0], array![5.0f64, 5.0]).unwrap();
        *de.config_mut() = config;
        let report = de.solve();
        assert!(
            report.fun < 1.0,
            "Should converge with integrality: f={}",
            report.fun
        );
        // x0 should be very close to an integer
        assert!(
            (report.x[0].round() - report.x[0]).abs() < 1e-6,
            "x0 should be integral: {}",
            report.x[0]
        );
    }

    #[test]
    fn test_wls_enabled() {
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let config = DEConfigBuilder::new()
            .seed(42)
            .maxiter(100)
            .popsize(10)
            .enable_wls(true)
            .build()
            .expect("popsize must be >= 4");

        let mut de =
            DifferentialEvolution::new(&sphere, array![-5.0f64, -5.0], array![5.0f64, 5.0])
                .unwrap();
        *de.config_mut() = config;
        let report = de.solve();
        assert!(
            report.fun < 1.0,
            "Should converge with WLS: f={}",
            report.fun
        );
    }
}

#[cfg(test)]
mod edge_case_and_error_tests {
    use super::*;

    #[test]
    fn test_all_fixed_variables() {
        let f = |x: &Array1<f64>| x[0] * x[0] + x[1] * x[1];
        let config = DEConfigBuilder::new()
            .seed(42)
            .maxiter(10)
            .popsize(10)
            .build()
            .expect("popsize must be >= 4");

        let mut de =
            DifferentialEvolution::new(&f, array![2.0f64, 3.0], array![2.0f64, 3.0]).unwrap();
        *de.config_mut() = config;
        let report = de.solve();
        assert!(report.success);
        assert_eq!(report.nit, 0);
        assert_eq!(report.nfev, 1);
        assert!((report.x[0] - 2.0).abs() < 1e-10);
        assert!((report.x[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_new_bounds_mismatch() {
        let f = |x: &Array1<f64>| x[0] * x[0];
        let result = DifferentialEvolution::new(&f, array![-5.0f64], array![5.0f64, 5.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_invalid_bounds() {
        let f = |x: &Array1<f64>| x[0] * x[0];
        let result = DifferentialEvolution::new(&f, array![5.0f64], array![-5.0f64]);
        assert!(result.is_err());
    }

    #[test]
    fn test_inf_objective_becomes_infinity() {
        let call_count = std::sync::atomic::AtomicUsize::new(0);
        let f = |x: &Array1<f64>| {
            let c = call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if c == 0 {
                f64::INFINITY
            } else {
                x.iter().map(|&xi| xi * xi).sum::<f64>()
            }
        };
        let config = DEConfigBuilder::new()
            .seed(42)
            .maxiter(50)
            .popsize(10)
            .build()
            .expect("popsize must be >= 4");

        let mut de =
            DifferentialEvolution::new(&f, array![-5.0f64, -5.0], array![5.0f64, 5.0]).unwrap();
        *de.config_mut() = config;
        let report = de.solve();
        assert!(report.fun.is_finite(), "Best fitness must be finite");
    }
}

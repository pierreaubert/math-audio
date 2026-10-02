use crate::{
    CheckpointFitness, DECheckpoint, DEConfigBuilder, DifferentialEvolution, LShadeConfig,
    ParallelConfig, Strategy,
};
use ndarray::{Array1, array};
use std::sync::atomic::{AtomicUsize, Ordering};

fn sphere(x: &Array1<f64>) -> f64 {
    (x[0] * x[0]) + ((x[1] - 0.35) * (x[1] - 0.35)) + (0.07 * x[0] * x[1])
}

fn config(strategy: Strategy, parallel: bool, seed: u64, maxiter: usize) -> crate::DEConfig {
    let mut builder = DEConfigBuilder::new()
        .seed(seed)
        .maxiter(maxiter)
        .popsize(8)
        .tol(0.0)
        .atol(0.0)
        .strategy(strategy)
        .parallel(ParallelConfig {
            enabled: parallel,
            num_threads: if parallel { Some(2) } else { None },
        });
    if matches!(strategy, Strategy::LShadeBin | Strategy::LShadeExp) {
        builder = builder.lshade(LShadeConfig {
            np_init: 8,
            np_final: 4,
            p: 0.2,
            arc_rate: 1.5,
            memory_size: 5,
        });
    }
    if matches!(strategy, Strategy::AdaptiveBin | Strategy::AdaptiveExp) {
        builder = builder.adaptive(crate::AdaptiveConfig {
            adaptive_mutation: true,
            wls_enabled: false,
            w_max: 0.8,
            w_min: 0.2,
            w_f: 0.7,
            w_cr: 0.6,
            f_m: 0.65,
            cr_m: 0.55,
            wls_prob: 0.0,
            wls_scale: 0.0,
        });
    }
    builder
        .build()
        .expect("valid continuation test configuration")
}

fn assert_same_report(left: &crate::DEReport, right: &crate::DEReport) {
    assert_eq!(left.x, right.x);
    assert_eq!(left.fun, right.fun);
    assert_eq!(left.success, right.success);
    assert_eq!(left.message, right.message);
    assert_eq!(left.nit, right.nit);
    assert_eq!(left.nfev, right.nfev);
    assert_eq!(left.population, right.population);
    assert_eq!(left.population_energies, right.population_energies);
}

fn interrupt_at_generation(
    strategy: Strategy,
    parallel: bool,
    seed: u64,
    maxiter: usize,
    generation: usize,
) -> DECheckpoint {
    interrupt_with_config(config(strategy, parallel, seed, maxiter), generation)
}

fn interrupt_with_config(config: crate::DEConfig, generation: usize) -> DECheckpoint {
    let mut solver =
        DifferentialEvolution::new(&sphere, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
    *solver.config_mut() = config;
    let mut saved = None;
    let mut save = |checkpoint: &DECheckpoint| {
        if checkpoint.generation == generation {
            saved = Some(checkpoint.clone());
            Err("test process interruption after a committed barrier".to_owned())
        } else {
            Ok(())
        }
    };
    let result = solver.solve_with_checkpoint(None, "sphere-fixture-v1", Some(&mut save));
    assert!(result.is_err(), "test callback should stop at its barrier");
    let state = saved.expect("generation callback should capture the barrier state");
    let encoded = serde_json::to_vec(&state).expect("checkpoint JSON serialization");
    serde_json::from_slice(&encoded).expect("checkpoint JSON round trip")
}

#[test]
fn adaptive_and_lshade_resume_match_uninterrupted_serial_and_parallel_runs() {
    let cases = [
        (Strategy::AdaptiveBin, false),
        (Strategy::AdaptiveBin, true),
        (Strategy::LShadeBin, false),
        (Strategy::LShadeBin, true),
    ];
    for (strategy, parallel) in cases {
        let seed = 0xa08_0000 + u64::from(parallel);
        let full_config = config(strategy, parallel, seed, 14);
        let mut uninterrupted =
            DifferentialEvolution::new(&sphere, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
        *uninterrupted.config_mut() = full_config;
        let expected = uninterrupted.solve();

        let checkpoint = interrupt_at_generation(strategy, parallel, seed, 14, 5);
        let mut resumed =
            DifferentialEvolution::new(&sphere, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
        *resumed.config_mut() = config(strategy, parallel, seed, 14);
        let actual = resumed
            .solve_with_checkpoint(Some(&checkpoint), "sphere-fixture-v1", None)
            .expect("valid exact continuation");
        assert_same_report(&expected, &actual);
    }
}

#[test]
fn best_and_rand_binomial_and_exponential_resume_at_initial_and_mid_run_barriers() {
    let cases = [
        (Strategy::Best1Bin, 0),
        (Strategy::Best1Exp, 4),
        (Strategy::Rand1Bin, 0),
        (Strategy::Rand1Exp, 4),
    ];
    for (index, (strategy, barrier)) in cases.into_iter().enumerate() {
        let seed = 0xa08_2000 + index as u64;
        let mut uninterrupted =
            DifferentialEvolution::new(&sphere, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
        *uninterrupted.config_mut() = config(strategy, false, seed, 11);
        let expected = uninterrupted.solve();

        let checkpoint = interrupt_with_config(config(strategy, false, seed, 11), barrier);
        let mut resumed =
            DifferentialEvolution::new(&sphere, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
        *resumed.config_mut() = config(strategy, false, seed, 11);
        let actual = resumed
            .solve_with_checkpoint(Some(&checkpoint), "sphere-fixture-v1", None)
            .expect("valid exact continuation");
        assert_same_report(&expected, &actual);
    }
}

#[test]
fn continuation_preserves_x0_integrality_and_adaptive_wls_state() {
    let seed = 0xa08_2345;
    let mut uninterrupted =
        DifferentialEvolution::new(&sphere, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
    let mut x0_integral = config(Strategy::Rand1Bin, false, seed, 9);
    x0_integral.x0 = Some(array![0.17, 0.51]);
    x0_integral.integrality = Some(vec![false, true]);
    *uninterrupted.config_mut() = x0_integral;
    let expected = uninterrupted.solve();
    let mut x0_integral_for_checkpoint = config(Strategy::Rand1Bin, false, seed, 9);
    x0_integral_for_checkpoint.x0 = Some(array![0.17, 0.51]);
    x0_integral_for_checkpoint.integrality = Some(vec![false, true]);
    let checkpoint = interrupt_with_config(x0_integral_for_checkpoint, 3);
    let mut resumed =
        DifferentialEvolution::new(&sphere, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
    let mut x0_integral_for_resume = config(Strategy::Rand1Bin, false, seed, 9);
    x0_integral_for_resume.x0 = Some(array![0.17, 0.51]);
    x0_integral_for_resume.integrality = Some(vec![false, true]);
    *resumed.config_mut() = x0_integral_for_resume;
    let actual = resumed
        .solve_with_checkpoint(Some(&checkpoint), "sphere-fixture-v1", None)
        .expect("x0/integrality continuation");
    assert_same_report(&expected, &actual);

    let mut uninterrupted =
        DifferentialEvolution::new(&sphere, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
    let mut adaptive_wls = config(Strategy::AdaptiveExp, true, seed + 1, 9);
    adaptive_wls.adaptive.wls_enabled = true;
    adaptive_wls.adaptive.wls_prob = 1.0;
    adaptive_wls.adaptive.wls_scale = 0.15;
    *uninterrupted.config_mut() = adaptive_wls;
    let expected = uninterrupted.solve();
    let mut adaptive_wls_for_checkpoint = config(Strategy::AdaptiveExp, true, seed + 1, 9);
    adaptive_wls_for_checkpoint.adaptive.wls_enabled = true;
    adaptive_wls_for_checkpoint.adaptive.wls_prob = 1.0;
    adaptive_wls_for_checkpoint.adaptive.wls_scale = 0.15;
    let checkpoint = interrupt_with_config(adaptive_wls_for_checkpoint, 3);
    let mut resumed =
        DifferentialEvolution::new(&sphere, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
    let mut adaptive_wls_for_resume = config(Strategy::AdaptiveExp, true, seed + 1, 9);
    adaptive_wls_for_resume.adaptive.wls_enabled = true;
    adaptive_wls_for_resume.adaptive.wls_prob = 1.0;
    adaptive_wls_for_resume.adaptive.wls_scale = 0.15;
    *resumed.config_mut() = adaptive_wls_for_resume;
    let actual = resumed
        .solve_with_checkpoint(Some(&checkpoint), "sphere-fixture-v1", None)
        .expect("adaptive WLS continuation");
    assert_same_report(&expected, &actual);
}

#[test]
fn terminal_pre_polish_checkpoint_resumes_and_polishes_once() {
    let seed = 0xa08_3456;
    let mut uninterrupted =
        DifferentialEvolution::new(&sphere, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
    let mut full_config = config(Strategy::Best1Bin, false, seed, 8);
    full_config.polish = Some(crate::PolishConfig {
        enabled: true,
        maxeval: 24,
    });
    *uninterrupted.config_mut() = full_config;
    let expected = uninterrupted.solve();

    let mut interrupt_config = config(Strategy::Best1Bin, false, seed, 8);
    interrupt_config.polish = Some(crate::PolishConfig {
        enabled: true,
        maxeval: 24,
    });
    let checkpoint = interrupt_with_config(interrupt_config, 8);
    let terminal = checkpoint.terminal.as_ref().expect("terminal state");
    assert!(!terminal.finalized);
    let mut resumed =
        DifferentialEvolution::new(&sphere, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
    let mut resume_config = config(Strategy::Best1Bin, false, seed, 8);
    resume_config.polish = Some(crate::PolishConfig {
        enabled: true,
        maxeval: 24,
    });
    *resumed.config_mut() = resume_config;
    let actual = resumed
        .solve_with_checkpoint(Some(&checkpoint), "sphere-fixture-v1", None)
        .expect("terminal state resumes through polishing");
    assert_same_report(&expected, &actual);
}

#[test]
fn legacy_uninterrupted_lshade_result_matches_pre_checkpoint_golden() {
    let seed = 0xa08_5eed;
    let mut solver =
        DifferentialEvolution::new(&sphere, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
    *solver.config_mut() = config(Strategy::LShadeBin, false, seed, 15);
    let report = solver.solve();

    assert_eq!(
        report.x,
        array![-0.004_919_928_421_372_892, 0.340_860_068_282_511_8]
    );
    assert_eq!(report.fun, -9.646_452_164_017_405e-6);
    assert_eq!(report.nit, 15);
    assert_eq!(report.nfev, 180);
    assert_eq!(report.population.shape(), &[7, 2]);
    assert_eq!(
        report.population_energies,
        array![
            -9.646_452_164_017_405e-6,
            0.004_119_392_258_304_544,
            0.004_595_184_354_712_096,
            0.005_969_934_809_332_606,
            0.010_875_071_544_890_29,
            0.015_571_101_664_894_369,
            0.026_230_079_800_600_3,
        ]
    );
}

#[test]
fn finalized_terminal_checkpoint_returns_without_re_evaluating_objective() {
    let calls = AtomicUsize::new(0);
    let objective = |x: &Array1<f64>| {
        calls.fetch_add(1, Ordering::SeqCst);
        sphere(x)
    };
    let mut solver =
        DifferentialEvolution::new(&objective, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
    *solver.config_mut() = config(Strategy::LShadeBin, false, 0xa08_1234, 8);
    let mut final_checkpoint = None;
    let mut save = |checkpoint: &DECheckpoint| {
        final_checkpoint = Some(checkpoint.clone());
        Ok(())
    };
    let expected = solver
        .solve_with_checkpoint(None, "sphere-terminal-v1", Some(&mut save))
        .expect("checkpointed run");
    let checkpoint = final_checkpoint.expect("final report checkpoint");
    assert!(
        checkpoint
            .terminal
            .as_ref()
            .is_some_and(|terminal| terminal.finalized)
    );

    calls.store(0, Ordering::SeqCst);
    let mut resumed =
        DifferentialEvolution::new(&objective, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
    *resumed.config_mut() = config(Strategy::LShadeBin, false, 0xa08_1234, 8);
    let actual = resumed
        .solve_with_checkpoint(Some(&checkpoint), "sphere-terminal-v1", None)
        .expect("completed exact state returns directly");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_same_report(&expected, &actual);
}

#[test]
fn rejects_identity_budget_seed_and_malformed_population_before_evaluation() {
    let checkpoint = interrupt_at_generation(Strategy::LShadeBin, false, 0x000a_0855, 12, 4);
    let calls = AtomicUsize::new(0);
    let objective = |x: &Array1<f64>| {
        calls.fetch_add(1, Ordering::SeqCst);
        sphere(x)
    };

    let mut wrong_identity =
        DifferentialEvolution::new(&objective, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
    *wrong_identity.config_mut() = config(Strategy::LShadeBin, false, 0x000a_0855, 12);
    assert!(
        wrong_identity
            .solve_with_checkpoint(Some(&checkpoint), "different-objective", None)
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let mut wrong_population_dimension = checkpoint.clone();
    wrong_population_dimension.population[0] = vec![0.0];
    let mut invalid_population_dimension =
        DifferentialEvolution::new(&objective, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
    *invalid_population_dimension.config_mut() =
        config(Strategy::LShadeBin, false, 0x000a_0855, 12);
    assert!(
        invalid_population_dimension
            .solve_with_checkpoint(Some(&wrong_population_dimension), "sphere-fixture-v1", None)
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let mut wrong_budget =
        DifferentialEvolution::new(&objective, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
    *wrong_budget.config_mut() = config(Strategy::LShadeBin, false, 0x000a_0855, 13);
    assert!(
        wrong_budget
            .solve_with_checkpoint(Some(&checkpoint), "sphere-fixture-v1", None)
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let mut wrong_seed =
        DifferentialEvolution::new(&objective, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
    *wrong_seed.config_mut() = config(Strategy::LShadeBin, false, 0x000a_0856, 12);
    assert!(
        wrong_seed
            .solve_with_checkpoint(Some(&checkpoint), "sphere-fixture-v1", None)
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let mut wrong_build = checkpoint.clone();
    wrong_build.build_identity.push_str("-changed");
    let mut invalid_build =
        DifferentialEvolution::new(&objective, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
    *invalid_build.config_mut() = config(Strategy::LShadeBin, false, 0x000a_0855, 12);
    assert!(
        invalid_build
            .solve_with_checkpoint(Some(&wrong_build), "sphere-fixture-v1", None)
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let mut malformed = checkpoint.clone();
    malformed.population[0][0] = f64::NAN;
    malformed.population_fitness[0] = CheckpointFitness::Finite(f64::NAN);
    let mut invalid =
        DifferentialEvolution::new(&objective, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
    *invalid.config_mut() = config(Strategy::LShadeBin, false, 0x000a_0855, 12);
    assert!(
        invalid
            .solve_with_checkpoint(Some(&malformed), "sphere-fixture-v1", None)
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let mut malformed_cursor = checkpoint.clone();
    malformed_cursor.rng_state[48] |= 0xf0;
    let mut invalid_cursor =
        DifferentialEvolution::new(&objective, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
    *invalid_cursor.config_mut() = config(Strategy::LShadeBin, false, 0x000a_0855, 12);
    assert!(
        invalid_cursor
            .solve_with_checkpoint(Some(&malformed_cursor), "sphere-fixture-v1", None)
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let mut wrong_best_index = checkpoint.clone();
    wrong_best_index.best_index = (checkpoint.best_index + 1) % checkpoint.population.len();
    let mut invalid_best_index =
        DifferentialEvolution::new(&objective, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
    *invalid_best_index.config_mut() = config(Strategy::LShadeBin, false, 0x000a_0855, 12);
    assert!(
        invalid_best_index
            .solve_with_checkpoint(Some(&wrong_best_index), "sphere-fixture-v1", None)
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let mut unpaired_best = checkpoint.clone();
    unpaired_best.best_params = vec![0.0, 0.0];
    let mut invalid_best_params =
        DifferentialEvolution::new(&objective, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
    *invalid_best_params.config_mut() = config(Strategy::LShadeBin, false, 0x000a_0855, 12);
    assert!(
        invalid_best_params
            .solve_with_checkpoint(Some(&unpaired_best), "sphere-fixture-v1", None)
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let mut malformed_archive = checkpoint.clone();
    malformed_archive
        .archive
        .as_mut()
        .expect("L-SHADE archive")
        .solutions
        .push(vec![0.0]);
    let mut invalid_archive =
        DifferentialEvolution::new(&objective, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
    *invalid_archive.config_mut() = config(Strategy::LShadeBin, false, 0x000a_0855, 12);
    assert!(
        invalid_archive
            .solve_with_checkpoint(Some(&malformed_archive), "sphere-fixture-v1", None)
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let mut adaptive_checkpoint =
        interrupt_at_generation(Strategy::AdaptiveBin, false, 0x000a_0857, 12, 4);
    adaptive_checkpoint
        .adaptive
        .as_mut()
        .expect("adaptive state")
        .f_m = f64::NAN;
    let mut invalid_adaptive =
        DifferentialEvolution::new(&objective, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
    *invalid_adaptive.config_mut() = config(Strategy::AdaptiveBin, false, 0x000a_0857, 12);
    assert!(
        invalid_adaptive
            .solve_with_checkpoint(Some(&adaptive_checkpoint), "sphere-fixture-v1", None)
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let mut terminal_checkpoint =
        interrupt_at_generation(Strategy::Best1Bin, false, 0x000a_0858, 12, 12);
    let terminal = terminal_checkpoint
        .terminal
        .as_mut()
        .expect("terminal state");
    terminal.finalized = true;
    terminal.final_params = Some(vec![f64::NAN, 0.0]);
    terminal.final_fitness = Some(CheckpointFitness::Finite(0.0));
    let mut invalid_terminal =
        DifferentialEvolution::new(&objective, array![-2.5, -2.0], array![2.5, 2.0]).unwrap();
    *invalid_terminal.config_mut() = config(Strategy::Best1Bin, false, 0x000a_0858, 12);
    assert!(
        invalid_terminal
            .solve_with_checkpoint(Some(&terminal_checkpoint), "sphere-fixture-v1", None)
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

use super::{BayesOptConfig, bayesian_multi_objective, bayesian_multi_objective_with_stop};
use ndarray::Array1;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

fn config() -> BayesOptConfig {
    let mut config = BayesOptConfig {
        bounds: vec![(0.0, 1.0)],
        initial_samples: 4,
        maxeval: 6,
        candidate_pool_size: 32,
        seed: Some(42),
        ..Default::default()
    };
    config.parallel.enabled = false;
    config
}

fn objective(x: &Array1<f64>) -> Vec<f64> {
    vec![x[0].powi(2), (x[0] - 1.0).powi(2)]
}

#[test]
fn pre_stopped_ehvi_does_not_evaluate() {
    let report = bayesian_multi_objective_with_stop(
        &|_| panic!("pre-stopped run must not evaluate"),
        config(),
        &|| true,
    )
    .unwrap();
    assert_eq!(report.nfev, 0);
    assert_eq!(report.nit, 0);
    assert!(report.population.is_empty());
    assert!(!report.success);
    assert_eq!(report.message, "stop requested");
}

#[test]
fn ehvi_stop_during_initial_design_keeps_completed_evaluations() {
    for parallel in [false, true] {
        let count = AtomicUsize::new(0);
        let stop = AtomicBool::new(false);
        let mut config = config();
        config.parallel.enabled = parallel;
        let report = bayesian_multi_objective_with_stop(
            &|x| {
                count.fetch_add(1, Ordering::SeqCst);
                stop.store(true, Ordering::SeqCst);
                objective(x)
            },
            config,
            &|| stop.load(Ordering::SeqCst),
        )
        .unwrap();
        assert_eq!(report.nfev, count.load(Ordering::SeqCst));
        assert_eq!(report.population.len(), report.nfev);
        assert!((1..=4).contains(&report.nfev));
        if !parallel {
            assert_eq!(report.nfev, 1);
        }
        assert_eq!(report.nit, 0);
        assert!(!report.success);
        assert_eq!(report.message, "stop requested");
        for point in report.population {
            assert_eq!(point.objectives, objective(&point.x));
        }
    }
}

#[test]
fn ehvi_stop_during_surrogate_work_does_not_admit_next_batch() {
    let evaluations = AtomicUsize::new(0);
    let surrogate_checks = AtomicUsize::new(0);
    let report = bayesian_multi_objective_with_stop(
        &|x| {
            evaluations.fetch_add(1, Ordering::SeqCst);
            objective(x)
        },
        config(),
        &|| {
            evaluations.load(Ordering::SeqCst) == 4
                && surrogate_checks.fetch_add(1, Ordering::SeqCst) >= 4
        },
    )
    .unwrap();
    assert_eq!(evaluations.load(Ordering::SeqCst), 4);
    assert_eq!(report.nfev, 4);
    assert_eq!(report.nit, 0);
    assert!(!report.success);
    assert_eq!(report.message, "stop requested");
}

#[test]
fn no_stop_ehvi_preserves_seeded_results() {
    let ordinary = bayesian_multi_objective(&objective, config()).unwrap();
    let controlled = bayesian_multi_objective_with_stop(&objective, config(), &|| false).unwrap();
    assert_eq!(ordinary.nfev, 6);
    assert_eq!(ordinary.nfev, controlled.nfev);
    assert_eq!(ordinary.nit, controlled.nit);
    assert_eq!(ordinary.message, controlled.message);
    for (a, b) in ordinary.population.iter().zip(&controlled.population) {
        assert_eq!(a.x, b.x);
        assert_eq!(a.objectives, b.objectives);
    }
}

#[test]
fn stop_observation_is_latched() {
    let count = AtomicUsize::new(0);
    let predicate = || count.fetch_add(1, Ordering::SeqCst) == 0;
    let stop = super::stop::StopCheck::new(&predicate);
    assert!(stop.requested());
    assert!(stop.requested());
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn ehvi_acquisition_stop_discards_partial_batch() {
    let xs = vec![Array1::from(vec![0.0]), Array1::from(vec![1.0])];
    let gp =
        super::gaussian_process::GaussianProcess::fit(&xs, &[0.0, 1.0], &[0.4], 1.0, 1e-8).unwrap();
    let candidates = vec![Array1::from(vec![0.25]), Array1::from(vec![0.75])];
    let checks = AtomicUsize::new(0);
    // Stop between scoring the first and second candidates, after setup.
    let predicate = || checks.fetch_add(1, Ordering::SeqCst) == 3;
    let stop = super::stop::StopCheck::new(&predicate);
    let selected = super::select::select_ehvi_batch(
        &[gp],
        &candidates,
        super::select::EhviFront {
            values: &[vec![0.0]],
            reference: &[2.0],
        },
        1,
        &config(),
        &mut super::misc::make_rng(Some(42)),
        &stop,
    );
    assert!(stop.observed());
    assert!(selected.is_empty());
    assert_eq!(checks.load(Ordering::SeqCst), 4);
}

#[test]
fn ehvi_stop_on_last_evaluation_is_reported() {
    let stopped = AtomicBool::new(false);
    let mut config = config();
    config.maxeval = 1;
    let report = bayesian_multi_objective_with_stop(
        &|x| {
            stopped.store(true, Ordering::SeqCst);
            objective(x)
        },
        config,
        &|| stopped.load(Ordering::SeqCst),
    )
    .unwrap();
    assert_eq!(report.nfev, 1);
    assert_eq!(report.message, "stop requested");
    assert!(!report.success);
}

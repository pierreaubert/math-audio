use super::*;
use ndarray::{Array1, array};
use std::sync::Mutex;

fn config(variant: NsgaVariant, seed: u64) -> NsgaConfig {
    NsgaConfig {
        variant,
        seed: Some(seed),
        population_size: 5,
        maxeval: 31,
        bounds: vec![(-1.0, 1.0), (-2.0, 2.0), (0.0, 0.0)],
        x0: Some(array![-1.0, 0.5, 0.0]),
        reference_partitions: 3,
        ..NsgaConfig::default()
    }
}
fn objectives(x: &Array1<f64>) -> Vec<f64> {
    vec![
        x.iter().map(|v| v * v).sum(),
        x.iter().map(|v| (v - 0.7).powi(2)).sum(),
        (x[0] + 1.0).powi(2) + x[1] * x[1],
    ]
}
type PopulationBits = Vec<(Vec<u64>, Vec<u64>, usize, u64)>;
fn population_bits(report: &NsgaReport) -> PopulationBits {
    report
        .population
        .iter()
        .map(|p| {
            (
                p.x.iter().map(|v| v.to_bits()).collect(),
                p.objectives.iter().map(|v| v.to_bits()).collect(),
                p.rank,
                p.crowding_distance.to_bits(),
            )
        })
        .collect()
}
fn completed(outcome: NsgaCheckpointOutcome) -> NsgaReport {
    match outcome {
        NsgaCheckpointOutcome::Completed(report) => report,
        NsgaCheckpointOutcome::Paused(_) => panic!("expected completion"),
    }
}

#[test]
fn checkpoint_serialized_resume_matches_every_candidate_and_population_bit() {
    for variant in [NsgaVariant::Nsga2, NsgaVariant::Nsga3] {
        for seed in [0, 42, 1234] {
            let config = config(variant, seed);
            let full_trace = Mutex::new(Vec::new());
            let full_f = |x: &Array1<f64>| {
                full_trace
                    .lock()
                    .unwrap()
                    .push(x.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                objectives(x)
            };
            let full = completed(
                nsga_checkpointed(&full_f, config.clone(), 3, "fixture-v1", None, |_| {
                    Ok(NsgaCheckpointAction::Continue)
                })
                .unwrap(),
            );
            assert_eq!(full.nfev, 31);
            let legacy_trace = Mutex::new(Vec::new());
            let legacy = nsga(
                &|x| {
                    legacy_trace
                        .lock()
                        .unwrap()
                        .push(x.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                    objectives(x)
                },
                config.clone(),
            )
            .unwrap();
            assert_eq!(*legacy_trace.lock().unwrap(), *full_trace.lock().unwrap());
            assert_eq!(population_bits(&legacy), population_bits(&full));
            for pause_at in [0, 1, 3, 5] {
                let trace = Mutex::new(Vec::new());
                let f = |x: &Array1<f64>| {
                    trace
                        .lock()
                        .unwrap()
                        .push(x.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                    objectives(x)
                };
                let paused =
                    nsga_checkpointed(&f, config.clone(), 3, "fixture-v1", None, |state| {
                        Ok(if state.generation() == pause_at {
                            NsgaCheckpointAction::Pause
                        } else {
                            NsgaCheckpointAction::Continue
                        })
                    })
                    .unwrap();
                let NsgaCheckpointOutcome::Paused(checkpoint) = paused else {
                    panic!("expected pause");
                };
                assert_eq!(checkpoint.evaluations(), 5 + pause_at * 5);
                assert_eq!(trace.lock().unwrap().len(), checkpoint.evaluations());
                let bytes = serde_json::to_vec(&checkpoint).unwrap();
                let restored = serde_json::from_slice(&bytes).unwrap();
                let mut terminal = None;
                let resumed = completed(
                    nsga_checkpointed(
                        &f,
                        config.clone(),
                        3,
                        "fixture-v1",
                        Some(&restored),
                        |state| {
                            terminal = Some(state.clone());
                            Ok(NsgaCheckpointAction::Continue)
                        },
                    )
                    .unwrap(),
                );
                assert_eq!(*trace.lock().unwrap(), *full_trace.lock().unwrap());
                assert_eq!(population_bits(&resumed), population_bits(&full));
                assert_eq!((resumed.nfev, resumed.nit), (full.nfev, full.nit));
                let terminal = terminal.unwrap();
                let done = completed(
                    nsga_checkpointed(
                        &|_| panic!("terminal resume evaluated objective"),
                        config.clone(),
                        3,
                        "fixture-v1",
                        Some(&terminal),
                        |_| Ok(NsgaCheckpointAction::Pause),
                    )
                    .unwrap(),
                );
                assert_eq!(population_bits(&done), population_bits(&full));
            }
        }
    }
}

fn initial_checkpoint() -> NsgaCheckpoint {
    match nsga_checkpointed(
        &objectives,
        config(NsgaVariant::Nsga2, 42),
        3,
        "fixture-v1",
        None,
        |_| Ok(NsgaCheckpointAction::Pause),
    )
    .unwrap()
    {
        NsgaCheckpointOutcome::Paused(state) => state,
        _ => panic!("expected pause"),
    }
}

#[test]
fn checkpoint_refuses_changed_identity_and_corrupt_state_before_evaluation() {
    let checkpoint = initial_checkpoint();
    let original = config(NsgaVariant::Nsga2, 42);
    for identity in ["changed", ""] {
        assert!(
            nsga_checkpointed(
                &|_| panic!("invalid identity evaluated"),
                original.clone(),
                3,
                identity,
                Some(&checkpoint),
                |_| Ok(NsgaCheckpointAction::Continue)
            )
            .is_err()
        );
    }
    let mut changed = Vec::new();
    macro_rules! change {
        ($field:ident, $value:expr) => {{
            let mut c = original.clone();
            c.$field = $value;
            changed.push(c);
        }};
    }
    change!(variant, NsgaVariant::Nsga3);
    change!(seed, Some(43));
    change!(maxeval, 32);
    change!(population_size, 6);
    change!(crossover_prob, 0.8);
    change!(mutation_prob, Some(0.2));
    change!(eta_c, 16.0);
    change!(eta_m, 21.0);
    change!(reference_partitions, 4);
    change!(x0, None);
    change!(bounds, vec![(-1.0, 1.1), (-2.0, 2.0), (0.0, 0.0)]);
    for c in changed {
        assert!(
            nsga_checkpointed(
                &|_| panic!("changed config evaluated"),
                c,
                3,
                "fixture-v1",
                Some(&checkpoint),
                |_| Ok(NsgaCheckpointAction::Continue)
            )
            .is_err()
        );
    }
    let original = serde_json::to_value(checkpoint).unwrap();
    for (field, replacement) in [
        ("version", serde_json::json!(2)),
        ("implementation", serde_json::json!("wrong")),
        ("build", serde_json::json!("wrong")),
        ("target", serde_json::json!("wrong")),
        ("objective_count", serde_json::json!(4)),
        ("evaluations", serde_json::json!(9)),
        ("generation", serde_json::json!(u64::MAX)),
        ("rng", serde_json::json!([1, 2])),
        ("population", serde_json::json!([])),
        ("checksum", serde_json::json!("bad")),
    ] {
        let mut value = original.clone();
        value["state"][field] = replacement;
        let decoded: NsgaCheckpoint = serde_json::from_value(value).unwrap();
        assert!(
            nsga_checkpointed(
                &|_| panic!("malformed state evaluated"),
                config(NsgaVariant::Nsga2, 42),
                3,
                "fixture-v1",
                Some(&decoded),
                |_| Ok(NsgaCheckpointAction::Continue)
            )
            .is_err()
        );
    }
}

#[test]
fn checkpoint_save_failure_and_invalid_objective_shape_do_not_acknowledge_pause() {
    let count = std::sync::atomic::AtomicUsize::new(0);
    let failure = nsga_checkpointed(
        &|x| {
            count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            objectives(x)
        },
        config(NsgaVariant::Nsga2, 42),
        3,
        "fixture-v1",
        None,
        |_| Err("disk full".into()),
    )
    .unwrap_err();
    assert!(failure.to_string().contains("disk full"));
    assert_eq!(count.load(std::sync::atomic::Ordering::Relaxed), 5);
    assert!(
        nsga_checkpointed(
            &|_| vec![1.0],
            config(NsgaVariant::Nsga2, 42),
            3,
            "fixture-v1",
            None,
            |_| panic!("invalid width must not save")
        )
        .is_err()
    );
    let mut short = config(NsgaVariant::Nsga2, 42);
    short.maxeval = 4;
    assert!(
        nsga_checkpointed(
            &|_| panic!("insufficient budget evaluated"),
            short,
            3,
            "fixture-v1",
            None,
            |_| panic!("insufficient budget saved")
        )
        .is_err()
    );
}

#[test]
fn checkpoint_serializes_nonfinite_sentinel_without_json_nulls() {
    let f = |_: &Array1<f64>| vec![f64::NAN, f64::NEG_INFINITY, 1.0];
    let mut serialized = Vec::new();
    let report = completed(
        nsga_checkpointed(
            &f,
            config(NsgaVariant::Nsga3, 42),
            3,
            "nonfinite-v1",
            None,
            |state| {
                serialized = serde_json::to_vec(state).unwrap();
                Ok(NsgaCheckpointAction::Continue)
            },
        )
        .unwrap(),
    );
    let state = serde_json::from_slice(&serialized).unwrap();
    let restored = completed(
        nsga_checkpointed(
            &|_| panic!("terminal sentinel replay evaluated"),
            config(NsgaVariant::Nsga3, 42),
            3,
            "nonfinite-v1",
            Some(&state),
            |_| Ok(NsgaCheckpointAction::Continue),
        )
        .unwrap(),
    );
    assert_eq!(population_bits(&report), population_bits(&restored));
}

#[test]
fn checkpoint_fresh_process_equivalence() {
    const CHILD_DIR: &str = "MATH_NSGA_CHECKPOINT_TEST_DIRECTORY";
    if let Some(directory) = std::env::var_os(CHILD_DIR) {
        let directory = std::path::PathBuf::from(directory);
        let variant = match std::fs::read_to_string(directory.join("variant"))
            .unwrap()
            .as_str()
        {
            "two" => NsgaVariant::Nsga2,
            "three" => NsgaVariant::Nsga3,
            _ => panic!("invalid test variant"),
        };
        let checkpoint =
            serde_json::from_slice(&std::fs::read(directory.join("checkpoint.json")).unwrap())
                .unwrap();
        let trace = Mutex::new(Vec::new());
        let report = completed(
            nsga_checkpointed(
                &|x| {
                    trace
                        .lock()
                        .unwrap()
                        .push(x.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                    objectives(x)
                },
                config(variant, 987),
                3,
                "process-fixture",
                Some(&checkpoint),
                |_| Ok(NsgaCheckpointAction::Continue),
            )
            .unwrap(),
        );
        let result = serde_json::json!({"population": population_bits(&report),
            "nfev": report.nfev, "nit": report.nit, "trace": *trace.lock().unwrap()});
        std::fs::write(
            directory.join("result.json"),
            serde_json::to_vec(&result).unwrap(),
        )
        .unwrap();
        return;
    }
    for (variant, label) in [(NsgaVariant::Nsga2, "two"), (NsgaVariant::Nsga3, "three")] {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "math-nsga-checkpoint-{}-{nonce}-{label}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).unwrap();
        let full_trace = Mutex::new(Vec::new());
        let report = completed(
            nsga_checkpointed(
                &|x| {
                    full_trace
                        .lock()
                        .unwrap()
                        .push(x.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                    objectives(x)
                },
                config(variant, 987),
                3,
                "process-fixture",
                None,
                |_| Ok(NsgaCheckpointAction::Continue),
            )
            .unwrap(),
        );
        let prefix = Mutex::new(Vec::new());
        let paused = nsga_checkpointed(
            &|x| {
                prefix
                    .lock()
                    .unwrap()
                    .push(x.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                objectives(x)
            },
            config(variant, 987),
            3,
            "process-fixture",
            None,
            |state| {
                Ok(if state.generation() == 2 {
                    NsgaCheckpointAction::Pause
                } else {
                    NsgaCheckpointAction::Continue
                })
            },
        )
        .unwrap();
        let NsgaCheckpointOutcome::Paused(state) = paused else {
            panic!("expected pause");
        };
        std::fs::write(directory.as_path().join("variant"), label).unwrap();
        std::fs::write(
            directory.as_path().join("checkpoint.json"),
            serde_json::to_vec(&state).unwrap(),
        )
        .unwrap();
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "nsga::checkpoint_tests::checkpoint_fresh_process_equivalence",
                "--nocapture",
            ])
            .env(CHILD_DIR, directory.as_path())
            .status()
            .unwrap();
        assert!(status.success());
        let result: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.as_path().join("result.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            result["population"],
            serde_json::to_value(population_bits(&report)).unwrap()
        );
        assert_eq!(result["nfev"], report.nfev);
        assert_eq!(result["nit"], report.nit);
        let suffix: Vec<Vec<u64>> = serde_json::from_value(result["trace"].clone()).unwrap();
        prefix.lock().unwrap().extend(suffix);
        assert_eq!(*prefix.lock().unwrap(), *full_trace.lock().unwrap());
        std::fs::remove_dir_all(&directory).unwrap();
    }
}

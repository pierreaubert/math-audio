//! Exact NSGA continuation at completed generation barriers.

use super::assign::assign_rank_and_crowding;
use super::individual::{advance_generation, initial_population, make_report};
use super::types::Individual;
use super::{NsgaConfig, NsgaReport, NsgaVariant};
use chacha20::ChaCha12Rng;
use ndarray::Array1;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicBool, Ordering};

const IMPLEMENTATION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    ";nsga-state-v1;chacha20-0.10.2;rand-0.10.3"
);

/// An exact, same-build NSGA generation checkpoint.
///
/// Serialize this value without modifying it. The caller's run identity must
/// bind objective code, measurement bytes, and all external numerical settings.
/// Recovery requires the same executable and target, and a deterministic
/// objective. This format does not promise equivalence across machines.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NsgaCheckpoint {
    state: Box<Snapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    version: u32,
    implementation: String,
    build: String,
    target: String,
    run_identity: String,
    config_identity: Vec<u64>,
    objective_count: usize,
    generation: usize,
    evaluations: usize,
    population: Vec<Member>,
    rng: Vec<u8>,
    checksum: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Member {
    parameters: Vec<u64>,
    objectives: Vec<u64>,
    rank: usize,
    crowding: u64,
}

impl NsgaCheckpoint {
    /// Return the number of fully selected generations.
    #[must_use]
    pub fn generation(&self) -> usize {
        self.state.generation
    }

    /// Return the number of objective evaluations already consumed.
    #[must_use]
    pub fn evaluations(&self) -> usize {
        self.state.evaluations
    }
}

/// Requested action after successfully saving a generation checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NsgaCheckpointAction {
    /// Continue searching under the original evaluation budget.
    Continue,
    /// Return the saved checkpoint before evaluating another candidate.
    Pause,
}

/// A completed run or a saved generation barrier.
#[derive(Debug, Clone)]
pub enum NsgaCheckpointOutcome {
    /// The original evaluation budget is exhausted.
    Completed(NsgaReport),
    /// A nonterminal checkpoint was saved and no further work was performed.
    Paused(NsgaCheckpoint),
}

/// A refused NSGA configuration, checkpoint, objective shape, or checkpoint save.
#[derive(Debug, Clone)]
pub struct NsgaCheckpointError {
    message: String,
}

impl std::fmt::Display for NsgaCheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for NsgaCheckpointError {}
fn error(message: impl Into<String>) -> NsgaCheckpointError {
    NsgaCheckpointError {
        message: message.into(),
    }
}

/// Run or resume NSGA with durable generation-barrier checkpoints.
///
/// `save` must persist the supplied checkpoint before returning an action.
/// A save error aborts the run. Terminal barriers always return `Completed`,
/// even when `save` requests a pause. Resuming a terminal barrier performs no
/// objective evaluations. Checkpoints retain population order separately from
/// the sorted report. Nonfinite objective values retain the solver's positive
/// infinity sentinel; inconsistent objective widths are refused.
///
/// # Errors
/// Returns an error for invalid numerical settings, insufficient initialization
/// budget, blank identity, unreadable executable identity, incompatible or
/// malformed checkpoint, inconsistent objective width, or failed persistence.
pub fn nsga_checkpointed<F, S>(
    f: &F,
    config: NsgaConfig,
    objective_count: usize,
    run_identity: &str,
    resume: Option<&NsgaCheckpoint>,
    mut save: S,
) -> Result<NsgaCheckpointOutcome, NsgaCheckpointError>
where
    F: Fn(&Array1<f64>) -> Vec<f64> + Sync,
    S: FnMut(&NsgaCheckpoint) -> Result<NsgaCheckpointAction, String>,
{
    let pop_size = validate_config(&config, objective_count, run_identity)?;
    let build = crate::de_checkpoint::running_build_identity().map_err(error)?;
    let target = format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS);
    let identity = config_identity(&config);
    let invalid_width = AtomicBool::new(false);
    let checked = |x: &Array1<f64>| {
        let values = f(x);
        if values.len() == objective_count {
            values
        } else {
            invalid_width.store(true, Ordering::Relaxed);
            // Keep the current generation well-shaped until the error propagates.
            vec![f64::INFINITY; objective_count]
        }
    };
    let (mut population, mut rng, mut evaluations, mut generation) =
        if let Some(checkpoint) = resume {
            let population = restore(
                checkpoint,
                &config,
                objective_count,
                run_identity,
                &identity,
                &build,
                &target,
                pop_size,
            )?;
            let bytes: [u8; 49] = checkpoint
                .state
                .rng
                .as_slice()
                .try_into()
                .map_err(|_| error("invalid RNG state length"))?;
            (
                population,
                ChaCha12Rng::deserialize_state(&bytes),
                checkpoint.evaluations(),
                checkpoint.generation(),
            )
        } else {
            let mut rng = match config.seed {
                Some(seed) => ChaCha12Rng::seed_from_u64(seed),
                None => ChaCha12Rng::from_rng(&mut rand::rng()),
            };
            let mut population = initial_population(&checked, &config, pop_size, &mut rng);
            assign_rank_and_crowding(&mut population);
            (population, rng, pop_size, 0)
        };
    loop {
        if invalid_width.load(Ordering::Relaxed) {
            return Err(error("objective vector width differs from objective_count"));
        }
        let mut state = Snapshot {
            version: 1,
            implementation: IMPLEMENTATION.into(),
            build: build.clone(),
            target: target.clone(),
            run_identity: run_identity.into(),
            config_identity: identity.clone(),
            objective_count,
            generation,
            evaluations,
            population: population
                .iter()
                .map(|member| Member {
                    parameters: member.x.iter().map(|v| v.to_bits()).collect(),
                    objectives: member.objectives.iter().map(|v| v.to_bits()).collect(),
                    rank: member.rank,
                    crowding: member.crowding_distance.to_bits(),
                })
                .collect(),
            rng: rng.serialize_state().to_vec(),
            checksum: String::new(),
        };
        state.checksum = checksum(&state);
        let checkpoint = NsgaCheckpoint {
            state: Box::new(state),
        };
        let action = save(&checkpoint)
            .map_err(|message| error(format!("NSGA checkpoint save failed: {message}")))?;
        if evaluations >= config.maxeval {
            return Ok(NsgaCheckpointOutcome::Completed(make_report(
                population,
                evaluations,
                generation,
            )));
        }
        if action == NsgaCheckpointAction::Pause {
            return Ok(NsgaCheckpointOutcome::Paused(checkpoint));
        }
        advance_generation(
            &checked,
            &config,
            &mut population,
            &mut rng,
            &mut evaluations,
        );
        generation += 1;
    }
}

fn validate_config(
    config: &NsgaConfig,
    objectives: usize,
    identity: &str,
) -> Result<usize, NsgaCheckpointError> {
    super::nsga_config::validate_config(config).map_err(|e| error(e.to_string()))?;
    if identity.trim().is_empty() || objectives == 0 {
        return Err(error("run identity and objective count must be nonempty"));
    }
    if config
        .bounds
        .iter()
        .any(|(lo, hi)| !lo.is_finite() || !hi.is_finite() || !(hi - lo).is_finite())
        || config
            .x0
            .as_ref()
            .is_some_and(|x| x.iter().any(|v| !v.is_finite()))
        || !config.crossover_prob.is_finite()
        || !(0.0..=1.0).contains(&config.crossover_prob)
        || config
            .mutation_prob
            .is_some_and(|p| !p.is_finite() || !(0.0..=1.0).contains(&p))
        || !config.eta_c.is_finite()
        || !config.eta_m.is_finite()
    {
        return Err(error(
            "NSGA checkpoint run requires finite bounds and valid probabilities",
        ));
    }
    let population = if config.population_size == 0 {
        config
            .bounds
            .len()
            .checked_mul(10)
            .ok_or_else(|| error("population size overflow"))?
            .max(40)
    } else {
        config.population_size
    }
    .max(4);
    if config.maxeval < population {
        return Err(error(
            "evaluation budget cannot initialize the full NSGA population",
        ));
    }
    Ok(population)
}

fn config_identity(config: &NsgaConfig) -> Vec<u64> {
    let mut identity = vec![
        config.bounds.len() as u64,
        config.population_size as u64,
        config.maxeval as u64,
        config.crossover_prob.to_bits(),
        config.eta_c.to_bits(),
        config.eta_m.to_bits(),
        config.reference_partitions as u64,
        u64::from(config.variant == NsgaVariant::Nsga3),
        u64::from(config.seed.is_some()),
        config.seed.unwrap_or_default(),
        u64::from(config.mutation_prob.is_some()),
        config.mutation_prob.unwrap_or_default().to_bits(),
        u64::from(config.x0.is_some()),
    ];
    for (lo, hi) in &config.bounds {
        identity.extend([lo.to_bits(), hi.to_bits()]);
    }
    if let Some(x0) = &config.x0 {
        identity.extend(x0.iter().map(|v| v.to_bits()));
    }
    identity
}

fn checksum(state: &Snapshot) -> String {
    // The executable and schema identities pin Debug's exact representation.
    // Clear the checksum field to avoid hashing the checksum itself.
    let mut payload = state.clone();
    payload.checksum.clear();
    crate::de_checkpoint::format_digest(Sha256::digest(format!("{payload:?}").as_bytes()))
}

#[expect(
    clippy::too_many_arguments,
    reason = "Restore validates each independent checkpoint identity before any objective call"
)]
fn restore(
    checkpoint: &NsgaCheckpoint,
    config: &NsgaConfig,
    objectives: usize,
    run_identity: &str,
    identity: &[u64],
    build: &str,
    target: &str,
    pop_size: usize,
) -> Result<Vec<Individual>, NsgaCheckpointError> {
    let state = &checkpoint.state;
    if state.version != 1
        || state.implementation != IMPLEMENTATION
        || state.build != build
        || state.target != target
        || state.run_identity != run_identity
        || state.config_identity != identity
        || state.objective_count != objectives
    {
        return Err(error("NSGA checkpoint identity mismatch"));
    }
    if state.checksum != checksum(state) {
        return Err(error("NSGA checkpoint checksum mismatch"));
    }
    let max_generations = (config.maxeval - pop_size).div_ceil(pop_size);
    let expected = pop_size
        .saturating_add(state.generation.saturating_mul(pop_size))
        .min(config.maxeval);
    if state.population.len() != pop_size
        || state.generation > max_generations
        || state.evaluations != expected
        || state.rng.len() != 49
        || state.rng[48] & 0xf0 != 0
    {
        return Err(error(
            "invalid NSGA checkpoint population, counters, or RNG state",
        ));
    }
    let mut population = Vec::with_capacity(pop_size);
    for member in &state.population {
        if member.parameters.len() != config.bounds.len() || member.objectives.len() != objectives {
            return Err(error("invalid NSGA checkpoint member dimensions"));
        }
        let x = Array1::from_iter(member.parameters.iter().map(|bits| f64::from_bits(*bits)));
        let values: Vec<_> = member
            .objectives
            .iter()
            .map(|bits| f64::from_bits(*bits))
            .collect();
        let crowding = f64::from_bits(member.crowding);
        if x.iter()
            .zip(&config.bounds)
            .any(|(v, (lo, hi))| !v.is_finite() || v < lo || v > hi)
            || values
                .iter()
                .any(|v| !(v.is_finite() || *v == f64::INFINITY))
            || crowding.is_nan()
            || crowding < 0.0
            || member.rank >= pop_size
        {
            return Err(error("invalid NSGA checkpoint member values"));
        }
        population.push(Individual {
            x,
            objectives: values,
            rank: member.rank,
            crowding_distance: crowding,
        });
    }
    let mut ranked = population.clone();
    assign_rank_and_crowding(&mut ranked);
    if population.iter().zip(&ranked).any(|(a, b)| {
        a.rank != b.rank || a.crowding_distance.to_bits() != b.crowding_distance.to_bits()
    }) {
        return Err(error("NSGA checkpoint rank or crowding mismatch"));
    }
    Ok(population)
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn malformed_state_with_valid_checksum_is_refused_before_objective() {
        let config = NsgaConfig {
            bounds: vec![(-1.0, 1.0)],
            population_size: 4,
            maxeval: 12,
            seed: Some(7),
            ..NsgaConfig::default()
        };
        let initial = nsga_checkpointed(
            &|x| vec![x[0] * x[0], (x[0] - 0.5).powi(2)],
            config.clone(),
            2,
            "validation-fixture",
            None,
            |_| Ok(NsgaCheckpointAction::Pause),
        )
        .unwrap();
        let NsgaCheckpointOutcome::Paused(initial) = initial else {
            panic!("expected pause");
        };
        for mutation in 0..10 {
            let mut checkpoint = initial.clone();
            match mutation {
                0 => checkpoint.state.evaluations += 1,
                1 => checkpoint.state.generation = usize::MAX,
                2 => checkpoint.state.rng[48] |= 0xf0,
                3 => checkpoint.state.population[0].parameters[0] = f64::NAN.to_bits(),
                4 => checkpoint.state.population[0].parameters[0] = 2.0_f64.to_bits(),
                5 => checkpoint.state.population[0].objectives[0] = f64::NEG_INFINITY.to_bits(),
                6 => checkpoint.state.population[0].rank = usize::MAX,
                7 => checkpoint.state.population[0].crowding = (-1.0_f64).to_bits(),
                8 => checkpoint.state.population[0].objectives.clear(),
                9 => checkpoint.state.population.clear(),
                _ => unreachable!(),
            }
            checkpoint.state.checksum = checksum(&checkpoint.state);
            let outcome = nsga_checkpointed(
                &|_| panic!("malformed checkpoint evaluated objective"),
                config.clone(),
                2,
                "validation-fixture",
                Some(&checkpoint),
                |_| panic!("malformed checkpoint reached persistence"),
            );
            assert!(outcome.is_err(), "mutation {mutation} was accepted");
        }
    }
}

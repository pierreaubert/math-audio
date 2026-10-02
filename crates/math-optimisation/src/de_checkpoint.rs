//! Versioned exact-continuation state for Differential Evolution.
//!
//! A checkpoint is emitted only at a generation barrier, after population
//! selection, L-SHADE reduction, adaptive updates, and best-member updates.
//! The caller-provided run identity must bind the objective and any opaque
//! callback/constraint semantics that are not represented by [`crate::DEConfig`].

use serde::{Deserialize, Serialize};

/// Current serialized format for [`DECheckpoint`].
pub const DE_CHECKPOINT_VERSION: u32 = 1;

/// Algorithm and RNG versions required to restore an exact DE state.
pub const DE_CHECKPOINT_IMPLEMENTATION_ID: &str = concat!(
    "math-optimisation/",
    env!("CARGO_PKG_VERSION"),
    ";de-state-v1;rand-0.10.3;chacha20-0.10.2;runtime-build-sha256-v1"
);

/// Finite objective value or the DE solver's normalized positive-infinity
/// sentinel. JSON cannot faithfully represent IEEE infinity as a number.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum CheckpointFitness {
    /// A finite objective value.
    Finite(f64),
    /// The solver's normalized value for a non-finite objective result.
    PositiveInfinity,
}

impl CheckpointFitness {
    pub(crate) fn from_solver(value: f64) -> Self {
        if value.is_finite() {
            Self::Finite(value)
        } else {
            Self::PositiveInfinity
        }
    }

    pub(crate) fn into_solver(self) -> f64 {
        match self {
            Self::Finite(value) => value,
            Self::PositiveInfinity => f64::INFINITY,
        }
    }

    pub(crate) fn is_valid(self) -> bool {
        match self {
            Self::PositiveInfinity => true,
            Self::Finite(value) => value.is_finite(),
        }
    }
}

/// Snapshot of the adaptive parameters at a generation barrier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DEAdaptiveCheckpoint {
    /// Current mutation-memory value.
    pub f_m: f64,
    /// Current crossover-memory value.
    pub cr_m: f64,
    /// Successful mutation values accumulated since the last update.
    pub successful_f: Vec<f64>,
    /// Successful crossover values accumulated since the last update.
    pub successful_cr: Vec<f64>,
    /// Current adaptive mutation weight.
    pub current_w: f64,
}

/// Snapshot of the ordered L-SHADE external archive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DEArchiveCheckpoint {
    /// Maximum number of stored solutions.
    pub capacity: usize,
    /// Archived solutions in their current selection order.
    pub solutions: Vec<Vec<f64>>,
}

/// Terminal DE status and, when finalized, the post-polish report values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DETerminalCheckpoint {
    /// Whether DE terminated successfully (convergence or callback stop).
    pub success: bool,
    /// Report message chosen by the solver at the termination barrier.
    pub message: String,
    /// Whether optional polishing and report finalization have completed.
    pub finalized: bool,
    /// Final report parameter vector, present only when finalized.
    pub final_params: Option<Vec<f64>>,
    /// Final report fitness, present only when finalized.
    pub final_fitness: Option<CheckpointFitness>,
    /// Additional objective evaluations performed by polishing.
    pub polish_evaluations: usize,
}

/// Exact DE continuation state captured at a safe generation barrier.
///
/// This state is valid only between runs with matching implementation,
/// executable build, target/runtime environment, configuration, and
/// caller-provided objective identities. It is a same-build recovery format,
/// not a promise of bitwise equivalence across toolchains or machines. It
/// contains no closures; the caller must reconstruct the same objective and
/// supply the same `run_identity` before resuming.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DECheckpoint {
    /// Serialized checkpoint format version.
    pub checkpoint_version: u32,
    /// Caller identity for measurement/objective/callback semantics.
    pub run_identity: String,
    /// Solver and RNG implementation identity.
    pub implementation_identity: String,
    /// Content fingerprint of the DE algorithm, adaptation, mutation, and
    /// initialization sources used by the checkpoint producer.
    pub solver_source_identity: String,
    /// Architecture and operating-system identity for target-dependent RNGs.
    pub target_identity: String,
    /// SHA-256 of the running executable, binding compiler, dependencies,
    /// features, and linked numeric implementation to this state.
    pub build_identity: String,
    /// Exact fingerprint of the DE configuration and bounds.
    pub configuration_fingerprint: String,
    /// Seed used by the original run.
    pub seed: u64,
    /// Completed generation count.
    pub generation: usize,
    /// Objective evaluation count before optional polishing.
    pub evaluations: usize,
    /// Population vectors in current solver order.
    pub population: Vec<Vec<f64>>,
    /// Population objective values in current solver order.
    pub population_fitness: Vec<CheckpointFitness>,
    /// Current best index used by mutation strategies.
    pub best_index: usize,
    /// Current best solution, retained separately from the population index.
    pub best_params: Vec<f64>,
    /// Current best objective value.
    pub best_fitness: CheckpointFitness,
    /// Portable state of the pinned ChaCha12 global RNG.
    pub rng_state: Vec<u8>,
    /// Adaptive strategy state, when the selected strategy uses it.
    pub adaptive: Option<DEAdaptiveCheckpoint>,
    /// Ordered L-SHADE archive, when the selected strategy uses it.
    pub archive: Option<DEArchiveCheckpoint>,
    /// Present after convergence, callback stop, or the generation limit.
    pub terminal: Option<DETerminalCheckpoint>,
}

pub(crate) fn solver_source_identity() -> &'static str {
    static SOURCE_IDENTITY: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SOURCE_IDENTITY.get_or_init(calculate_solver_source_identity)
}

pub(crate) fn running_build_identity() -> Result<String, String> {
    calculate_running_build_identity()
}

fn calculate_running_build_identity() -> Result<String, String> {
    use sha2::Digest;
    use std::io::Read;

    let executable = std::env::current_exe()
        .map_err(|error| format!("cannot identify current executable: {error}"))?;
    let mut file = std::fs::File::open(&executable).map_err(|error| {
        format!(
            "cannot read current executable {}: {error}",
            executable.display()
        )
    })?;
    let mut hasher = sha2::Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("cannot hash current executable: {error}"))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("sha256:{}", format_digest(hasher.finalize())))
}

fn format_digest(digest: impl IntoIterator<Item = u8>) -> String {
    let mut result = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        write!(&mut result, "{byte:02x}").expect("writing digest to String cannot fail");
    }
    result
}

fn calculate_solver_source_identity() -> String {
    use sha2::{Digest, Sha256};

    const SOURCES: &[(&str, &str)] = &[
        ("de_checkpoint.rs", include_str!("de_checkpoint.rs")),
        (
            "differential_evolution_mod/differential_evolution.rs",
            include_str!("differential_evolution_mod/differential_evolution.rs"),
        ),
        (
            "differential_evolution.rs",
            include_str!("differential_evolution.rs"),
        ),
        ("impl_helpers.rs", include_str!("impl_helpers.rs")),
        ("adaptive_state.rs", include_str!("adaptive_state.rs")),
        ("external_archive.rs", include_str!("external_archive.rs")),
        (
            "init_latin_hypercube.rs",
            include_str!("init_latin_hypercube.rs"),
        ),
        ("init_random.rs", include_str!("init_random.rs")),
        ("apply_integrality.rs", include_str!("apply_integrality.rs")),
        ("apply_wls.rs", include_str!("apply_wls.rs")),
        ("argmin.rs", include_str!("argmin.rs")),
        (
            "crossover_binomial.rs",
            include_str!("crossover_binomial.rs"),
        ),
        (
            "crossover_exponential.rs",
            include_str!("crossover_exponential.rs"),
        ),
        ("deconfig.rs", include_str!("deconfig.rs")),
        ("deconfig_builder.rs", include_str!("deconfig_builder.rs")),
        ("lshade.rs", include_str!("lshade.rs")),
        ("mutation.rs", include_str!("mutation.rs")),
        ("strategy.rs", include_str!("strategy.rs")),
        ("types.rs", include_str!("types.rs")),
        ("parallel_eval.rs", include_str!("parallel_eval.rs")),
        ("mutant_adaptive.rs", include_str!("mutant_adaptive.rs")),
        ("mutant_best1.rs", include_str!("mutant_best1.rs")),
        ("mutant_best2.rs", include_str!("mutant_best2.rs")),
        (
            "mutant_current_to_best1.rs",
            include_str!("mutant_current_to_best1.rs"),
        ),
        (
            "mutant_current_to_pbest1.rs",
            include_str!("mutant_current_to_pbest1.rs"),
        ),
        ("mutant_rand1.rs", include_str!("mutant_rand1.rs")),
        ("mutant_rand2.rs", include_str!("mutant_rand2.rs")),
        (
            "mutant_rand_to_best1.rs",
            include_str!("mutant_rand_to_best1.rs"),
        ),
    ];
    let mut hasher = Sha256::new();
    for (path, source) in SOURCES {
        hasher.update(path.as_bytes());
        hasher.update([0]);
        hasher.update(source.as_bytes());
        hasher.update([0xff]);
    }
    format!("sha256:{}", format_digest(hasher.finalize()))
}

impl DECheckpoint {
    /// Return the serialized checkpoint schema version.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.checkpoint_version
    }
}

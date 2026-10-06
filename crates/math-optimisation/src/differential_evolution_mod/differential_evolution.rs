use super::super::adaptive_state::AdaptiveState;
pub(crate) use super::super::argmin::argmin;
use super::super::deconfig::DEConfig;
use super::super::dereport::DEReport;
pub use super::super::error::{DEError, Result};
use super::super::strategy::Strategy;
use super::super::types::CallbackAction;
use super::super::types::Crossover;
use super::super::types::DEIntermediate;
use super::super::types::Init;
use super::super::types::PenaltyTuple;
use crate::de_checkpoint::{
    CheckpointFitness, DE_CHECKPOINT_IMPLEMENTATION_ID, DE_CHECKPOINT_VERSION,
    DEAdaptiveCheckpoint, DECheckpoint, DETerminalCheckpoint, running_build_identity,
    solver_source_identity,
};
use chacha20::ChaCha12Rng;
use ndarray::{Array1, Array2, Zip};
use oxiblas_ndarray::blas::matvec;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Callback called after a complete DE checkpoint barrier.
pub type DECheckpointCallback<'a> =
    dyn FnMut(&DECheckpoint) -> std::result::Result<(), String> + 'a;

/// Differential Evolution optimizer.
///
/// A population-based stochastic optimizer for continuous functions.
/// Use [`DifferentialEvolution::new`] to create an instance, configure
/// with [`config_mut`](Self::config_mut), then call [`solve`](Self::solve).
pub struct DifferentialEvolution<'a, F>
where
    F: Fn(&Array1<f64>) -> f64 + Sync,
{
    pub(in super::super) func: &'a F,
    pub(in super::super) lower: Array1<f64>,
    pub(in super::super) upper: Array1<f64>,
    pub(in super::super) config: DEConfig,
}

struct DECheckpointRuntime<'a> {
    generation: usize,
    evaluations: usize,
    population: &'a Array2<f64>,
    energies: &'a Array1<f64>,
    best_index: usize,
    best_fitness: f64,
    best_params: &'a Array1<f64>,
    rng: &'a ChaCha12Rng,
    adaptive: Option<&'a AdaptiveState>,
    archive: Option<&'a Arc<RwLock<super::super::external_archive::ExternalArchive>>>,
    terminal: Option<DETerminalCheckpoint>,
}

impl<'a, F> DifferentialEvolution<'a, F>
where
    F: Fn(&Array1<f64>) -> f64 + Sync,
{
    /// Creates a new DE optimizer with objective `func` and bounds [lower, upper].
    ///
    /// # Errors
    ///
    /// Returns `DEError::BoundsMismatch` if `lower` and `upper` have different lengths.
    /// Returns `DEError::InvalidBounds` if any lower bound exceeds its corresponding upper bound.
    pub fn new(func: &'a F, lower: Array1<f64>, upper: Array1<f64>) -> Result<Self> {
        if lower.len() != upper.len() {
            return Err(DEError::BoundsMismatch {
                lower_len: lower.len(),
                upper_len: upper.len(),
            });
        }

        // Validate that lower <= upper for all dimensions
        for i in 0..lower.len() {
            if lower[i] > upper[i] {
                return Err(DEError::InvalidBounds {
                    index: i,
                    lower: lower[i],
                    upper: upper[i],
                });
            }
        }

        Ok(Self {
            func,
            lower,
            upper,
            config: DEConfig::default(),
        })
    }

    /// Mutable access to configuration
    pub fn config_mut(&mut self) -> &mut DEConfig {
        &mut self.config
    }

    /// Run the optimization and return a report.
    pub fn solve(&mut self) -> DEReport {
        self.solve_with_checkpoint(None, "", None)
            .expect("fresh DE runs do not fail checkpoint validation or persistence")
    }

    /// Run DE with optional exact-continuation restore and generation-barrier saves.
    ///
    /// `run_identity` must identify the objective and any opaque callback or
    /// constraint semantics. Exact continuation requires a deterministic seed.
    /// The checkpoint callback is invoked after initialization, every complete
    /// generation, and final report polishing; an error stops the run while
    /// leaving persistence policy to the caller.
    pub fn solve_with_checkpoint(
        &mut self,
        resume: Option<&DECheckpoint>,
        run_identity: &str,
        mut checkpoint_callback: Option<&mut DECheckpointCallback<'_>>,
    ) -> Result<DEReport> {
        use super::super::apply_integrality::apply_integrality;
        use super::super::apply_wls::apply_wls_in_place;
        use super::super::crossover_binomial::binomial_crossover_clamp_into;
        use super::super::crossover_exponential::exponential_crossover_clamp_into;
        use super::super::init_latin_hypercube::init_latin_hypercube;
        use super::super::init_random::init_random;
        use super::super::mutant_adaptive::mutant_adaptive_into;
        use super::super::mutant_best1::mutant_best1_into;
        use super::super::mutant_best2::mutant_best2_into;
        use super::super::mutant_current_to_best1::mutant_current_to_best1_into;
        use super::super::mutant_current_to_pbest1::mutant_current_to_pbest1_into;
        use super::super::mutant_rand_to_best1::mutant_rand_to_best1_into;
        use super::super::mutant_rand1::mutant_rand1_into;
        use super::super::mutant_rand2::mutant_rand2_into;
        use super::super::mutation::Mutation;
        use super::super::parallel_eval::evaluate_population_parallel_slice;
        use rayon::prelude::*;
        use std::cell::RefCell;

        // Thread-local scratch used to copy a slice into an Array1 before calling
        // the user-provided objective function (which expects &Array1<f64>).
        thread_local! {
            static X_SCRATCH: RefCell<Array1<f64>> = RefCell::new(Array1::zeros(0));
        }

        let n = self.lower.len();
        let exact_mode = resume.is_some() || checkpoint_callback.is_some();
        if exact_mode {
            if self.config.seed.is_none() {
                return Err(DEError::InvalidCheckpoint {
                    message: "exact continuation requires an explicit deterministic seed".into(),
                });
            }
            if run_identity.trim().is_empty() {
                return Err(DEError::InvalidCheckpoint {
                    message: "exact continuation requires a non-empty caller run identity".into(),
                });
            }
        }
        let build_identity = if exact_mode {
            Some(
                running_build_identity()
                    .map_err(|message| DEError::InvalidCheckpoint { message })?,
            )
        } else {
            None
        };
        let target_identity = exact_mode.then(checkpoint_target_identity);
        if let Some(checkpoint) = resume {
            self.validate_checkpoint(
                checkpoint,
                run_identity,
                build_identity
                    .as_deref()
                    .expect("resumption computes a build identity"),
                target_identity
                    .as_deref()
                    .expect("resumption computes a target identity"),
            )?;
        }

        // Identify fixed (equal-bounds) and free variables
        let mut is_free: Vec<bool> = Vec::with_capacity(n);
        for i in 0..n {
            is_free.push((self.upper[i] - self.lower[i]).abs() > 0.0);
        }
        let n_free = is_free.iter().filter(|&&b| b).count();
        let _n_equal = n - n_free;
        if n_free == 0 {
            if exact_mode {
                return Err(DEError::InvalidCheckpoint {
                    message: "exact continuation is unavailable when all variables are fixed"
                        .into(),
                });
            }
            // All fixed; just evaluate x = lower
            let x_fixed = self.lower.clone();
            let mut x_eval = x_fixed.clone();
            if let Some(mask) = &self.config.integrality {
                apply_integrality(&mut x_eval, mask, &self.lower, &self.upper);
            }
            let f = (self.func)(&x_eval);
            return Ok(DEReport {
                x: x_eval,
                fun: f,
                success: true,
                message: "All variables fixed by bounds".into(),
                nit: 0,
                nfev: 1,
                population: Array2::zeros((1, n)),
                population_energies: Array1::from(vec![f]),
            });
        }

        let is_lshade = matches!(
            self.config.strategy,
            Strategy::LShadeBin | Strategy::LShadeExp
        );
        let initial_npop = if is_lshade {
            self.config.lshade.initial_population_size(n_free)
        } else {
            self.config.popsize * n_free
        };
        let mut npop = initial_npop;
        let max_nfev = if is_lshade {
            self.config.maxiter * npop
        } else {
            0
        };
        let _bounds_span = &self.upper - &self.lower;

        if self.config.disp {
            eprintln!(
                "DE Init: {} dimensions ({} free), population={}, maxiter={}",
                n, n_free, npop, self.config.maxiter
            );
            eprintln!(
                "  Strategy: {:?}, Mutation: {:?}, Crossover: CR={:.3}",
                self.config.strategy, self.config.mutation, self.config.recombination
            );
            eprintln!(
                "  Tolerances: tol={:.2e}, atol={:.2e}",
                self.config.tol, self.config.atol
            );
        }

        // Timing toggle via env var
        let timing_enabled = std::env::var("AUTOEQ_DE_TIMING")
            .map(|v| v != "0")
            .unwrap_or(false);

        // RNG
        let mut rng: ChaCha12Rng = if let Some(state) = resume {
            let serialized_rng: [u8; 49] = state
                .rng_state
                .as_slice()
                .try_into()
                .expect("checkpoint validation checked the ChaCha12 state length");
            ChaCha12Rng::deserialize_state(&serialized_rng)
        } else if let Some(seed) = self.config.seed {
            ChaCha12Rng::seed_from_u64(seed)
        } else {
            let mut thread_rng = rand::rng();
            ChaCha12Rng::from_rng(&mut thread_rng)
        };

        // Initialize population in [lower, upper]
        let mut pop = if let Some(state) = resume {
            checkpoint_matrix(&state.population, n)?
        } else {
            match self.config.init {
                Init::LatinHypercube => {
                    if self.config.disp {
                        eprintln!("  Using Latin Hypercube initialization");
                    }
                    init_latin_hypercube(n, npop, &self.lower, &self.upper, &is_free, &mut rng)
                }
                Init::Random => {
                    if self.config.disp {
                        eprintln!("  Using Random initialization");
                    }
                    init_random(n, npop, &self.lower, &self.upper, &is_free, &mut rng)
                }
            }
        };
        npop = pop.nrows();

        // Evaluate energies (objective + penalties) for a fresh run only.
        if self.config.disp && resume.is_none() {
            eprintln!("  Evaluating initial population of {} individuals...", npop);
        }

        // Build thread-safe energy function that includes penalties
        let func_ref = self.func;
        let penalty_ineq_vec: Vec<PenaltyTuple> = self
            .config
            .penalty_ineq
            .iter()
            .map(|(f, w)| (f.clone(), *w))
            .collect();
        let penalty_eq_vec: Vec<PenaltyTuple> = self
            .config
            .penalty_eq
            .iter()
            .map(|(f, w)| (f.clone(), *w))
            .collect();
        let linear_penalty = self.config.linear_penalty.clone();
        let no_penalties =
            penalty_ineq_vec.is_empty() && penalty_eq_vec.is_empty() && linear_penalty.is_none();

        let energy_fn = Arc::new(move |x: &[f64]| -> f64 {
            // Copy the slice into a reusable per-thread Array1 so the public
            // objective function signature (&Array1<f64>) is preserved without
            // a per-evaluation heap allocation.
            let energy = X_SCRATCH.with(|xs| {
                let mut scratch = xs.borrow_mut();
                if scratch.len() != x.len() {
                    *scratch = Array1::zeros(x.len());
                }
                scratch
                    .as_slice_mut()
                    .expect("contiguous")
                    .copy_from_slice(x);
                let x_arr = &*scratch;

                if no_penalties {
                    // Fast path: most DE benchmarks/use cases have no penalties,
                    // so avoid constructing empty iterators and the linear-penalty
                    // branch inside every evaluation.
                    (func_ref)(x_arr)
                } else {
                    let base = (func_ref)(x_arr);
                    let mut p = 0.0;
                    for (f, w) in &penalty_ineq_vec {
                        let v = f(x_arr);
                        let viol = v.max(0.0);
                        p += w * viol * viol;
                    }
                    for (h, w) in &penalty_eq_vec {
                        let v = h(x_arr);
                        p += w * v * v;
                    }
                    if let Some(ref lp) = linear_penalty {
                        let ax = matvec(&lp.a, x_arr);
                        Zip::from(&ax)
                            .and(&lp.lb)
                            .and(&lp.ub)
                            .for_each(|&v, &lo, &hi| {
                                if v < lo {
                                    let d = lo - v;
                                    p += lp.weight * d * d;
                                } else if v > hi {
                                    let d = v - hi;
                                    p += lp.weight * d * d;
                                }
                            });
                    }
                    base + p
                }
            });
            if energy.is_finite() {
                energy
            } else {
                f64::INFINITY
            }
        });

        let (mut energies, mut nfev, t_integrality, t_eval_init) = if let Some(state) = resume {
            (
                Array1::from_iter(
                    state
                        .population_fitness
                        .iter()
                        .copied()
                        .map(CheckpointFitness::into_solver),
                ),
                state.evaluations,
                std::time::Duration::ZERO,
                std::time::Duration::ZERO,
            )
        } else {
            // Prepare population for evaluation (apply integrality constraints).
            // Borrow the population directly when no integrality is required.
            let t_integrality0 = Instant::now();
            let eval_pop = if let Some(mask) = &self.config.integrality {
                let mut ep = pop.clone();
                for i in 0..npop {
                    let mut row = ep.row_mut(i);
                    apply_integrality(&mut row, mask, &self.lower, &self.upper);
                }
                std::borrow::Cow::Owned(ep)
            } else {
                std::borrow::Cow::Borrowed(&pop)
            };
            let t_integrality = t_integrality0.elapsed();
            let t_eval0 = Instant::now();
            let energies = evaluate_population_parallel_slice(
                eval_pop.as_ref(),
                energy_fn.clone(),
                &self.config.parallel,
            );
            drop(eval_pop); // release the borrow of `pop` before the mutable main loop
            (energies, npop, t_integrality, t_eval0.elapsed())
        };
        if timing_enabled {
            eprintln!(
                "TIMING init: integrality={:.3} ms, eval={:.3} ms",
                t_integrality.as_secs_f64() * 1e3,
                t_eval_init.as_secs_f64() * 1e3
            );
        }

        // Report initial population statistics
        let pop_mean = energies.mean().unwrap_or(0.0);
        let pop_std = energies.std(0.0);
        if self.config.disp {
            eprintln!(
                "  Initial population: mean={:.6e}, std={:.6e}",
                pop_mean, pop_std
            );
        }

        // If x0 provided, override the best member
        if resume.is_none()
            && let Some(x0) = &self.config.x0
        {
            let mut x0c = x0.clone();
            // Clip to bounds using ndarray
            for i in 0..x0c.len() {
                x0c[i] = x0c[i].clamp(self.lower[i], self.upper[i]);
            }
            if let Some(mask) = &self.config.integrality {
                apply_integrality(&mut x0c, mask, &self.lower, &self.upper);
            }
            let f0 = self.energy(&x0c);
            nfev += 1;
            // find current best
            let (best_idx, _best_f) = argmin(&energies);
            pop.row_mut(best_idx).assign(&x0c.view());
            energies[best_idx] = f0;
        }

        let (mut best_idx, mut best_f, mut best_x) = if let Some(state) = resume {
            (
                state.best_index,
                state.best_fitness.into_solver(),
                Array1::from_vec(state.best_params.clone()),
            )
        } else {
            let (best_index, best_fitness) = argmin(&energies);
            (best_index, best_fitness, pop.row(best_index).to_owned())
        };

        if self.config.disp {
            eprintln!(
                "  Initial best: fitness={:.6e} at index {}",
                best_f, best_idx
            );
            let param_summary: Vec<String> = (0..best_x.len() / 3)
                .map(|i| {
                    let freq = 10f64.powf(best_x[i * 3]);
                    let q = best_x[i * 3 + 1];
                    let gain = best_x[i * 3 + 2];
                    format!("f{:.0}Hz/Q{:.2}/G{:.2}dB", freq, q, gain)
                })
                .collect();
            eprintln!("  Initial best params: [{}]", param_summary.join(", "));
        }

        if self.config.disp {
            eprintln!("DE iter {:4}  best_f={:.6e}", 0, best_f);
        }

        // Initialize adaptive state if adaptive strategies are enabled
        let mut adaptive_state = if let Some(state) = resume {
            state.adaptive.as_ref().map(|adaptive| AdaptiveState {
                f_m: adaptive.f_m,
                cr_m: adaptive.cr_m,
                successful_f: adaptive.successful_f.clone(),
                successful_cr: adaptive.successful_cr.clone(),
                current_w: adaptive.current_w,
            })
        } else if matches!(
            self.config.strategy,
            Strategy::AdaptiveBin | Strategy::AdaptiveExp
        ) || self.config.adaptive.adaptive_mutation
        {
            Some(AdaptiveState::new(&self.config.adaptive))
        } else {
            None
        };

        // Initialize external archive for L-SHADE strategies
        let external_archive: Option<Arc<RwLock<super::super::external_archive::ExternalArchive>>> =
            if let Some(state) = resume {
                state.archive.as_ref().map(|archive| {
                    Arc::new(RwLock::new(
                        super::super::external_archive::ExternalArchive::from_checkpoint(archive),
                    ))
                })
            } else if matches!(
                self.config.strategy,
                Strategy::LShadeBin | Strategy::LShadeExp
            ) {
                Some(Arc::new(RwLock::new(
                    super::super::external_archive::ExternalArchive::with_population_size(
                        npop,
                        self.config.lshade.arc_rate,
                    ),
                )))
            } else {
                None
            };

        // Main loop
        let mut terminal_state = resume.and_then(|state| state.terminal.clone());
        let mut success = terminal_state.as_ref().is_some_and(|state| state.success);
        let mut message = terminal_state
            .as_ref()
            .map_or_else(String::new, |state| state.message.clone());
        let mut nit = resume.map_or(0, |state| state.generation);
        let mut accepted_trials;
        let mut improvement_count;

        let mut t_build_tot = std::time::Duration::ZERO;
        let mut t_eval_tot = std::time::Duration::ZERO;
        let mut t_select_tot = std::time::Duration::ZERO;
        let mut t_iter_tot = std::time::Duration::ZERO;

        if resume.is_none() && checkpoint_callback.is_some() {
            let initial_checkpoint = self.create_checkpoint(
                run_identity,
                build_identity
                    .as_deref()
                    .expect("checkpoint callbacks require exact mode"),
                target_identity
                    .as_deref()
                    .expect("checkpoint callbacks require exact mode"),
                DECheckpointRuntime {
                    generation: 0,
                    evaluations: nfev,
                    population: &pop,
                    energies: &energies,
                    best_index: best_idx,
                    best_fitness: best_f,
                    best_params: &best_x,
                    rng: &rng,
                    adaptive: adaptive_state.as_ref(),
                    archive: external_archive.as_ref(),
                    terminal: None,
                },
            )?;
            store_checkpoint(&mut checkpoint_callback, &initial_checkpoint)?;
        }

        // Reusable trial matrix and per-trial parameter storage, allocated once
        // outside the hot loop.  They are only reallocated when L-SHADE shrinks
        // the population.
        let mut trials_buf: Array2<f64> = Array2::zeros((npop, n));
        // Plain `f64` vectors for F/CR and trial energies: writes are to disjoint
        // indices and the parallel loop joins before the sequential read phase,
        // so atomics are unnecessary here.
        let mut trial_f: Vec<f64> = vec![0.0; npop];
        let mut trial_cr: Vec<f64> = vec![0.0; npop];
        let mut trial_energies: Vec<f64> = vec![0.0; npop];

        // One scratch vector per rayon thread for the mutant vector.
        thread_local! {
            static MUTANT_SCRATCH: RefCell<Array1<f64>> = RefCell::new(Array1::zeros(0));
        }

        // Pre-extract contiguous slices for bound clipping; avoids repeated
        // `as_slice()` calls inside the per-individual hot path.
        let lower_slice = self.lower.as_slice().expect("contiguous");
        let upper_slice = self.upper.as_slice().expect("contiguous");

        // Guarded fast path for the benchmark's default configuration:
        // Best1Bin strategy, default dithered mutation, no penalties, no
        // integrality, no WLS, and no adaptive/L-SHADE machinery. This path
        // inlines the trial build and evaluation, removing strategy/crossover
        // enum dispatch and the penalty-free energy wrapper. Non-default configs
        // fall through to the general loop below unchanged.
        let use_fast_path = npop >= 3
            && matches!(self.config.strategy, Strategy::Best1Bin)
            && matches!(self.config.mutation, Mutation::Range { min: 0.0, max: 2.0 })
            && self.config.penalty_ineq.is_empty()
            && self.config.penalty_eq.is_empty()
            && self.config.linear_penalty.is_none()
            && self.config.integrality.is_none()
            && !self.config.adaptive.wls_enabled
            && adaptive_state.is_none()
            && !is_lshade
            && self.config.parallel.enabled;

        if terminal_state.is_none() {
            for iter in (nit + 1)..=self.config.maxiter {
                nit = iter;
                accepted_trials = 0;
                improvement_count = 0;

                let iter_start = Instant::now();

                // Generate all trials into the reusable pre-allocated matrix, then
                // evaluate in parallel.
                let t_build0 = Instant::now();

                if use_fast_path {
                    // Fast path for the default Best1Bin/binomial/no-penalty
                    // configuration. Mirrors the general loop's raw-pointer layout
                    // but removes strategy/crossover enum dispatch, skips F/CR
                    // storage, and calls the objective directly instead of through
                    // the Arc penalty wrapper.
                    let func_ref = self.func;
                    let recombination = self.config.recombination;
                    let seed = self.config.seed;
                    let trials_addr = trials_buf.as_mut_ptr() as usize;
                    let trial_energies_ptr = trial_energies.as_mut_ptr() as usize;
                    (0..npop).into_par_iter().with_min_len(16).for_each(|i| {
                        let trials_ptr = trials_addr as *mut f64;
                        let mut local_rng: SmallRng = if let Some(base_seed) = seed {
                            SmallRng::seed_from_u64(
                                base_seed
                                    .wrapping_add((iter as u64) << 32)
                                    .wrapping_add(i as u64),
                            )
                        } else {
                            let mut thread_rng = rand::rng();
                            SmallRng::from_rng(&mut thread_rng)
                        };

                        let f = local_rng.random_range(0.0..2.0);
                        let cr = recombination;

                        let trial_ptr = unsafe { trials_ptr.add(i * n) };
                        let mut trial_row =
                            unsafe { ndarray::ArrayViewMut1::from_shape_ptr((n,), trial_ptr) };

                        MUTANT_SCRATCH.with(|ms| {
                            let mut scratch = ms.borrow_mut();
                            if scratch.len() != n {
                                *scratch = Array1::zeros(n);
                            }
                            mutant_best1_into(&mut scratch, i, &pop, best_idx, f, &mut local_rng);
                            let target_row = pop.row(i);
                            let target_slice = target_row.as_slice().expect("contiguous row");
                            let mutant_slice = scratch.as_slice().expect("contiguous");
                            let trial_slice = trial_row.as_slice_mut().expect("contiguous row");
                            binomial_crossover_clamp_into(
                                trial_slice,
                                target_slice,
                                mutant_slice,
                                cr,
                                lower_slice,
                                upper_slice,
                                &mut local_rng,
                            );
                        });

                        let trial_slice = trial_row.as_slice().expect("contiguous row");
                        let energy = X_SCRATCH.with(|xs| {
                            let mut x = xs.borrow_mut();
                            if x.len() != n {
                                *x = Array1::zeros(n);
                            }
                            x.as_slice_mut()
                                .expect("contiguous")
                                .copy_from_slice(trial_slice);
                            (func_ref)(&x)
                        });

                        unsafe {
                            *(trial_energies_ptr as *mut f64).add(i) = if energy.is_finite() {
                                energy
                            } else {
                                f64::INFINITY
                            };
                        }
                    });
                } else {
                    // Pre-sort indices for adaptive/L-SHADE strategies to avoid re-sorting in the loop
                    let sorted_indices = if matches!(
                        self.config.strategy,
                        Strategy::AdaptiveBin
                            | Strategy::AdaptiveExp
                            | Strategy::LShadeBin
                            | Strategy::LShadeExp
                    ) {
                        let mut indices: Vec<usize> = (0..npop).collect();
                        indices.sort_by(|&a, &b| {
                            energies[a]
                                .partial_cmp(&energies[b])
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        indices
                    } else {
                        Vec::new() // Not needed for other strategies
                    };

                    // Write directly into the pre-allocated trial matrix using a raw address.
                    // Rows are disjoint, so this is safe despite the parallel write access.
                    let trials_addr = trials_buf.as_mut_ptr() as usize;
                    let trial_f_ptr = trial_f.as_mut_ptr() as usize;
                    let trial_cr_ptr = trial_cr.as_mut_ptr() as usize;
                    let trial_energies_ptr = trial_energies.as_mut_ptr() as usize;
                    let eval_fn = energy_fn.clone();
                    // Strategy/crossover type is constant for the whole iteration; resolve
                    // it once instead of inside every parallel task.
                    let crossover_type = self.config.strategy.crossover();
                    (0..npop).into_par_iter().with_min_len(16).for_each(|i| {
                        let trials_ptr = trials_addr as *mut f64;
                        // Create a fast, deterministic per-individual RNG from a composite
                        // seed. SmallRng seeds much cheaper than StdRng while preserving
                        // reproducibility for a fixed configuration/seed.
                        let mut local_rng: SmallRng = if let Some(base_seed) = self.config.seed {
                            SmallRng::seed_from_u64(
                                base_seed
                                    .wrapping_add((iter as u64) << 32)
                                    .wrapping_add(i as u64),
                            )
                        } else {
                            // Use thread_rng for unseeded runs
                            let mut thread_rng = rand::rng();
                            SmallRng::from_rng(&mut thread_rng)
                        };

                        // Sample mutation factor and crossover rate (adaptive or fixed)
                        let (f, cr) = if let Some(ref adaptive) = adaptive_state {
                            // Use adaptive parameter sampling
                            let adaptive_f = adaptive.sample_f(&mut local_rng);
                            let adaptive_cr = adaptive.sample_cr(&mut local_rng);
                            (adaptive_f, adaptive_cr)
                        } else {
                            // Use fixed or dithered parameters
                            (
                                self.config.mutation.sample(&mut local_rng),
                                self.config.recombination,
                            )
                        };

                        let trial_ptr = unsafe { trials_ptr.add(i * n) };
                        let mut trial_row =
                            unsafe { ndarray::ArrayViewMut1::from_shape_ptr((n,), trial_ptr) };

                        MUTANT_SCRATCH.with(|ms| {
                            let mut scratch = ms.borrow_mut();
                            if scratch.len() != n {
                                *scratch = Array1::zeros(n);
                            }

                            // Generate mutant in place based on strategy
                            match self.config.strategy {
                                Strategy::Best1Bin | Strategy::Best1Exp => {
                                    mutant_best1_into(
                                        &mut scratch,
                                        i,
                                        &pop,
                                        best_idx,
                                        f,
                                        &mut local_rng,
                                    );
                                }
                                Strategy::Rand1Bin | Strategy::Rand1Exp => {
                                    mutant_rand1_into(&mut scratch, i, &pop, f, &mut local_rng);
                                }
                                Strategy::Rand2Bin | Strategy::Rand2Exp => {
                                    mutant_rand2_into(&mut scratch, i, &pop, f, &mut local_rng);
                                }
                                Strategy::CurrentToBest1Bin | Strategy::CurrentToBest1Exp => {
                                    mutant_current_to_best1_into(
                                        &mut scratch,
                                        i,
                                        &pop,
                                        best_idx,
                                        f,
                                        &mut local_rng,
                                    );
                                }
                                Strategy::Best2Bin | Strategy::Best2Exp => {
                                    mutant_best2_into(
                                        &mut scratch,
                                        i,
                                        &pop,
                                        best_idx,
                                        f,
                                        &mut local_rng,
                                    );
                                }
                                Strategy::RandToBest1Bin | Strategy::RandToBest1Exp => {
                                    mutant_rand_to_best1_into(
                                        &mut scratch,
                                        i,
                                        &pop,
                                        best_idx,
                                        f,
                                        &mut local_rng,
                                    );
                                }
                                Strategy::AdaptiveBin | Strategy::AdaptiveExp => {
                                    if let Some(ref adaptive) = adaptive_state {
                                        mutant_adaptive_into(
                                            &mut scratch,
                                            i,
                                            &pop,
                                            &sorted_indices,
                                            adaptive.current_w,
                                            f,
                                            &mut local_rng,
                                        );
                                    } else {
                                        mutant_rand1_into(&mut scratch, i, &pop, f, &mut local_rng);
                                    }
                                }
                                Strategy::LShadeBin | Strategy::LShadeExp => {
                                    let pbest_size =
                                        super::super::mutant_current_to_pbest1::compute_pbest_size(
                                            self.config.lshade.p,
                                            pop.nrows(),
                                        );
                                    let archive_ref =
                                        external_archive.as_ref().and_then(|a| a.read().ok());
                                    mutant_current_to_pbest1_into(
                                        &mut scratch,
                                        i,
                                        &pop,
                                        &sorted_indices,
                                        pbest_size,
                                        archive_ref.as_deref(),
                                        f,
                                        &mut local_rng,
                                    );
                                }
                            };

                            // Crossover from target row into the trial row.
                            let target_row_view = pop.row(i);
                            let target_slice = target_row_view.as_slice().expect("contiguous row");
                            let mutant_slice = scratch.as_slice().expect("contiguous");
                            let trial_slice = trial_row.as_slice_mut().expect("contiguous row");
                            match crossover_type {
                                Crossover::Binomial => binomial_crossover_clamp_into(
                                    trial_slice,
                                    target_slice,
                                    mutant_slice,
                                    cr,
                                    lower_slice,
                                    upper_slice,
                                    &mut local_rng,
                                ),
                                Crossover::Exponential => exponential_crossover_clamp_into(
                                    trial_slice,
                                    target_slice,
                                    mutant_slice,
                                    cr,
                                    lower_slice,
                                    upper_slice,
                                    &mut local_rng,
                                ),
                            }
                        });

                        // Apply WLS if enabled, in place on the trial row.
                        if self.config.adaptive.wls_enabled
                            && local_rng.random::<f64>() < self.config.adaptive.wls_prob
                        {
                            apply_wls_in_place(
                                &mut trial_row,
                                &self.lower,
                                &self.upper,
                                self.config.adaptive.wls_scale,
                                &mut local_rng,
                            );
                        }

                        // Apply integrality if provided
                        if let Some(mask) = &self.config.integrality {
                            apply_integrality(&mut trial_row, mask, &self.lower, &self.upper);
                        }

                        let trial_slice = trial_row.as_slice().expect("contiguous row");
                        let energy = eval_fn(trial_slice);

                        // Safe: each thread writes to distinct indices and the loop
                        // joins before the values are read in the selection phase.
                        unsafe {
                            *(trial_f_ptr as *mut f64).add(i) = f;
                            *(trial_cr_ptr as *mut f64).add(i) = cr;
                            *(trial_energies_ptr as *mut f64).add(i) = energy;
                        }
                    });
                }

                let t_build = t_build0.elapsed();
                let t_eval = std::time::Duration::ZERO;
                nfev += npop;

                let t_select0 = Instant::now();
                // Selection phase: update population based on trial results
                for (i, trial_energy) in trial_energies.iter().enumerate() {
                    let f = trial_f[i];
                    let cr = trial_cr[i];

                    // Selection: replace if better
                    if *trial_energy <= energies[i] {
                        // For L-SHADE: add replaced solution to archive
                        if let Some(ref archive) = external_archive
                            && *trial_energy < energies[i]
                            && let Ok(mut arch) = archive.write()
                        {
                            arch.add_with_rng(pop.row(i).to_owned(), &mut rng);
                        }
                        pop.row_mut(i).assign(&trials_buf.row(i));
                        energies[i] = *trial_energy;
                        accepted_trials += 1;

                        // Update adaptive parameters if improvement
                        if let Some(ref mut adaptive) = adaptive_state {
                            adaptive.record_success(f, cr);
                        }

                        // Track if this is an improvement over the current best
                        if *trial_energy < best_f {
                            improvement_count += 1;
                        }
                    }
                }
                let t_select = t_select0.elapsed();

                // L-SHADE: linear population size reduction
                if is_lshade {
                    let current_npop = self
                        .config
                        .lshade
                        .current_population_size(n_free, nfev, max_nfev);
                    if current_npop < pop.nrows() {
                        let mut indices: Vec<usize> = (0..pop.nrows()).collect();
                        indices.sort_by(|&a, &b| {
                            energies[a]
                                .partial_cmp(&energies[b])
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let keep: Vec<usize> = indices.into_iter().take(current_npop).collect();

                        let mut new_pop = Array2::zeros((current_npop, n));
                        let mut new_energies = Array1::zeros(current_npop);
                        for (new_i, &old_i) in keep.iter().enumerate() {
                            new_pop.row_mut(new_i).assign(&pop.row(old_i));
                            new_energies[new_i] = energies[old_i];
                        }
                        pop = new_pop;
                        energies = new_energies;
                        npop = current_npop;
                        trials_buf = Array2::zeros((npop, n));
                        trial_f = vec![0.0; npop];
                        trial_cr = vec![0.0; npop];
                        trial_energies = vec![0.0; npop];

                        if let Some(ref archive) = external_archive
                            && let Ok(mut arch) = archive.write()
                        {
                            arch.resize(self.config.lshade.current_archive_size(current_npop));
                        }
                    }
                }

                t_build_tot += t_build;
                t_eval_tot += t_eval;
                t_select_tot += t_select;
                let iter_dur = iter_start.elapsed();
                t_iter_tot += iter_dur;

                if timing_enabled && (iter <= 5 || iter % 10 == 0) {
                    eprintln!(
                        "TIMING iter {:4}: build={:.3} ms, eval={:.3} ms, select={:.3} ms, total={:.3} ms",
                        iter,
                        t_build.as_secs_f64() * 1e3,
                        t_eval.as_secs_f64() * 1e3,
                        t_select.as_secs_f64() * 1e3,
                        iter_dur.as_secs_f64() * 1e3,
                    );
                }

                // Update adaptive parameters after each generation
                if let Some(ref mut adaptive) = adaptive_state {
                    adaptive.update(&self.config.adaptive, iter, self.config.maxiter);
                }

                // Update best solution after generation
                let (new_best_idx, new_best_f) = argmin(&energies);
                if is_lshade {
                    // L-SHADE reorders/shrinks the population before best selection.
                    // Its p-best mutation does not use best_idx, but keep this
                    // internal index valid for checkpoint validation and reporting.
                    best_idx = new_best_idx;
                }
                if new_best_f < best_f {
                    best_idx = new_best_idx;
                    best_f = new_best_f;
                    best_x.assign(&pop.row(best_idx));
                }

                // Convergence check
                let pop_mean = energies.mean().unwrap_or(0.0);
                let pop_std = energies.std(0.0);
                let convergence_threshold = self.config.atol + self.config.tol * pop_mean.abs();

                if self.config.disp {
                    eprintln!(
                        "DE iter {:4}  best_f={:.6e}  std={:.3e}  accepted={}/{}, improved={}",
                        iter, best_f, pop_std, accepted_trials, npop, improvement_count
                    );
                }

                // Callback
                let mut callback_stopped = false;
                if let Some(ref mut cb) = self.config.callback {
                    let intermediate = DEIntermediate {
                        x: best_x.clone(),
                        fun: best_f,
                        convergence: pop_std,
                        iter,
                    };
                    match cb(&intermediate) {
                        CallbackAction::Stop => {
                            success = true;
                            message = "Optimization stopped by callback".to_string();
                            callback_stopped = true;
                        }
                        CallbackAction::Continue => {}
                    }
                }

                if !callback_stopped
                    && iter >= self.config.min_convergence_iter
                    && pop_std <= convergence_threshold
                {
                    success = true;
                    message = format!(
                        "Converged: std(pop_f)={:.3e} <= threshold={:.3e}",
                        pop_std, convergence_threshold
                    );
                }

                if !callback_stopped && !success && iter == self.config.maxiter {
                    message = format!("Maximum iterations reached: {}", self.config.maxiter);
                }

                if callback_stopped || success || iter == self.config.maxiter {
                    terminal_state = Some(DETerminalCheckpoint {
                        success,
                        message: message.clone(),
                        finalized: false,
                        final_params: None,
                        final_fitness: None,
                        polish_evaluations: 0,
                    });
                }
                if checkpoint_callback.is_some() {
                    let checkpoint = self.create_checkpoint(
                        run_identity,
                        build_identity
                            .as_deref()
                            .expect("checkpoint callbacks require exact mode"),
                        target_identity
                            .as_deref()
                            .expect("checkpoint callbacks require exact mode"),
                        DECheckpointRuntime {
                            generation: iter,
                            evaluations: nfev,
                            population: &pop,
                            energies: &energies,
                            best_index: best_idx,
                            best_fitness: best_f,
                            best_params: &best_x,
                            rng: &rng,
                            adaptive: adaptive_state.as_ref(),
                            archive: external_archive.as_ref(),
                            terminal: terminal_state.clone(),
                        },
                    )?;
                    store_checkpoint(&mut checkpoint_callback, &checkpoint)?;
                }

                if terminal_state.is_some() {
                    break;
                }
            }
        }

        if terminal_state.is_none() {
            success = false;
            message = format!("Maximum iterations reached: {}", self.config.maxiter);
            terminal_state = Some(DETerminalCheckpoint {
                success,
                message: message.clone(),
                finalized: false,
                final_params: None,
                final_fitness: None,
                polish_evaluations: 0,
            });
            if checkpoint_callback.is_some() {
                let checkpoint = self.create_checkpoint(
                    run_identity,
                    build_identity
                        .as_deref()
                        .expect("checkpoint callbacks require exact mode"),
                    target_identity
                        .as_deref()
                        .expect("checkpoint callbacks require exact mode"),
                    DECheckpointRuntime {
                        generation: nit,
                        evaluations: nfev,
                        population: &pop,
                        energies: &energies,
                        best_index: best_idx,
                        best_fitness: best_f,
                        best_params: &best_x,
                        rng: &rng,
                        adaptive: adaptive_state.as_ref(),
                        archive: external_archive.as_ref(),
                        terminal: terminal_state.clone(),
                    },
                )?;
                store_checkpoint(&mut checkpoint_callback, &checkpoint)?;
            }
        }

        if self.config.disp {
            eprintln!("DE finished: {}", message);
        }

        // Resume an already finalized report directly; otherwise polish once,
        // then persist a complete terminal snapshot.
        let (final_x, final_f, polish_nfev) = if let Some(terminal) = terminal_state
            .as_ref()
            .filter(|terminal| terminal.finalized)
        {
            (
                Array1::from_vec(
                    terminal
                        .final_params
                        .clone()
                        .expect("validated finalized checkpoint includes parameters"),
                ),
                terminal
                    .final_fitness
                    .expect("validated finalized checkpoint includes fitness")
                    .into_solver(),
                terminal.polish_evaluations,
            )
        } else {
            let (final_x, final_f, polish_nfev) = if let Some(ref polish_cfg) = self.config.polish {
                if polish_cfg.enabled {
                    self.polish(&best_x)
                } else {
                    (best_x.clone(), best_f, 0)
                }
            } else {
                (best_x.clone(), best_f, 0)
            };
            terminal_state = Some(DETerminalCheckpoint {
                success,
                message: message.clone(),
                finalized: true,
                final_params: Some(final_x.to_vec()),
                final_fitness: Some(CheckpointFitness::from_solver(final_f)),
                polish_evaluations: polish_nfev,
            });
            if checkpoint_callback.is_some() {
                let checkpoint = self.create_checkpoint(
                    run_identity,
                    build_identity
                        .as_deref()
                        .expect("checkpoint callbacks require exact mode"),
                    target_identity
                        .as_deref()
                        .expect("checkpoint callbacks require exact mode"),
                    DECheckpointRuntime {
                        generation: nit,
                        evaluations: nfev,
                        population: &pop,
                        energies: &energies,
                        best_index: best_idx,
                        best_fitness: best_f,
                        best_params: &best_x,
                        rng: &rng,
                        adaptive: adaptive_state.as_ref(),
                        archive: external_archive.as_ref(),
                        terminal: terminal_state.clone(),
                    },
                )?;
                store_checkpoint(&mut checkpoint_callback, &checkpoint)?;
            }
            (final_x, final_f, polish_nfev)
        };

        if timing_enabled {
            eprintln!(
                "TIMING total: build={:.3} s, eval={:.3} s, select={:.3} s, iter_total={:.3} s",
                t_build_tot.as_secs_f64(),
                t_eval_tot.as_secs_f64(),
                t_select_tot.as_secs_f64(),
                t_iter_tot.as_secs_f64()
            );
        }

        Ok(self.finish_report(
            pop,
            energies,
            final_x,
            final_f,
            success,
            message,
            nit,
            nfev + polish_nfev,
        ))
    }
}

impl<'a, F> DifferentialEvolution<'a, F>
where
    F: Fn(&Array1<f64>) -> f64 + Sync,
{
    fn create_checkpoint(
        &self,
        run_identity: &str,
        build_identity: &str,
        target_identity: &str,
        runtime: DECheckpointRuntime<'_>,
    ) -> Result<DECheckpoint> {
        let seed = self.config.seed.ok_or_else(|| DEError::InvalidCheckpoint {
            message: "exact continuation requires an explicit deterministic seed".into(),
        })?;
        let archive = runtime
            .archive
            .map(|archive| {
                archive
                    .read()
                    .map(|archive| archive.checkpoint_snapshot())
                    .map_err(|_| DEError::InvalidCheckpoint {
                        message: "L-SHADE archive lock is poisoned at checkpoint barrier".into(),
                    })
            })
            .transpose()?;
        let checkpoint = DECheckpoint {
            checkpoint_version: DE_CHECKPOINT_VERSION,
            run_identity: run_identity.to_owned(),
            implementation_identity: DE_CHECKPOINT_IMPLEMENTATION_ID.to_owned(),
            solver_source_identity: solver_source_identity().to_owned(),
            target_identity: target_identity.to_owned(),
            build_identity: build_identity.to_owned(),
            configuration_fingerprint: self.configuration_fingerprint(),
            seed,
            generation: runtime.generation,
            evaluations: runtime.evaluations,
            population: runtime
                .population
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect(),
            population_fitness: runtime
                .energies
                .iter()
                .copied()
                .map(CheckpointFitness::from_solver)
                .collect(),
            best_index: runtime.best_index,
            best_params: runtime.best_params.to_vec(),
            best_fitness: CheckpointFitness::from_solver(runtime.best_fitness),
            rng_state: runtime.rng.serialize_state().to_vec(),
            adaptive: runtime.adaptive.map(|adaptive| DEAdaptiveCheckpoint {
                f_m: adaptive.f_m,
                cr_m: adaptive.cr_m,
                successful_f: adaptive.successful_f.clone(),
                successful_cr: adaptive.successful_cr.clone(),
                current_w: adaptive.current_w,
            }),
            archive,
            terminal: runtime.terminal,
        };
        self.validate_checkpoint(&checkpoint, run_identity, build_identity, target_identity)?;
        Ok(checkpoint)
    }

    fn validate_checkpoint(
        &self,
        checkpoint: &DECheckpoint,
        run_identity: &str,
        build_identity: &str,
        target_identity: &str,
    ) -> Result<()> {
        let invalid = |message: String| DEError::InvalidCheckpoint { message };
        if checkpoint.checkpoint_version != DE_CHECKPOINT_VERSION {
            return Err(invalid(format!(
                "format version {} is unsupported; expected {DE_CHECKPOINT_VERSION}",
                checkpoint.checkpoint_version
            )));
        }
        if run_identity.trim().is_empty() || checkpoint.run_identity != run_identity {
            return Err(invalid("objective/run identity mismatch".into()));
        }
        if checkpoint.implementation_identity != DE_CHECKPOINT_IMPLEMENTATION_ID {
            return Err(invalid(format!(
                "solver/RNG implementation mismatch: saved {:?}, current {:?}",
                checkpoint.implementation_identity, DE_CHECKPOINT_IMPLEMENTATION_ID
            )));
        }
        let source_identity = solver_source_identity();
        if checkpoint.solver_source_identity != source_identity {
            return Err(invalid(format!(
                "solver source mismatch: saved {:?}, current {:?}",
                checkpoint.solver_source_identity, source_identity
            )));
        }
        if checkpoint.target_identity != target_identity {
            return Err(invalid(format!(
                "target mismatch: saved {:?}, current {:?}",
                checkpoint.target_identity, target_identity
            )));
        }
        if checkpoint.build_identity != build_identity {
            return Err(invalid(format!(
                "executable build mismatch: saved {:?}, current {:?}; exact continuation requires the same executable build",
                checkpoint.build_identity, build_identity
            )));
        }
        if checkpoint.configuration_fingerprint != self.configuration_fingerprint() {
            return Err(invalid(
                "DE bounds, seed, budget, or strategy configuration differs from saved state"
                    .into(),
            ));
        }
        if self.config.seed != Some(checkpoint.seed) {
            return Err(invalid("random seed mismatch".into()));
        }
        if checkpoint.rng_state.len() != 49 {
            return Err(invalid(format!(
                "ChaCha12 RNG state has {} bytes; expected 49",
                checkpoint.rng_state.len()
            )));
        }
        // ChaCha12's serialized word cursor uses 68 bits; deserialize_state
        // masks the high nibble of byte 48. Reject those noncanonical bits
        // before restoring so malformed JSON cannot silently alias a stream.
        if checkpoint.rng_state[48] & 0xf0 != 0 {
            return Err(invalid(
                "ChaCha12 RNG word cursor exceeds its canonical 68-bit range".into(),
            ));
        }

        let is_lshade = matches!(
            self.config.strategy,
            Strategy::LShadeBin | Strategy::LShadeExp
        );
        let is_adaptive = matches!(
            self.config.strategy,
            Strategy::AdaptiveBin | Strategy::AdaptiveExp
        ) || self.config.adaptive.adaptive_mutation;
        let free_dimensions = self
            .lower
            .iter()
            .zip(self.upper.iter())
            .filter(|(lower, upper)| (**upper - **lower).abs() > 0.0)
            .count();
        if free_dimensions == 0 {
            return Err(invalid("all DE variables are fixed by their bounds".into()));
        }
        let initial_population = if is_lshade {
            self.config.lshade.initial_population_size(free_dimensions)
        } else {
            self.config.popsize * free_dimensions
        };
        let mut expected_population = initial_population;
        let mut expected_evaluations = initial_population + usize::from(self.config.x0.is_some());
        let max_evaluations = self.config.maxiter.saturating_mul(initial_population);
        if checkpoint.generation > self.config.maxiter {
            return Err(invalid(format!(
                "saved generation {} exceeds configured maximum {}",
                checkpoint.generation, self.config.maxiter
            )));
        }
        for _ in 0..checkpoint.generation {
            expected_evaluations = expected_evaluations.saturating_add(expected_population);
            if is_lshade {
                expected_population = self.config.lshade.current_population_size(
                    free_dimensions,
                    expected_evaluations,
                    max_evaluations,
                );
            }
        }
        if checkpoint.evaluations != expected_evaluations {
            return Err(invalid(format!(
                "saved evaluation count {} is inconsistent with generation {} (expected {})",
                checkpoint.evaluations, checkpoint.generation, expected_evaluations
            )));
        }
        if checkpoint.population.len() != expected_population
            || checkpoint.population_fitness.len() != expected_population
        {
            return Err(invalid(format!(
                "saved population shape is inconsistent: expected {expected_population} rows, got {} rows and {} fitness values",
                checkpoint.population.len(),
                checkpoint.population_fitness.len()
            )));
        }
        for (index, vector) in checkpoint.population.iter().enumerate() {
            validate_checkpoint_vector(vector, &self.lower, &self.upper).map_err(|reason| {
                invalid(format!("invalid population vector {index}: {reason}"))
            })?;
            if !checkpoint.population_fitness[index].is_valid() {
                return Err(invalid(format!(
                    "population fitness {index} is NaN or negative infinity"
                )));
            }
        }
        if checkpoint.best_index >= expected_population {
            return Err(invalid(format!(
                "best member index {} is outside population size {expected_population}",
                checkpoint.best_index
            )));
        }
        validate_checkpoint_vector(&checkpoint.best_params, &self.lower, &self.upper)
            .map_err(|reason| invalid(format!("invalid best vector: {reason}")))?;
        if !checkpoint.best_fitness.is_valid() {
            return Err(invalid("best fitness is NaN or negative infinity".into()));
        }
        let minimum_fitness = checkpoint
            .population_fitness
            .iter()
            .copied()
            .map(CheckpointFitness::into_solver)
            .fold(f64::INFINITY, f64::min);
        if checkpoint.best_fitness.into_solver() != minimum_fitness {
            return Err(invalid(
                "saved best fitness does not match the population minimum".into(),
            ));
        }
        if checkpoint.population_fitness[checkpoint.best_index].into_solver() != minimum_fitness {
            return Err(invalid(
                "saved best index does not point to a minimum-fitness member".into(),
            ));
        }
        if !checkpoint
            .population
            .iter()
            .zip(&checkpoint.population_fitness)
            .any(|(vector, fitness)| {
                vector == &checkpoint.best_params
                    && fitness.into_solver() == checkpoint.best_fitness.into_solver()
            })
        {
            return Err(invalid(
                "saved best vector is not paired with the saved minimum fitness in the current population".into(),
            ));
        }

        if is_adaptive != checkpoint.adaptive.is_some() {
            return Err(invalid(
                "adaptive strategy state does not match the configured strategy".into(),
            ));
        }
        if let Some(adaptive) = &checkpoint.adaptive {
            if !adaptive.f_m.is_finite()
                || !adaptive.cr_m.is_finite()
                || !adaptive.current_w.is_finite()
                || adaptive.successful_f.iter().any(|value| !value.is_finite())
                || adaptive
                    .successful_cr
                    .iter()
                    .any(|value| !value.is_finite())
            {
                return Err(invalid("adaptive state contains a non-finite value".into()));
            }
            if !adaptive.successful_f.is_empty() || !adaptive.successful_cr.is_empty() {
                return Err(invalid(
                    "adaptive state was saved outside a generation barrier".into(),
                ));
            }
        }

        if is_lshade != checkpoint.archive.is_some() {
            return Err(invalid(
                "L-SHADE archive presence does not match the configured strategy".into(),
            ));
        }
        if let Some(archive) = &checkpoint.archive {
            let expected_capacity = if checkpoint.generation == 0 {
                super::super::external_archive::ExternalArchive::with_population_size(
                    initial_population,
                    self.config.lshade.arc_rate,
                )
                .checkpoint_snapshot()
                .capacity
            } else {
                self.config.lshade.current_archive_size(expected_population)
            };
            if archive.capacity != expected_capacity || archive.solutions.len() > archive.capacity {
                return Err(invalid(
                    "L-SHADE archive capacity or length does not match the current population"
                        .into(),
                ));
            }
            for (index, vector) in archive.solutions.iter().enumerate() {
                validate_checkpoint_vector(vector, &self.lower, &self.upper).map_err(|reason| {
                    invalid(format!("invalid archive vector {index}: {reason}"))
                })?;
            }
        }

        if checkpoint.generation == self.config.maxiter
            && checkpoint.terminal.is_none()
            && self.config.maxiter != 0
        {
            return Err(invalid(
                "maximum-generation checkpoint is missing terminal status".into(),
            ));
        }
        if let Some(terminal) = &checkpoint.terminal {
            if terminal.finalized {
                let final_params = terminal.final_params.as_deref().ok_or_else(|| {
                    invalid("finalized terminal state is missing final parameters".into())
                })?;
                let final_fitness = terminal.final_fitness.ok_or_else(|| {
                    invalid("finalized terminal state is missing final fitness".into())
                })?;
                validate_checkpoint_vector(final_params, &self.lower, &self.upper)
                    .map_err(|reason| invalid(format!("invalid final vector: {reason}")))?;
                if !final_fitness.is_valid() {
                    return Err(invalid("final fitness is NaN or negative infinity".into()));
                }
                if self
                    .config
                    .polish
                    .as_ref()
                    .is_none_or(|polish| !polish.enabled)
                    && terminal.polish_evaluations != 0
                {
                    return Err(invalid(
                        "checkpoint records polish evaluations when polishing is disabled".into(),
                    ));
                }
            } else if terminal.final_params.is_some()
                || terminal.final_fitness.is_some()
                || terminal.polish_evaluations != 0
            {
                return Err(invalid(
                    "unfinished terminal state contains finalization data".into(),
                ));
            }
        }
        Ok(())
    }

    fn configuration_fingerprint(&self) -> String {
        let config = &self.config;
        let mut fingerprint = format!(
            "v1;maxiter={};popsize={};mutation={:?};strategy={:?};crossover={:?};init={:?};updating={:?};parallel={:?};lshade={:?};adaptive={:?};integrality={:?};polish={:?};disp={};callback_present={};ineq_count={};eq_count={}",
            config.maxiter,
            config.popsize,
            config.mutation,
            config.strategy,
            config.crossover,
            config.init,
            config.updating,
            config.parallel,
            config.lshade,
            config.adaptive,
            config.integrality,
            config.polish,
            config.disp,
            config.callback.is_some(),
            config.penalty_ineq.len(),
            config.penalty_eq.len(),
        );
        append_fingerprint_float(&mut fingerprint, "tol", config.tol);
        if config.min_convergence_iter != 0 {
            fingerprint.push_str(&format!(
                ";min_convergence_iter={}",
                config.min_convergence_iter
            ));
        }
        append_fingerprint_float(&mut fingerprint, "atol", config.atol);
        append_fingerprint_float(&mut fingerprint, "recombination", config.recombination);
        fingerprint.push_str(&format!(";seed={:?}", config.seed));
        append_fingerprint_vector(&mut fingerprint, "lower", self.lower.iter().copied());
        append_fingerprint_vector(&mut fingerprint, "upper", self.upper.iter().copied());
        if let Some(initial) = &config.x0 {
            append_fingerprint_vector(&mut fingerprint, "x0", initial.iter().copied());
        } else {
            fingerprint.push_str(";x0=none");
        }
        for (index, (_, weight)) in config.penalty_ineq.iter().enumerate() {
            append_fingerprint_float(&mut fingerprint, &format!("ineq_weight_{index}"), *weight);
        }
        for (index, (_, weight)) in config.penalty_eq.iter().enumerate() {
            append_fingerprint_float(&mut fingerprint, &format!("eq_weight_{index}"), *weight);
        }
        if let Some(linear) = &config.linear_penalty {
            fingerprint.push_str(&format!(
                ";linear_shape={}x{}",
                linear.a.nrows(),
                linear.a.ncols()
            ));
            append_fingerprint_float(&mut fingerprint, "linear_weight", linear.weight);
            append_fingerprint_vector(&mut fingerprint, "linear_a", linear.a.iter().copied());
            append_fingerprint_vector(&mut fingerprint, "linear_lb", linear.lb.iter().copied());
            append_fingerprint_vector(&mut fingerprint, "linear_ub", linear.ub.iter().copied());
        } else {
            fingerprint.push_str(";linear=none");
        }
        fingerprint
    }
}

fn checkpoint_matrix(rows: &[Vec<f64>], dimension: usize) -> Result<Array2<f64>> {
    let values = rows.iter().flatten().copied().collect();
    Array2::from_shape_vec((rows.len(), dimension), values).map_err(|error| {
        DEError::InvalidCheckpoint {
            message: format!("saved population matrix cannot be restored: {error}"),
        }
    })
}

fn checkpoint_target_identity() -> String {
    let endian = if cfg!(target_endian = "little") {
        "little"
    } else {
        "big"
    };
    let runtime_features = runtime_numeric_features();
    format!(
        "{}-{}-{}-{}bit-{};numeric-features={}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        std::env::consts::FAMILY,
        usize::BITS,
        endian,
        runtime_features,
    )
}

fn runtime_numeric_features() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        let features = [
            ("sse2", std::arch::is_x86_feature_detected!("sse2")),
            ("sse3", std::arch::is_x86_feature_detected!("sse3")),
            ("ssse3", std::arch::is_x86_feature_detected!("ssse3")),
            ("sse4.1", std::arch::is_x86_feature_detected!("sse4.1")),
            ("sse4.2", std::arch::is_x86_feature_detected!("sse4.2")),
            ("avx", std::arch::is_x86_feature_detected!("avx")),
            ("avx2", std::arch::is_x86_feature_detected!("avx2")),
            ("fma", std::arch::is_x86_feature_detected!("fma")),
        ];
        return features
            .into_iter()
            .filter_map(|(name, present)| present.then_some(name))
            .collect::<Vec<_>>()
            .join(",");
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "neon".to_owned();
    }
    #[allow(unreachable_code)]
    "baseline".to_owned()
}

fn validate_checkpoint_vector(
    vector: &[f64],
    lower: &Array1<f64>,
    upper: &Array1<f64>,
) -> std::result::Result<(), String> {
    if vector.len() != lower.len() || vector.len() != upper.len() {
        return Err(format!(
            "dimension mismatch: expected {}, got {}",
            lower.len(),
            vector.len()
        ));
    }
    for (index, ((value, lower), upper)) in vector.iter().zip(lower).zip(upper).enumerate() {
        if !value.is_finite() {
            return Err(format!("value {index} is not finite"));
        }
        if value < lower || value > upper {
            return Err(format!("value {index} is outside configured bounds"));
        }
    }
    Ok(())
}

fn append_fingerprint_float(fingerprint: &mut String, name: &str, value: f64) {
    fingerprint.push_str(&format!(";{name}={:016x}", value.to_bits()));
}

fn append_fingerprint_vector(
    fingerprint: &mut String,
    name: &str,
    values: impl Iterator<Item = f64>,
) {
    fingerprint.push_str(&format!(";{name}="));
    for (index, value) in values.enumerate() {
        if index > 0 {
            fingerprint.push(',');
        }
        fingerprint.push_str(&format!("{:016x}", value.to_bits()));
    }
}

fn store_checkpoint(
    callback: &mut Option<&mut DECheckpointCallback<'_>>,
    checkpoint: &DECheckpoint,
) -> Result<()> {
    if let Some(callback) = callback.as_mut() {
        (**callback)(checkpoint).map_err(|message| DEError::CheckpointSave { message })?;
    }
    Ok(())
}

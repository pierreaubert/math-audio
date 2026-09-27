# 0.5.15

## New features

- Added `cobra`, a SACOBRA-style surrogate-assisted constrained optimiser
  (RBF models per objective/constraint, distance-requirement cycle,
  self-adjusting output transform, stagnation kicks, COBYLA polish).
- Added `constrained_bayesian_optimization`, a single-trust-region
  constrained BO arm with constrained EI.
- Added `RbfSurrogate` with cubic/Gaussian/multiquadric kernels and
  holdout model selection (`fit_auto`).
- CMA-ES gained native inequality constraints with adaptive-penalty merit,
  IPOP restarts, and a diagonal-covariance mode.
- Added the `benchmark-constrained` suite (G06, G08, G24, G04, G09,
  rosenbrock-disk) with budget-normalised rungs; COBRA solves 5/6 problems
  against 2/6 for the CMA-ES baseline.

# 0.5.14

## Bug fixes

- Differential evolution now uses its seeded RNG when replacing entries in a full
  external archive, removing an unseeded random draw from L-SHADE runs.
- Added `ExternalArchive::add_with_rng` while retaining the existing `add` API.

## Tests

- Added deterministic archive-replacement and zero-capacity RNG-state regressions.

# 0.5.13

## Improvements

- CMA-ES now uses a lazy eigendecomposition schedule with a reused scratch
  buffer instead of cloning and factorizing the covariance every generation.
- Levenberg-Marquardt accepts an analytic Jacobian via `LMConfig::jacobian`
  (`JacobianFn`) instead of forward finite-differences only.
- Documented the CMA-ES bound-clipping bias near walls.

## Bug fixes

- `happy_cat` registry entries now point at `happycat` (same landscape);
  the deprecated duplicate no longer triggers deprecation warnings.

# 0.5.12

## Performance

- CMA-ES and DE minimise memory copies by pre-allocating generation scratch
  buffers and offspring pools.

## Tests

- Fixed a flaky CMA-ES caller-pool test: workers are now identified by
  thread name and the worker-count assertion tolerates pool scheduling.

# 0.5.11

## Performance

- DE and CMA-ES hot paths are ~7.3x faster on the evo benchmark (551.55 ms
  -> 75.12 ms): reusable buffers, thread-local scratch, raw-slice
  mutation/crossover loops, fused parallel trial build+eval, a guarded
  Best1Bin fast path, and no materialised B*D transform matrix in CMA-ES.
- Removed memory copies from mutation, crossover, and selection loops.

## Internal

- `benchmark_convergence` moved out of `benches/` (it is a binary, not a
  benchmark) and the duplicate benchmark target removed.
- Migrated to the `rand` `RngExt` API.

## Tests

- Added proptest property tests and criterion `de_bench`/`cmaes_bench`
  benchmarks.

# 0.5.10

## Bug fixes

- `apply_wls` now uses the documented Cauchy perturbation distribution instead
  of a uniform perturbation while still clipping candidates to bounds.
- Clarified the Levenberg-Marquardt finite-difference step documentation.

## Tests

- Added direct tests for mixed-integer rounding/clamping, linear-penalty
  stacking, and WLS shape/bounds behavior.

# 0.5.9

## Bug fixes

- **L-SHADE population reduction wired into main loop**: `current_population_size`
  is now called every generation for `LShadeBin` / `LShadeExp` strategies. The
  worst individuals are culled and the external archive is resized accordingly.
  Initial L-SHADE population now uses `lshade.initial_population_size(n_free)`
  instead of the generic `popsize * n_free`.
- **NaN/Inf guards in DE selection**: `argmin` now skips non-finite values,
  treating them as `+inf`. The energy closure and `impl_helpers::energy`
  sanitize non-finite results to `f64::INFINITY`, preventing a single bad
  evaluation from silently corrupting the optimisation.
- **Adaptive mutation uses Cauchy for F**: `AdaptiveState::sample_f` now draws
  from a Cauchy distribution (heavy tails allow occasional large mutation
  factors) as required by JADE/SHADE literature. `sample_cr` continues to use
  Normal.
- **`init_sobol` replaced with proper Halton sequence**: The old implementation
  used non-coprime bases for `dim > 4` (cycling via `dim % 10`). It is now a
  true Halton sequence using the first `n` primes as Van der Corput bases.
  `init_sobol` is retained as a deprecated alias; internal code migrates to
  `init_halton`.
- **LM Jacobian step bounded**: `jacobian_step` now uses `eps * (1 + |x|)`
  instead of the unbounded `eps * |x|`, preventing catastrophic cancellation
  when `|x|` is huge.
- **LM linear solver stricter threshold**: Singularity threshold in
  `solve_linear_system` raised from `1e-30` to `1e-12` so near-singular matrices
  are rejected rather than trusted to produce wild steps.
- **BO deduplication threshold scaled by dimension**: The hard-coded `1e-18`
  squared-distance threshold in `is_new_point` is now `1e-12 / n`, avoiding
  the admission of near-duplicates in low dimensions and excessive rejection
  in high dimensions.
- **Removed no-op `PolishConfig::algo` field**: The field was documented as a
  local-optimizer selector but was completely ignored. Removing it eliminates
  API clutter and misleading expectations.
- **COBYLA fixed-dimension scaling**: When `lb == ub`, `rhocur` now uses `rho`
  instead of `f64::EPSILON`, keeping the simplex edge on the same scale as
  free dimensions and avoiding ill-conditioned `simi` entries.

# 0.5.8

## New features

- Added `continuous_area` module for continuous-prior loss integration. It is
  generic over dimension via const generics (`Prior<const D: usize>` /
  `Quadrature<const D: usize>`), with `Uniform` / axis-aligned `Gaussian` /
  `Custom` priors, Sobol / Latin-Hypercube / Gauss–Legendre tensor-product
  quadrature, and `ExpectedValue` / `WorstCase` / `CVaR` scalarisations.
  `evaluate_area_loss` collapses a per-position loss `L(params, p)` into one
  outer-loop scalar suitable for any optimiser. Reuses the existing
  `init_sobol`, `init_latin_hypercube`, and `differential_evolution` (for the
  `WorstCase` inner search). Includes hand-derived Beasley-Springer-Moro
  inverse-normal CDF and Abramowitz-Stegun erf for the Gaussian inverse-CDF
  transform.

## Performance

- Added parallelisation where it can help: CMA-ES

# 0.5.7

## New features

- Added a generic Gaussian-process Bayesian optimisation backend for expensive
  bounded continuous objectives. It supports Matérn-5/2 ARD kernels with
  marginal-likelihood lengthscale fitting, Cholesky-based GP solves, latent
  posterior-uncertainty stopping, Sobol-seeded initial designs, EI, real
  Monte-Carlo q-EI, Thompson acquisition, parallel batch evaluation through
  `ParallelConfig`, and real Monte-Carlo qEHVI for small expensive Pareto
  searches.

# 0.5.6

- Fix a 1 line bug in the new cobyla implementation.

# 0.5.5

- Added ISRES and Cobyla. Used them to drop the dependency to nlopt.

# 0.5.4

## Switch to oxiblas-ndarray for BLAS operations

Replaced ndarray's built-in matrix-vector multiply with oxiblas-ndarray's
pure-Rust BLAS implementation in the linear penalty evaluation.

- `mod.rs`, `impl_helpers.rs`: `lp.a.dot(&x.view())` replaced with
  `matvec(&lp.a, &x.to_owned())` from oxiblas-ndarray.
- Added `oxiblas-ndarray` dependency.

Cargo version 0.5.3 -> 0.5.4.

# 0.5.3

- Added Sobol initialisation which was in another crate for historical reasons

# 0.5.2

- Added LSHADE algorithm

# 0.5.1

- Bug fixes

# 0.5.0 == 0.4.27

- Split from AutoEQ
- Added a Levenberg-Marquardt bounded nonlinear least-squares solver

# Review: math-optimisation (spark, 2026-08-04)

Scope: performance + correctness review of `crates/math-optimisation`
(v0.5.12, ~15 kLOC: DE/L-SHADE, CMA-ES, LM, COBYLA-native, ISRES,
NSGA-II/III, GP-Bayesian, continuous-area quadrature). Method:
read-only review agent + maintainer verification. No code changed.

## Correctness

- DE core is careful (per review): fixed-bounds short-circuit,
  bounds validation, `Cow::Borrowed` no-copy fast path, NaN guards
  (`energy.is_finite()`), non-finite→inf mapping in `argmin`.
- CMA-ES maps NaN/inf to inf (`cmaes.rs:220,261,268,495-496`), clips to
  `[0,1]` (`:244`) and validates bounds/x0/lambda (`:133-165`);
  cc/cs/c1/cmu/damps/chi_n constants look standard (`:177-182`); hsig
  stall logic present (`:313-323,345`). Caveat: bound handling is pure
  clipping, biasing covariance near walls — fine for box problems (RIR
  gains/Q) but document it.
- LM: `project()` clips x0/steps; dense Gauss-elim with partial pivot,
  singular threshold 1e-12. Issues: (a) forward finite-difference
  Jacobian costs n_residual-evals per param per iter with no analytic-J
  path surfaced (matters for autodiff/analog curve-fit consumers);
  (b) fixed 1e-12 pivot cutoff is scale-blind — large-gain residuals
  can false-trigger singular.

## Performance

- P1 (verified): CMA-ES eigendecomposes **every generation**
  (`cmaes.rs:360`, `SymmetricEigen::new(covariance.clone())` — note the
  clone). Correct but O(n³)/gen; fine for n≤20 audio fits, wasteful at
  n=100+. Cache decomposition across generations (standard lazy-eig
  trick) and drop the clone.
- `parallel_eval` exists for population evaluation; DE benches
  (`benches/{de,cmaes}_bench.rs`) present.

## Prioritized actions

- P1: lazy eigendecomposition + remove covariance clone in CMA-ES.
- P1: analytic-Jacobian hook for LM (natural fit for math-autodiff /
  math-analog fitting consumers).
- P2: scale-aware LM pivot threshold; document CMA-ES clipping bias.

## Verdict

Careful, well-guarded implementations; the costs are compute-structure
issues (eig-per-gen, finite-diff-only LM), not bugs. Both P1s directly
serve the analog/autodiff fitting story.

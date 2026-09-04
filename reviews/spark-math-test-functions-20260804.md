# Review: math-test-functions (spark, 2026-08-04)

Scope: performance + correctness review of `crates/math-test-functions`
(~110 benchmark functions). Method: read-only review agent +
maintainer reproduction by execution. No repo files changed (scratch
probe lived in `/tmp`, incidental `Cargo.lock` rewrite reverted).

## Correctness

P0 bugs, both reproduced by execution (`Err(Any..)` = panicked):

- `rosenbrock([])` panics: `for i in 0..x.len() - 1`
  (`functions/rosenbrock.rs:10`) underflows on empty input. `len == 1`
  silently returns `0.0`, which is wrong (1-D Rosenbrock at `x!=1` is
  `(1-x)^2`). `levy.rs` already shows the right guard pattern
  (`if n == 0 { return 0.0; }`); `pinter.rs` (`x[n-1]`) has the same
  empty-input panic.
- `eggholder([1.0])` panics (`ndarray: index out of bounds`,
  reproduced): hard-indexes `x[0], x[1]` (`functions/eggholder.rs:10-12`)
  and silently ignores dims beyond 2. Needs a dimension assert.

Near-P0 formula divergence (found by diff, masked by tests):

- `happy_cat.rs:13` computes `((s-n).powi(2)).powf(0.25)` = `|s-n|^0.5`,
  while `happycat.rs:13` computes `|s-n|.powf(0.25)` = `|s-n|^0.25`.
  Both claim min `0.0` at `[-1,-1]` (`get.rs:628,642`) so registry tests
  pass at the optimum only. Both cannot match the reference — pick the
  cited reference formulation and delete/alias the other entry.

## Performance

- Per-function fns are thin scalar loops over `Array1`; no hot-loop
  pathology. `benches/eval.rs` exists. Not a concern for this crate.

## Prioritized actions

- P0: empty-input guards (`rosenbrock`, `pinter`, audit the rest for
  `x.len()-1` / `x[n-1]` patterns); dimension asserts on fixed-dim
  functions (`eggholder` et al.).
- P0: resolve the `happy_cat` vs `happycat` exponent against the cited
  reference; remove one.
- P1: audit sweep for hard-indexed `x[0..2]` functions ignoring extra dims.

## Verdict

Benchmark values feed every optimiser comparison, so wrong-function
bugs are high-leverage. The two panics are execution-confirmed; the
happy-cat divergence undermines benchmark trust. Fix before the next
optimiser bake-off.

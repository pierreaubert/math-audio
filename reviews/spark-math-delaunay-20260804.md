# Review: math-delaunay (spark, 2026-08-04)

Scope: performance + correctness review of `crates/math-delaunay`
(v0.5.3, d3-delaunay port over `delaunator`). Method: parallel
read-only review agent + maintainer spot-verification. No code changed.

## Correctness

Strengths (verified): translation-invariant conditioning is real —
`bbox_scale_sq` (`delaunay/misc.rs:15-46`), invariant circumcenter scale
(`delaunay.rs:322-326`), NaN guard (`misc.rs:90-98`), pinned by tests
for micro-scale, translated, large-scale, and NaN inputs; area
conservation property test (`tests/property_tests.rs:51-85`).

Issues (all verified by reading):

- P1: `Voronoi::new` validates bounds only with `debug_assert!`
  (`voronoi.rs:38-46`). Release callers passing `xmin>=xmax`/NaN/Inf get
  silent garbage cells. Promote to a fallible constructor or clamp.
- P1: `Delaunay::new` does no finite-coordinate validation before
  `triangulate` (`delaunay.rs:47-56`); NaN/Inf propagate into
  backend-defined behavior.
- P2: `find(x,y,start)` (`:240-257`) and `circumcenter(t)` (`:300-333`)
  index unchecked — out-of-range `start`/`t` panics. `step`
  (`:260-262`) indexes `inedges[i]` *before* the `is_empty` guard, so
  empty-set/out-of-range calls panic with index-out-of-bounds rather
  than a clean error.
- P2: `cell_polygon` dedup epsilons are bbox-relative `1e-9*scale`
  (`voronoi.rs:144,532,600-602,648-649`) — fine at unit scale, but
  sub-epsilon slivers on large stages may survive; document the scale
  assumption.

## Performance

- Voronoi precomputes all circumcenters up front (`:47-56`); fine for
  typical point counts. `find` documents local-search limits — acceptable.
- No hot-loop allocation issues found; backend `delaunator` dominates.

## Prioritized actions

- P1: release-mode validation for `Voronoi::new` bounds and
  `Delaunay::new` finiteness (return `Result` or sanitize).
- P2: bounds-checked `find`/`step`/`circumcenter` or documented
  panics + tests pinning the contract.
- P2: document epsilon scale assumptions for large-coordinate stages.

## Verdict

Numerically careful port with good regression tests. Main gap is
release-mode input validation, not math. Good enough for Voronoi-cell
RIR interpolation use after the P1 hardening.

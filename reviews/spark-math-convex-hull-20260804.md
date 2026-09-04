# Review: math-convex-hull (spark, 2026-08-04)

Scope: performance + correctness review of `crates/math-convex-hull`
(Quickhull 3D). Method: parallel read-only review agent + maintainer
spot-verification of headline claims against source. No code changed.

## Correctness

- P0 (verified): wrong vertex slice in furthest-point search.
  `quickhull_3d` deduplicates into `unique_vertices` (`quickhull.rs:177`)
  and all `outside_points` indices address that slice (`:215`), but the
  main loop calls `find_face_with_furthest_point(&hull_faces, vertices)`
  (`:268`) with the *original* input. The cached path usually masks it,
  but the fallback linear search in `furthest_point` (`:89-106`) then
  measures distances at misaligned coordinates whenever dedup removed
  points. Fix: pass `&unique_vertices`. Silent wrong-hull risk, not just
  perf.
- P1 (verified): epsilon inconsistency. Dedup/simplex use scale-aware
  `relative_eps` (`:174-184`), but `HullFace::is_visible_from` uses
  absolute `EPSILON` (`:76-78`) and `Face::is_visible_from` hardcodes
  `1e-10` (`types.rs:176`). Large coords (1e6+) misclassify visibility;
  tiny coords (<1e-10) break. Thread one tolerance through `HullFace`.
- P1 (verified): degenerate-normal fallback is arbitrary.
  `HullFace::new` falls back to `(0,0,1)` on zero-area triangles
  (`:50-53`) instead of rejecting the face, creating a bogus plane.
- P2: `find_initial_simplex` line-seed filter (`:496-498`) discards
  points with `projection_len < 0.0`, suspect for valid off-axis points;
  needs a targeted degenerate-input test.

Strengths: tetrahedron seed with outward-normal fixup, cached
normal/plane constant per face (`:30-38`), simplex-centroid orientation
check (`:324-340`), good invariant property tests (all-points-inside,
dup-invariance).

## Performance

- Adaptive face compaction (`:248-266`) and parallel thresholds for
  visible-face search exist; main remaining cost is the O(faces) linear
  furthest-point scan per iteration — acceptable for audio geometry
  sizes. No redundant-allocation hot loop found beyond that.
- `SymmetricEigen`-style clones: none here; but `compact_faces` runs on a
  fixed schedule (`iterations % 500`) — fine.

## Prioritized actions

- P0: pass `&unique_vertices` at `:268`; add a duplicate-points hull
  regression test.
- P1: unify epsilon handling; reject (don't fake) degenerate faces.
- P2: property test with translated/large-scale inputs.

## Verdict

One real (verified) correctness bug tied to duplicate inputs, plus
tolerance hygiene issues. Otherwise sound. Fix the P0 before using this
for mesh-derived RIR work.

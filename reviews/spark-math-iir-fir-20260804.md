# Review: math-iir-fir (spark, 2026-08-04)

Scope: performance + correctness review of `crates/math-iir-fir`
(RBJ biquads, FIR, SVF, warped biquad, crossovers, filtfilt,
phase_smooth). Method: read-only review agent + maintainer
verification. No code changed.

## Correctness

Strong overall (verified): RBJ core covers LP/HP/BP/Notch/Peak/Shelf/
AllPass plus Orfanidis shelves (`iir/biquad.rs:386-472`) and Vicanek
`PeakMatched` (`:473-518`), degenerate-`a0` passthrough fallback
(`:521-536`), `frequency_margin = nyquist*sqrt(eps)` clamping
(`:110,:269`). Good dual API: clamping `new` (`:98-173`) vs strict
`try_new` (`:208-249`). Fast magnitude via precomputed `r_up/r_dw`
(`:546-551,:885-904`) matches `complex_response` (`:870-882`). SVF is
Zavalishin TPT (`svf.rs:127-217`). `WarpedBiquad` degenerates to stock
biquad at `lambda=0` (`iir/warped_biquad.rs:38-140`).

Issues (verified):

- P1: `Fir` constructors validate with `debug_assert!` only
  (`fir.rs:73-74,109-116`, also `lowpass` `:108-115`). Release builds
  accept empty coeffs / non-positive rates / above-Nyquist cutoffs and
  fail downstream. Same release-vs-debug class as the delaunay finding.
- P2: `np_log_result` allocates per call (`:908-912`) while
  `np_log_result_into` reuses buffers — fine as API, but audit internal
  callers (PEQ banks) to use the `_into` variant in loops.
- P2: `ScopedFlushToZero` (`denormals.rs:55-93`) is opt-in and not wired
  into any `process*` path — either wire it or document that callers
  own denormal handling.

## Performance

- FIR uses doubled circular buffer for contiguous reads (`:42-58,
  :335-375`) plus symmetric half-multiply path (`:598-631`) with NEON/
  AVX2 f64 variants (`:640-784`). Windowed-sinc design. No complaints;
  this is the best-engineered DSP core in the workspace.

## Prioritized actions

- P1: fallible/release-checked FIR constructors (or documented panics).
- P2: `_into`-only discipline inside bank/PEQ loops; decide FTZ policy.
- P2 (feature for autodiff axis): analytic biquad-coefficient
  gradients would let `math-autodiff` differentiate through *this*
  crate's designs instead of its own reimplementation.

## Verdict

The strongest crate reviewed. One release-validation gap, otherwise
minor API hygiene. Candidate reference implementation for the
workspace's filter math.

# Review: math-autodiff (spark, 2026-08-04) — new crate: accuracy, performance, SOTA

Scope: accuracy + performance + SOTA review of `crates/math-autodiff`
(~8.5 kLOC, frequency-domain differentiable LTI: tensor/module/fft/
recursion/iir/loss/optim/system). Method: read-only review agent
(baseline: prior `reviews/autodiff-20260818.md`, deltas re-verified) +
maintainer verification. No code changed.

## Accuracy

Prior-review fixes confirmed landed (verified): `sigmoid` clamps to
`[1e-6,1-1e-6]` closing the 0/0-NaN at Nyquist (`iir/biquad.rs:41-44`);
`ParallelDelay` validates bins/channels (`delay.rs:464-477`);
`SosFilter` rejects unstable/non-finite poles (`iir/sos_filter.rs:76-84`)
with O(K·M) VJP backward (`:317`); FFT plan cache (`fft.rs:97-108`) and
`process_with_scratch` reuse (`:316-332,:393-419,:550-640`); in-place
`Sgd::step` via `Zip` (`optim.rs:47-50`).

Remaining (all low severity, verified):

- P2: `Series::backward` recomputes the full forward on cache miss plus
  one full-tensor `clone()` per module (`system.rs:121-125,155-166`);
  fingerprint-gated cache is cold on the first step. Same pattern in
  `Parallel::backward` (`:320+`). Correct, doubles first-step forward
  cost — warm the cache or accept and document.
- P2: full-Jacobian `Array5` path still exists
  (`iir/response.rs:586-587,670`, `sos_frequency_response_jacobian*`
  `:665,731`). Live `SosFilter::backward` no longer calls it, but the
  `pub` API invites O(K²·M·N) use. Deprecate or gate behind a feature.

## Performance

- FFT plan cache + scratch reuse landed; SGD in-place. The remaining
  cost is the Series/Parallel clone-per-module pattern above and the
  legacy Array5 path. `benches/biquad_bench.rs` + 12 integration tests
  + 6 match examples (`peq/geq/svf/fdn`) give good coverage.

## SOTA gaps (prioritized)

1. Loss surface is MSE-only (`loss.rs:39-95`: `mse_loss`,
   `magnitude_mse_loss`). SOTA differentiable-audio needs multi-scale
   spectral loss, Bark/ERB weighting, and temporal-envelope terms —
   directly consumable from `math-dsp` psychoacoustics.
2. No perceptual/regularization priors (smoothness, sparsity on PEQ
   gains) — needed so FDN/PEQ matching doesn't overfit notches.
3. Gradient-through-geometry (delaunay/voronoi interpolation weights)
   is future work; note as non-goal for now.
4. Second-order or adaptive optimizers beyond SGD for ill-conditioned
   room fits (link `math-optimisation` LM/CMA-ES as outer loops).

## Prioritized actions

- P1: multi-scale + perceptual loss terms (biggest SOTA step).
- P2: Series/Parallel cache warming; deprecate Array5 Jacobian API.
- P2: PEQ smoothness/sparsity priors; optimiser bridge to
  `math-optimisation`.

## Verdict

Correct core with the scary NaN/stability holes already closed. The gap
to SOTA is almost entirely the loss function, not the engine — one
focused milestone.

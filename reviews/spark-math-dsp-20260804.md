# Review: math-dsp (spark, 2026-08-04)

Scope: performance + correctness review of `crates/math-dsp` (v0.5.27,
~28 kLOC) plus accuracy/SOTA gaps where it touches the RIR / autodiff /
analog axis. Method: read-only review agent + maintainer verification
of headline claims. No code changed.

## Correctness

Above-average edge-case discipline (verified): `fast_exp2` clamps
`[-126,126]` (`fast_math.rs:58-60`); EBUR128 rejects `channels==0` and
out-of-range rates (`ebur128/ebu_r128.rs:51-59`); ESPRIT guards
empty/near-zero SVD (`esprit.rs:142-144,158-170`); FDN derives per-line
`g_i = 10^(-3*m_i/(RT60*SR))` (`fdn.rs:99-108`); RMS uses f64
accumulator (`detector.rs:73-76`).

Issues (verified by reading):

- P1: `true_peak.rs:47-66` is NOT BS.1770-4 — Catmull-Rom 3-tap
  heuristic with hold boundary, and `self.peak = max_abs` keeps only the
  *last* window, so block-max callers silently get last-value-not-max.
  The sibling `ebur128/true_peak_detector.rs` uses the proper 48 kHz FIR
  table and `ebu_r128.rs:67-72` warns non-48 kHz true-peak is
  approximate: two divergent "true peak" paths under one name.
- P2: `fdn.rs:162` hard-clamps output to ±4.0 (verified, with comment
  admitting the rationale). Masks instability instead of guaranteeing
  it — acceptable as a safety net, but a non-unitary user matrix can
  sit in permanent saturation silently. Prefer a stability check or at
  least a saturation flag.

## Performance

- SIMD utilities + flush-to-zero helpers present with boundary tests
  (`simd/flush.rs:102-158`). No new hot-loop pathology reported; FDN
  per-line gains are precomputed, not per-sample.

## SOTA gaps (dsp-adjacent)

- No psychoacoustic loss surface for autodiff consumers (Bark/ERB
  spectral loss, temporal masking); `math-autodiff/loss.rs` is MSE-only
  (see autodiff review).
- FDN late-tail has no frequency-dependent decay control hooking into
  `math-rir` band metrics; deconvolution/transfer-matrix paths
  (`binaural_matrix`) could serve RIR equalization but are unlinked.

## Prioritized actions

- P1: unify the two true-peak paths (document which is canonical;
  fix `peak` to be a running max or rename to last-window).
- P2: FDN saturation reporting / unitary-matrix validation helper.
- P2 (feature): Bark/ERB weighting helpers consumable by autodiff loss.

## Verdict

Solid core with honest edge-case handling. The true-peak duality is the
one correctness-adjacent trap; the rest is feature-linking work toward
the RIR/autodiff axis.

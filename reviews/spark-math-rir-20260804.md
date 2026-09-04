# Review: math-rir (spark, 2026-08-04) — new crate: accuracy, performance, SOTA

Scope: accuracy + performance + SOTA feature review of
`crates/math-rir` (v0.5.8, ~3.5 kLOC). Method: read-only review agent
+ maintainer verification of headline claims. No code changed.

## Accuracy

Strong ISO 3382 path (verified): Schroeder backward integration in f64
(`metrics/decay_curve.rs:94-98`), dB relative to peak with -300 dB
floor (`:115`), EDT/T20/T30 via `fit_db_range(0,-10 / -5,-25 / -5,-35)`
with negative-slope guard (`iso3382_metrics.rs:162-173`),
C50/C80/D50/Ts from direct-sound TOA with `MIN_POSITIVE` guards
(`:134-158`), NaN-sentinel `EMPTY_METRICS` + `quality_verdict()` /
`fit_is_valid()` (r²≥0.95) (`:49-103`). Edge cases handled: empty RIR,
`sample_rate≤0`, `start≥len` → `EMPTY_METRICS` (`:112-122`);
non-finite total → empty curve (`decay_curve.rs:80-107`); `order==0`
passes input through (`bands.rs:99-115`); silent-RIR TOA `None` with
fallback to 0 (`detection/misc.rs:18-20`, `iso3382_metrics.rs:119`).

Concerns:

- P1 (verified): Chu-style two-pass noise cutoff
  (`metrics/misc.rs:98-150`) is self-described as "±20 ms vs Lundeby"
  with no Lundeby iteration and no 10%-tail SNR check before fitting —
  fine for T20/T30 on clean RIRs, but T30 on noisy/short RIRs can fit
  noise. Add Lundeby refinement + pre-fit SNR gate.
- P2 (verified): `linear_fit_indexed` sums `x=i*dt` absolutely
  (`metrics/misc.rs:15-27`); late-window absolute times magnify
  subtractive cancellation in `n·sxx-sx²`. Center the time base per
  fit window (standard fix, cheap).

## Performance

- `DecayCurve::from_rir_with_workspace` and `BandpassWorkspace` reuse
  buffers — good. No hot-loop pathology reported; f64 Schroeder sums
  are the right call for dynamic range.

## SOTA gaps (prioritized)

1. Lundeby late-onset/noise-floor iteration (above) — table stakes for
   ISO 3382 credibility.
2. Band-integrated EDT/T20 with fractional-octave filterbank linkage
   (own `bands.rs` exists; wire it to metrics + report per-band r²).
3. Mixing-time/echo-density output already present (`mixing_time.rs`,
   `segmentation.rs`) — promote to first-class report fields with
   uncertainty, matching SOTA RIR tooling (e.g. ITA-Toolbox/AKA style
   outputs).
4. Multi-position averaging (ISO 3382 §measurement positions) and
   just-noticeable-difference (JND) flags per parameter.

## Prioritized actions

- P1: Lundeby refinement + SNR gate; centered time base in fits.
- P2: per-band ISO metrics wired through `bands.rs`.
- P2: multi-position averaging + JND verdicts.

## Verdict

Textbook-correct core with honest edge-case handling; the documented
±20 ms cutoff approximation is the accuracy ceiling to lift. Closest of
the three new crates to SOTA already.

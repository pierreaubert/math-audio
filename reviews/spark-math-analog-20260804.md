# Review: math-analog (spark, 2026-08-04) — new crate: accuracy, performance, SOTA

Scope: accuracy + performance + SOTA review of `crates/math-analog`
(v0.5.2: harmonics/static/Hammerstein/tape/transformer models +
`AnalogProcessor` block contract). Method: read-only review agent
(which ran `cargo test -p math-analog --tests`, incl. realtime
zero-alloc and spectral-matrix suites, all passing) + maintainer
verification. No code changed.

## Accuracy

Strong contract discipline (verified): `ProcessSpec` validation
(`process.rs:22-33`); oversize-block/length rejection before state
mutation (`:183-207`); `AnalogModel::from_id` rejects unknown ids
(`lib.rs:77-87`); one sample loop per model, no virtual dispatch
(`lib.rs:107-155`).

Input/output hygiene is consistent (verified): `sanitize_sample`
clamps to ±16, zeroes non-finite (`:211-217`); `finite_output` clamps
to ±128 + denormal flush (`:220-236`); `ControlSmoother::advance` uses
1e-6 settle epsilon with snap-on-reset for click-controlled bypass
(`:141-148`). Curve math is careful: `tanh_drive` guard
(`curves.rs:9-17`), f32 transfer with f64 overflow fallback
(`:21-41`, tested `:89-100`), finite at `f32::MAX` (`:77-86`); stable
ADAA forms (`harmonics.rs:366,371-374`).

Watch items (per review, lower severity): clamp levels (±16/±128) are
reasonable but arbitrary — calibrate against named hardware headroom;
smoother time-constant reconfiguration across sample rates needs a
pinned test (partial coverage exists in `process.rs:250-254`).

## Performance

- Zero-alloc realtime path is tested (`tests/realtime.rs`), static
  per-model dispatch avoids vtable overhead, denormal flushing keeps
  worst-case determinism. `benches/harmonics.rs` present. No hot-loop
  issues found.

## SOTA gaps (prioritized)

1. Model coverage: SOTA analog-modeling suites (Neural Amp Modeler /
   differentiable waveshaper literature) center on *measured-device
   capture* — the `fitting` feature + `math-optimisation` LM bridge is
   the right start; add a documented capture→fit→validate loop with
   error metrics (see existing `reports/fitting.md`).
2. Oversampling discipline: stateful models need documented
   oversampling guidance (aliasing vs CPU tradeoff) — `reports/alias-
   reference.md` exists; promote conclusions into API docs/defaults.
3. Component-level (WDF/state-space) modeling vs static curves —
   `reports/wdf-vs-state-space.md` scopes it; keep as staged roadmap,
   not v1 scope.
4. Listening-protocol grounding (`analog-20260813-listening-protocol.md`)
   should gate what "SOTA" means for euphonic color models — keep the
   blind-test loop attached to new models.

## Prioritized actions

- P1: documented capture→fit→validate loop with numeric acceptance
  thresholds; oversampling guidance in API docs.
- P2: calibrate clamp levels to hardware headroom figures; pin
  smoother reconfiguration test across rates.
- P2: staged WDF/state-space roadmap per existing reports.

## Verdict

The most production-disciplined of the three new crates (contracts,
hygiene, realtime tests). SOTA work is modeling-coverage and
capture-loop depth, not engine repair.

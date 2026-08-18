# Analog coloration roadmap — 2026-08-18

This roadmap supersedes the phasing in
[analog-20260813.md](analog-20260813.md) for the work that remains. It is
grounded in the current `crates/math-analog` implementation and the evidence
ledger in [analog-20260813-scope.md](analog-20260813-scope.md). All naming,
calibration, realtime, and truthful-claims rules from the 2026-08-13 plan
still apply and are not restated here.

## Current state recap

Done and verified (see scope ledger and `crates/math-analog/reports/`):

- Crate foundation: `ProcessSpec`, `AnalogProcessor`, checked
  prepare/reset/process contracts, allocation-free steady state, deterministic
  reset, NaN/Inf sanitization, denormal flushing (`process.rs`).
- Five model families behind append-only IDs 0–4: `HarmonicModel` (Chebyshev
  H2/H3 of `tanh`), `StaticColorModel` (tanh/softclip/hardclip),
  `HammersteinModel` (≤5 Chebyshev branches, one-pole branch filters),
  `TapeModel`, `TransformerModel` (both explicitly stylized).
- First-order ADAA on the memoryless models via `math-dsp`.
- Calibration (`0 VU = −18 dBFS`), offline analysis module (harmonics, IMD,
  transient, alias reference, BS.1770 level matching).
- Contract, spectral-matrix, realtime-allocation, and performance test
  suites; three report examples; criterion benches.
- SOTF standalone Analog Color plugin, host-owned Off/2x/4x quality, and
  factory/engine/preset regressions.

Known gaps that drive this roadmap:

- Hammerstein, Tape, and Transformer have **no antialiasing path**; only the
  memoryless models support ADAA1. Alias control for stateful models is
  entirely host oversampling.
- No Wiener (pre-filter) structure, no branch filters above one-pole, no way
  to fit branch coefficients from measurements.
- No fitted/measured model exists; all coefficients are synthetic.
- Stateful models are stylized one-pole constructions: no hysteresis, no
  record/replay EQ, no sag, no slew limiting, no wow/flutter/noise/hum
  modules.
- `analysis.rs` uses a naive per-bin DFT and a rectangular-only window.
- Scalar f32 per-sample loops; no use of `math-dsp` SIMD helpers.
- `ControlSmoother::reconfigure` ignores its configured time constant
  (hardcodes 10 ms).

## Guiding constraints for every phase

- Truthful naming: curve, “-style” coloration, measured model, and component
  model stay distinct in metadata and docs.
- No allocation, locking, logging, or unbounded iteration in processing.
- Append-only model IDs; existing Saturation and Analog Color presets never
  change meaning.
- Every new model ships with the standard evidence set: spectral matrix,
  IMD, transient, alias-reference, performance, and allocation reports
  regenerated under `crates/math-analog/reports/`.
- One oversampling owner. In-crate additions must remain composable building
  blocks, not a second competing resampling path for SOTF.

## Phase A — Robustness and analysis tooling (foundation, no new sound)

Low-risk fixes that everything below depends on.

1. Fix `ControlSmoother::reconfigure` to honor the configured time constant
   (`process.rs`); add a regression test that re-prepare preserves smoothing
   time.
2. Replace the naive DFT in `analysis.rs` with the workspace FFT from
   `math-dsp`, keeping the public report types and the
   one-sided-amplitude convention bit-comparable on the existing fixtures.
3. Add windowed analysis variants (Hann, Blackman, flat-top) with declared
   coherent-gain correction, filling the single-variant `WindowKind` enum.
4. Add swept-sine / log-chirp capture helpers (generation + deconvolution)
   as an offline module, the capture side of Phase C fitting.

Exit criterion: existing reports regenerate with matching numbers on the
coherent-tone fixtures; long-record analysis cost drops from O(N·bins) to
FFT-bound.

## Phase B — Close the antialiasing gap

1. Extend ADAA1 coverage to the Hammerstein branches. Closed-form
   antiderivatives already exist for T2/T3 of `tanh` in `harmonics.rs`;
   generalize them per branch order (recurrence over Chebyshev
   antiderivatives) with derivative-check unit tests per order.
2. Add ADAA2 (second-order) for the memoryless curves where a closed-form
   second antiderivative exists (tanh, softclip); document the numerical
   fallback near zero exactly as `math-dsp` does.
3. Give Tape and Transformer an explicit `AntiAliasing` option: ADAA1 on the
   static `tanh` core where valid, plus documented guidance that host 2x
   remains the recommended control for the stateful parts. Do not claim ADAA
   makes a stateful nonlinearity alias-free.
4. Regenerate `reports/alias-reference.md` with the new paths and extend
   `tests/spectral_matrix.rs` to assert folded-bin reduction for every model
   family, not only the memoryless ones.

Exit criterion: every model family has a measured in-crate alias-reduction
option, and the spectral matrix shows folded energy below the provisional
guard (<50% of Off in the 10 kHz fixture) for all five families.

## Phase C — Measurement-driven model fitting (offline)

This unlocks the plan's Phase 4 ("first richer model") with real provenance.

1. Add a feature-gated offline `fitting` module (default off; pulls in
   `math-optimisation` only when enabled) that identifies parallel-Hammerstein
   coefficients from captured records:
   - input: captured response + known stimulus, or swept-sine harmonic
     impulse responses from Phase A;
   - fit: per-branch gains and one-pole cutoffs via DE followed by
     Levenberg–Marquardt polish on a harmonic-amplitude/phase cost;
   - validation split: held-out levels, frequencies, and programme material
     are never used for fitting.
2. Define a frozen-coefficient container with provenance metadata (target
   description, capture chain, sample rate, date, fit tool version, hash of
   the capture) so fitted models are immutable, auditable, and
   versioned.
3. Add a fit-quality report example: captured vs. model harmonic
   distribution, two-tone IMD, transient recovery, on both fit and held-out
   data.

Exit criterion: a synthetic round-trip (fit a known Hammerstein from its own
render, recover coefficients within tolerance) plus the report tooling, all
offline and feature-gated so the runtime dependency graph is unchanged.

## Phase D — First fitted richer model: Console/Preamp

The plan's Tier-2 candidate, now with machinery to do it honestly.

1. Implement `ConsolePreampModel` (new append-only model ID) as a
   Wiener–Hammerstein structure:
   - calibrated input gain;
   - coupling high-pass (pre-filter, i.e. true Wiener stage — requires
     extending `HammersteinModel` with a pre-filter slot);
   - mild frequency-dependent asymmetry;
   - level-dependent compression via the existing envelope machinery;
   - high-frequency output pole;
   - level-matched output trim.
2. Ship it first with the existing synthetic `generic_coloration()`
   coefficients relabeled truthfully as stylized, then replace them with
   fitted coefficients (Phase C) once a measured target or openly documented
   reference is supplied.
3. Extend branch filters from one-pole to biquad/SOS via `math-iir-fir`,
   keeping per-branch order bounded and prepared-time allocation only.
4. Compare directly against the H2/H3 baseline and static models in a new
   `reports/console-preamp.md`, including level-matched conditions.

Exit criterion: the richer model demonstrates repeatable behavior (spectral,
IMD, transient) not explained by loudness or the H2/H3 baseline, with
coefficient provenance recorded.

## Phase E — Stateful model upgrades

Ordered by increasing risk; each item is independently shippable behind new
append-only model IDs or new defaulted-off parameters.

1. **Tape, honestly stylized → measured-capable**: record/replay EQ curves,
   head-bump low-frequency resonance, level-dependent HF loss, and a real
   hysteresis option (Jiles–Atherton or Preisach) with a documented bounded
   iteration/fallback. Time constants become parameters instead of the
   hardcoded 8/80 ms values, with per-source-frame smoothing.
2. **Transformer**: low-frequency saturation from a proper flux state,
   replacing the leaky clamped heuristic; keep the stylized mode for preset
   compatibility.
3. **Power-supply sag** as a shared sub-module (envelope-driven rail
   compression) usable by Console/Preamp and future component models.
4. **Slew-rate limiting** module with explicit V/µs parameterization.
5. **Defect modules**, all defaulting to off and individually gated:
   wow/flutter (pitch LFO on a short modulated delay), noise floor, hum,
   crosstalk. These must never be bundled into a model's default behavior.

Exit criterion per item: stateful-model test battery (convergence from rest,
reset from saturation, bounded extreme-input response, hysteresis loop
direction where claimed, sample-rate-independent time constants) plus the
standard reports.

## Phase F — Component-level tier (only with a reference)

Per the 2026-08-13 plan, this tier starts only when a schematic or measured
unit exists. Preparatory work that is safe to do now:

1. A bounded nonlinear solver utility (capped fixed-point/Newton iterations,
   deterministic convergence check, finite click-safe fallback) as an
   internal primitive, stress-tested at extreme states.
2. Evaluation spike: WDF vs. bounded state-space for a single diode clipper,
   with a position paper on CPU, accuracy, and code complexity before any
   component library is committed.
3. If adopted: diode clipper, triode stage, and a tone stack as the first
   three components, each validated against a published reference or
   measurement.

Exit criterion for the spike: a written decision record; for components:
measured agreement within a pre-registered tolerance on the validation
matrix.

## Phase G — Performance and SIMD

Do this after the model surface stabilizes (Phases A–D), not before.

1. SIMD inner loops for the Chebyshev recurrence and the static curves using
   `math-dsp::simd`, behind runtime detection with a scalar fallback; prove
   bit-compatible-or-tighter tolerance in the spectral matrix.
2. Convert `tests/performance.rs` worst-callback report into a hard bound on
   CI hardware (provisional guard: ≤25% of a 2,048-frame/48 kHz callback
   period, 12 channels, worst model).
3. Denormal-stress benchmark coverage for the stateful models.

Exit criterion: worst-callback bound enforced in tests; SIMD shows a
measurable win on the 6/12-channel criterion matrix without changing
reported spectra.

## Phase H — SOTF integration follow-through

Not in this repository, tracked here for completeness:

1. Run the listening protocol
   ([analog-20260813-listening-protocol.md](analog-20260813-listening-protocol.md))
   with the level-matching helper; fill in the intentionally empty result
   fields before any accuracy claim is made.
2. Choose the first embedded `AnalogColorStage` placement (candidate:
   pre-detector in the compressor) only after the standalone plugin has
   listening evidence.
3. Wire any new model IDs through factory/catalog/engine/bridge/UI surfaces
   with preset round-trip tests, one PR per model.

## Priority and dependency summary

| Phase | Theme | Depends on | Risk |
| --- | --- | --- | --- |
| A | Robustness + analysis tooling | — | Low |
| B | Antialiasing for all models | — | Low |
| C | Offline fitting + provenance | A | Medium |
| D | Console/Preamp fitted model | B, C | Medium |
| E | Stateful upgrades (tape/transformer/sag/slew/defects) | B | Medium–High |
| F | Component-level tier | external reference unit | High |
| G | SIMD + hard perf bounds | A–D surface stable | Low |
| H | SOTF listening + embedded stage | D, E listening evidence | Process |

Recommended execution order: A and B immediately (independent, unlock
everything), then C → D, with E.1 starting once B lands. F is gated on an
external reference; G is scheduled after D; H follows SOTF-side.

## Open decisions carried forward

1. 2x vs. 4x default quality — still pending benchmark-machine confirmation
   of the provisional CPU/alias guards.
2. First measured target for Phase C/D fitting (device, revision, capture
   chain) — the single external blocker for all hardware-named work.
3. Hysteresis formulation for tape (Jiles–Atherton vs. Preisach) — decide at
   Phase E.1 design time with a small accuracy/CPU spike.
4. Whether `fitting` stays in `math-analog` behind a feature or moves to a
   separate `math-analog-fitting` crate if its dependency weight grows.

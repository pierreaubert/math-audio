# Component-model frozen reference — Phase F (IDs 6–8)

Status: **frozen reference**. This note pins the published equations,
parameter values, and validation tolerances for the three component-tier
models BEFORE their implementation. Tolerances below are pre-registered:
an implementation that misses them does not ship as a component model.

Naming tier: these are **component models** (documented circuits solved
from published equations), not measured-hardware models. No device claim
is made; validation is against the cited published references.

## Sources (all inspected 2026-09-27)

- Koren triode equations + 12AX7A parameters (Glass Audio 1996, updated
  2001/2008): `https://www.normankoren.com/Audio/Tubemodspice_article.html`
  and `https://www.normankoren.com/Audio/Tube_params.html`
- Shockley diode equation:
  `https://en.wikipedia.org/wiki/Shockley_diode_equation`
- TMB tone-stack topology narrative + production values:
  `https://robrobinette.com/How_The_TMB_Tone_Stack_Works.htm`
- Yeh & Smith, "Discretization of the '59 Fender Bassman Tone Stack",
  DAFx-06 (symbolic transfer function, SPICE-verified at t=m=l=0.5),
  bundled in `https://robrobinette.com/images/Guitar/Bassman/Fender_Bassman_5F6-A_Circuit_Kuehnel.pdf`
- Kuehnel, "The Fender Bassman 5F6A Circuit" (same PDF): schematic
  values 56k slope, 250pF treble, 0.02uF bass+mid; 1M next-stage load
  convention shared with the Robinette page.

## ID 6 — DiodeClipperModel (Shockley shunt clipper)

Circuit: input → series R → node Vc → (C to ground || antiparallel
Shockley diodes to ground); output = Vc.

- Diode current (one direction): `I = Is * (exp(V / (N * Vt)) - 1)`,
  `Vt = 0.02585 V` (thermal voltage, T = 300 K). Antiparallel pair:
  `I = 2 * Is * sinh(V / (N * Vt))`.
- Defaults (model defaults, not datasheet values): `R = 10 kΩ`,
  corner `10 kHz` (`C = 1 / (2π·R·corner)`); silicon flavor
  `Is = 2.0 nA, N = 1.8` (predicts ~0.50 V clipping at 100 uA);
  germanium flavor `Is = 0.2 uA, N = 1.0` (predicts ~0.16 V at 100 uA).
- Level convention: 0 dBFS = 1 V peak at the clipper input.
- Solver: trapezoidal companion model + Newton via
  `solve_bounded_nonlinear`. As-built (amended 2026-09-27 after f32
  correction-resolution analysis; gates unchanged): R-scaled residual so
  the tolerance is in volts, tol `1e-4` V, max 64 iterations, rails ±2 V
  (proven to contain every solution: diodes nail the node under 0.85 V),
  initial guess = exact linear node voltage clamped to ±0.75 V, exponent
  argument clamped to ±80. A non-converged iterate within 1 mV is
  accepted (and counted); only a truly lost solve holds the previous
  sample. Both outcomes increment saturating fallback counters.

Pre-registered gates:

- DC transfer is odd-symmetric within `1e-4` V and monotonic.
- Measured clipping threshold within ±5% of the Shockley closed-form
  prediction at the test current, for both flavors.
- Small-signal gain 1 ± 0.1 dB below threshold.
- Passivity pre-trim: |out| <= |in| + 1e-6 for in-band sine input.
- Solver fallback rate < 0.1% of samples on the spectral-matrix
  fixture; every fallback holds (no click: step <= active signal peak).
- Finite/bounded for ±16 inputs at ±36 dB drive extremes.

## ID 7 — TriodeStageModel (Koren 12AX7A common-cathode stage)

Koren triode equations (Tube_params.html Eq. 1):

- `E1 = (EP / KP) * ln(1 + exp(KP * (1 / MU + EG / sqrt(KVB + EP^2))))`
- `IP = (E1^EX / KG1) * (1 + sgn(E1))`
- 12AX7A parameters: `MU = 101.24, EX = 1.267, KG1 = 1002.9`,
  `KP = 699.73, KVB = 300.0, VCT = 0.00`.

Stage (documented operating point, fixed bias):

- `Vb = 350 V`, `Rp = 150 kΩ` (Koren's published "typical load line"),
  next-stage grid leak `1 MΩ` (Robinette tone-stack source), AC plate
  load `Rp || 1 MΩ`, fixed grid bias `-1.5 V` (inside Koren's published
  0..-4 V curve region).
- Quiescent point solved at design time from the equations above:
  `Vp_q = 204.174 V, Ip_q = 0.9722 mA`, `gm = 1.85 mS`. The implementation
  re-verifies this Q against the Koren equation and the DC load line
  within `1e-3` (relative) in tests.
- CORRECTION (2026-09-27): the pre-registered "AC gain -241.8" was a
  design hand-calculation error (forgot the plate resistance). The true
  small-signal gain is `-gm * (Rac || rp) = -68.2`, confirmed three ways:
  the implementation's implicit solve, direct gm/gp differentiation of
  the published equation in tests, and hand derivation. Normalization,
  unity-gain, and THD gates are unaffected (all relative to the measured
  gain); only the informational design number was wrong.
- Level convention: 1 V peak grid swing per 0 dBFS; input/output
  coupling high-passes at 8 Hz (model defaults); the output is the
  plate AC normalized by the numeric small-signal gain measured at
  prepare time (deterministic; documented unity small-signal convention).
  The stage inverts (documented).
- Solver: per-sample Newton on the plate node via
  `solve_bounded_nonlinear`. As-built (amended 2026-09-27; gates
  unchanged): tol `5e-3` V plate (clears the slope·ulp floor of ~3e-3 V
  in the saturation knee), max 64 iterations, physical bounds `[0, Vb]`,
  two-phase solve (Newton from previous plate, then a fixed 17-point
  grid restart on failure — breaks rail ping-pong on
  cutoff/saturation transitions), loose acceptance 0.25 V plate with
  hold-previous only when truly lost. Stable softplus evaluation (no exp
  overflow for any finite grid).
- Documented limitations: no grid current (Koren triode has none for
  EG < 0; positive-grid drive stays bounded but unmodeled — Leach
  extension is future work); fixed bias (no cathode RC); static AC load
  (the plate solve uses `Rp || 1M` without output-coupling-cap memory —
  exact for small signals, approximate in deep clipping where the true
  circuit's cap state matters; explicit cap state is future work).

Pre-registered gates:

- Pinned Q satisfies Koren + load line within `1e-3` relative.
- Small-signal gain 1 ± 0.5 dB post-normalization (-36 dBFS, 1 kHz).
- Output inverts (sine correlation with input < 0).
- THD rises monotonically over a drive sweep (-12..+24 dB).
- Finite/bounded for ±16 inputs; positive-grid drive stays finite.

## ID 8 — ToneStackModel (Yeh FMV/TMB network)

Topology: exactly Yeh & Smith Fig. 1 (ideal source, unloaded output —
Yeh verified loading-independent). Nodes: IN → R4(slope) → S;
IN → C1 → T; R1 treble pot T → W(wiper/output) → X with segments
(1-t)R1 / tR1; C2 bass cap S → X; R2 bass variable resistor X → Y with
value l·R2; R3 mid pot Y → Z(wiper) → GND with segments (1-m)R3 / mR3;
C3 mid cap S → Z.

- Value sets: schematic ('59, DEFAULT — the set Yeh SPICE-verified):
  `R4 = 56 kΩ, C1 = 250 pF, C2 = C3 = 0.02 uF, R1 = 250 kΩ`,
  `R2 = 1 MΩ, R3 = 25 kΩ`. Production variant: `R4 = 100 kΩ`,
  `C2 = 0.1 uF` (Robinette: shipped units + reissues). Both face the
  symbolic oracle in tests.
- Pot segments clamp to a 1 Ω wiper/contact minimum (documented).
- Derivation: resistive MNA matrix + descriptor-to-explicit state-space
  reduction (3 capacitor-voltage states), bilinear discretization with
  the exact causal input absorption. Coefficient math in f64 at
  prepare/control time; per-sample loop is a 3-state f32 filter.
- Knobs 0..1 with 10 ms smoothing; matrices rebuild when a smoothed
  knob moves more than `1e-6` (static otherwise — deterministic and
  partition-independent). Bass sweeps l linearly in v1 (Yeh models a
  log taper; log taper is deferred future work — see note).
- Level convention: 0 dB reference = |H(1 kHz)| with knobs at noon
  (-11.74 dB raw for the schematic set); the reference is fixed at
  prepare time per value set and does NOT track knob moves.

Pre-registered gates (default schematic set unless noted):

- State-space response vs Yeh symbolic H(s) oracle: <= 0.25 dB,
  20 Hz–15 kHz, at 4+ knob settings including extremes. Both value
  sets face the oracle.
- DC gain H(z=1) < -90 dB.
- Post-normalization |H(1 kHz)| at noon = 0 ± 0.1 dB.
- Noon scoop: minimum over 400–1000 Hz >= 6 dB below the 100 Hz
  magnitude (design prototype: 9.6 dB; dip sits ~700 Hz, see note).
- Bass knob full sweep at 40 Hz: increasing, range >= 10 dB
  (prototype: 15 dB). Mid knob at 500 Hz: strictly increasing,
  range >= 4 dB. Treble knob at 8 kHz: strictly increasing,
  range >= 8 dB.
- 48/96 kHz responses agree within 0.5 dB below 18 kHz.
- Both value sets finite at all knob extremes (0/1 corners).

## As-built shared-solver fix (2026-09-27)

`solve_bounded_nonlinear`'s stall exit now reports convergence when the
stalled iterate satisfies the tolerance (matching the exhausted-budget
exit). Rationale: near a steep root the Newton correction drops below the
stall threshold while the residual is already inaudible, so the old exit
reported false with the root found. No in-crate caller depended on the old
behavior (verified: only the new component models call it); the existing
solver test still passes and a dedicated regression test pins the new exit.

## Shared gates (all three)

- Existing contract battery extended: finite outputs, bit-exact
  callback-partition independence, reset == fresh prepare, no
  allocation after prepare, denormal stress survival.
- New append-only IDs 6/7/8; existing presets unchanged in meaning.
- Worst-callback provisional guard (25% of a 2048-frame/48 kHz
  callback, 12 channels, worst model, release) still holds with the
  new models included.
- Antialiasing decision (explicit): implicit Newton solves have no
  closed-form antiderivative, so in-crate ADAA is not applicable;
  internal oversampling would violate the one-oversampling-owner rule.
  Alias control for IDs 6/7 is documented host oversampling; ID 8 is
  linear (no aliasing). A characterization test records finite alias
  reports for the new models instead of the 50%-folded-energy guard,
  which stays scoped to families 0–5.

## Design investigation notes (kept, not gates)

- A first prototype wired the treble wiper backwards and the bass pot
  in series with the signal path; both were caught by behavior probes
  (inverted treble action, bass-up reducing LF) and corrected against
  Yeh Fig. 1 before this reference was frozen.
- The published "mid scoop at 500 Hz, knobs at noon" narrative
  reproduces as a ~700 Hz dip with Yeh's schematic values; the gate
  above uses a 400–1000 Hz window rather than the narrated 500 Hz.
- Yeh sweeps the bass control logarithmically; v1 sweeps l linearly.
  The symbolic oracle validates either sweep since it is written in l.

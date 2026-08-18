# Analog initiative scope and evidence ledger

This local ledger records the decisions that can be made from the supplied
plan and current repository state. It is not a hardware-validation report.

## Current scope decisions

| Decision | Current value | Evidence or limitation |
| --- | --- | --- |
| Library ownership | Separate `math-analog` crate | Keeps model equations and state out of SOTF host/plugin metadata. |
| First target | Generic mathematical coloration | No hardware unit, schematic, capture, or target coefficients were supplied. |
| Calibration | `0 VU = -18 dBFS` | Implemented in `level.rs` and covered by unit tests. |
| Default standalone quality | Host-owned 2x | Analog Color defaults to quality index 1; host wrapper owns resampling and reports its latency. |
| Hardware claims | None | Tape/Transformer are explicitly stylized; Hammerstein’s built-in coefficients are synthetic. |
| Embedded stage | Not selected | Standalone placement is validated before any plugin-specific signal-flow choice. |
| Formal alias/CPU budget | Provisional local fixture guards selected | CPU: worst 2,048-frame/48 kHz callback ≤25% of callback period; alias: host 2x/4x folded-bin amplitude <50% of Off in the declared 10 kHz fixture. These are engineering guards, not hardware-model claims; release acceptance remains open. |

## Evidence available locally

- [analog-20260813-baseline.md](analog-20260813-baseline.md) records the
  existing Saturation QA CPU/allocation/DC baseline plus harmonic, IMD,
  direct-folded-alias, latency, and preset round-trip measurements.
- [math-analog/reports/alias-reference.md](crates/math-analog/reports/alias-reference.md)
  records the synthetic high-rate reference comparison.
- [math-analog/reports/analysis.md](crates/math-analog/reports/analysis.md)
  records harmonic, THD/THD+N, two-tone IMD, and transient fixtures.
- [math-analog/reports/model-matrix.md](crates/math-analog/reports/model-matrix.md)
  records the same synthetic harmonic, IMD, transient, DC, and finite-output
  fixture across all five serialized model families.
- [math-analog/reports/performance.md](crates/math-analog/reports/performance.md)
  records the bounded callback matrix and allocation test commands.
- `math-analog::analysis::level_match_candidate` provides offline
  BS.1770/EBU R128 loudness matching with an explicit sample-peak ceiling for
  future listening fixtures; it does not substitute for human preference or
  discrimination results.
- [analog-20260813-listening-protocol.md](analog-20260813-listening-protocol.md)
  defines the render set, blinded trial procedure, matching fields, and
  result schema; it intentionally contains no human or hardware result.
- [analog-20260813-host-validation.md](../sotf/analog-20260813-host-validation.md)
  records the standalone host-owned Off/2x/4x alias-ordering fixture.
- The SOTF factory regression `analog_color_host_owns_quality_for_every_model_family`
  verifies that both host-owned quality modes report wrapper latency and that
  the plugin does not retain a second oversampling owner for all five model
  IDs.
- The filtered SOTF facade suite `cargo test -p sotf-plugins analog_color`
  passes its four Analog Color catalog/factory/host tests.
- SOTF engine regressions now verify every non-default Analog Color field at
  the factory conversion boundary and preserve all non-default fields through
  a serialized preset round-trip.
- Current broad verification also passes `cargo check --workspace` and
  `cargo clippy --workspace -- -W warnings` in both repositories. The full
  math workspace test command is not treated as a passing gate here because
  unrelated convex-hull exporter tests require filesystem access outside the
  sandbox; the focused analog and integration suites remain the authoritative
  regression evidence for this change.

## Open evidence gates

The overall plan is not hardware-model complete until a reference target and
target-specific acceptance budget are supplied. The local provisional guards
above are synthetic engineering checks, not that target-specific evidence.
The following remain intentionally open:

- implementation issue and required remote SOTF code review (Tea/Gitea is
  unavailable in this environment);
- measured target or schematic, fitted coefficients, provenance, and held-out
  spectral/IMD/transient validation;
- blinded level-matched listening comparison using the new offline matching
  primitive;
- confirmation of the provisional CPU/alias guards on the selected benchmark
  machine and broader stimuli;
- a deliberate embedded `AnalogColorStage` placement.

The local implementation therefore exposes truthful generic and stylized
models and does not label them as emulations of named hardware.

# QA process for all workspace crates — 2026-08-18

## Objective

Provide a single command `just qa` that runs the complete QA process for the
whole workspace, and `just qa-{crate}` commands that run the QA process for a
single crate. The process mirrors the CI gates so failures are caught locally
before pushing.

Current state: `just qa` only runs `qa-math`, which runs the `simd-fuzzer`
binary from `math-dsp`. Everything else (per-crate tests, property tests,
examples, benches, coverage) is not covered by any single entry point.

## Decisions (from brainstorming)

- Per-crate QA includes the full ladder: fmt-check, clippy with `-D warnings`,
  unit + integration + doc tests, coverage gate, plus crate-specific extras
  (fuzzers, examples, bench smokes).
- `just qa` mirrors the CI gates (workspace clippy with `plotly`, 90 %
  workspace coverage gate).
- Fuzzers and property tests run at full strength; criterion benches are
  compile-checked and smoke-run with `--quick` (full bench runs stay in
  `just bench`).
- Implementation is pure Justfile (no new tooling, no shell scripts).

## Recipe structure

One private parameterized ladder recipe plus thin public wrappers. Just runs
recipe lines sequentially and stops on first failure, giving fail-fast per
crate for free.

```just
_qa crate threshold:
    cargo fmt -p {{crate}} -- --check
    cargo clippy -p {{crate}} --all-targets -- -D warnings
    cargo test -p {{crate}} --lib --release
    cargo test -p {{crate}} --tests --release
    cargo test -p {{crate}} --doc
    cargo llvm-cov -p {{crate}} --lib --summary-only --release --fail-under-lines {{threshold}}
```

Public recipes use short crate names (matching the existing
`examples-autodiff` naming style):

| Recipe | Crate | Extras |
|---|---|---|
| `qa-analog` | math-analog | run the 5 report examples (regenerate `crates/math-analog/reports/`), `harmonics` bench smoke |
| `qa-autodiff` | math-autodiff | run the 6 `*_match` examples, `biquad_bench` smoke |
| `qa-convex-hull` | math-convex-hull | none beyond the ladder |
| `qa-delaunay` | math-delaunay | none beyond the ladder |
| `qa-dsp` | math-dsp | `simd-fuzzer` full strength; AVX2 check+test pass gated on `arch() == "x86_64"`; `wav2csv` build check |
| `qa-iir-fir` | math-iir-fir | run the 5 examples; `filter_bench` bin + bench smokes |
| `qa-optimisation` | math-optimisation | run the 5 `optde_*` examples; `run-de` build check; `de_bench`/`cmaes_bench` smokes |
| `qa-rir` | math-rir | `iso3382` bench smoke only (no examples exist) |
| `qa-test-functions` | math-test-functions | run examples; `plot-functions` build check with `plotly` feature |

Bench smoke = `cargo bench -p <crate> --no-run` (compile check) plus
`cargo bench -p <crate> --bench <name> -- --quick` where the bench exists.

`qa-math` remains as an alias of `qa` for backwards compatibility.

## Aggregator

```just
qa: qa-analog qa-autodiff qa-convex-hull qa-delaunay qa-dsp qa-iir-fir qa-optimisation qa-rir qa-test-functions
    cargo clippy --workspace --all-targets --features plotly -- -D warnings
    cargo llvm-cov --lib --summary-only --release --fail-under-lines 90
```

## Coverage thresholds

Per-crate threshold = current measured line coverage, rounded down to a whole
percent. This acts as a ratchet: thresholds may be raised later but never
silently lowered. Implementation measures each crate with `cargo llvm-cov`
before filling in the numbers.

The workspace gate stays at 90 % to mirror CI. If the workspace currently
measures below 90 %, that is reported to the user rather than weakening the
gate.

## CI and documentation updates

- `.github/workflows/ci.yml`: the `qa-math` job runs `just qa` instead of
  `just qa-math` (the separate coverage-gate job already exists).
- Root `AGENTS.md`: the "Build and Test Commands" section documents
  `just qa` and `just qa-{crate}`.

## Failure behavior

Fail-fast: within a crate, the ladder stops at the first failing step;
`just qa` stops at the first failing crate. No parallel or best-effort mode —
output stays readable and semantics match CI.

## Error handling

- Missing tools (`cargo-llvm-cov`, criterion harness) surface as ordinary
  cargo errors; `install-rustup` already installs `cargo-llvm-cov`.
- The AVX2 pass only runs on `x86_64` hosts; on Apple Silicon it is skipped
  with a printed note.
- Examples that require the `plotly` feature are build-checked with the
  feature enabled rather than run, avoiding chromedriver requirements.

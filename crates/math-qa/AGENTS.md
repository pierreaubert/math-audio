# math-qa (lib: `math_qa`, version: 0.1.0)

Wolfram Engine cross-validation for math-audio: 9 closed-form `.wls`
oracles plus Rust comparison tests.

## Layout

```
src/
  lib.rs            rel_error, complex compare helpers, golden/live
                    reference loader, QA_RESULT emitter
wolfram/
  *.wls             One closed-form oracle per case (RawJSON to stdout)
  goldens/*.json    Engine-blessed references (via `just qa-goldens`)
tests/
  wolfram_*.rs      One comparison target per manifest case
  qa_contract.rs    Manifest <-> files completeness gate
validation-manifest.toml   Case ids, tolerances, tiers
```

## Test tiers

| Tier | Cases | Cost |
|------|-------|------|
| `smoke` | biquad, fft, schroeder, comb, third-octave, svf, test-functions | seconds |
| `deep` | waterfall-spots, morlet-spots (direct DFT/CWT summation) | ~1 min (mostly engine-side) |

## Reference resolution

Each comparison test resolves its reference live when `WOLFRAMSCRIPT`
points at an activated engine, otherwise from the checked-in golden.
With neither available the test prints `SKIPPED` and passes. The
contract test (`qa_contract`) fails while any manifest case lacks a
golden — that is the honest pending-engine signal, cleared by
`just qa-goldens`.

## Tolerances

Recorded per case in `validation-manifest.toml`. Rules of thumb:

- Same-math f64 comparisons (biquad, SVF, test functions): 1e-12..1e-6.
- Sample-rounded comparisons (FFT, waterfall, Morlet spot dB): 1e-6..1e-3.
- Fit-derived quantities (T30): 1e-4 (window-edge quantization).

## Adding a case

1. Write `wolfram/<name>.wls` emitting one compact RawJSON object as
   its last stdout line (`schema_version`, `case`, params, values).
2. Generate the golden: `WOLFRAMSCRIPT=wolframscript just qa-goldens`
   (or per-file: `wolframscript -file wolfram/<name>.wls`).
3. Write `tests/wolfram_<name>.rs` comparing via `math_qa::reference`
   and emitting `QA_RESULT:` on success.
4. List the case in `validation-manifest.toml`.

## Testing

```bash
cargo test -p math-qa --release
cargo check -p math-qa && cargo clippy -p math-qa
just qa-goldens   # needs an activated Wolfram Engine
```

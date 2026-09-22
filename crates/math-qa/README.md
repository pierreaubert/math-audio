# math-qa

Independent cross-validation of math-audio against the Wolfram Engine.

Each case pairs a closed-form Wolfram oracle (`wolfram/*.wls`) with a
Rust comparison test (`tests/wolfram_*.rs`). References resolve live
when `WOLFRAMSCRIPT` is set, otherwise from checked-in goldens
(`wolfram/goldens/`, produced with `just qa-goldens`).

Covered: biquad and SVF frequency responses, FFT peak calibration,
Schroeder/T30 decay, M1 comb first-dip, M2 third-octave SPL, M4
waterfall STFT spots, M5 Morlet CWT spots, optimisation test functions.

See [AGENTS.md](AGENTS.md) and `validation-manifest.toml` for the case
list, tiers, and tolerances.

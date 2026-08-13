# Synthetic alias-reference report

This is a deterministic, synthetic comparison of the `HarmonicModel` direct
and ADAA paths against a high-rate render. It is an aliasing proxy, not a
hardware-model accuracy score: the error also includes the declared
reconstruction filter transition band and any state evolution difference
between sample rates.

Generate it from the repository root with:

```text
rtk cargo run -p math-analog --example alias_reference_report --offline
```

The report uses a 48 kHz, 4,800-frame, 10 kHz sine at amplitude 0.8, 24 dB
drive, muted H2/H3 branches, and a 127-tap Blackman-windowed sinc FIR with
cutoff `0.45 / factor` before decimation. The comparison is sample-aligned;
the helper zero-extends record boundaries, so the captured values below are
best interpreted with the documented guard-interval caveat.

Captured output on 2026-08-13:

```text
sample_rate_hz=48000
tone_hz=10000
base_frames=4800
input_amplitude=0.8
drive_db=24
h2_db=-120
h3_db=-120
fir=127-tap Blackman-windowed sinc, cutoff=0.45/factor
columns=anti_aliasing factor base_rms reference_rms error_rms error_peak error_level_db
Off 2 0.958186269 0.907638609 0.229590163 0.401740432 -12.780934334
Off 4 0.958186269 0.906785607 0.206375733 0.414556146 -13.706827164
Adaa1 2 0.838684142 0.887181282 0.304204702 0.573132157 -10.336681366
Adaa1 4 0.838684142 0.906500459 0.456382215 0.879318476 -6.813426018
```

The exact values are part of the reproducibility artifact for this declared
synthetic setup. They must not be interpreted as a measured hardware target,
an absolute alias budget, or evidence that ADAA and host oversampling are
interchangeable.

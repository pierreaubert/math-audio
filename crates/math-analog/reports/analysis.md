# Synthetic harmonic, IMD, and transient report

This artifact exercises the deterministic offline analysis helpers against a
synthetic `HarmonicModel`. It is a reproducibility fixture, not a measured
hardware target or a perceptual quality claim.

Generate it with:

```text
rtk cargo run -p math-analog --example analysis_report --offline
```

The fixture uses 48 kHz, 4,800 samples, a rectangular window, one-sided
amplitude normalization, no zero-padding, 12 dB drive, H2 at -18 dB, and H3 at
-24 dB. The harmonic and IMD records are coherent for the listed tones. The
transient summary is computed over the first 256 samples of the harmonic
record.

Captured output on 2026-08-18:

```text
sample_rate_hz=48000
record_length=4800
model=HarmonicModel, drive_db=12, h2_db=-18, h3_db=-24
convention=rectangular one-sided amplitude, no zero-padding
harmonic fundamental=1.136563301 h2=0.101872988 h3=0.145714045 h4=0.039769039 h5=0.016192872
distortion thd=0.160928726 thd_plus_n=0.161400661 alias_rms=0.000000000 alias_level_db=-inf
imd tone_a=0.714015961 tone_b=0.330459684 2f1-f2=0.061607081 2f2-f1=0.057547688
transient peak=1.091778159 peak_index=12 rms=0.818400264 dc=0.057019427
```

The aliases are reported separately from desired in-band harmonics. All
normalization and record metadata are intentionally printed by the example so
that the numbers cannot be mistaken for zero-padded resolution or a hidden
window correction.

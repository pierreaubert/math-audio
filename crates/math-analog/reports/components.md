# Component-tier evidence report (IDs 6-8)

This artifact records deterministic implementation evidence for the
component-tier models against their frozen references in
`../references/component-references.md`: Shockley clipper thresholds and
symmetry, triode gain/THD behavior, and tone-stack level/scoop/sweeps. It
is implementation evidence, not a hardware fit or listening result.

Generate it with:

```text
rtk cargo run --release -p math-analog --example component_report --offline
```

The fixture renders settled coherent sines at 48 kHz (14,400 settle +
4,800 measured samples, rectangular-window one-sided amplitude). Clipper
threshold rows drive a 100 Hz sine at 5 V peak; triode rows sweep drive at
1 kHz; tone-stack rows use the schematic value set unless noted.

Captured output on 2026-09-27:

```text
sample_rate_hz=48000 settle=14400 measure=4800
fixture=settled coherent sines, rectangular one-sided amplitude
clipper flavor=Silicon sine100hz_peak5v_out=0.636092 solves=19200 fallbacks=0
clipper flavor=Germanium sine100hz_peak5v_out=0.217200 solves=19200 fallbacks=0
clipper symmetry h1=0.632197 h2=0.000000 h3=0.157238
triode gain_magnitude=68.2133
triode drive_db=0 in=0.016 gain_db=-0.004 thd=0.000618 solves=19200 fallbacks=0
triode drive_db=-12 in=0.5 gain_db=-12.005 thd=0.004849 solves=19200 fallbacks=0
triode drive_db=0 in=0.5 gain_db=-0.029 thd=0.019395 solves=19200 fallbacks=0
triode drive_db=12 in=0.5 gain_db=11.526 thd=0.083380 solves=19200 fallbacks=0
triode drive_db=24 in=0.5 gain_db=15.531 thd=0.314513 solves=19600 fallbacks=0
tonestack values=Schematic59 noon_1khz_db=0.010 scoop_db=9.45
tonestack values=Production noon_1khz_db=0.010 scoop_db=10.87
tonestack sweep knob=bass hz=40 db=[-5.10, 7.71, 9.18, 9.64, 9.84]
tonestack sweep knob=mid hz=500 db=[-2.63, -1.45, 0.23, 1.71, 2.89]
tonestack sweep knob=treble hz=8000 db=[-1.65, 3.74, 7.09, 9.51, 11.41]
```

Reading notes:

- Clipper sine peaks (0.636/0.217 V) sit just above the settled-DC
  Shockley predictions (0.573/0.201 V) because the shunt capacitor
  conducts at 100 Hz; the DC agreement itself is gated in unit tests.
  H2 reads exactly zero (odd symmetry); all fixture solves converged.
- Triode small-signal gain is unity (-0.004 dB at -36 dBFS input) and
  THD rises monotonically with drive. The +24 dB row needed 400
  two-phase retries (19,600 solves for 19,200 samples); all converged.
- Tone-stack noon 1 kHz reads +0.010 dB for both value sets (level
  reference), the noon scoop is 9-11 dB deep, and every knob sweep
  increases monotonically with the pre-registered range.

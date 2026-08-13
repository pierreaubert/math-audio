# Analog coloration baseline — 2026-08-13

This is the reproducible Phase 0 baseline captured from the existing SOTF
Saturation plugin. The Saturation DSP and its public modes were not changed by
the analog-color implementation.

Command, run from the SOTF repository with an isolated Cargo target:

```text
cargo run -p sotf-plugin-saturation --features qa --bin qa-saturation
```

Observed output on the current macOS benchmark machine:

```text
Saturation Asymmetric stereo 4x path: 1.22% CPU, zero callback allocations, DC L/R=0.000422/-0.000421
```

Latest rerun on the same machine during this implementation pass:

```text
Saturation Asymmetric stereo 4x path: 1.16% CPU, zero callback allocations, DC L/R=0.000422/-0.000421
```

The CPU figure is timing-sensitive; the stable evidence in both runs is the
zero-allocation assertion and the matching DC values.

The fixture is stereo, 48 kHz, 1,024-frame callbacks, an 8 kHz sine, the
Asymmetric mode, 10 dB drive, 4x host oversampling, 0.75 dynamic amount, and
0.5 mix. The reported CPU is the average of 1,000 callbacks; the allocation
assertion covers a warmed steady-state callback path.

This artifact is only the baseline that the existing QA binaries can prove. A
complete hardware-modeling baseline still needs a separately reproducible
high-rate-reference alias comparison and a level-matched listening capture.
No hardware target or measured reference unit was supplied by the plan.

## Extended Phase 0 report

The read-only baseline report command adds deterministic harmonic, two-tone
IMD, alias-bin, latency, and preset round-trip measurements without changing
the Saturation DSP:

```text
rtk cargo run -p sotf-plugin-saturation --features qa --bin qa-saturation-baseline --offline
```

Captured output on 2026-08-13:

```text
sample_rate_hz=48000
record_length=4800
mode=Asymmetric drive=10 tone=2 mix=1 output_gain_db=0
dynamic_amount=0 dc_blocker=false use_adaa=false
harmonic latency_samples=0 dc=0.008629830 h1=1.250706315 h2=0.015194566 h3=0.363833070 h4=0.010399882
imd f1=1.068327069 f2=0.352569491 2f1-f2=0.185123444 2f2-f1=0.027983695
alias factor=1 latency_samples=0 folded_18khz=0.394341290
alias factor=2 latency_samples=512 folded_18khz=0.004882309
alias factor=4 latency_samples=512 folded_18khz=0.007086035
preset_roundtrip=true
preset_json={"dc_blocker_enabled":true,"drive":2.0,"dynamic_amount":0.0,"dynamic_attack_ms":5.0,"dynamic_release_ms":50.0,"exciter_freq":3000.0,"mix":0.5,"mode":"Soft Clip","output_gain_db":0.0,"oversampling":"2x","tone":1.5,"use_adaa":true}
```

The 4x value is not lower than the 2x value for this particular direct
folded-bin fixture, so no monotonic quality claim is made for existing
Saturation. The Analog Color host fixture uses its own declared measurement
and ordering assertion. The report remains a baseline, not a formal alias
budget or hardware comparison.

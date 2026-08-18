# Console/Preamp model report

This report describes the synthetic `ConsolePreampModel` structure introduced
by the 2026-08-18 roadmap. It is not a measurement or a hardware emulation.

The prepared path is:

```text
coupling high-pass → asymmetric ADAA1 tanh → envelope compression → output low-pass → DC blocker
```

The model is selected by append-only `AnalogModel` ID 5. Its default trim is
explicit (`-1.5 dB`) and its controls are smoothed at the same 10 ms control
time as the existing model families. The pre/output filters are prepared with
`math-iir-fir::Biquad`; no processing callback allocation or topology change is
allowed.

The default coefficients are synthetic. A fitted model may replace them only
after a capture, fit/validation split, and provenance record have been supplied
through the feature-gated fitting module.

The level-matched comparison fixture uses the same 48 kHz, two-channel,
240,000-frame amplitude-modulated 1 kHz input for the H2/H3 `HarmonicModel`
baseline and Console/Preamp.  The console output is matched to baseline
integrated loudness with the explicit 0.95 sample-peak ceiling before the
harmonic comparison. Generate it with:

```text
rtk cargo run -p math-analog --example console_preamp_report --offline
```

Captured output on 2026-08-18:

```text
sample_rate_hz=48000 frames=240000 channels=2 fixture=synthetic_amplitude_modulated_1kHz
level_match baseline_lufs=-5.537642 console_lufs=-7.345974 requested_gain_db=1.808332 applied_gain_db=1.808332 peak_limited=false baseline_peak=0.660380 console_peak=0.478244
harmonics_level_matched baseline_h1=0.519943357 console_h1=0.524894834 baseline_h2=0.041733537 console_h2=0.008339898 baseline_h3=0.005668402 console_h3=0.017959217 baseline_h4=0.003054583 console_h4=0.002133388 baseline_h5=0.000576875 console_h5=0.000805707
```

These repeatable differences characterize the declared synthetic structures
after loudness matching; they do not establish measured-device accuracy or a
perceptual preference.

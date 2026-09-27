# Synthetic analog model-family report

This artifact characterizes every serialized `AnalogModel` family with the
same deterministic fixture. It is implementation evidence for finite,
bounded, spectral, IMD, and transient behavior; it is not a hardware fit,
listening result, or claim that one model is perceptually better.

Generate it with:

```text
rtk cargo run -p math-analog --example model_matrix_report --offline
```

The fixture uses 48 kHz, 4,800-frame coherent 1 kHz and 1.5 kHz tones,
rectangular-window one-sided amplitude measurements, 12 dB drive, full amount
and mix, neutral character, and the Harmonics model's H2/H3 controls at -18
and -24 dB. Memoryless models use direct evaluation for this report; the
host-owned oversampling comparison is recorded separately.

Captured output on 2026-09-27 (recapture: rows 0-5 match the 2026-08-18
capture to <=1e-4 relative except Console/Preamp H2/IMD2 (<=0.4% relative,
-90 dBFS absolute) — cross-machine float noise amplified by that model's
feedback loop; no Console code path changed. Rows 6-8 are new):

```text
sample_rate_hz=48000 record_length=4800
fixture=synthetic coherent sine/two-tone, rectangular one-sided amplitude
columns=model id finite h1 h2 h3 thd thd_plus_n imd_2f1_minus_f2 imd_2f2_minus_f1 transient_peak transient_rms dc
model=Harmonics id=0 true 1.137981415 0.102202408 0.147058055 0.157370687 0.162522361 0.061812267 0.057784081 1.091821313 0.820843637 0.059277922
model=Static id=1 true 1.117642045 0.000038093 0.187800020 0.168032333 0.172625765 0.061162140 0.024360286 0.987124681 0.802990854 0.031855982
model=Hammerstein id=2 true 1.039624572 0.048731379 0.149347141 0.151108891 0.153765708 0.040519789 0.034574781 0.951968312 0.744808853 0.043165646
model=Tape-style id=3 true 0.888116360 0.000078372 0.184627280 0.207886383 0.219071254 0.072085373 0.023100263 0.765557349 0.642497599 0.024026150
model=Transformer-style id=4 true 0.747137964 0.000130287 0.144815072 0.193826497 0.202239171 0.053461056 0.018572498 0.648914576 0.538789451 0.020563077
model=Console/Preamp-style id=5 true 0.734383881 0.009536693 0.122395709 0.167169616 0.178811267 0.042062923 0.017365443 0.818331242 0.622231245 0.029348282
model=DiodeClipper id=6 true 0.632150114 0.000082134 0.157210574 0.248691887 0.279546589 0.080382012 0.011635053 0.536344945 0.463783085 0.017303348
model=TriodeStage id=7 true 1.884234428 0.152898505 0.037442796 0.083543941 0.084357664 0.038805660 0.038504727 2.061194658 1.348879337 -0.122546054
model=ToneStack id=8 true 0.500896096 0.000245954 0.000167162 0.000593701 0.031819921 0.000612854 0.000198395 0.625427186 0.363926351 0.027761543
```

The rows are directly comparable only for this declared synthetic setup.
They do not establish hardware accuracy, a formal CPU or alias budget, or a
held-out target-model advantage.

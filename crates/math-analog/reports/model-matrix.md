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

Captured output on 2026-08-18:

```text
sample_rate_hz=48000 record_length=4800
fixture=synthetic coherent sine/two-tone, rectangular one-sided amplitude
columns=model id finite h1 h2 h3 thd thd_plus_n imd_2f1_minus_f2 imd_2f2_minus_f1 transient_peak transient_rms dc
model=Harmonics id=0 true 1.137981415 0.102202408 0.147058070 0.157370687 0.162522092 0.061812267 0.057784081 1.091821313 0.820843637 0.059277922
model=Static id=1 true 1.117642045 0.000038090 0.187800020 0.168032333 0.172625765 0.061162122 0.024360294 0.987124681 0.802990854 0.031855978
model=Hammerstein id=2 true 1.039624572 0.048731379 0.149347141 0.151108891 0.153765708 0.040519789 0.034574781 0.951968312 0.744808853 0.043165646
model=Tape-style id=3 true 0.888116479 0.000078379 0.184627295 0.207886353 0.219070897 0.072085358 0.023100259 0.765557289 0.642497599 0.024026169
model=Transformer-style id=4 true 0.747137964 0.000130280 0.144815072 0.193826497 0.202238649 0.053461056 0.018572504 0.648914635 0.538789451 0.020563077
model=Console/Preamp-style id=5 true 0.734385550 0.009504283 0.122397013 0.167167589 0.178810552 0.042063780 0.017374111 0.818331242 0.622231245 0.029348241
```

The rows are directly comparable only for this declared synthetic setup.
They do not establish hardware accuracy, a formal CPU or alias budget, or a
held-out target-model advantage.

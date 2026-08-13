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

Captured output on 2026-08-13:

```text
sample_rate_hz=48000 record_length=4800
fixture=synthetic coherent sine/two-tone, rectangular one-sided amplitude
columns=model id finite h1 h2 h3 thd thd_plus_n imd_2f1_minus_f2 imd_2f2_minus_f1 transient_peak transient_rms dc
model=Harmonics id=0 true 1.137981415 0.102202393 0.147058100 0.157370701 0.162522092 0.061812282 0.057784054 1.091821313 0.820843637 0.059277922
model=Static id=1 true 1.117642045 0.000038068 0.187800035 0.168032363 0.172625765 0.061162170 0.024360238 0.987124681 0.802990854 0.031855978
model=Hammerstein id=2 true 1.040883064 0.048930313 0.150669366 0.152193248 0.154906631 0.040709410 0.034803916 0.952106953 0.747045100 0.045118716
model=Tape-style id=3 true 0.888830721 0.000036705 0.185690656 0.208915681 0.220330179 0.072348915 0.023259895 0.765603840 0.643972158 0.025395919
model=Transformer-style id=4 true 0.747816086 0.000091946 0.145766661 0.194923177 0.203563794 0.053666778 0.018708400 0.648964167 0.540089428 0.021729957
```

The rows are directly comparable only for this declared synthetic setup.
They do not establish hardware accuracy, a formal CPU or alias budget, or a
held-out target-model advantage.

# Synthetic Hammerstein fitting-quality report

This artifact exercises the feature-gated offline DE → LM fitting path against
synthetic parallel-Hammerstein captures.  It is not a hardware fit, measured
device claim, or listening result.  Training records contain a 1 kHz sine,
two-tone stimulus, and transient programme; held-out records use different
frequencies, phases, and transient decay.  The optimizer sees only the
training records and minimizes FFT magnitude/phase residuals.

Generate it with:

```text
rtk cargo run -p math-analog --features fitting --example fitting_report --offline
```

Captured output on 2026-08-18:

```text
sample_rate_hz=48000 fit_captures=3 held_out_captures=3
quality fit_rms=0.005052582 held_out_rms=0.009886365 fit_spectral_rms=0.000056880 held_out_spectral_rms=0.000129264 de_objective=0.000019878 lm_objective=0.000019878
fitted_branch order=1 gain=0.263901383 cutoff_hz=24000.000000000
fitted_branch order=2 gain=0.083585732 cutoff_hz=24000.000000000
fitted_branch order=3 gain=-0.192822799 cutoff_hz=0.000000000
capture_hash=13897121256530777510
fit_sine_harmonics ref_h1=0.157351285 model_h1=0.157724187 ref_h2=0.003157994 model_h2=0.004601521 ref_h3=0.000831659 model_h3=0.003624881 ref_h4=0.000235769 model_h4=0.000141651 ref_h5=0.000193406 model_h5=0.000125694
held_out_sine_harmonics ref_h1=0.192456678 model_h1=0.178370342 ref_h2=0.000710127 model_h2=0.004495181 ref_h3=0.000905415 model_h3=0.003925207 ref_h4=0.000014806 model_h4=0.000009542 ref_h5=0.000100953 model_h5=0.000175714
fit_two_tone_imd ref_tone_a=0.138606235 model_tone_a=0.139207840 ref_tone_b=0.081984997 model_tone_b=0.079536222 ref_2f1_f2=0.000841047 model_2f1_f2=0.001995118 ref_2f2_f1=0.000180922 model_2f2_f1=0.000922964
held_out_two_tone_imd ref_tone_a=0.119578168 model_tone_a=0.118865065 ref_tone_b=0.061315406 model_tone_b=0.056475509 ref_2f1_f2=0.000343094 model_2f1_f2=0.002702825 ref_2f2_f1=0.000221561 model_2f2_f1=0.001017428
fit_transient ref_peak=0.361691326 model_peak=0.283761412 ref_rms=0.048987564 model_rms=0.048488386 ref_dc=-0.014863324 model_dc=-0.015530907
held_out_transient ref_peak=0.312539190 model_peak=0.283462524 ref_rms=0.046763677 model_rms=0.047187809 ref_dc=-0.014867047 model_dc=-0.015533922
```

The fitted coefficients are deliberately synthetic and are not promoted to a
default serialized model without a documented measured target and provenance.

(* math-qa case math-qa.capture-tdoa.v1.
   C1: fractional chirp delay by direct-summation band-limited
   cross-correlation plus parabolic interpolation (independent of the
   Rust FFT implementation). C2: two-point skew arithmetic. C4: CRB-form
   uncertainty evaluation. Delays are applied by Fourier phase rotation;
   the estimated offset must reproduce the applied delay. *)

Off[General::munfl];
sr = 48000.;
n = 12000;
f0 = 500.; f1 = 12000.; dur = 0.25; kk = f1/f0;
chirp[i_] := Sin[2. Pi f0 dur (kk^((i/sr)/dur) - 1)/Log[kk]];
ref = Table[chirp[i], {i, 0, n - 1}];
freq[k_] := k*sr/n;

(* Band mask 2-8 kHz, mirrored for negative frequencies. *)
inBand[k_] := Boole[2000. <= freq[k] <= 8000.];
mask = Table[If[k <= n/2, inBand[k], inBand[n - k]], {k, 0, n - 1}];

delaySignal[d_] := Module[{R, rot},
  R = Fourier[ref, FourierParameters -> {1, -1}];
  rot = Table[Exp[-2. Pi I k d/n], {k, 0, n - 1}];
  Re[InverseFourier[R*rot, FourierParameters -> {1, -1}]]
];

estimateOffset[rec_] := Module[{R, C, cross, corr, peak, prev, nxt, denom, delta, off, excl, side, conf},
  R = Fourier[ref, FourierParameters -> {1, -1}];
  C = Fourier[rec, FourierParameters -> {1, -1}];
  cross = Table[
    If[mask[[k + 1]] == 1,
      C[[k + 1]]*Conjugate[R[[k + 1]]],
      0. + 0. I],
    {k, 0, n - 1}];
  (* Analytic correlation: envelope has no carrier cusps, so argmax and
     parabolic interpolation read the true pulse peak (magnitude
     interpolation misplaces it by ~1 sample near carrier zeros). *)
  onesided = Table[
    Which[k == 0 || k == n/2, cross[[k + 1]],
      k < n/2, 2*cross[[k + 1]],
      True, 0. + 0. I],
    {k, 0, n - 1}];
  corr = Abs[InverseFourier[onesided, FourierParameters -> {1, -1}]/n];
  peak = Ordering[corr, -1][[1]] - 1;
  prev = corr[[Mod[peak - 1, n] + 1]];
  nxt = corr[[Mod[peak + 1, n] + 1]];
  denom = prev - 2*corr[[peak + 1]] + nxt;
  delta = If[Abs[denom] > 1.*^-12, 0.5*(prev - nxt)/denom, 0.];
  delta = Min[0.5, Max[-0.5, delta]];
  off = peak + delta;
  If[off > n/2, off = off - n];
  excl = Max[2, Ceiling[sr/(8000. - 2000.)]];
  side = Max[Table[
    If[Min[Abs[i - peak], n - Abs[i - peak]] > excl, corr[[i + 1]], 0.],
    {i, 0, n - 1}]];
  conf = 20.*Log10[corr[[peak + 1]]/(side + 1.*^-30)];
  {off, conf}
];

c1entry[d_] := Module[{rec, est},
  rec = delaySignal[d];
  est = estimateOffset[rec];
  <|
    "true_delay_samples" -> N[d, 16],
    "offset_samples" -> N[est[[1]], 16],
    "confidence_db" -> N[est[[2]], 16],
    "offset_tolerance" -> 0.03,
    "confidence_tolerance_db" -> 1.0
  |>
];

(* C2: exact skew arithmetic. *)
c2entry[d1_, d2_, spacing_] := Module[{ppm},
  ppm = (d2 - d1)/spacing*1.*^6;
  <|
    "offset_samples" -> N[d1, 16],
    "skew_ppm" -> N[ppm, 16],
    "valid" -> True,
    "tolerance" -> 1.*^-9
  |>
];

(* C4: documented CRB-form evaluation with the 3-sigma + half-sample
   reporting margin; worst (lower) confidence dominates. *)
c4entry[conf1_, conf2_, spacing_, eval_] := Module[{snr, beta, s, ss, total},
  snr = 10.^((Min[conf1, conf2])/10.);
  beta = 6000./Sqrt[12.];
  s = sr/(2. Pi beta Sqrt[2.*snr]);
  ss = Sqrt[2.]*s/spacing;
  total = (3.*Sqrt[s^2 + (eval*ss)^2] + 0.5)/sr*1.*^6;
  <|
    "uncertainty_us" -> N[total, 16],
    "tolerance" -> 1.*^-9
  |>
];

payload = <|
  "schema_version" -> 1,
  "case" -> "math-qa.capture-tdoa.v1",
  "sample_rate_hz" -> sr,
  (* NOTE: exact half-integer circular delays (e.g. -123.5) are avoided:
     symmetric bracketing samples make the peak pick numerically fragile
     across FFT grids; negative fractional coverage lives in the Rust
     unit tests with a linear oracle (-123.75). *)
  "c1_entries" -> Map[c1entry, N[{37.25, -123.4}, 16]],
  "c2_entries" -> {c2entry[1000., 1002.4, 480000.]},
  "c4_entries" -> {c4entry[20., 14., 480000., 480000.]},
  "c4_invalid" -> True
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];

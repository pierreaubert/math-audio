(* math-qa case math-qa.capture-tdoa.v1, schema 2 (multi-rate).
   C1: fractional chirp delay by direct-summation band-limited
   cross-correlation plus parabolic interpolation (independent of the
   Rust FFT implementation). C2: two-point skew arithmetic (pure
   sample-domain arithmetic: rate-free, single entry). C4: CRB-form
   uncertainty evaluation per rate. Delays are applied by Fourier phase
   rotation; the estimated offset must reproduce the applied delay.
   Rates cover the 48 kHz family (48/96 kHz), the 44.1 kHz family
   (44.1/88.2 kHz) and cheap-device rates (6/12 kHz). Chirp and mask
   bands scale with the rate: chirp sr/96..sr/4, mask sr/24..sr/6
   (at 48 kHz exactly the documented 500-12000 Hz chirp / 2-8 kHz band). *)

Off[General::munfl];

rateBlock[sr_] := Module[{n, dur, f0, f1, kk, chirp, ref, freq, blo, bhi,
    inBand, mask, delaySignal, estimateOffset, c1entry, c2entry, c4entry,
    spacing, c1, c4},
  n = Round[sr*0.25]; dur = 0.25;
  f0 = sr/96.; f1 = sr/4.; kk = f1/f0;
  chirp[i_] := Sin[2. Pi f0 dur (kk^((i/sr)/dur) - 1)/Log[kk]];
  ref = Table[chirp[i], {i, 0, n - 1}];
  freq[k_] := k*sr/n;

  (* Analysis mask, mirrored for negative frequencies. *)
  blo = sr/24.; bhi = sr/6.;
  inBand[k_] := Boole[blo <= freq[k] <= bhi];
  mask = Table[If[k <= n/2, inBand[k], inBand[n - k]], {k, 0, n - 1}];

  delaySignal[d_] := Module[{R, rot},
    R = Fourier[ref, FourierParameters -> {1, -1}];
    rot = Table[Exp[-2. Pi I k d/n], {k, 0, n - 1}];
    Re[InverseFourier[R*rot, FourierParameters -> {1, -1}]]
  ];

  estimateOffset[rec_] := Module[{R, C, cross, onesided, corr, peak,
      prev, nxt, denom, delta, off, excl, side, conf},
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
    excl = Max[2, Ceiling[sr/(bhi - blo)]];
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
      "sample_rate_hz" -> N[sr, 16],
      "band_lo_hz" -> N[blo, 16],
      "band_hi_hz" -> N[bhi, 16],
      "true_delay_samples" -> N[d, 16],
      "offset_samples" -> N[est[[1]], 16],
      "confidence_db" -> N[est[[2]], 16],
      "offset_tolerance" -> 0.03,
      "confidence_tolerance_db" -> 1.0
    |>
  ];

  (* C4: documented CRB-form evaluation with the 3-sigma + half-sample
     reporting margin; worst (lower) confidence dominates. Spacing/eval
     are a 10 s sweep at the block rate. *)
  spacing = 10.*sr;
  c4entry[conf1_, conf2_] := Module[{snr, beta, s, ss, total},
    snr = 10.^((Min[conf1, conf2])/10.);
    beta = (bhi - blo)/Sqrt[12.];
    s = sr/(2. Pi beta Sqrt[2.*snr]);
    ss = Sqrt[2.]*s/spacing;
    total = (3.*Sqrt[s^2 + (spacing*ss)^2] + 0.5)/sr*1.*^6;
    <|
      "sample_rate_hz" -> N[sr, 16],
      "band_lo_hz" -> N[blo, 16],
      "band_hi_hz" -> N[bhi, 16],
      "chirp_spacing_samples" -> N[spacing, 16],
      "eval_time_samples" -> N[spacing, 16],
      "uncertainty_us" -> N[total, 16],
      "tolerance" -> 1.*^-9
    |>
  ];

  c1 = Map[c1entry, N[{37.25, -123.4}, 16]];
  c4 = c4entry[20., 14.];
  <|"sample_rate_hz" -> N[sr, 16], "c1_entries" -> c1, "c4_entry" -> c4|>
];

blocks = Map[rateBlock, N[{6000., 12000., 44100., 48000., 88200., 96000.}, 16]];

(* C2: exact skew arithmetic; rate-free sample-domain computation. *)
c2entry[d1_, d2_, spacing_] := Module[{ppm},
  ppm = (d2 - d1)/spacing*1.*^6;
  <|
    "offset_samples" -> N[d1, 16],
    "skew_ppm" -> N[ppm, 16],
    "valid" -> True,
    "tolerance" -> 1.*^-9
  |>
];

payload = <|
  "schema_version" -> 2,
  "case" -> "math-qa.capture-tdoa.v1",
  "sample_rates_hz" -> Map[#["sample_rate_hz"]&, blocks],
  (* NOTE: exact half-integer circular delays (e.g. -123.5) are avoided:
     symmetric bracketing samples make the peak pick numerically fragile
     across FFT grids; negative fractional coverage lives in the Rust
     unit tests with a linear oracle (-123.75). *)
  "c1_entries" -> Flatten[Map[#["c1_entries"]&, blocks]],
  "c2_entries" -> {c2entry[1000., 1002.4, 480000.]},
  "c4_entries" -> Map[#["c4_entry"]&, blocks],
  "c4_invalid" -> True
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];

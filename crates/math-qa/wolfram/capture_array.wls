(* math-qa case math-qa.capture-array.v1, schema 2 (multi-rate).
   C5: least-squares DOA from exact free-field pair delays via LinearSolve
   (independent of the Rust Cramer solver). C6: geometry residuals and
   scale calibration. C3: direct textbook windowed-sinc interpolation of a
   closed-form two-tone signal at fractional positions (independent of the
   Rust polyphase table: Hann/129-tap/direct vs Kaiser/65-tap/table).
   Rates cover the 48 kHz family (48/96 kHz), the 44.1 kHz family
   (44.1/88.2 kHz) and cheap-device rates (6/12 kHz). C3 tone frequencies
   scale with the rate (440/2900 Hz at 48 kHz) so every rate exercises the
   same relative band. *)

Off[General::munfl];
c = 343.;

rateBlock[sr_] := Module[{positions, dirOf, predDelay, doaEntry, c6block,
    micLen, fa, fb, tone, sincHannInterp, c3entry},
  positions = {{0., 0., 0.}, {0.12, 0., 0.}, {0., 0.12, 0.}, {0., 0., 0.12}};
  dirOf[azDeg_, elDeg_] := Module[{az = azDeg Degree, el = elDeg Degree},
    {Cos[el] Cos[az], Cos[el] Sin[az], Sin[el]}];
  predDelay[pa_, pb_, dir_] := -((pb - pa).dir)/c*sr;

  doaEntry[azDeg_, elDeg_] := Module[{dir, A, d, u, nrm, est, az, el, resid},
    dir = dirOf[azDeg, elDeg];
    A = Table[-(positions[[b]] - positions[[a]])*sr/c, {a, 1, 3}, {b, a + 1, 4}] ~Flatten~ 1;
    d = Table[predDelay[positions[[a]], positions[[b]], dir], {a, 1, 3}, {b, a + 1, 4}] ~Flatten~ 1;
    u = LinearSolve[Transpose[A].A, Transpose[A].d];
    nrm = Norm[u];
    est = u/nrm;
    az = ArcTan[est[[1]], est[[2]]]/Degree;
    el = ArcSin[Clip[est[[3]], {-1., 1.}]]/Degree;
    resid = Sqrt[Mean[(d - A.est)^2]];
    <|
      "sample_rate_hz" -> N[sr, 16],
      "azimuth_deg" -> N[azDeg, 16],
      "elevation_deg" -> N[elDeg, 16],
      "est_azimuth_deg" -> N[az, 16],
      "est_elevation_deg" -> N[el, 16],
      "rms_residual_samples" -> N[resid, 16],
      "angle_tolerance_deg" -> 0.05,
      "residual_tolerance" -> 1.*^-9
    |>
  ];

  (* C6: residuals against declared geometry + scale calibration. Scale the
     array by 1.03 (tape error) and recover the factor from exact delays. *)
  c6block[] := Module[{dir, big, pred, pairs, resid, num, den, scale, rms},
    dir = dirOf[30., 10.];
    big = 1.03*positions;
    pred = Table[predDelay[positions[[a]], positions[[b]], dir], {a, 1, 3}, {b, a + 1, 4}] ~Flatten~ 1;
    pairs = Table[predDelay[big[[a]], big[[b]], dir], {a, 1, 3}, {b, a + 1, 4}] ~Flatten~ 1;
    resid = pairs - pred;
    num = pred.pairs; den = pred.pred;
    scale = num/den;
    rms = Sqrt[Mean[(scale*pred - pairs)^2]];
    <|
      "sample_rate_hz" -> N[sr, 16],
      "max_residual_vs_true_samples" -> N[Max[Abs[resid]], 16],
      "scale_factor" -> N[scale, 16],
      "scale_rms_samples" -> N[rms, 16],
      "scale_tolerance" -> 1.*^-9
    |>
  ];

  (* ---- C3 resampler ---- *)
  micLen = 2000;
  fa = 440.*sr/48000.; fb = 2900.*sr/48000.;
  tone[n_] := Sin[2. Pi fa n/sr] + 0.5*Sin[2. Pi fb n/sr + 1.];
  sincHannInterp[s_, half_] := Module[{i0, frac, acc, gain},
    i0 = Floor[s]; frac = s - i0; acc = 0.; gain = 0.;
    Do[
      Module[{x = k - frac, u = N[k/half], h},
        h = If[Abs[x] < 1.*^-12, 1., Sin[Pi x]/(Pi x)]*0.5*(1. + Cos[Pi u]);
        gain += h;
        If[0 <= i0 + k < micLen, acc += h*tone[i0 + k]]],
      {k, -half, half}];
    acc/gain
  ];

  c3entry[m_, frac_] := Module[{s = m + frac, v},
    v = sincHannInterp[N[s, 16], 64];
    <|
      "sample_rate_hz" -> N[sr, 16],
      "tone_lo_hz" -> N[fa, 16],
      "tone_hi_hz" -> N[fb, 16],
      "index" -> m,
      "offset_samples" -> N[frac, 16],
      "value" -> N[v, 16],
      "tolerance_abs" -> 3.*^-3
    |>
  ];

  <|
    "sample_rate_hz" -> N[sr, 16],
    "c5_entries" -> {doaEntry[30., 10.], doaEntry[-45., 25.]},
    "c6" -> c6block[],
    "c3_entries" -> {c3entry[100, 0.25], c3entry[500, 0.5], c3entry[1000, 0.75]}
  |>
];

blocks = Map[rateBlock, N[{6000., 12000., 44100., 48000., 88200., 96000.}, 16]];

payload = <|
  "schema_version" -> 2,
  "case" -> "math-qa.capture-array.v1",
  "sample_rates_hz" -> Map[#["sample_rate_hz"]&, blocks],
  "sound_speed_m_s" -> c,
  "c5_entries" -> Flatten[Map[#["c5_entries"]&, blocks]],
  "c6_entries" -> Map[#["c6"]&, blocks],
  "c3_entries" -> Flatten[Map[#["c3_entries"]&, blocks]]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];

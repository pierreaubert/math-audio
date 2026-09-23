(* math-qa case math-qa.capture-array.v1.
   C5: least-squares DOA from exact free-field pair delays via LinearSolve
   (independent of the Rust Cramer solver). C6: geometry residuals and
   scale calibration. C3: direct textbook windowed-sinc interpolation of a
   closed-form two-tone signal at fractional positions (independent of the
   Rust polyphase table: Hann/129-tap/direct vs Kaiser/65-tap/table). *)

Off[General::munfl];
sr = 48000.;
c = 343.;

(* ---- C5/C6 geometry ---- *)
positions = {{0., 0., 0.}, {0.12, 0., 0.}, {0., 0.12, 0.}, {0., 0., 0.12}};
dirOf[azDeg_, elDeg_] := Module[{az = azDeg Degree, el = elDeg Degree},
  {Cos[el] Cos[az], Cos[el] Sin[az], Sin[el]}];
predDelay[pa_, pb_, dir_] := -((pb - pa).dir)/c*sr;

doaEntry[azDeg_, elDeg_] := Module[{dir, pairs, A, d, u, nrm, est, az, el, resid},
  dir = dirOf[azDeg, elDeg];
  pairs = Flatten[Table[{a, b}, {a, 1, 3}, {b, a + 1, 4}], 1];
  A = Table[-(positions[[b]] - positions[[a]])*sr/c, {a, 1, 3}, {b, a + 1, 4}] ~Flatten~ 1;
  d = Table[predDelay[positions[[a]], positions[[b]], dir], {a, 1, 3}, {b, a + 1, 4}] ~Flatten~ 1;
  u = LinearSolve[Transpose[A].A, Transpose[A].d];
  nrm = Norm[u];
  est = u/nrm;
  az = ArcTan[est[[1]], est[[2]]]/Degree;
  el = ArcSin[Clip[est[[3]], {-1., 1.}]]/Degree;
  resid = Sqrt[Mean[(d - A.est)^2]];
  <|
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
c6block[] := Module[{dir, big, pairs, pred, resid, num, den, scale, rms},
  dir = dirOf[30., 10.];
  big = 1.03*positions;
  pairs = Flatten[Table[{a, b}, {a, 1, 3}, {b, a + 1, 4}], 1];
  pred = Table[predDelay[positions[[a]], positions[[b]], dir], {a, 1, 3}, {b, a + 1, 4}] ~Flatten~ 1;
  pairs = Table[predDelay[big[[a]], big[[b]], dir], {a, 1, 3}, {b, a + 1, 4}] ~Flatten~ 1;
  resid = pairs - pred;
  num = pred.pairs; den = pred.pred;
  scale = num/den;
  rms = Sqrt[Mean[(scale*pred - pairs)^2]];
  <|
    "max_residual_vs_true_samples" -> N[Max[Abs[resid]], 16],
    "scale_factor" -> N[scale, 16],
    "scale_rms_samples" -> N[rms, 16],
    "scale_tolerance" -> 1.*^-9
  |>
];

(* ---- C3 resampler ---- *)
tone[n_] := Sin[2. Pi 440. n/sr] + 0.5*Sin[2. Pi 2900. n/sr + 1.];
sincHannInterp[s_, half_] := Module[{i0, frac, acc, gain},
  i0 = Floor[s]; frac = s - i0; acc = 0.; gain = 0.;
  Do[
    Module[{x = k - frac, u = N[k/half], h},
      h = If[Abs[x] < 1.*^-12, 1., Sin[Pi x]/(Pi x)]*0.5*(1. + Cos[Pi u]);
      gain += h;
      If[0 <= i0 + k < 2000, acc += h*tone[i0 + k]]],
    {k, -half, half}];
  acc/gain
];

c3entry[m_, frac_] := Module[{s = m + frac, v},
  v = sincHannInterp[N[s, 16], 64];
  <|
    "index" -> m,
    "offset_samples" -> N[frac, 16],
    "value" -> N[v, 16],
    "tolerance_abs" -> 3.*^-3
  |>
];

payload = <|
  "schema_version" -> 1,
  "case" -> "math-qa.capture-array.v1",
  "sample_rate_hz" -> sr,
  "sound_speed_m_s" -> c,
  "c5_entries" -> {doaEntry[30., 10.], doaEntry[-45., 25.]},
  "c6" -> c6block[],
  "c3_entries" -> {c3entry[100, 0.25], c3entry[500, 0.5], c3entry[1000, 0.75]}
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];

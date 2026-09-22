(* math-qa case math-qa.schroeder-t60.v1.
   Exact Schroeder decay of a noise-free exponential IR
   h[n] = Exp[-alpha n/sr] with alpha chosen for T60 = 0.5 s.
   Backward cumulative energy is exactly exponential, so the -5..-35 dB
   least-squares fit recovers the planted T60 up to sampling. Catches
   Schroeder-integration, anchoring, and fit-range slips. *)

Off[General::munfl];
sr = 48000.;
t60plant = 0.5;
dur = 2.0;
n = Round[dur sr];
alpha = N[3 Log[10.]/t60plant, 16];
h = Table[N[Exp[-alpha k/sr], 16], {k, 0, n - 1}];
energy = Reverse[Accumulate[Reverse[h^2]]];
total = energy[[1]];
curve = 10 Log10[energy/total];
cross[thr_] := First[First[Position[curve, v_ /; v <= thr]]];
i5 = cross[-5.];
i35 = cross[-35.];
window = curve[[i5 ;; i35]];
times = N[(Range[i5, i35] - 1)/sr, 16];
fit = LinearModelFit[Transpose[{times, window}], t, t];
slope = fit["ParameterTableEntries"][[2, 1]];
t60 = -60./slope;
probes = {0., 0.1, 0.25, 0.5};
probeVals = Table[
  curve[[Clip[Round[t sr] + 1, {1, n}]]],
  {t, probes}
];
payload = <|
  "schema_version" -> 1,
  "case" -> "math-qa.schroeder-t60.v1",
  "sample_rate_hz" -> sr,
  "planted_t60_s" -> t60plant,
  "fit_range_db" -> {-5., -35.},
  "t60_s" -> N[t60, 16],
  "slope_db_per_s" -> N[slope, 16],
  "probe_times_s" -> probes,
  "probe_db" -> N[probeVals, 16]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];

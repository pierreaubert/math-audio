(* math-qa case math-qa.comb-first-dip.v1.
   Pure two-path comb formulas for a direct sound plus one reflection of
   relative linear gain g after a delay dt: first dip at 1/(2 dt),
   peak-to-peak ripple 20 log10((1+g)/(1-g)), extra path c dt with
   c = 343 m/s. The Rust M1 primitive must reproduce these from picked
   peaks on unit-delta IRs. *)

Off[General::munfl];
c = 343.;
g = 0.5;
(* Pure comb formulas. At 2 ms the 1-8 kHz bandpass rings of the two
   deltas overlap, shifting the picked gain ~1e-3 off the formula, so
   that entry carries a wider tolerance. *)
dts = {0.002, 0.005, 0.010};
tols = {5.*^-3, 1.*^-5, 1.*^-5};
entry[dt_, tol_] := <|
  "delay_s" -> dt,
  "delay_ms" -> N[1000 dt, 16],
  "gain_db" -> N[20 Log10[g], 16],
  "path_difference_m" -> N[c dt, 16],
  "first_dip_hz" -> N[1/(2 dt), 16],
  "comb_ripple_db" -> N[20 Log10[(1 + g)/(1 - g)], 16],
  "tolerance" -> tol
|>;
payload = <|
  "schema_version" -> 1,
  "case" -> "math-qa.comb-first-dip.v1",
  "sound_speed_m_s" -> c,
  "relative_gain" -> g,
  "entries" -> MapThread[entry, {N[dts, 16], tols}]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];

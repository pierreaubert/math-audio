(* math-qa case math-qa.svf-response.v1.
   Bilinear-discretized analog lowpass prototype (the exact mapping the
   TPT state-variable filter implements): H(s) = 1/(sn^2 + k sn + 1)
   with sn = j tan(pi f/fs)/g, g = tan(pi fc/fs), k = 1/Q, evaluated at
   the QA frequencies. Catches coefficient, damping, and mixing slips in
   the Rust SVF response. *)

Off[General::munfl];
sr = 48000.;
fc = 1000.;
q = N[1/Sqrt[2], 16];
k = 1/q;
g = N[Tan[Pi fc/sr], 16];
freqs = {100., 500., 1000., 2000., 5000.};

lp[f_] := Module[{sn = I N[Tan[Pi f/sr], 16]/g},
  1/(sn^2 + k sn + 1)
];

complexPair[value_] := {N[Re[value], 16], N[Im[value], 16]};
payload = <|
  "schema_version" -> 1,
  "case" -> "math-qa.svf-response.v1",
  "sample_rate_hz" -> sr,
  "cutoff_hz" -> fc,
  "q" -> q,
  "mapping" -> "bilinear-tpt",
  "freqs_hz" -> freqs,
  "response_re_im" -> (complexPair[lp[#]] & /@ freqs)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];

(* math-qa case math-qa.biquad-response.v1.
   Independent evaluation of the RBJ cookbook lowpass design equations
   (the same published formulas the Rust Biquad constructor implements)
   at the exact QA parameters. Catches coefficient-transcription and
   response-evaluation slips, not the cookbook itself. *)

Off[General::munfl];
sr = 48000.;
fc = 1000.;
q = N[1/Sqrt[2], 16];
freqs = {20., 100., 500., 1000., 2000., 5000., 10000., 20000.};

w0 = N[2 Pi fc/sr, 16];
cw = Cos[w0];
alpha = Sin[w0]/(2 q);
b0 = (1 - cw)/2;
b1 = 1 - cw;
b2 = (1 - cw)/2;
a0 = 1 + alpha;
a1 = -2 cw;
a2 = 1 - alpha;

response[f_] := Module[{w = N[2 Pi f/sr, 16], z},
  z = Exp[-I w];
  (b0 + b1 z + b2 z^2)/(a0 + a1 z + a2 z^2)
];

complexPair[value_] := {N[Re[value], 16], N[Im[value], 16]};
payload = <|
  "schema_version" -> 1,
  "case" -> "math-qa.biquad-response.v1",
  "sample_rate_hz" -> sr,
  "cutoff_hz" -> fc,
  "q" -> q,
  "design" -> "RBJ-cookbook-lowpass",
  "freqs_hz" -> freqs,
  "response_re_im" -> (complexPair[response[#]] & /@ freqs)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];

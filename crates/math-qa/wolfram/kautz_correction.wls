(* math-qa case math-qa.kautz-correction.v1.
   Dry-plus-bank Kautz correction H(f) = 1 + Sum[g_k * B_k(f)] evaluated by
   direct z-domain summation (independent of the Rust recurrence): per
   section, A1 = -2 r Cos[theta], A2 = r^2 with r = Exp[-Pi f/(Q sr)] and
   theta = 2 Pi f/sr; basis B = Sqrt[1-r^2]*(1-r^2)/(1+A1/z+A2/z^2)*chain
   with z = Exp[I 2 Pi f/sr]; the chain accumulates the allpass
   (A2+A1/z+1/z^2)/(1+A1/z+A2/z^2) in section order. Three sample rates
   cover the 44.1 kHz, 48 kHz and 96 kHz acceptance clocks. *)

Off[General::munfl];

rateBlock[sr_, modes_, gains_] := Module[{pts, entry},
  pts = Union[
    Table[20.*(500./20.)^(i/23.), {i, 0, 23}],
    modes[[All, 1]]];
  entry[f_] := Module[{z, chain = 1. + 0. I, h = 0. + 0. I, r, th, a1, a2, den, b},
    z = Exp[I 2. Pi f/sr];
    Do[
      r = Exp[-Pi modes[[k, 1]]/(modes[[k, 2]]*sr)];
      th = 2. Pi modes[[k, 1]]/sr;
      a1 = -2.*r*Cos[th]; a2 = r*r;
      den = 1. + a1/z + a2/z^2;
      b = Sqrt[1. - r^2]*(1. - r^2)/den*chain;
      h += gains[[k]]*b;
      chain *= (a2 + a1/z + 1./z^2)/den,
      {k, Length[modes]}];
    h = 1. + h;
    <|
      "sample_rate_hz" -> N[sr, 16],
      "freq_hz" -> N[f, 16],
      "re" -> N[Re[h], 16],
      "im" -> N[Im[h], 16],
      "tolerance" -> 1.*^-9
    |>
  ];
  <|
    "sample_rate_hz" -> N[sr, 16],
    "modes" -> Map[N[#, 16]&, modes, {2}],
    "gains" -> N[gains, 16],
    "points" -> Map[entry, N[pts, 16]]
  |>
];

blocks = {
  rateBlock[44100., {{80., 6.}, {120., 4.}}, {1.5, -1.}],
  rateBlock[48000., {{80., 6.}, {120., 4.}}, {1.5, -1.}],
  rateBlock[96000., {{80., 6.}, {120., 4.}}, {1.5, -1.}],
  rateBlock[48000., {{100., 8.}}, {2.}]
};

payload = <|
  "schema_version" -> 1,
  "case" -> "math-qa.kautz-correction.v1",
  "sample_rates_hz" -> Map[#["sample_rate_hz"]&, blocks],
  "blocks" -> blocks
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];

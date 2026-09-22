(* math-qa case math-qa.morlet-spots.v1.
   Direct Morlet-correlation summation (no FFT) for a 1 kHz tone burst,
   using the exact wavelet convention from the Rust side: 3 sine cycles
   inside +/-2 sigma, unit-energy kernels, sine-at-centre reads 1.0.
   Catches kernel-shape, normalization, and calibration slips. *)

Off[General::munfl];
sr = 48000.;
fb = 1000.;
burstStart = 0.1;
burstDur = 0.05;
dur = 0.6;
n = Round[dur sr];
x[k_] := Module[{t = N[k/sr, 16]},
  If[Round[burstStart sr] <= k < Round[(burstStart + burstDur) sr],
    N[Sin[2 Pi fb t], 16],
    0.
  ]
];

cycles = 3.;
kernel[fc_] := Module[{sigma, half},
  sigma = N[cycles/(2 Pi fc) sr, 16];
  half = Ceiling[2 sigma];
  Table[
    Module[{g = N[Exp[-0.5 (m/sigma)^2], 16], ph = N[2 Pi fc m/sr, 16]},
      {m, g Cos[ph], g Sin[ph]}
    ],
    {m, -half, half}
  ]
];

valueAt[fc_, center_] := Module[{ker = kernel[fc], acc = 0. + 0. I, idx, en, norm, cal},
  en = Total[ker[[All, 2]]^2 + ker[[All, 3]]^2];
  norm = Sqrt[en];
  cal = Total[Sqrt[ker[[All, 2]]^2 + ker[[All, 3]]^2]]/(2 norm);
  Do[
    idx = center + ker[[j, 1]];
    If[0 <= idx < n,
      acc += (ker[[j, 2]] - I ker[[j, 3]]) x[idx]
    ],
    {j, Length[ker]}
  ];
  N[Abs[acc]/(norm cal), 16]
];

perOct = 12.;
fmin = 50.;
fmax = Min[16000., 0.9 sr/2];
grid = Table[fmin 2^(k/perOct), {k, 0, Ceiling[perOct Log2[fmax/fmin]]}];
grid = Select[grid, # <= fmax &];
nearest[target_] := First[MinimalBy[grid, Abs[# - target] &]];
(* Rows stay above the heatmap's -30 dB display floor at the chosen
   times; deeper cells clamp on the Rust side and are out of scope. *)
rows = nearest /@ {500., 1000.};
timesMs = {99., 100., 125.};
centers = Round[# sr/1000] & /@ timesMs;
payload = <|
  "schema_version" -> 1,
  "case" -> "math-qa.morlet-spots.v1",
  "sample_rate_hz" -> sr,
  "burst_hz" -> fb,
  "cycles" -> cycles,
  "freqs_per_octave" -> perOct,
  "rows_hz" -> N[rows, 16],
  "times_ms" -> timesMs,
  "centers" -> centers,
  "magnitude" -> Outer[valueAt, rows, centers]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];

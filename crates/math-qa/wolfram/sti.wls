(* math-qa case math-qa.sti.v1.
   Closed-form IEC 60268-16:2020 references for indirect (Schroeder-MTF)
   STI. The Wolfram Language has no native STI primitive, so the oracle
   is the standard itself in closed form: analytic modulation transfer
   of an exponential energy decay, the Annex A apparent-SNR mapping with
   +/-15 dB rails, per-band MTI means, and the Table A.1 alpha/beta
   combination. The Rust side runs analyze_sti on synthetic IRs whose
   band-energy envelopes match the oracle assumptions up to octave-filter
   transients, carrier ripple, and truncation (all small; per-entry
   tolerances below). Catches MTF-normalization, TI-mapping, MTI-mean,
   and weighting-factor slips, plus typos in the IEC constants. *)

Off[General::munfl];
sr = 48000.;
modfreqs = {0.63, 0.8, 1., 1.25, 1.6, 2., 2.5, 3.15, 4., 5., 6.3, 8., 10., 12.5};
centers = {125., 250., 500., 1000., 2000., 4000., 8000.};
alpha = {0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173};
beta = {0.085, 0.078, 0.065, 0.011, 0.047, 0.095};

(* Annex A transmission index: apparent SNR clipped at +/-15 dB. *)
ti[m_] := If[m <= 0., 0., If[m >= 1., 1., (Clip[10. Log10[m/(1. - m)], {-15., 15.}] + 15.)/30.]];

(* Table A.1 combination of the seven band MTIs. *)
combine[mti_] := Clip[
  alpha . mti - Sum[beta[[k]] Sqrt[mti[[k]] mti[[k + 1]]], {k, 1, 6}],
  {0., 1.}
];

(* Analytic Schroeder MTF of e(t) = Exp[-a t]: a/Sqrt[a^2 + (2 Pi fm)^2].
   Energy decay rate a = 6 ln 10/RT60 for 60 dB energy decay in RT60. *)
analyticMtf[fm_, rt60_] := Module[{a = N[6. Log[10.]/rt60, 16]},
  1./Sqrt[1. + (2. Pi fm/a)^2.]
];

(* One entry: 14x7 MTF rows (modulation freq, then band), TI rows, MTI
   per band, and combined STI. Analytic MTF is band-independent. *)
entry[name_, rt60_, tolMtf_, tolSti_] := Module[{mtf, tiss, mval, mti, sti},
  mtf = If[rt60 === None,
    Table[1., {14}, {7}],
    Table[analyticMtf[fm, rt60], {fm, modfreqs}, {b, 7}]
  ];
  tiss = Map[ti, mtf, {2}];
  mval = Mean[Flatten[tiss]];
  mti = Table[mval, {7}];
  sti = combine[mti];
  <|
    "name" -> name,
    "rt60_s" -> If[rt60 === None, Null, N[rt60, 16]],
    "mtf" -> N[mtf, 16],
    "transmission_indices" -> N[tiss, 16],
    "mti" -> N[mti, 16],
    "sti" -> N[sti, 16],
    "tolerance_mtf" -> tolMtf,
    "tolerance_sti" -> tolSti
  |>
];

(* Absolute tolerances: ~2x the measured octave-filter-transient
   deviation (worst cell always 12.5 Hz x 125 Hz band). The chain tail
   (MTI/STI) averages and saturates the MTF deviation, hence tighter.
   Blessed values also recorded in validation-manifest.toml. *)
entries = {
  entry["impulse", None, 8.*^-2, 5.*^-3],
  entry["rt60_0.5", 0.5, 2.5*^-2, 5.*^-3],
  entry["rt60_1.0", 1., 1.*^-2, 2.5*^-3],
  entry["rt60_2.0", 2., 5.*^-3, 2.*^-3]
};

payload = <|
  "schema_version" -> 1,
  "case" -> "math-qa.sti.v1",
  "sample_rate_hz" -> sr,
  "modulation_frequencies_hz" -> N[modfreqs, 16],
  "octave_centers_hz" -> N[centers, 16],
  "alpha" -> N[alpha, 16],
  "beta" -> N[beta, 16],
  (* Rust IR construction contract for the exponential entries:
     carriers = Sum[Sin[2 Pi fc t]], amplitude envelope
     Exp[-3 ln 10 t/RT60], duration RT60*1.5 + 0.1 s. *)
  "ir_duration_s" -> <|"rt60_0.5" -> 0.85, "rt60_1.0" -> 1.6, "rt60_2.0" -> 3.1|>,
  "entries" -> entries
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];

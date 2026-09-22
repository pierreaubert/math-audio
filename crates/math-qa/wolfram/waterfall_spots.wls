(* math-qa case math-qa.waterfall-spots.v1.
   Direct windowed-DFT summation (no FFT) of a decaying 440 Hz sine at
   selected STFT frames/bins, using the exact waterfall convention from
   the Rust side: 32 ms symmetric Hann, 2 ms hop, frames centred from
   sample 0, next-pow-2 FFT length, raw unnormalized power. Catches
   windowing, framing, padding, and spectrum-scaling slips. *)

Off[General::munfl];
sr = 48000.;
f0 = 440.;
t60 = 0.6;
dur = 0.6;
n = Round[dur sr];
alpha = N[3 Log[10.]/t60, 16];
x[k_] := N[Exp[-alpha k/sr] Sin[2 Pi f0 k/sr], 16];

winMs = 32.;
hopMs = 2.;
wlen = Round[winMs sr/1000];
hop = Round[hopMs sr/1000];
nfft = 2^Ceiling[Log2[wlen]];
hann[m_] := N[0.5 (1 - Cos[2 Pi m/(wlen - 1)]), 16];

powerAt[center_, bin_] := Module[{s = 0. + 0. I, idx},
  Do[
    idx = center - Quotient[wlen, 2] + m;
    If[0 <= idx < n,
      s += x[idx] hann[m] Exp[-I 2 Pi bin m/nfft]
    ],
    {m, 0, wlen - 1}
  ];
  N[Abs[s]^2, 16]
];

(* Frame centres in samples from t0 = 0; bins for df = sr/nfft. *)
centers = {0, 15 hop, 50 hop};
bins = {8, 19, 40};
df = N[sr/nfft, 16];
payload = <|
  "schema_version" -> 1,
  "case" -> "math-qa.waterfall-spots.v1",
  "sample_rate_hz" -> sr,
  "mode_hz" -> f0,
  "mode_t60_s" -> t60,
  "window_ms" -> winMs,
  "hop_ms" -> hopMs,
  "n_fft" -> nfft,
  "frame_centres" -> centers,
  "frame_times_ms" -> N[1000. centers/sr, 16],
  "bins" -> bins,
  "bin_freqs_hz" -> N[df bins, 16],
  "power" -> Outer[powerAt, centers, bins]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];

(* math-qa case math-qa.fft-peak.v1.
   Unnormalized DFT ({1,-1} FourierParameters) of an exact integer-cycle
   1 kHz tone at 48 kHz. The peak bin reads exactly N/2; off-peak bins
   read ~0. Catches FFT normalization, windowing, and peak-calibration
   slips on the Rust side. *)

Off[General::munfl];
sr = 48000.;
f = 1000.;
n = 48000;
tone = Table[N[Sin[2 Pi f k/sr], 16], {k, 0, n - 1}];
spec = Fourier[tone, FourierParameters -> {1, -1}];
mags = Abs[spec];
peakBin = 1000 + 1;
payload = <|
  "schema_version" -> 1,
  "case" -> "math-qa.fft-peak.v1",
  "sample_rate_hz" -> sr,
  "tone_hz" -> f,
  "n_samples" -> n,
  "normalization" -> "unnormalized-forward",
  "peak_bin" -> 1000,
  "peak_mag" -> N[mags[[peakBin]], 16],
  "mirror_mag" -> N[mags[[n - 1000 + 1]], 16],
  "off_bins" -> {999, 1001, 2000},
  "off_mags" -> N[mags[[{999, 1001, 2000} + 1]], 16]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];

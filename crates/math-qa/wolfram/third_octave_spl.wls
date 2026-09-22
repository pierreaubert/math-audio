(* math-qa case math-qa.third-octave-spl.v1.
   Two steady sines (1 kHz at amplitude 1, 2 kHz at amplitude 1/2): the
   1 kHz third-octave band is the peak (0 dB after peak normalization)
   and the 2 kHz band reads exactly 20 log10(1/2). Hann sidelobes sit
   below -25 dB in every other band, which is the documented neighbor
   ceiling the Rust M2 primitive must respect. *)

Off[General::munfl];
payload = <|
  "schema_version" -> 1,
  "case" -> "math-qa.third-octave-spl.v1",
  "sample_rate_hz" -> 48000.,
  "tones_hz" -> {1000., 2000.},
  "tone_amps" -> {1., 0.5},
  "peak_band_hz" -> 1000.,
  "peak_band_db" -> 0.,
  "second_band_hz" -> 2000.,
  "second_band_db" -> N[20 Log10[1./2.], 16],
  "neighbor_ceiling_db" -> -25.
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];

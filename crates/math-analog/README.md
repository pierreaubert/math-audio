# math-analog

`math-analog` contains host-independent analog-style coloration models for
realtime audio. It owns model mathematics and per-channel state; it does not
own SOTF parameter metadata, plugin schemas, graph topology, or oversampling
wrappers.

## Current model boundary

The crate currently provides five truthful model families:

- `HarmonicModel`: a controlled H2/H3 baseline using
  `u = tanh(drive * x)` and
  `v = u + h2 * (2u² - 1) + h3 * (4u³ - 3u)`. H2/H3 are branch strengths
  relative to the fundamental, not independent spectral oscillators.
- `StaticColorModel`: bounded `tanh`, `x / (1 + |x|)`, and clamp curves.
  These are memoryless “style” curves, not tube, tape, or transformer models.
- `HammersteinModel`: at most five user-supplied Chebyshev branches followed
  by first-order output filters. It is a generic model until fitted
  coefficients and measurement provenance are supplied. `AnalogModel::from_id(2)`
  selects a documented synthetic generic coloration preset; it is not fitted
  hardware and has no measurement provenance, while `HammersteinModel::new()`
  remains the unity identity-branch constructor.
- Public memoryless curve helpers preserve the legacy normalized soft-clip,
  tube-style, tape-style, and asymmetric equations; the SOTF Saturation
  wrapper delegates to these helpers rather than duplicating the equations.
- `TapeModel` and `TransformerModel`: bounded stateful stylized targets with
  explicit slow memory/flux equations. Their names describe the target
  family, not a measured hardware claim.

The common `character` macro is neutral at `0.5`. For the memoryless and
Hammerstein families it applies a bounded pre-curve bias, making the mapping
explicit and continuously controllable; the stateful families map it to their
documented memory/flux equations.

Memoryless tanh and clipping branches reuse `math-dsp` ADAA processors where
available. Every model has checked preparation, deterministic reset, in-place
interleaved processing, sample-rate-aware control smoothing, finite-input
sanitization, a final DC blocker, and zero reported latency. Preparation is
the allocation boundary; processing does not grow buffers or change channel
topology.

The stateful targets do not include an internal resampler. A host integration
must own 2x/4x oversampling and report any wrapper latency exactly once.

The offline analysis helpers record sample rate, record length, bin spacing,
one-sided normalization, rectangular window, and coherent gain for harmonic
and two-tone intermodulation reports. They report out-of-band components at
their folded bins and mark them as aliases. A transient helper reports peak,
RMS, peak position, and DC. `downsample_reference` and
`compare_alias_reference` provide a separate 127-tap Blackman-windowed sinc
reference path for comparing base-rate renders with 2x/4x high-rate renders;
their result is an aliasing proxy and includes the declared reconstruction
filter transition band. None of these helpers uses zero-padding as evidence
of increased resolution.

`level_match_candidate` is an offline listening-test helper. It uses the
existing BS.1770/EBU R128 integrated loudness meter to compute a candidate gain
relative to a reference, then limits that gain by an explicit sample-peak
ceiling. It creates no preference or discrimination result; human listening
must still be blinded, randomized, and recorded separately.

The `model_matrix_report` example applies one declared synthetic fixture to
all five serialized model families and records harmonic, THD/THD+N, IMD,
transient, DC, and finite-output rows. It is characterization evidence, not a
hardware-validation or listening claim.

The verification surface includes a 44.1/48/96/192 kHz harmonic matrix at
-36/-24/-18/-12/-6/-1 dBFS and 50/100/1k/5k/near-Nyquist test frequencies,
H2-H10 in-band checks, direct-versus-ADAA alias reports, 1/2/6/12-channel
benchmarks, callback-partition tests, a worst-callback timing report, and an
allocation counter around warmed steady-state processing. The benchmark and
allocation setup are outside the processing contract; they never run on the
audio callback.

SOTF plugin integration is deliberately outside this crate. Hardware-specific
coefficients still require a measured reference, while the included stateful
models remain explicitly stylized targets.

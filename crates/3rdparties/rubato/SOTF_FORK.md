# SOTF Rubato 5.0.0 fork: ramp sizing and prepared cutoffs

Date: 2026-09-28 (rebased from the 1.0.1 fork the same day). Source:
github `HEnquist/rubato` tag `v5.0.0`, commit
`6b72d0f9d8843c6623c818751730764aefcd0525`. Upstream is dual-licensed
MIT OR Apache-2.0; attribution remains in `LICENSE-APACHE`,
`LICENSE-MIT` and `LICENSE.txt`. `Cargo.toml.orig` is the pristine
upstream manifest and `.cargo_vcs_info.json` retains the upstream
commit. This is the shared SOTF rubato fork: DAW users (resampler plugin,
host oversampler, convolution IR, binaural HRIR) and capture build against
this path dependency. The DAW workspace carries a single
`audioadapter-buffers` 5.x.

## What upstream v5 already provides

Upstream fixed the gross ramp-sizing defect (issue #136) in 5.0.0: size
estimates now average the inverse-ratio step sizes and correct for the
ramp overshoot, which matches this fork's analytic candidate. v5 also
provides the `avg_t_ratio` / `compute_t_ratio_increment` /
`advance_index` helpers, `Option` cutoffs with `cutoff()` reporting,
`Fft::new_custom` (explicit sub-chunks and window), the
`Adjustable` / `Resizable` trait split, the `Slip` clutch, and
`audioadapter` 5. The FFT users take `new_custom` with their
historical sub-chunk counts and `BlackmanHarris2` window, which is
bit-identical to v1 behavior (verified, see below); `Fft::new` would
auto-select sub-chunks and change delay and geometry.

## Exact trajectory and frame planning

Let `p0=last_index`, `h0=1/current_ratio`, `h1=1/target_ratio`, and `M`
be the output count for the next block. The kernel updates the inverse
ratio **before** producing each output:

```
delta = (h1-h0)/M
h_j = h0 + j*delta
p_j = p0 + j*h0 + (h1-h0)*j*(j+1)/(2*M),  j=1..M
p_M = p0 + M*(h0+h1)/2 + (h1-h0)/2
```

Upstream's closed form is used only as a seed. The fork replays the
scalar recurrence to check the candidate, decreasing or increasing it
at the boundary until the actual floating-point endpoint fits and the
next count does not. There is no epsilon or relaxed comparison. Stock
v5 omits this replay and over-counts by one frame on some ramped
blocks (8 of 42 production scenarios in the differential check);
the replay converges to the exact boundary-valid count from any seed.
Planning is O(M) scalar additions; no allocation, audio interpolation
or filter generation occurs during planning.

For fixed output `M`, required input is `ceil(actual_p_M + L + 1)`.
Upstream v5 kept `+L`, which can allow sinc interpolation's adjacent
table to inspect an unfilled input slot near the last fractional
phase. Constant-rate fixed-output block consumption can therefore
differ by one input frame at startup; the global raw anchor sequence
stays continuous because consumed input is subtracted from
`last_index`. The reported filter delay is unchanged.

Zero-output blocks use zero increment, retain the raw anchor before
subtracting consumed input, and advance the current ratio to the
target as upstream does. Construction rejects NaN/infinite/nonpositive
original ratios, nonfinite reciprocals, and invalid/nonfinite relative
bounds or their derived endpoints before scalar planning (v5.0.0
accepts NaN/inf ratios; upstream schedules that fix post-5.0).

## Shared position API

`InputPositions` is an owned, cloneable, fused exact-size iterator.
`Async::input_positions_next()` snapshots the planned block without
advancing state or allocating. Both sinc and polynomial kernels now
consume this same iterator, so an EOF consumer can reproduce the exact
floating-point positions without copying the recurrence. The iterator
funnels through upstream's `step_index` helper, which performs the
same floating-point operations in the same order as the previous
inline stepping, so snapshots and kernels cannot silently diverge.
`last_input_index()` exposes the anchor relative to the next block;
`input_history_frames()` reports retained history for support
diagnostics/tests.

The iterator's values are interpolation anchors, **not physical signal
centers, clock timestamps, or PDC delay**. For sinc they locate the
FIR window, so EOF accounting must independently derive its center
convention. A later ratio/chunk/reset change invalidates a previous
snapshot's prediction.

## Deferred history and maximum capacities

Stock v5 retains upstream's history sizing and panics on deep
deferred-input trajectories (`sinc_interpolator` index overflow,
reproduced 3/3 against v5.0.0). At nominal 8/384 and relative 0.5,
the maximum inverse step is 96 input frames. With a 64-tap sinc and
one-frame chunks, after 96 deferred-input blocks the anchor is -159.
An instantaneous change to relative 2 requests its next anchor at
-135. Upstream retained only 128 history frames, so that request
precedes the buffer.

Let `hmax=max_relative/original_ratio` and
`rmax=original_ratio*max_relative`. Before a block, deferred input
may place the anchor approximately one maximum step behind the
ordinary `-(L+1)` boundary. The fork retains:

```
H = 2*L + ceil(hmax) + 2
output_frames_max = ceil((max_chunk_size+hmax)*rmax) + 1
```

The extra retained frames cover a complete deferred step plus integer
endpoint/left-interpolation guards. The private inner-kernel call
accepts `H` explicitly, uses it as the raw buffer offset, and copies
the last `H` frames between blocks. The raw anchor, initial phase and
sinc dot-product math are unchanged. For Fast64 at the extreme 384->8
pair this adds 98 frames/channel. The output bound replaces the
unexplained upstream `+10` (retained in v5) with the possible input
debt converted at the highest permitted ratio; callers must query the
new maximum before allocating output storage.

The same extension fixes the short-polynomial history failure
(nearest, chunk 1, `p0=-3.25`, 0.7->1.9, first anchor -2.27256 but
only two retained frames). The tests exercise all polynomial degrees,
including 0.125<->8 ratio jumps and changing chunk sizes.

Fixed-output maximum-input storage retains the upstream bound:
`ceil(max_chunk*hmax)+2+floor(L/2)`. The initial polynomial anchor
`-L/2` is its largest startup requirement; later anchors are near
`-(L+1)`. The new endpoint guard remains within that bound throughout
the tested matrix.

## Cutoff-bank ownership and selection

The bank keeps one empty slot for the active boxed interpolator.
Selecting another valid slot takes that box, replaces the active box,
and moves the old box into the previous empty slot. There is no
destruction or allocation on selection. Invalid indices return false
without mutation. Every table has the same rounded length,
interpolation type and oversampling factor, so selection does not
change timing or history geometry. Each slot records its resolved
cutoff so `cutoff()` keeps reporting the active table after
selection; `new_sinc` tables are built through `InnerSinc::new` to
initialize v5's combined-sinc scratch.

Reset returns to slot zero as well as the original ratio; it clears
history in place. Repeated slot transfers and reset are covered by a
cold allocation probe and an audio comparison against a fresh nominal
instance. A bank's cutoff ratios must be positive and finite. As with
upstream, callers must supply valid sinc parameters; the SOTF presets
use finite, positive cutoffs and oversampling factors >=128.

The production grid uses eight log intervals/octave over relative
[0.5,2], capped at absolute ratio 1 and deduplicated, with the exact
nominal table retained in slot zero and an additional
min(nominal,1)*0.999 table for small negative clock drift when it lies
in the allowed range. Select the largest cutoff not exceeding
`min(actual_backend_ratio,target_ratio,1)` for the entire ramp, then
reevaluate for the following block. A zero-output block still advances
the current ratio as described above. Precompute the bank at
construction because dynamic mode is a realtime switch.

Grid quantization can lower bandwidth by up to `1-2^(-1/8)=8.30%`
between ordinary neighboring points; the .999 anchor limits the
common small negative-drift case to 0.1% relative to nominal. Worst
case: 18 High tables, approximately 4.5 MiB of coefficients; nominal
ratio 1 requires 10 tables, approximately 2.5 MiB. Coefficient storage
is independent of channel count. Instant table changes preserve
timing/history but are not a promise of click-free cutoff automation
or transient-free switching; steep filter-response changes can still
alter output abruptly.

This is a gross-alias correction, not universal stopband
certification. With Fast64 at 48->24 kHz, the 18 kHz probe changes
from -2.029 dB alias to -140.973 dB after selecting the low cutoff.
That is a deep-stop tone. The parent's independent FIR analysis shows
substantially weaker attenuation immediately around the new Nyquist,
especially at very low ratios. No uniform 60 dB boundary rejection is
claimed. Filter length/cutoff/transition redesign remains separate.

## Verification and limits

- **812 upstream library tests pass**, plus 5 doc-tests.
- **8 fork integration tests pass** (`tests/dynamic_ramp.rs`,
  ported to the v5 API: `Some` cutoffs, `Adjustable`/`Resizable`
  imports):
  - Independent brute-force frame-count oracle and exact 170-output
    checkpoint; reversed ramps.
  - Sinc nearest/linear/quadratic/cubic, lengths 8/64/128/256,
    chunks 1/7/256, both fixed modes, repeated instant/ramped
    ratio changes.
  - Requested nominal pairs 8/384, 8/192, 44.1/48, 48/44.1, 192/8
    and 384/8, with relative 0.5..2.
  - Explicit deferred-history phase regression and every
    polynomial degree.
  - Absolute linear source-coordinate oracle, including large
    ratio jumps and chunk changes; its rounding allowance is
    derived from three floating-point arithmetic operations, not
    measured production error.
  - Offset/partial-input sentinel checks; used/reset versus fresh
    bank audio equivalence.
  - Nonfinite ratio validation before planning.
- **Differential check of the rebase** (throwaway crate, v1 fork vs
  rebased fork, f32): 0/42 production and 0/18 extreme scenarios
  diverge in per-block counts; deferred scenarios panic on neither
  side; checkpoint 170/170; FFT geometry and samples bit-identical
  (0.0); nonfinite construction rejected identically. Residual
  sample differences are ~1e-8..1e-7 from upstream's v3+ kernel
  rework (dot-product strategy, Horner polynomials).
- **Production regressions pass**: `sotf-plugin-resampler`
  (50 lib + 45 integration, including the five `dynamic_cutoff`
  bank/bit-exact/alias/zero-alloc tests), `sotf-host`
  oversampling, convolution IR and binaural HRIR suites.
- **Clippy passes with warnings denied** (`cargo clippy
  --all-targets -- -D warnings`).
- Modified upstream Rust files are limited to `asynchro.rs`,
  `asynchro_fast.rs`, `asynchro_sinc.rs` and `lib.rs`; the last
  only reexports `InputPositions`. `Cargo.toml` adds the
  `dynamic_ramp` test target.

The acceptance envelope is the concrete SOTF audio-rate/quality
matrix above, not all conceivable finite floating-point ratios,
unlimited chunks, arbitrary custom interpolators, or allocation sizes
beyond available memory. Oversampling factor 1 with non-nearest sinc
interpolation panics identically on the v1 and v5 bases (probed) and
is outside the SOTF presets (factors >= 128). The fork still does not
redesign sinc
transition-band quality, and EOF duration integration belongs to the
separate wrapper accounting task. The wrapper integration and
independent production tests are tracked in
`sotf-daw/audit/resampler-prototype.md` in the aggregate checkout.

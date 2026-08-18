# Analog coloration listening protocol

This is a protocol and result template, not a listening result. The current
repository has no measured hardware capture and no human preference or
discrimination data.

## Render set

Render the same multichannel programme through the following conditions, with
the same sample rate, channel order, start offset, and duration:

1. Dry reference.
2. H2-only Harmonics (`h3_db` muted).
3. H3-only Harmonics (`h2_db` muted).
4. Combined H2/H3 Harmonics.
5. Static coloration.
6. Generic synthetic Hammerstein coloration.
7. Tape-style target.
8. Transformer-style target.
9. Captured hardware reference, only after a documented target and capture
   chain are supplied.

Use coherent synthetic fixtures for repeatable DSP checks and at least one
non-silent musical programme for listening. Keep a guard interval around
stateful renders and document the discarded samples. Do not add noise, hum,
wow, flutter, or crosstalk unless a separate test explicitly requests them.

## Matching and blinding

1. Measure each render with the offline
   `math_audio_analog::analysis::level_match_candidate` helper.
2. Use integrated BS.1770/EBU R128 loudness for the requested gain.
3. Choose one common sample-peak ceiling no greater than full scale and no
   lower than the dry reference peak. Apply the helper's `applied_gain_db` to
   every candidate.
4. Verify the post-gain sample peak and record both requested and applied
   gain. A peak-limited candidate is not an exact loudness match and must be
   flagged to the listener or excluded from the comparison.
5. Assign opaque labels with a fresh random seed for each session. Keep the
   label-to-condition map outside the listener's view until all trials are
   complete. The person preparing the map must not conduct the preference
   scoring when practical.

Do not infer audibility from a loudness or peak number. The matching step only
controls a confound; it is not evidence that one model is more accurate.

## Trial record

Record one row per trial with:

```text
session_id,listener_id,trial_id,reference_label,candidate_label,
abx_answer,abx_correct,preference_label,confidence_1_to_5,
reference_lufs,candidate_lufs,requested_gain_db,applied_gain_db,
reference_peak,candidate_peak,peak_ceiling,peak_limited,notes
```

Use at least three independent ABX trials per comparison for a smoke test and
pre-register the final trial count, exclusions, and analysis before unblinding.
Report discrimination separately from preference. Summarize listener count,
correct/incorrect ABX responses, confidence, preference counts, and any
peak-limited trials; do not report a louder condition as preferred-model
evidence.

## Current status

The level-matching helper and deterministic synthetic render fixtures are
implemented and tested. The following fields remain intentionally unfilled:

- measured hardware target, calibration, and capture provenance;
- held-out hardware validation;
- blinded human trials and results;
- release decision based on the recorded data.

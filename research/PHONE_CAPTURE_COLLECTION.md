# Phone Capture Collection Spec

Goal: build a small, structured phone-speaker-to-laptop-mic dataset that teaches the model the capture domain, not just one exact device pair.

## Principle

Do not collect one random pile of recordings.

Collect a small matrix of controlled conditions so the model sees:
- the same reciter under multiple capture conditions
- multiple reciters under the same capture conditions
- enough variation to learn the channel family, not one exact setup

## Scope

Phase 1 should be intentionally small:
- 5 target reciters first
- 3 capture conditions per reciter
- 2 repeats per condition
- 10 to 15 seconds per recording

Recommended first 5 reciters:
- `ahmed_talib_hameed`
- `maher_al_muaiqly`
- `raad_al_kurdi`
- `mishary_al_afasy`
- `yasser_al_dossary`

Why these first:
- they cover the observed failure cluster
- they give nearby confusions and one strong attractor class

## Capture Matrix

Use the same laptop and app path for Phase 1.
Change only one or two physical factors at a time.

Phase 1 conditions:

1. `near_mid_quiet`
- phone volume: medium
- phone distance to laptop mic: 15 to 20 cm
- phone angle: facing mic
- room: quiet

2. `mid_mid_quiet`
- phone volume: medium
- phone distance to laptop mic: 40 to 60 cm
- phone angle: facing mic
- room: quiet

3. `mid_high_room`
- phone volume: high
- phone distance to laptop mic: 40 to 60 cm
- phone angle: 30 to 45 degrees off-axis
- room: normal room noise

Two repeats per condition:
- `rep1`
- `rep2`

That yields:
- 5 reciters
- 3 conditions
- 2 repeats
- 30 recordings total

This is enough for a real first in-domain training set.

## What Must Stay Fixed Within One Phase

Keep these fixed during Phase 1:
- laptop
- laptop mic
- app capture path
- phone model if only one phone is available
- source clip identity for each reciter

Reason:
- if too many variables change at once, we cannot tell what helped
- Phase 1 is for learning the channel family from one hardware pair under several physical conditions

## What To Change In Phase 2

Only after Phase 1 is trained and evaluated:
- second phone model, if available
- second laptop or external mic, if available
- another room type
- lower phone volume edge case

Phase 2 extends hardware coverage.
Phase 1 establishes whether real in-domain data fixes the current problem at all.

## Source Audio Rules

For each reciter:
- choose 1 to 2 clean source clips not already used in the held-out `phone_mic` eval set
- use stable, speech-dominant sections
- avoid intros, silence, nasheed, crowd noise, or heavy background effects

Do not train on the exact current eval clips:
- `app/tests/repro/*.wav`
- `research/datasets/audio_test_sets/phone_mic/*/*.wav`

Those stay held out.

## Recording Procedure

For each recording:

1. Start from a chosen clean source clip for one reciter.
2. Play it from the phone speaker.
3. Capture it through the laptop mic using the same app path used in real usage.
4. Record 10 to 15 seconds.
5. Keep the phone stationary during one recording.
6. Leave 1 to 2 seconds of context at the start if needed, but avoid long silence.

Do not:
- speak during capture
- move the phone around continuously
- change volume mid-recording
- stack multiple reciters in one capture

## Naming Convention

File name format:

`<reciter>__<condition>__<repeat>__src-<clip_id>.wav`

Examples:
- `ahmed_talib_hameed__near_mid_quiet__rep1__src-113.wav`
- `maher_al_muaiqly__mid_high_room__rep2__src-109.wav`

Rules:
- use lowercase
- use double underscore between fields
- keep the source clip id in the name

## Folder Layout

Store raw captures here:

```text
research/datasets/audio_phone_train_raw/
  ahmed_talib_hameed/
    ahmed_talib_hameed__near_mid_quiet__rep1__src-113.wav
    ahmed_talib_hameed__near_mid_quiet__rep2__src-113.wav
    ...
  maher_al_muaiqly/
  mishary_al_afasy/
  raad_al_kurdi/
  yasser_al_dossary/
```

Store capture metadata here:

```text
research/datasets/audio_phone_train_raw/_meta/
  capture_manifest.csv
```

## Capture Manifest

Create one row per recording with these columns:

```text
file,reciter,source_clip_id,phone_model,laptop_model,mic_path,condition,room,volume_level,distance_cm,angle_deg,repeat,notes
```

Example:

```text
ahmed_talib_hameed__near_mid_quiet__rep1__src-113.wav,ahmed_talib_hameed,113,pixel6,asus_g14,browser_mic,near_mid_quiet,quiet,medium,20,0,1,
```

Why this matters:
- lets us split by condition later
- lets us compare failures by physical setup
- keeps the data useful beyond one experiment

## Train / Val / Eval Rule

Split by recording condition or source clip, not randomly by windows.

For Phase 1:
- train on 2 conditions
- validate on the remaining condition
- keep the current `phone_mic` eval suite fully untouched

Example:
- train: `near_mid_quiet`, `mid_mid_quiet`
- validate: `mid_high_room`
- eval: existing `phone_mic` suite only

This tests whether the model generalizes to an unseen capture condition.

## Minimum Quality Gate

A recording is acceptable only if:
- the reciter is clearly audible
- clipping is not severe
- there is no accidental second voice
- at least 8 seconds are usable

Reject and re-record if:
- the phone is too quiet
- the signal is dominated by room noise
- the capture is truncated
- there is handling noise or accidental movement

## Why This Is Better Than “My Phone And My Laptop”

If you record only one exact setup once, the model may memorize:
- one phone EQ
- one mic path
- one distance
- one room

This spec avoids that by varying the physical channel in a controlled way.

The data scientist pattern is:
- control the factors
- vary a few relevant ones
- keep metadata
- hold out one condition
- measure generalization, not memorization

## Commit Guidance

The collection spec itself is safe to commit immediately.

Recommended commit scope:
- this file
- optionally a short README link

Do not commit actual captured training audio until you decide the repo should store it.

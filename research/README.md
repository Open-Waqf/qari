# 🧪 Qari Finder — Research Lab (Data + Training + Parity)

This folder contains the **end-to-end ML pipeline** for Qari Finder:

- build the dataset (`features.npz`) from raw audio
- train the reciter classifier (`qari_model.keras` + exported SavedModel)
- evaluate on regression suites (**golden + challenge**) and an optional coverage suite (**golden_autofill**)
- **prove parity** between Python (trainer) and the browser (extractor + TFJS)

If you’re new: follow the **Quickstart**. If you’re senior: jump to **Runbooks** and **Parity / Regression Gates**.

---

## TL;DR Quickstart (the “safe” workflow)

> Run commands from **repo root** unless noted.

### 0) Prereqs

- **Python 3.10+** (3.11 is fine for training; see TFJS conversion note below)
- **ffmpeg** (required for mp3 decode in some environments)
- **Node 18+** (only needed for running the web app + parity checks)

### 1) Create env + install deps

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

pip install -r research/requirements.txt
python -m playwright install
````

### 2) Export “audio physics” (DSP matrices)

This locks the MFCC math used by both Python and the browser.

```bash
python research/tools/export_ears.py
```

Outputs:

* `research/models/audio_config.json` (source of truth for Python tooling)
* `app/public/models/audio_config.bin` (runtime payload for the app)

### 3) Build background noise class (optional but recommended)

```bash
python research/tools/extract_background.py --noise_mins 20 --reset
```

### 4) Data audit and sufficiency gate (optional but recommended)

```bash
python research/tools/audit_dataset.py --data datasets/audio --min_files 3 --fail
```

### 5) Prepare dataset (features)

```bash
python research/prepare_data.py
```

Outputs:

* `models/features.npz`
* updates `app/public/models/reciters_map.json` (class order is append-only)

### 6) Sanity check dataset (post-cap + post-gating truth)

```bash
python research/tools/check_balance.py --file models/features.npz --fail-hard
```

### 7) Train (3-way split + train-only normalization)

```bash
python research/train.py
```

Outputs:

* `models/qari_model.keras`
* `models/qari_model_export/` (SavedModel)
* `app/public/models/normalization.json`

### 8) Evaluate suites

Curated suites (baseline-regressed):

```bash
python research/evaluate_model.py --suite both
```

Coverage suite (no baseline; used for coverage gating):

```bash
python research/evaluate_model.py --suite golden_autofill --min_class_coverage 1.0
```

### 9) Export model parity sample (Python vs TFJS)

```bash
python research/tools/export_model_parity.py
```

### 10) Collect real phone-capture training data

Use the structured spec in:

```text
research/PHONE_CAPTURE_COLLECTION.md
```

---

## Test Suites: golden vs challenge vs golden_autofill

All evaluation suites live under:

```
datasets/audio_test_sets/<suite_name>/<class_name>/*.wav|*.mp3
```

### 1) `golden/` — curated regression suite (stable, baselined)

* small and curated
* can be **unseen audio** (preferred)
* answers: “did we break something important?”
* baseline file: `research/models/golden_baseline.json`

### 2) `challenge/` — stress / adversarial suite (baselined)

* harder real-world conditions: noise, reverb, overlaps, phone artifacts
* answers: “are we improving robustness?”
* baseline file: `research/models/challenge_baseline.json`

### 3) `golden_autofill/` — generated coverage filler (rebuildable, NOT baselined)

* auto-generated to ensure **coverage** (avoid 0-support classes)
* used for: coverage gating (`--min_class_coverage`) and sanity
* **no baseline comparison by default**
* safe to overwrite by scripts

---

## Project Layout

```
research/
  prepare_data.py           # raw audio -> MFCC images -> models/features.npz
  train.py                  # 3-way split + train-only normalization + export
  evaluate_model.py         # suite evaluator (golden/challenge/golden_autofill/any suite folder)
  requirements.txt
  README.md

  models/
    audio_config.json       # DSP matrices (source of truth)
    golden_baseline.json    # regression baselines (curated golden)
    challenge_baseline.json # regression baselines (curated challenge)

  tools/
    export_ears.py          # generate audio_config.json + audio_config.bin
    verify_matrix.py        # Python MFCC vs app MFCC check (Playwright)
    generate_golden.py      # exports golden_parity.json for MFCC math
    export_model_parity.py  # exports model_parity.json for Keras vs TFJS
    check_balance.py        # dataset sanity on features.npz (pairing + caps + uniq groups)
    audit_dataset.py        # slow “true duration” scan (optional)
    extract_background.py   # build/refresh _background class audio
    build_golden_suite.py   # create/refresh golden_autofill coverage suite
```

---

## Core Invariants (things we *must not* break)

1. **Group split is by source file**

* `prepare_data.py` assigns `groups` per source file.
* `train.py` uses group-aware splits so the same file never appears in both train and eval.

2. **Clean/Dirty pairing is preserved**

* each window adds **two samples**: `clean` then `dirty`
* many scripts assume: `X[0] clean, X[1] dirty, X[2] clean, ...`
* `tools/check_balance.py --fail-hard` verifies this

3. **Normalization is computed from TRAIN ONLY**

* `mean/std` computed from **train-clean** samples only
* written to: `app/public/models/normalization.json`

4. **Preprocessing parity between Python and TS**

* MFCC matrices come from `audio_config.json` → used on both sides
* gain clamp parity is required: `MAX_GAIN = 5.0` (Python) matches `maxGain=5` (TS)

---

## Runbooks (common tasks)

### A) I changed preprocessing constants (SR / FFT / hop / mel / MFCC)

```bash
python research/tools/export_ears.py
python research/prepare_data.py
python research/train.py
python research/evaluate_model.py --suite both

# Optional: validate DSP parity with running app
npm run dev -- --host
python research/tools/verify_matrix.py
```

### B) I added a new reciter (new class folder)

```bash
python research/prepare_data.py
python research/tools/check_balance.py --fail-hard
python research/train.py
python research/evaluate_model.py --suite both
```

### C) Golden suite has lots of 0-support classes

That means your curated `golden/` doesn’t cover most classes. Don’t pollute it—generate a separate coverage suite:

```bash
python research/tools/build_golden_suite.py \
  --src datasets/audio \
  --dst datasets/audio_test_sets/golden_autofill \
  --n 2 \
  --seed 42 \
  --clean

python research/evaluate_model.py --suite golden_autofill --min_class_coverage 1.0
```

### D) I need to retrain and ship a new model to the app

```bash
python research/prepare_data.py
python research/tools/check_balance.py --fail-hard
python research/train.py
python research/evaluate_model.py --suite both

# Optional: parity artifact for debugging
python research/tools/export_model_parity.py
```

App artifacts live in:

* `app/public/models/reciters_map.json`
* `app/public/models/normalization.json`
* `app/public/models/audio_config.bin`
* `app/public/models/tfjs_model/` (after conversion)

---

## Parity & Regression Gates (what to run before merging)

### 1) Dataset integrity gate

```bash
python research/tools/check_balance.py --file models/features.npz --fail-hard
```

### 2) DSP parity gate (MFCC math)

```bash
npm run dev -- --host
python research/tools/verify_matrix.py
```

### 3) Model parity gate (Keras vs TFJS)

```bash
python research/tools/export_model_parity.py
```

### 4) Regression evaluation gate (curated suites)

```bash
python research/evaluate_model.py --suite both
```

### 5) Coverage evaluation gate (optional, no baseline)

```bash
python research/evaluate_model.py --suite golden_autofill --min_class_coverage 1.0
```

---

## TFJS Conversion (SavedModel → TFJS)

### Recommended: convert in WSL (Windows users)

```bash
python3.10 -m venv venv_export
source venv_export/bin/activate
pip install tensorflow==2.15.0 tensorflowjs==4.17.0

tensorflowjs_converter     --input_format=tf_saved_model     --output_format=tfjs_graph_model     ./models/qari_model_export     ../app/public/models/tfjs_model
```

---

## Troubleshooting

### librosa can’t decode mp3 / durations are 0

Install `ffmpeg` and ensure it’s on PATH.

### verify_matrix.py can’t find the app

* run `npm run dev -- --host`
* ensure debug mode exposes the parity hook

### TFJS conversion fails on Windows

Use WSL and the pinned conversion env above.

### Split-risk warning: uniq_groups < 4

That class has too few source files. Group splits may omit it from val/test.

Options:

* add more raw files
* implement forced-coverage splitting for tiny classes

---

## Privacy note

See `research/PRIVACY_POLICY.md`. Intended posture is **local-only** audio processing (no raw audio uploads).

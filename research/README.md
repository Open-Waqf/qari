# 🧪 Qari Finder — Research Lab (Data + Training + Parity)

This folder contains the **end‑to‑end ML pipeline** for Qari Finder:

- build the dataset (`features.npz`) from raw audio
- train the reciter classifier (`qari_model.keras` + exported SavedModel)
- evaluate on regression suites (golden + challenge)
- **prove parity** between Python (trainer) and the browser (extractor + TFJS)

If you’re new: follow the **Quickstart**. If you’re senior: jump to **Runbooks** and **Parity / Regression Gates**.

---

## TL;DR Quickstart (the “safe” workflow)

> Run from repo root unless noted.

### 0) Prereqs

- **Python 3.10+** (3.11 is fine for training; see TFJS conversion note below)
- **ffmpeg** (required for `pydub` / some audio decodes)
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
```

### 2) Export “audio physics” (DSP matrices)

This locks the math used by both Python and the browser.

```bash
python research/tools/export_ears.py
```

Outputs:

- `research/models/audio_config.json` (source of truth)
- `app/public/models/audio_config.bin` (runtime payload)

### 3) Build background noise class (optional but recommended)

```bash
python research/tools/extract_background.py --noise_mins 20 --reset
```

### 4) Prepare dataset (features)

```bash
python research/prepare_data.py
```

Outputs:

- `models/features.npz`
- updates `app/public/models/reciters_map.json`

### 5) Sanity check dataset (post-cap + post-gating truth)

```bash
python research/tools/check_balance.py --file models/features.npz --fail-hard \
  --audio-config models/audio_config.json
```

### 6) Train (3-way split + train-only normalization)

```bash
python research/train.py
```

Outputs:

- `models/qari_model.keras`
- `models/qari_model_export/` (SavedModel)
- `app/public/models/normalization.json`

### 7) Evaluate regression suites (golden + challenge)

```bash
python research/evaluate_model.py --suite both
```

### 8) Export model parity sample (Python vs TFJS)

```bash
python research/tools/export_model_parity.py
```

---

## Project Layout

```
research/
  prepare_data.py           # raw audio -> MFCC images -> models/features.npz
  train.py                  # 3-way split + train-only normalization + export
  evaluate_model.py         # regression suites (golden/challenge)
  requirements.txt
  README.md

  models/
    audio_config.json       # DSP matrices (source of truth)
    golden_baseline.json    # evaluation baseline snapshots
    challenge_baseline.json

  tools/
    export_ears.py          # generate audio_config.json + audio_config.bin
    verify_matrix.py        # Python MFCC vs app MFCC check (Playwright)
    generate_golden.py      # exports golden_parity.json for MFCC math
    export_model_parity.py  # exports model_parity.json for Keras vs TFJS
    check_balance.py        # post-prepare dataset sanity (pairing + caps)
    audit_dataset.py        # slow “true duration” scan (pydub decode)
    extract_background.py   # build _background class audio
    build_golden_suite.py   # create/refresh golden suite coverage
    sync_manager.py         # Colab hybrid workflow
    Qari_Trainer.ipynb      # Colab notebook
```

---

## Core Invariants (things we *must not* break)

These are the rules that keep training metrics honest and the app consistent:

1) **Group split is by source file**

- In `prepare_data.py`, `groups` is set per source file.
- In `train.py`, we split with `GroupShuffleSplit` so the same file never appears in both train and eval.

2) **Clean/Dirty pairing is preserved**

- For each window we add **two samples**: `clean` then `dirty`.
- Many scripts assume the pattern: `X[0] clean, X[1] dirty, X[2] clean, ...`.
- `tools/check_balance.py --fail-hard` verifies this.

3) **Normalization is computed from TRAIN ONLY**

- `mean/std` are computed from **train-clean** samples only.
- Written to `app/public/models/normalization.json`.

4) **Preprocessing parity between Python and TS**

- MFCC matrices come from `audio_config.json` → used both sides.
- Gain clamp parity is required: `MAX_GAIN = 5.0` (Python) matches `maxGain=5` (TS).

---

## Runbooks (common tasks)

### A) “I changed preprocessing constants” (SR / FFT / hop / mel / MFCC)

1. Re-export matrices:
   ```bash
   python research/tools/export_ears.py
   ```
2. Re-generate MFCC parity truth (optional but recommended):
   ```bash
   python research/tools/generate_golden.py
   ```
3. Rebuild features + retrain:
   ```bash
   python research/prepare_data.py
   python research/train.py
   python research/evaluate_model.py --suite both
   ```
4. Validate DSP parity with the running app:
   ```bash
   # In another terminal
   npm run dev -- --host
   python research/tools/verify_matrix.py
   ```

### B) “I added a new reciter (new class folder)”

1. Add folder: `datasets/audio/<new_reciter_name>/...`.
2. Run prepare (it will append new class at end):
   ```bash
   python research/prepare_data.py
   ```
3. Sanity check:
   ```bash
   python research/tools/check_balance.py --fail-hard
   ```
4. Retrain + evaluate:
   ```bash
   python research/train.py
   python research/evaluate_model.py --suite both
   ```

### C) “Golden suite has lots of zero-support classes”

That means your regression suite doesn’t cover most classes.

Quick fix: build/refresh a golden suite with N files per class:

```bash
python research/tools/build_golden_suite.py \
  --src datasets/audio \
  --dst datasets/audio_test_sets/golden \
  --n 2 \
  --seed 42 \
  --clean

python research/evaluate_model.py --suite golden --update_baseline
```

### D) “I need to retrain and ship a new model to the app”

```bash
python research/prepare_data.py
python research/tools/check_balance.py --fail-hard
python research/train.py
python research/evaluate_model.py --suite both

# optional parity artifacts for debugging
python research/tools/generate_golden.py
python research/tools/export_model_parity.py
```

Artifacts used by the app live in:

- `app/public/models/reciters_map.json`
- `app/public/models/normalization.json`
- `app/public/models/audio_config.bin`
- `app/public/models/tfjs_model/` (after conversion)

---

## Parity & Regression Gates (what to run before merging)

### 1) Dataset integrity gate

```bash
python research/tools/check_balance.py --file models/features.npz --fail-hard
```

This catches:

- broken clean/dirty pairing
- missing/invalid mapping indices
- split-risk warning (few unique files in a class)

### 2) DSP parity gate (MFCC math)

**Goal:** Python MFCC math == browser MFCC math

1. Start app:

```bash
npm run dev -- --host
```

2. Run parity checker:

```bash
python research/tools/verify_matrix.py
```

### 3) Model parity gate (Keras vs TFJS)

**Goal:** same MFCC input → (nearly) same probabilities

```bash
python research/tools/export_model_parity.py
```

Then load the app in debug mode and compare against `app/public/models/model_parity.json`.

### 4) Regression evaluation gate

```bash
python research/evaluate_model.py --suite both
```

Tip: keep baselines in `research/models/*_baseline.json` updated only when you *intend* to accept a new baseline.

---

## TFJS Conversion (SavedModel → TFJS)

### Recommended: convert in WSL (Windows users)

TensorFlowJS conversion is often painful on native Windows due to protobuf / binary conflicts.

In WSL (Ubuntu):

```bash
python3.10 -m venv venv_export
source venv_export/bin/activate
pip install tensorflow==2.15.0 tensorflowjs==4.17.0

tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_graph_model \
  ./models/qari_model_export \
  ./app/public/models/tfjs_model
```

---

## Script Reference (what each script is for)

### Main pipeline

- `research/prepare_data.py`
    - raw audio → normalized chunks → MFCC images → `models/features.npz`
    - maintains `reciters_map.json` order
    - enforces MFCC width/shape expectations

- `research/train.py`
    - **3-way split** (train/val/test) with **GroupShuffleSplit**
    - train-only normalization; writes `normalization.json`
    - exports Keras + SavedModel

- `research/evaluate_model.py`
    - runs evaluation on `datasets/audio_test_sets/{golden,challenge}`
    - produces detailed per-class report + confusion matrix
    - supports baseline snapshots (`research/models/*_baseline.json`)

### Tools

- `research/tools/export_ears.py` — exports DSP matrices to JSON + BIN
- `research/tools/generate_golden.py` — exports `golden_parity.json` (MFCC truth)
- `research/tools/verify_matrix.py` — verifies Python MFCC == browser MFCC (requires running app + debug hook)
- `research/tools/export_model_parity.py` — emits a fixed MFCC + expected probs for TFJS parity
- `research/tools/check_balance.py` — dataset sanity on `features.npz` (pairing, caps, unique files)
- `research/tools/audit_dataset.py` — slow deep scan of raw audio minutes per class (pydub decode)
- `research/tools/extract_background.py` — generates/refreshes `_background` audio
- `research/tools/build_golden_suite.py` — ensure golden suite has minimum per-class coverage
- `research/tools/sync_manager.py` + `research/tools/Qari_Trainer.ipynb` — Colab hybrid workflow

---

## Troubleshooting

### “pydub can’t decode mp3” / durations are 0

Install ffmpeg and ensure it’s on PATH.

### “verify_matrix.py can’t find the app”

- run `npm run dev -- --host` (required so Playwright can hit the server)
- ensure debug mode exposes `window.checkParity`

### “TFJS conversion fails on Windows”

Use WSL and the pinned conversion env shown above.

### “My eval has tons of support: 0 classes”

Your test suite doesn’t include files for those classes. Build the golden suite:

```bash
python research/tools/build_golden_suite.py --clean --n 2
```

### “Split-risk warning: uniq_groups < 4”

That class has too few source files. Group splits may omit it from val/test.
Options:

- add more raw files
- implement a split retry/forced coverage policy for tiny classes

---

## Privacy note

See `research/PRIVACY_POLICY.md`. The intended posture is **local-only** audio processing (no raw audio uploads).

# 🧪 Qari Finder Research Lab

The Python environment for training, evaluating, and converting the Reciter Identification Model.

## 📂 Project Structure

* **`datasets/`**: Contains all heavy data (`audio/`, `audio_test_set/`, `features.npz`). Ignored by Git (except the
  Golden Set).
* **`models/`**: Stores trained models (`.h5`) and baselines (`.json`).
* **`tools/`**: Helper scripts for auditing, physics export, and noise generation.
* **Root Scripts:** The main pipeline (`prepare_data.py`, `train.py`, `evaluate_model.py`).

## 🎯 Workflow

1. **Math Export (`tools/export_ears.py`):** Generates the physics matrices (`audio_config.json`) that ensure the App
   and Python "hear" the same way.
2. **Dataset Audit (`tools/audit_dataset.py`):** Checks your `datasets/audio/` folder to see which reciters need more
   data.
3. **Data Collection:** Place training audio in `datasets/audio/<reciter_name>/`.
4. **Golden Set:** Place unseen test audio in `datasets/audio_test_set/<reciter_name>/` (Required for quality gates).
5. **Background Noise:** Generate noise/silence for the `_background` class using `tools/export_background_manual.py`.
6. **Feature Extraction (`prepare_data.py`):** Converts MP3s into spectrogram tensors (`datasets/features.npz`).
7. **Training (`train.py`):** Trains the Keras model (`models/qari_model.h5`).
8. **Evaluation (`evaluate_model.py`):** Checks the model against the Golden Set to ensure no regressions.
9. **Conversion (`convert_wizard.py`):** Converts the valid Keras model to TensorFlow.js format.

## 🚀 Commands

### 1. Setup Environment

```bash
python -m venv venv
# Windows: .\venv\Scripts\activate | Mac: source venv/bin/activate
pip install -r requirements.txt

```

### 2. Export Physics (The "Ears")

Run this first to generate the shared configuration.

```bash
python tools/export_ears.py

```

### 3. Generate Background Noise (Phase 1 Essential)

Downloads or processes noise files (rain, typing, silence) to train the "Unknown" class.
**Note:** You must have the ESC-50 dataset unzipped in `datasets/sound_datasets/esc50/`.

```bash
# Deletes old files (--reset) and generates 35 mins of noise
python tools/export_background_manual.py --noise_mins 35 --reset

```

**Output location:** `datasets/audio/_background/`
**Example files:** `rain_1-54023.wav`, `silence_synthetic_001.wav`

### 4. Audit Dataset

Check for data imbalance (identifies "Low" or "High" duration reciters).

```bash
python tools/audit_dataset.py

```

### 5. Prepare & Train

```bash
python prepare_data.py
python train.py

```

### 6. Evaluate (The Quality Gate) 🛡️

Run this to compare your new model against the baseline.

```bash
python evaluate_model.py

```

### 7. Verify Math Parity

If you change the DSP logic, run this to ensure Python's output matches the App's expectations.

```bash
python tools/verify_matrix.py

```

### 8. Convert to Web

```bash
python convert_wizard.py

```

## 🔬 File Guide

* `prepare_data.py`: Main feature extraction script with "bad mic" augmentation.
* `train.py`: Neural network training with Global Average Pooling (GAP).
* `evaluate_model.py`: Strict no-regression tester using the Golden Test Set.
* `convert_wizard.py`: Handles Keras to TFJS conversion (fixes Windows-specific missing `.so` errors).
* `tools/export_ears.py`: Generates `audio_config.json`.
* `tools/audit_dataset.py`: Scans audio files for true duration.
* `tools/export_background_manual.py`: Generates named noise clips for the background class.
# 🧪 Qari Finder Research Lab

The Python environment for training, evaluating, and converting the Reciter Identification Model.

## 🎯 Workflow

1. **Math Export (`export_ears.py`):** Generates the physics matrices (`audio_config.json`) that ensure the App and
   Python "hear" the same way.
2. **Dataset Audit (`audit_dataset.py`):** Checks your `audio/` folder to see which reciters need more data.
3. **Data Collection:** Place training audio in `audio/<reciter_name>/`.
4. **Golden Set:** Place unseen test audio in `audio_test_set/<reciter_name>/` (Required for quality gates).
5. **Feature Extraction (`prepare_data.py`):** Converts MP3s into spectrogram tensors (`features.npz`).
6. **Training (`train.py`):** Trains the Keras model (`qari_model.h5`).
7. **Evaluation (`evaluate_model.py`):** Checks the model against the Golden Set to ensure no regressions.
8. **Conversion (`convert_wizard.py`):** Converts the valid Keras model to TensorFlow.js format.

## 🚀 Commands

### 1. Setup Environment

```bash
python -m venv venv
# Windows: .\venv\Scripts\activate | Mac: source venv/bin/activate
pip install -r requirements.txt

```

### 2. Export Physics (The "Ears")

Run this first to generate the shared configuration for the app and feature extractor.

```bash
python export_ears.py

```

### 3. Audit Dataset

Check for data imbalance (identifies "Low" or "High" duration reciters).

```bash
python audit_dataset.py

```

### 4. Prepare & Train

```bash
python prepare_data.py
python train.py

```

### 5. Evaluate (The Quality Gate) 🛡️

Run this to compare your new model against the baseline.

```bash
python evaluate_model.py

```

### 6. Verify Math Parity

If you change the DSP logic, run this to ensure Python's output matches the App's expectations.

```bash
python verify_matrix.py

```

### 7. Convert to Web

```bash
python convert_wizard.py

```

## 🔬 File Guide

* `export_ears.py`: Generates `audio_config.json` containing Mel filterbanks and DFT matrices.
* `audit_dataset.py`: Performs a deep scan of audio files to report true durations using Pydub.
* `prepare_data.py`: Main feature extraction script with "bad mic" augmentation.
* `train.py`: Neural network training with Global Average Pooling (GAP).
* `evaluate_model.py`: Strict no-regression tester using the Golden Test Set.
* `verify_matrix.py`: Debug tool to verify feature extraction parity against a sine wave.
* `convert_wizard.py`: Handles Keras to TFJS conversion (fixes Windows-specific missing `.so` errors).
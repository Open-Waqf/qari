# 🧪 Qari Finder Research Lab

The Python research environment for training, evaluating, and converting the **Reciter Identification Model**.

This lab supports two workflows:

1. **☁️ Hybrid (Recommended):** Code locally, Train on Colab (GPU), Sync results back.
2. **💻 Local Hero:** Run everything on your machine (CPU/GPU).

---

## 📂 Project Structure

* **`datasets/`**: Contains raw data (`audio/`, `audio_test_set/`, `features.npz`). *Ignored by Git
  except `audio_test_set/`.*
* **`models/`**: Stores trained model artifacts (`.h5`, `saved_model/`).
* **`tools/`**: Helper scripts for syncing, auditing, and physics export.
* **Root Scripts:** The main pipeline (`prepare_data.py`, `train.py`, `export_tfjs.py`).

---

## 🛠️ Setup & Prerequisites

### 1. Python Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

### 2. Audio "Physics" Configuration

Run this **first**. It generates `audio_config.json`, which ensures the Web App and Python "hear" audio exactly the same
way.

```bash
python tools/export_ears.py

```

### 3. Background Noise Generation

Downloads or processes noise files (rain, typing, silence) to train the "Unknown" class.

```bash
# Deletes old files (--reset) and generates 20 mins of noise
python tools/export_background_manual.py --noise_mins 20 --reset

```

---

## ☁️ Option A: The "Hybrid" Workflow (Recommended)

Use your local machine for coding and data collection, but let Google Colab's powerful GPUs handle the training.

### Step 1: Push Code & Data 🚀

Run this command to upload your local `research/`code, app public models, and `datasets/audio/` to Google Drive.

```bash
python tools/sync_manager.py --push

```

### Step 2: Train on Colab 🧠

1. Open **`Qari_Trainer.ipynb`** in Google Colab.
2. Mount Google Drive.
3. Run the **"Train"** and **"Convert"** cells.
4. The notebook automatically saves the trained model (`export model format`) and the web model (`tfjs_model/`) back to
   your Drive.

### Step 3: Pull Results 📥

Download the trained brain back to your local machine. This updates your app's `public/models` folder automatically.

```bash
python tools/sync_manager.py --pull

```

---

## 💻 Option B: The "Local Hero" Workflow

If you prefer to run everything offline on your own machine.

### 1. Audit Dataset

Check for data imbalance (identifies "Low" or "High" duration reciters).

```bash
python tools/audit_dataset.py

```

### 2. Prepare Features

Extracts spectrograms from audio files.

```bash
python prepare_data.py

```

### 3. Train Model

Trains the neural network. Now uses `model.export()` to create a clean SavedModel folder.

```bash
python train.py

```

### 4. Evaluate (Quality Gate) 🛡️

Compare your new model against the baseline to ensure no regressions.

```bash
python evaluate_model.py

```

### 5. Convert to Web

Converts the Keras model to a TensorFlow.js **Graph Model**.

* **Output:** `../app/public/models/tfjs_model/`

---

## 🔬 File Guide

### 🔄 Sync Tools

* **`tools/sync_manager.py`**: The bridge between Local and Cloud.
* `--push`: Uploads code/audio to Drive.
* `--pull`: Downloads trained `.h5` and `tfjs_model` folder to your App.


* **`Qari_Trainer.ipynb`**: The Colab Notebook for high-speed training.

### 🧠 Research Scripts

* **`prepare_data.py`**: Feature extraction with augmentation and index locking.
* **`train.py`**: Training script. Defines the `SpecAugment` layer and exports clean SavedModels.
* **`export_tfjs.py`**: Robust converter. Handles the Keras 3 -> SavedModel -> TFJS pipeline.
* **`evaluate_model.py`**: Strict no-regression testing.

### 🛠️ Utilities

* **`tools/export_ears.py`**: Generates `audio_config.json` (the physics config).
* **`tools/verify_matrix.py`**: Verifies mathematical parity between Python and JS.
* **`tools/audit_dataset.py`**: Scans audio files for true duration and class balance.
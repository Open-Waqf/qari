# 🧪 Qari Finder Research Lab

The Python research environment for training, evaluating, and converting the **Reciter Identification Model**.

This lab supports two workflows:

1. **☁️ Hybrid (Recommended):** Code locally, Train on Colab (GPU), Sync results back.
2. **💻 Local Hero:** Run everything on your machine (CPU/GPU). **Note:** Windows users require **WSL** for the final
   conversion step.

---

## 📂 Project Structure

* **`datasets/`**: Contains raw data (`audio/`, `audio_test_set/`, `features.npz`). *Ignored by Git
  except `audio_test_set/`.*
* **`models/`**: Stores trained model artifacts (`.keras`, `qari_model_export/`).
* **`tools/`**: Helper scripts for syncing, auditing, and physics export.
* **Root Scripts:** The main pipeline (`prepare_data.py`, `train.py`, `evaluate_model.py`).

---

## 🛠️ Setup & Prerequisites

### 1. Python Environment (Training & Prep)

For data preparation and training, you can use standard Windows Python.

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

```

### 2. Audio "Physics" Configuration

Run this **first**. It generates `audio_config.json`, which ensures the Web App (Client) and Python (Server) "hear"
audio exactly the same way.

* **Standard:** `SR=22050`, `FFT=512`.

```bash
python tools/export_ears.py

```

### 3. Background Noise Generation

Downloads or processes noise files (rain, typing, silence) to train the "Unknown" class.

```bash
# Deletes old files (--reset) and generates 20 mins of noise
python tools/export_background.py --noise_mins 20 --reset

```

---

## ☁️ Option A: The "Hybrid" Workflow (Recommended)

Use your local machine for coding and data collection, but let Google Colab's powerful GPUs handle the training and
conversion.

### Step 1: Push Code & Data 🚀

Run this command to upload your local `research/` code and `datasets/` to Google Drive.

```bash
python tools/sync_manager.py --push

```

### Step 2: Train on Colab 🧠

1. Open **`tools/Qari_Trainer.ipynb`** in Google Colab.
2. Mount Google Drive.
3. Run the **"Train"** and **"Convert"** cells.
4. The notebook automatically saves the trained model and the converted web model (`model.json`) back to your Drive.

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

Extracts spectrograms from audio files into `models/features.npz`.

```bash
python prepare_data.py

```

### 3. Train Model

Trains the neural network. Now uses `model.export()` to create a clean `SavedModel` folder in
`models/qari_model_export`.

```bash
python train.py

```

### 4. Evaluate (Quality Gate) 🛡️

Compare your new model against the baseline to ensure no regressions.

```bash
python evaluate_model.py

```

### 5. Convert to Web (The "WSL" Step) ⚠️

**Windows Users:** You **cannot** run this step in standard Windows PowerShell due to dependency conflicts between
`tensorflowjs` and `protobuf`. You **must** use **WSL (Ubuntu)**.

**1. Open WSL Terminal (Ubuntu):**

```bash
cd /mnt/c/.../qari-finder/research

```

**2. Set up the "Golden" Conversion Environment (Python 3.10):**
*Note: Python 3.12 is not supported by TF 2.15.*

```bash
# Install Python 3.10 if missing
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update && sudo apt install python3.10 python3.10-venv -y

# Create & Activate Environment
rm -rf venv_export
python3.10 -m venv venv_export
source venv_export/bin/activate

# Install the Golden Version Combo
pip install --upgrade pip
pip install tensorflow==2.15.0 tensorflowjs==4.17.0

```

**3. Run the Conversion:**

```bash
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    ./models/qari_model_export \
    ../app/public/models/tfjs_model

```

---

## 🔬 File Guide

### 🔄 Sync Tools

* **`tools/sync_manager.py`**: The bridge between Local and Cloud.
* `--push`: Uploads code/audio to Drive.
* `--pull`: Downloads trained `.keras` and web models to your App.


* **`tools/Qari_Trainer.ipynb`**: The Colab Notebook for high-speed training.

### 🧠 Research Scripts

* **`prepare_data.py`**: Feature extraction with augmentation and index locking.
* **`train.py`**: Training script. Defines the `SpecAugment` layer and exports clean SavedModels.
* **`evaluate_model.py`**: Strict no-regression testing.

### 🛠️ Utilities

* **`tools/export_ears.py`**: Generates `audio_config.json` (the physics config: 22050Hz / 512 FFT).
* **`tools/verify_matrix.py`**: Verifies mathematical parity between Python and JS.
* **`tools/audit_dataset.py`**: Scans audio files for true duration and class balance.
* **`check_balance.py`**: Check balance of the data from features.npz
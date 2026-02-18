# 🧪 Qari Finder Research Lab

The Python research environment for training, evaluating, and converting the **Reciter Identification Model**.

This lab ensures that the "Brain" trained in Python is mathematically identical to the "Ears" running in the browser.

1. **☁️ Hybrid (Recommended):** Code locally, Train on Colab (GPU), Sync results back.
2. **💻 Local Hero:** Run everything on your machine (CPU/GPU). **Note:** Windows users require **WSL** for the final
   conversion step.

---

## 📂 Project Structure

* **`datasets/`**: Contains raw data (`audio/`, `audio_test_set/`, `features.npz`).
* **`models/`**: Stores trained model artifacts (`.keras`, `qari_model_export/`).
* **`tools/`**: Helper scripts for syncing, auditing, and physics export.
* **`prepare_data.py`**: The main feature extraction pipeline with SpecAugment.
* **`train.py`**: Training script using `model.export()` for clean `SavedModel` folders.
* **`generate_golden.py`**: 🆕 **The Judge.** Generates the mathematical truth file used by the Web App to verify signal
  parity.

---

## 🛠️ The "Golden Parity" Workflow

*The most critical step to ensure the App "hears" audio exactly like the Trainer.*

1. **Export Physics:** Run `python tools/export_ears.py`. This generates `audio_config.json`, which contains the exact
   FFT and Mel matrices your model was trained on.
2. **Generate Truth:** Run `python generate_golden.py`. This creates a deterministic signal and calculates the "perfect"
   MFCC result using Python’s math. It saves this to `app/public/models/golden_parity.json`.
3. **Verify:** Open the Web App with `?debug=1` and click **Math** in the debug panel. If the JS matches the Python
   result within `0.00005`, your pipeline is stable.

---

## ⚙️ Setup & Prerequisites

### 1. Python Environment (Training & Prep)

For data preparation and training, you can use standard Windows Python.

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

```

### 2. Audio "Physics" Configuration

Run this **first**. It ensures the Web App (Client) and Python (Server) use the same mathematical "lens."

* **Constants:** `SR=22050`, `FFT=512`, `Hop=256`, `MFCC=40`.

```bash
python tools/export_ears.py

```

### 3. Background Noise Generation

Generates synthetic noise to train the "Unknown" class.

```bash
python tools/export_background.py --noise_mins 20 --reset

```

---

## ☁️ Option A: The "Hybrid" Workflow (Recommended)

1. **Push:** `python tools/sync_manager.py --push` (Uploads local code/audio to Drive).
2. **Train:** Open **`tools/Qari_Trainer.ipynb`** in Google Colab. Run the "Train" and "Convert" cells.
3. **Pull:** `python tools/sync_manager.py --pull` (Downloads results back to `app/public/models`).

---

## 💻 Option B: The "Local Hero" Workflow

### 1. Prepare & Train

```bash
python prepare_data.py   # Extracts spectrograms into models/features.npz
python train.py          # Trains and creates models/qari_model_export
python evaluate_model.py # Strict no-regression testing

```

### 2. Convert to Web (The "WSL" Step) ⚠️

**Windows Users:** You **must** use **WSL (Ubuntu)** for this step due to binary conflicts between `tensorflowjs` and
`protobuf` on Windows.

**Open WSL Terminal:**

```bash
# Set up the Golden Conversion Environment (Python 3.10)
python3.10 -m venv venv_export
source venv_export/bin/activate
pip install tensorflow==2.15.0 tensorflowjs==4.17.0

# Run the Conversion
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    ./models/qari_model_export \
    ../app/public/models/tfjs_model

```

---

## 🔬 File Guide

### 🧠 Research Scripts

* **`prepare_data.py`**: Extracts features using Magnitude + Natural Log math.
* **`train.py`**: Defines the `SpecAugment` dummy layer for TFJS compatibility.
* * **`evaluate_model.py`**: Strict no-regression testing.
* **`generate_golden.py`**: Creates a reference JSON to prove Python and JS parity.

### 🛠️ Utilities

* **`tools/export_ears.py`**: Exports the mathematical matrices to `audio_config.json`.
* **`tools/verify_matrix.py`**: Cross-checks raw matrix multiplication against Python logic.
* **`tools/audit_dataset.py`**: Identifies class imbalance in the training data.
* **`tools/check_balance.py`**: Check balance of the data from features.npz
* 
### 🔄 Sync Tools

* **`tools/sync_manager.py`**: The bridge to Google Drive.
* **`tools/Qari_Trainer.ipynb`**: Colab Notebook for GPU-accelerated training.
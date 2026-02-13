# 🧪 Qari Finder Research Lab

The Python environment for training the Reciter Identification Model.

## 🎯 Workflow

1. **Data Collection:** Place MP3 files in `audio/<reciter_name>/`.
2. **Math Export (`export_ears.py`):** Generates the FFT/DCT matrices (`audio_config.json`) used by both Python and the
   App.
3. **Feature Extraction (`prepare_data.py`):** Converts MP3s into `.npz` files (Spectrograms).
4. **Training (`train.py`):** Trains the Keras model (`qari_model.h5`) on the `.npz` data.
5. **Conversion (`convert_wizard.py`):** Converts the Keras model to TensorFlow.js format (bypassing Windows errors).

## 🗂️ Dataset Format

Structure your `audio/` folder like this:

```text
audio/
  ├── al_afasy/
  │     ├── 001.mp3
  │     └── 002.mp3
  ├── sudais/
  │     ├── 001.mp3
  └── ...

```

## 🚀 Commands

### 1. Setup Environment (Windows Safe)

We use specific versions to avoid JAX/Flax errors on Windows.

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

```

### 2. Prepare Data

Run this if you add new audio files:

```bash
python prepare_data.py

```

### 3. Train Model

Trains the neural network.

```bash
python train.py

```

* **Output:** `qari_model.h5`
* **Note:** Ignore the "Conversion Failed" error at the end of this script.

### 4. Convert to Web (The Wizard Step)

Run this to generate the files for the app (fixes the `inference.so` missing error).

```bash
python convert_wizard.py

```

* **Output:** `../app/public/models/tfjs_model/`

## 🔬 Files

* `train.py`: Main training script.
* `convert_wizard.py`: Special script to handle TFJS conversion on Windows.
* `export_ears.py`: Generates physics matrices (`audio_config.json`).
* `requirements.txt`: Pinned dependencies for stability.
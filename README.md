# 🔍 Qari Finder: The "Shazam" for Quran Reciters

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://opensource.org/licenses/GPL-3.0)
[![Tech: TensorFlow.js](https://img.shields.io/badge/AI-TensorFlow.js-orange.svg)](https://js.tensorflow.org/)
[![Platform: PWA/Android/iOS](https://img.shields.io/badge/Platform-PWA%20%7C%20Android%20%7C%20iOS-green.svg)](https://capacitorjs.com/)

**Qari Finder** is an AI-powered recognition engine that identifies Quran reciters (Qaris) from live audio in real-time.
Unlike traditional apps, it performs **100% of processing locally on your device**, ensuring your spiritual moments
remain private, secure, and offline-capable.

---

## ✨ Key Features

* **⚡ Real-time Recognition:** Identifies 20+ famous reciters instantly using client-side AI.
* **🔒 Privacy-First:** Audio **never** leaves your device. No cloud APIs, no server costs, no latency.
* **🌊 Dynamic Visualizer:** A "Siri-style" Halo visualizer that responds to voice energy in real-time.
* **📱 Universal Support:** Progressive Web App (PWA) that works on iOS, Android, and Desktop.

---

## 🏗️ Project Architecture

A high-performance monorepo designed for strict signal processing parity:

| Component       | Stack                        | Purpose                                       |
|-----------------|------------------------------|-----------------------------------------------|
| **`/app`**      | Vite, Lit, TypeScript, TF.js | The consumer-facing application & DSP engine. |
| **`/research`** | Python, Keras, Librosa       | The AI Lab for dataset generation & training. |

---

## 🛠️ The Technical Moat (Engineering Deep Dive)

Qari Finder bridges the gap between Python research and JavaScript production with strict mathematical guarantees.

### 1. Strict Signal Parity ⚖️

We don't just "hope" the JS works. We mathematically prove it.

* **The Problem:** Python (`librosa`) and JS (`WebAudio`) often calculate math differently (floating point drift).
* **Our Solution:** A **Golden Master** test suite.
* `research/generate_golden.py` creates a deterministic signal + expected MFCC matrices.
* The App runs a **Strict Parity Test** on boot, asserting that JS inference matches Python training data within a
  `0.00005` error margin.

### 2. Robust Audio Pipeline 🎧

* **48kHz Support:** Most Android phones run at 48kHz. Our `AudioWorklet` uses **Hermite Cubic Interpolation** to
  downsample to the model's 22.05kHz requirement without aliasing artifacts.
* **Smart Gating:** Adaptive noise floor calibration prevents background hum from triggering the AI.

### 3. Session State Machine 🧠

Instead of raw frame-by-frame predictions, we use a time-based state machine:

* **Candidate:** A reciter is detected but not confirmed.
* **Locked:** After `N` seconds of stability, the UI locks.
* **Hysteresis:** Switching reciters requires strong, sustained evidence (preventing "glitch" switches).

---

## 🚀 Quick Start

### 1. Run the App

```bash
cd app
npm install
npm run dev

```

Open `http://localhost:5173`.

### 2. Developer Debugging

Append `?debug=1` to the URL (e.g., `http://localhost:5173/?debug=1`) to open the **Engineering Panel**.

* **🐞 Math Parity:** Run the Strict Parity test live in the browser.
* **🔁 Loopback:** Inject audio files directly into the DSP pipeline (bypassing the mic).
* **📊 Stats:** Monitor latency, RMS, and TensorFlow backend status.

### 3. Training the Brain

```bash
cd research
pip install -r requirements.txt
# Place audio clips in datasets/audio/
python prepare_data.py  # Generates spectrograms
python train.py         # Trains the CNN & exports to TFJS

```

---

## 📜 License

Distributed under the **GPL-3.0-or-later** license.

> **Privacy Note:** This application architecture guarantees privacy by design. It does not contain code to record,
> store, or transmit user audio to any external server.
# 🔍 Qari Finder: The "Shazam" for Quran Reciters

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://opensource.org/licenses/GPL-3.0)
[![Tech: TensorFlow.js](https://img.shields.io/badge/AI-TensorFlow.js-orange.svg)](https://js.tensorflow.org/)
[![Platform: PWA/Android/iOS](https://img.shields.io/badge/Platform-PWA%20%7C%20Android%20%7C%20iOS-green.svg)](https://capacitorjs.com/)

**Qari Finder** is an AI-powered recognition engine that identifies Quran reciters (Qaris) from live audio in real-time.
Unlike traditional apps, it performs all processing **locally in your browser**, ensuring your spiritual moments remain
private and secure.

---

## ✨ Key Features

* **⚡ Real-time Recognition:** Identify 20+ famous reciters instantly using advanced signal processing.
* **🔒 Privacy-First:** Audio never leaves your device. We use TensorFlow.js for local inference—no cloud uploads
  required.
* **🌊 Dynamic Visualizer:** A "Siri-style" waveform responds to live audio input.
* **📱 Universal Support:** Installable as a PWA or deployed as a native app via Capacitor.
* **🎯 High Accuracy:** Custom CNN model trained on 9,000+ audio clips with 98% accuracy.

---

## 🏗️ Project Architecture

This is a monorepo designed for high-performance audio analysis:

| Component       | Stack                        | Purpose                                      |
|:----------------|:-----------------------------|:---------------------------------------------|
| **`/app`**      | Vite, Lit, TypeScript, TF.js | The consumer-facing Web/Mobile application.  |
| **`/research`** | Python, Keras, Librosa       | The AI Lab for data prep and model training. |

---

## 🚀 Quick Start

### 1. Run the Web App

```bash
cd app
npm install
npm run dev

```

Open `http://localhost:5173` to start identifying reciters.

### 2. Retrain or Add New Reciters

```bash
cd research
pip install -r requirements.txt
# Place audio in datasets/audio/ and then:
python prepare_data.py
python train.py

```

This generates a new model and automatically formats it for the web app.

---

## 🛠️ Technical Moat

* **Deterministic Math:** Custom FFT/DCT implementation ensures 100% parity between Python training and JavaScript
  inference.
* **Smart Smoothing:** Uses a moving average of predictions and a "Clarity Gap" threshold to prevent flickering results.
* **Optimized Audio:** Utilizes `AudioWorklet` for low-latency resampling.

---

## 📜 License & Compliance

Distributed under the **GPL-3.0-or-later** license.

**Privacy Note:** This app does not record, store, or transmit any audio data.
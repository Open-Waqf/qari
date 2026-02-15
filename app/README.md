# 📱 Qari Finder App

The frontend application for Qari Finder. It handles microphone input, audio visualization, feature extraction, and
neural network inference.

## 🧠 How it Works

The app mimics the Python training pipeline inside the browser using **WebAssembly** and **WebGL**:

1. **The Ear (Microphone):** Captures raw audio at 48kHz.
2. **The Resampler (AudioWorklet):** Downsamples audio to 16kHz in a separate thread to match the AI's training data.
3. **The Visualizer (Canvas):** Draws the raw waveform in real-time.
4. **The Extractor (Custom FFT):** Converts 3-second audio clips into Spectrogram images (40x186 pixels) using custom
   matrix math.
5. **The Brain (TFJS):** Feeds the image into a CNN model to predict the reciter.

## 🏗️ Architecture

* `src/main.ts`: Entry point.
* `src/services/audio-manager.ts`: Manages Mic permissions and Audio Context.
* `src/services/inference-engine.ts`: Runs the TensorFlow model loop.
* `src/lib/audio/custom-extractor.ts`: The "Math Bridge" that ensures JS numbers match Python numbers.
* `public/processors/resampler-processor.js`: The multi-threaded audio downsampler.

## 📦 Key Commands

```bash
# Start Development Server
npm run dev

# Build for Production
npm run build

# Preview Production Build
npm run preview

```

---

## Debug & Test Toolkit

### Enable debug logging

Append `?debug=1` to the app URL to enable verbose logs inside the inference engine (feature stats, gate stats, top-k,
etc).

Example:

* `http://localhost:5173/?debug=1`

### Debug Hotkeys (Shift + …)

| Hotkey      | What it does                                                                                                                                                   | Output file                           | What it validates                                                                                                   |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| **Shift+T** | **FILETEST**: run offline inference on a selected audio file using the same TFJS + extractor pipeline                                                          | none (you choose a file)              | “Model works on known audio” baseline (no mic / no worklet timing)                                                  |
| **Shift+M** | **MICCAP (post-worklet)**: capture ~8s of **16k PCM chunks coming out of the resampler worklet**, save WAV, then run sliding-window analysis (`runMicCapTest`) | `miccap_16k_<ts>.wav` (PCM WAV @ 16k) | What the model actually receives from the live pipeline (resampler + browser audio graph)                           |
| **Shift+R** | **RAW MIC RECORD (pre-worklet)**: record the browser’s `MediaStream` using `MediaRecorder` for ~8s                                                             | `raw_mic_<ts>.webm` / `.ogg` (Opus)   | What the mic stream sounds like **before** our worklet/resampler/graph (also captures browser processing artifacts) |
| **Shift+L** | **FILE LOOPBACK**: play an audio file through an `AudioContext` + worklet to simulate “live” resampling, then analyze                                          | none (you choose a file)              | Resampler correctness and “live-like” transport without real mic hardware                                           |
| **Shift+F** | Toggle Far-Field mode (echo cancel / NS / AGC) and request engine restart                                                                                      | none                                  | Whether browser-side processing is causing domain shift                                                             |
| **Shift+N** | Toggle RMS normalization in the inference pipeline                                                                                                             | none                                  | Sensitivity to level differences / gain                                                                             |
| **Shift+E** | Toggle pre-emphasis in the inference pipeline                                                                                                                  | none                                  | Sensitivity to high-freq emphasis                                                                                   |
| **Shift+C** | Toggle CMVN inside feature extraction                                                                                                                          | none                                  | Sensitivity to per-clip normalization                                                                               |

### What each recorder captures (important)

* **Shift+R (Raw mic)** records the **browser MediaStream** (compressed Opus). This is *pre-worklet*. Great for
  listening to “what the browser mic stream sounds like”.
* **Shift+M (MicCap16k)** captures the **post-worklet 16k Float32 chunks** (the same chunks fed into
  `handleIncomingAudio`). This is the best “what the model actually saw” capture.
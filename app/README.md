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

## 🧪 Debug Hotkeys (Audio + Inference)

These hotkeys are meant to isolate *where* errors come from: file decode, resampling, live mic capture, or model DSP
flags.

### Inference DSP toggles (affect model input)

- **Shift+N**: Toggle RMS Normalize (inferenceEngine)
- **Shift+E**: Toggle Pre-Emphasis (inferenceEngine)
- **Shift+C**: Toggle CMVN (inferenceEngine)
- **Shift+F**: Toggle Far Field (AudioManager)

> Tip: Keep flags constant during comparisons. CMVN can drastically change input distribution unless the model was
> trained with CMVN.

### Mic capture / recording tools

- **Shift+R**: Record **RAW mic stream** using `MediaRecorder` on `getUserMedia()` stream  
  Output: `raw_mic_<ts>.webm` or `.ogg` (Opus).  
  Purpose: hear/test what the browser + mic capture produces *before* the worklet pipeline.

- **Shift+M**: MICCAP capture of **post-worklet 16k PCM** chunks  
  First press starts capture (8s). Second press stops/saves.  
  Output: `miccap_16k_<ts>.wav` (PCM16 WAV @ 16k). Also runs `runMicCapTest()` on the captured signal.  
  Purpose: capture exactly what the model receives in the live pipeline.

### File-based evaluation

- **Shift+T**: FILETEST – run inference on a selected audio file
    - Decodes file -> converts to mono -> resamples to 16k -> scans 3s windows.
    - If the selected file is a direct 16k WAV, it uses the direct WAV fast-path.

### Pipeline simulation

- **Shift+L**: LOOPBACK – play a file through a forced 48k AudioContext + worklet, collect chunks, downsample to 16k,
  then run `runMicCapTest()`.
  Purpose: stress-test resampling / buffering like live mode.
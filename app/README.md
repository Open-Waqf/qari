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
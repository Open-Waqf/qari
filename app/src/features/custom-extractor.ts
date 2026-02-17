import * as tf from '@tensorflow/tfjs';

export class CustomAudioExtractor {
    private melBasis: tf.Tensor | null = null;
    private dctMatrix: tf.Tensor | null = null;
    private dftReal: tf.Tensor | null = null;
    private dftImag: tf.Tensor | null = null;
    private window: tf.Tensor | null = null;
    private isReady = false;

    async loadConfig(url: string = '/models/audio_config.json') {
        if (this.isReady) return;
        const res = await fetch(url);
        const data = await res.json();

        this.melBasis = tf.tensor(data.mel_basis);
        this.dctMatrix = tf.tensor(data.dct_matrix);
        this.dftReal = tf.tensor(data.dft_real);
        this.dftImag = tf.tensor(data.dft_imag);
        this.window = tf.tensor1d(data.window);
        this.isReady = true;
    }

    /**
     * Process signal -> MFCCs [40, Frames, 1]
     * @param opts.strictShape If true (default), forces 2.0s duration (171 frames). Set false for parity tests.
     */
    extractFullClip(signal: Float32Array, opts?: { cmvn?: boolean; strictShape?: boolean }): tf.Tensor {
        if (!this.isReady) throw new Error("Audio Config not loaded");

        const cmvn = !!opts?.cmvn;
        // 🟢 FIX: Default to TRUE for model safety, but allow FALSE for unit tests
        const strictShape = opts?.strictShape ?? true;

        return tf.tidy(() => {
            let processedSignal = signal;
            const TARGET_LEN = 44100;

            // 🟢 APPLY STRICT PADDING ONLY IF REQUESTED
            if (strictShape) {
                if (signal.length > TARGET_LEN) {
                    processedSignal = signal.subarray(0, TARGET_LEN);
                } else if (signal.length < TARGET_LEN) {
                    processedSignal = new Float32Array(TARGET_LEN);
                    processedSignal.set(signal);
                }
            }

            const frameSize = 512;
            const hopSize = 256;

            // Use processedSignal (which might be padded OR raw 512)
            const framesCount = Math.floor((processedSignal.length - frameSize) / hopSize) + 1;

            const flatBuffer = new Float32Array(framesCount * frameSize);
            for (let i = 0; i < framesCount; i++) {
                const start = i * hopSize;
                flatBuffer.set(processedSignal.subarray(start, start + frameSize), i * frameSize);
            }

            const signalTensor = tf.tensor2d(flatBuffer, [framesCount, frameSize]);
            const windowed = tf.mul(signalTensor, this.window!);
            const windowedT = windowed.transpose();

            const realPart = tf.matMul(this.dftReal!, windowedT);
            const imagPart = tf.matMul(this.dftImag!, windowedT);
            const mag = tf.sqrt(tf.add(tf.square(realPart), tf.square(imagPart)));

            const melEnergies = tf.matMul(this.melBasis!, mag);
            const logMel = tf.log(tf.add(melEnergies, 1e-6));

            let mfcc = tf.matMul(this.dctMatrix!, logMel); // [40, Frames]

            if (cmvn) {
                const mean = tf.mean(mfcc, 1, true);
                const var_ = tf.mean(tf.square(mfcc.sub(mean)), 1, true);
                const std = tf.sqrt(var_.add(1e-6));
                mfcc = mfcc.sub(mean).div(std);
            }

            return mfcc.expandDims(-1);
        });
    }

}

/**
 * Builds a windowed-sinc FIR lowpass filter kernel (Blackman window).
 * Ported from ResamplerProcessor.js for parity.
 */
function buildFIR(fs: number, cutoffHz: number, taps = 63): Float32Array {
    const cutoff = Math.min(cutoffHz, 0.45 * fs);
    const M = taps - 1;
    const fc = cutoff / fs;
    const h = new Float32Array(taps);
    let sum = 0;

    for (let i = 0; i < taps; i++) {
        const n = i - M / 2;
        const sinc = (n === 0) ? 2 * fc : Math.sin(2 * Math.PI * fc * n) / (Math.PI * n);
        // Blackman window
        const w = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / M) + 0.08 * Math.cos((4 * Math.PI * i) / M);
        h[i] = sinc * w;
        sum += h[i];
    }
    // Normalize gain to 1.0
    for (let i = 0; i < taps; i++) h[i] /= sum;
    return h;
}

/**
 * 🟢 SYNCHRONOUS FIR RESAMPLER
 * Replaces the naive linear interpolation with a proper Lowpass -> Interpolate chain.
 * This prevents aliasing in your debug/test paths.
 */
export function downsampleBuffer(buffer: Float32Array, inputRate: number, outputRate: number): Float32Array {
    if (inputRate === outputRate) return buffer;

    // 1. FIR Filtering (Anti-Aliasing)
    // Cutoff is 0.45 * Nyquist of the lower rate
    const cutoffHz = 0.45 * Math.min(inputRate, outputRate);
    const h = buildFIR(inputRate, cutoffHz);
    const taps = h.length;
    const halfTaps = Math.floor(taps / 2);

    const inputLen = buffer.length;
    const filtered = new Float32Array(inputLen);

    // Direct Convolution (OK for debug/tests on < 10s clips)
    for (let i = 0; i < inputLen; i++) {
        let sum = 0;
        for (let j = 0; j < taps; j++) {
            const tapIdx = j;
            // Align filter center with current sample
            const bufIdx = i - j + halfTaps;
            if (bufIdx >= 0 && bufIdx < inputLen) {
                sum += buffer[bufIdx] * h[tapIdx];
            }
        }
        filtered[i] = sum;
    }

    // 2. Linear Interpolation (Resampling)
    const ratio = inputRate / outputRate;
    const newLength = Math.round(inputLen / ratio);
    const result = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
        const originalPos = i * ratio;
        const idx = Math.floor(originalPos);
        const frac = originalPos - idx;

        const s0 = (idx >= 0 && idx < inputLen) ? filtered[idx] : 0;
        const s1 = (idx + 1 >= 0 && idx + 1 < inputLen) ? filtered[idx + 1] : 0;

        result[i] = s0 + (s1 - s0) * frac;
    }

    return result;
}

export const customExtractor = new CustomAudioExtractor();
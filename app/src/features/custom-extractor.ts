import * as tf from '@tensorflow/tfjs';

export class CustomAudioExtractor {
    private melBasis: tf.Tensor | null = null;
    private dctMatrix: tf.Tensor | null = null;
    private dftReal: tf.Tensor | null = null;
    private dftImag: tf.Tensor | null = null;
    private window: tf.Tensor | null = null;
    private isReady = false;

    // Memory Pooling. Allocate these once for the strict path.
    // 44100 length = 171 frames of 512 samples
    private pooledPaddedSignal = new Float32Array(44100);
    private pooledFlatBuffer = new Float32Array(171 * 512);

    async loadConfig(url: string = '/models/audio_config.bin') {
        if (this.isReady) return;
        const res = await fetch(url);
        const buffer = await res.arrayBuffer();

        // Wrap the raw memory in a Float32 view (instant, zero parsing)
        const flatData = new Float32Array(buffer);

        // 🟢 Slice the flat array back into our matrices based on known shapes
        let offset = 0;

        // 1. mel_basis [40, 257] -> 10,280 elements
        const melLen = 40 * 257;
        this.melBasis = tf.tensor2d(flatData.subarray(offset, offset + melLen), [40, 257]);
        offset += melLen;

        // 2. dct_matrix [40, 40] -> 1,600 elements
        const dctLen = 40 * 40;
        this.dctMatrix = tf.tensor2d(flatData.subarray(offset, offset + dctLen), [40, 40]);
        offset += dctLen;

        // 3. window [512] -> 512 elements
        const winLen = 512;
        this.window = tf.tensor1d(flatData.subarray(offset, offset + winLen));
        offset += winLen;

        // 4. dft_real [257, 512] -> 131,584 elements
        const dftLen = 257 * 512;
        this.dftReal = tf.tensor2d(flatData.subarray(offset, offset + dftLen), [257, 512]);
        offset += dftLen;

        // 5. dft_imag [257, 512] -> 131,584 elements
        this.dftImag = tf.tensor2d(flatData.subarray(offset, offset + dftLen), [257, 512]);

        this.isReady = true;
        console.log("⚡ Binary Audio Config Loaded Instantly!");
    }

    /**
     * Process signal -> MFCCs [40, Frames, 1]
     * @param opts.strictShape If true (default), forces 2.0s duration (171 frames). Set false for parity tests.
     */
    extractFullClip(signal: Float32Array, opts?: { cmvn?: boolean; strictShape?: boolean }): tf.Tensor {
        if (!this.isReady) throw new Error("Audio Config not loaded");

        const cmvn = !!opts?.cmvn;
        // Default to TRUE for model safety, but allow FALSE for unit tests
        const strictShape = opts?.strictShape ?? true;

        return tf.tidy(() => {
            let processedSignal = signal;
            const TARGET_LEN = 44100;

            // Use pooled array instead of `new Float32Array`
            if (strictShape) {
                if (signal.length > TARGET_LEN) {
                    processedSignal = signal.subarray(0, TARGET_LEN);
                } else if (signal.length < TARGET_LEN) {
                    this.pooledPaddedSignal.fill(0); // clear old data
                    this.pooledPaddedSignal.set(signal);
                    processedSignal = this.pooledPaddedSignal;
                }
            }

            const frameSize = 512;
            const hopSize = 256;
            const framesCount = Math.floor((processedSignal.length - frameSize) / hopSize) + 1;

            // Use pooled flat buffer if sizes match (they will 99.9% of the time in production)
            let currentFlatBuffer: Float32Array;
            if (framesCount === 171) {
                currentFlatBuffer = this.pooledFlatBuffer;
            } else {
                currentFlatBuffer = new Float32Array(framesCount * frameSize);
            }

            for (let i = 0; i < framesCount; i++) {
                const start = i * hopSize;
                currentFlatBuffer.set(processedSignal.subarray(start, start + frameSize), i * frameSize);
            }

            const signalTensor = tf.tensor2d(currentFlatBuffer, [framesCount, frameSize]);
            const windowed = tf.mul(signalTensor, this.window!);
            const windowedT = windowed.transpose();

            const realPart = tf.matMul(this.dftReal!, windowedT);
            const imagPart = tf.matMul(this.dftImag!, windowedT);
            const mag = tf.sqrt(tf.add(tf.square(realPart), tf.square(imagPart)));

            const melEnergies = tf.matMul(this.melBasis!, mag);
            const logMel = tf.log(tf.add(melEnergies, 1e-6));

            let mfcc = tf.matMul(this.dctMatrix!, logMel);

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
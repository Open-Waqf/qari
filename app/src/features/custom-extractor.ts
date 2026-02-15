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

    // Process a full 3-second clip (48000 samples) -> Image (40, 186)
    extractFullClip(signal: Float32Array, opts?: { cmvn?: boolean }): tf.Tensor {
        if (!this.isReady) throw new Error("Audio Config not loaded");

        const cmvn = !!opts?.cmvn;

        return tf.tidy(() => {
            const frameSize = 512;
            const hopSize = 256;
            const framesCount = Math.floor((signal.length - frameSize) / hopSize) + 1;

            const flatBuffer = new Float32Array(framesCount * frameSize);
            for (let i = 0; i < framesCount; i++) {
                const start = i * hopSize;
                flatBuffer.set(signal.subarray(start, start + frameSize), i * frameSize);
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

            // ✅ CMVN per coefficient across time (axis=1)
            if (cmvn) {
                const mean = tf.mean(mfcc, 1, true);
                const var_ = tf.mean(tf.square(mfcc.sub(mean)), 1, true);
                const std = tf.sqrt(var_.add(1e-6));
                mfcc = mfcc.sub(mean).div(std);
            }

            return mfcc.expandDims(-1); // [40, Frames, 1]
        });
    }

}

export function downsampleBuffer(buffer: Float32Array, inputRate: number, outputRate: number): Float32Array {
    if (inputRate === outputRate) return buffer;
    const ratio = inputRate / outputRate;
    const newLength = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
        const originalIndex = i * ratio;
        const index1 = Math.floor(originalIndex);
        const index2 = Math.ceil(originalIndex);
        const weight = originalIndex - index1;
        result[i] = (buffer[index1] || 0) * (1 - weight) + (buffer[index2] || 0) * weight;
    }
    return result;
}

export const customExtractor = new CustomAudioExtractor();
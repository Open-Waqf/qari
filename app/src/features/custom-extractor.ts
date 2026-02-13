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
    extractFullClip(signal: Float32Array): tf.Tensor {
        if (!this.isReady) throw new Error("Audio Config not loaded");

        return tf.tidy(() => {
            const frameSize = 512;
            const hopSize = 256;
            const framesCount = Math.floor((signal.length - frameSize) / hopSize) + 1;

            // 1. Manually create the 2D Tensor (Flatten first to please TS)
            const flatBuffer = new Float32Array(framesCount * frameSize);
            for (let i = 0; i < framesCount; i++) {
                const start = i * hopSize;
                flatBuffer.set(signal.subarray(start, start + frameSize), i * frameSize);
            }

            // Shape: [Frames, 512]
            const signalTensor = tf.tensor2d(flatBuffer, [framesCount, frameSize]);

            // 2. Apply Window
            const windowed = tf.mul(signalTensor, this.window!);

            // 3. FFT (Matrix Mult) - Transpose needed: [512, Frames]
            const windowedT = windowed.transpose();

            // [257, 512] @ [512, Frames] = [257, Frames]
            const realPart = tf.matMul(this.dftReal!, windowedT);
            const imagPart = tf.matMul(this.dftImag!, windowedT);

            const mag = tf.sqrt(tf.add(tf.square(realPart), tf.square(imagPart)));

            // 4. Mel [40, 257] @ [257, Frames] = [40, Frames]
            const melEnergies = tf.matMul(this.melBasis!, mag);
            const logMel = tf.log(tf.add(melEnergies, 1e-6));

            // 5. DCT [40, 40] @ [40, Frames] = [40, Frames]
            const mfcc = tf.matMul(this.dctMatrix!, logMel);

            // 6. Final Shape: [Height(40), Width(Time), 1]
            return mfcc.expandDims(-1);
        });
    }
}

export const customExtractor = new CustomAudioExtractor();
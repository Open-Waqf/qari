import * as tf from '@tensorflow/tfjs';
import {audioManager} from '../core/audio-manager';
import {customExtractor} from '../features/custom-extractor';

//run in console await window.checkParity("/test_sine.wav");
export async function checkAudioParity(input: string | number[]) {

    await tf.ready();
    // 1. Ensure Config is Loaded (Idempotent)
    await customExtractor.loadConfig();

    let signal: Float32Array;

    // === PATH A: Raw Data Injection (From Python) ===
    if (Array.isArray(input)) {
        // Python sends a standard array, we convert to Float32
        signal = new Float32Array(input);
    }
    // === PATH B: Manual File Test (From Console) ===
    else if (typeof input === 'string') {
        console.log(`💿 Loading File: ${input}`);
        const context = audioManager.context;
        const response = await fetch(input);
        const arrayBuffer = await response.arrayBuffer();
        const audioBuffer = await context.decodeAudioData(arrayBuffer);

        let rawData = audioBuffer.getChannelData(0);

        // Optional: Downsample if your file isn't 22050 (Simple check)
        // Note: Your extractor expects 22050 input.
        if (audioBuffer.sampleRate !== 22050) {
            console.warn("⚠️ Parity file is not 22050Hz. Results may drift.");
            // You can call downsampleBuffer here if you want to be strict
            // rawData = downsampleBuffer(rawData, audioBuffer.sampleRate, 22050);
        }

        // Slice first frame (512 samples) for direct comparison
        signal = rawData.slice(0, 512);
    } else {
        throw new Error("Invalid input. Expected string (url) or number[] (signal).");
    }

    // 2. Run the Extractor
    // We pass the signal to your class.
    // It returns a Tensor [40, Frames, 1]
    const tensorResult = customExtractor.extractFullClip(signal);

    // 3. Extract Data
    const values = await tensorResult.data();

    // Cleanup Tensors to avoid memory leaks during test loops
    tensorResult.dispose();

    console.log(`📤 Parity: Computed ${values.length} values.`);
    return Array.from(values);
}
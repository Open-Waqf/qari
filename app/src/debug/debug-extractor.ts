import * as tf from '@tensorflow/tfjs';
import {audioManager} from '../core/audio-manager';
import {customExtractor, downsampleBuffer} from '../features/custom-extractor';

export async function checkAudioParity(input: string | number[] = "/test_sine.wav") {
    // ... (Keep existing code unchanged) ...
    await tf.ready();
    await customExtractor.loadConfig();

    let signal: Float32Array;

    if (Array.isArray(input)) {
        signal = new Float32Array(input);
    } else if (typeof input === 'string') {
        console.log(`💿 Loading File: ${input}`);
        const context = audioManager.context;
        const response = await fetch(input);
        const arrayBuffer = await response.arrayBuffer();
        const audioBuffer = await context.decodeAudioData(arrayBuffer);

        let rawData = audioBuffer.getChannelData(0);
        if (audioBuffer.sampleRate !== 22050) {
            console.warn("⚠️ Parity file is not 22050Hz. Results may drift.");
            rawData = downsampleBuffer(rawData, audioBuffer.sampleRate, 22050) as any;
        }
        signal = rawData.slice(0, 512);
    } else {
        throw new Error("Invalid input. Expected string (url) or number[] (signal).");
    }

    const tensorResult = customExtractor.extractFullClip(signal, {strictShape: false});
    const values = await tensorResult.data();
    tensorResult.dispose();

    console.log(`📤 Parity: Computed ${values.length} values.`);
    return Array.from(values);
}

// ✅ UPDATED: Now returns a Report Object
export async function checkStrictParity() {
    await tf.ready();
    await customExtractor.loadConfig();

    const res = await fetch('/models/golden_parity.json');
    if (!res.ok) throw new Error("Golden Parity file missing! Run generate_golden.py first.");

    const golden = await res.json();
    const signal = new Float32Array(golden.signal);
    const expected = golden.expected_mfcc as number[][];

    // 1. Run JS Implementation
    const tensorRaw = customExtractor.extractFullClip(signal, {strictShape: false});

    // 2. Transpose
    const tensor = tensorRaw.transpose([1, 0, 2]);
    const actual = await tensor.array() as number[][][];

    tensorRaw.dispose();
    tensor.dispose();

    let maxError = 0;
    let totalError = 0;
    let count = 0;
    let failures = 0;

    // 3. Safety Check
    if (actual.length !== expected.length) {
        console.error(`❌ SHAPE MISMATCH: JS=${actual.length} frames, PY=${expected.length} frames`);
        failures++;
    }

    const framesToTest = Math.min(actual.length, expected.length);

    for (let i = 0; i < framesToTest; i++) {
        for (let j = 0; j < expected[0].length; j++) {
            const jsVal = actual[i][j][0];
            const pyVal = expected[i][j];

            const diff = Math.abs(jsVal - pyVal);
            maxError = Math.max(maxError, diff);
            totalError += diff;
            count++;

            if (diff > 0.5) {
                console.error(`❌ GROSS FAILURE at Frame ${i} Coeff ${j}: JS=${jsVal.toFixed(4)} PY=${pyVal.toFixed(4)}`);
                failures++;
                if (failures > 5) break;
            }
        }
    }

    const avgError = totalError / Math.max(1, count);

    // Log to console as well
    console.log(`📊 Parity Report: MaxErr=${maxError.toFixed(6)} AvgErr=${avgError.toFixed(6)}`);
    const passed = failures === 0 && maxError < 0.05;

    if (passed) {
        console.log("%c✅ PARITY PASSED: JS matches Python!", "color: #0f0; font-size: 14px; font-weight: bold;");
    } else {
        console.error("%c❌ PARITY FAILED: Precision drift too high.", "color: #f00; font-size: 14px; font-weight: bold;");
    }

    // ✅ RETURN REPORT
    return {
        passed,
        maxError,
        avgError,
        frameCount: framesToTest
    };
}
import * as tf from '@tensorflow/tfjs';
import {inferenceEngine} from '../model/inference-engine';

export async function runModelParityCheck() {
    try {
        console.log("🔍 Fetching golden parity data...");
        const res = await fetch('/models/model_parity.json');
        if (!res.ok) throw new Error("model_parity.json not found. Run export_model_parity.py first!");

        const data = await res.json();

        // Convert the flat array back into a TFJS Tensor
        const inputTensor = tf.tensor4d(data.input_mfcc, data.shape);

        // Ensure the TFJS model is loaded
        // 🟢 FIX: Check 'isReady' instead of 'isModelCached'
        if (!(inferenceEngine as any).isReady) {
            await inferenceEngine.setup();
        }

        console.log("🧠 Running TFJS prediction...");
        // @ts-ignore - reaching into the private model for testing
        const model = inferenceEngine.model;
        if (!model) throw new Error("Model failed to load.");

        const tfjsResult = model.predict(inputTensor) as tf.Tensor;
        const tfjsProbs = await tfjsResult.data();

        // Mathematically compare Keras vs TFJS
        const expected = data.expected_probs;
        let maxDiff = 0;

        for (let i = 0; i < expected.length; i++) {
            const diff = Math.abs(expected[i] - tfjsProbs[i]);
            if (diff > maxDiff) maxDiff = diff;
        }

        console.log(`📊 Max Drift: ${maxDiff.toFixed(8)}`);

        // Return the report data so the DebugPanel can display it nicely
        return {
            passed: maxDiff < 0.001,
            maxDiff: maxDiff
        };

    } catch (err: any) {
        console.error(err);
        throw err;
    }
}
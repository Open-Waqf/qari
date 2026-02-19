import * as tf from "@tensorflow/tfjs";
// Import all potential backends
import "@tensorflow/tfjs-backend-wasm";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-backend-cpu";
import {setWasmPaths} from "@tensorflow/tfjs-backend-wasm";

export async function forceWasmBackend(debug = false): Promise<void> {
    setWasmPaths("/tfjs-wasm/");

    // The order of preference: fastest to slowest
    const preferredBackends = ['wasm', 'webgl', 'cpu'];
    let activeBackend = '';

    for (const backend of preferredBackends) {
        try {
            if (debug) console.log(`🧠 Attempting to initialize backend: ${backend}...`);
            await tf.setBackend(backend);
            await tf.ready();

            activeBackend = tf.getBackend();

            // If it successfully set the backend we requested, break the loop
            if (activeBackend === backend) {
                if (debug) console.log(`✅ TFJS successfully locked to: ${activeBackend}`);
                break;
            }
        } catch (e) {
            if (debug) console.warn(`⚠️ Failed to initialize '${backend}':`, e);
        }
    }

    if (!activeBackend) {
        throw new Error("Critical: No TensorFlow backends could be initialized. The app cannot run.");
    }

    // Graceful warning rather than a hard crash
    if (activeBackend !== 'wasm') {
        console.warn(`Performance Warning: Running on '${activeBackend}' instead of 'wasm'. Inference may consume more battery.`);
    }
}
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-wasm";
import {setWasmPaths} from "@tensorflow/tfjs-backend-wasm";

export async function forceWasmBackend(debug = false): Promise<void> {
    // Serve from app origin (works for Vite dev, PWA, and Capacitor)
    setWasmPaths("/tfjs-wasm/");

    await tf.setBackend("wasm");
    await tf.ready();

    const backend = tf.getBackend();
    if (debug) console.log(`🧠 TFJS backend = ${backend}`);

    // No-regression rule: if not wasm, stop.
    if (backend !== "wasm") {
        throw new Error(`TFJS backend '${backend}' (expected 'wasm'). Check wasm files are served (no 404).`);
    }
}
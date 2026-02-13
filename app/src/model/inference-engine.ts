import * as tf from '@tensorflow/tfjs';
import {customExtractor} from '../features/custom-extractor';
import {RingBuffer} from './ring-buffer';

// CONSTANT for System ID (Do not translate this)
export const STATE_IDLE = "Analyzing...";

export class InferenceEngine {
    private static instance: InferenceEngine;
    private model: tf.LayersModel | null = null;
    private labels: string[] = [];
    private normalization = {mean: 0, std: 1};
    private buffer: RingBuffer;
    private lastPredictionTime = 0;
    private recentScores: number[][] = [];
    private silenceThreshold = 0.02; // Increased to ignore fans/AC
    private silenceCounter = 0;

    setThreshold(newThreshold: number) {
        this.silenceThreshold = Math.max(newThreshold, 0.002); // Never go below 0.002
        console.log(`🎯 New Sensitivity Floor: ${this.silenceThreshold.toFixed(5)}`);
    }

    private constructor() {
        this.buffer = new RingBuffer(3, 16000);
    }

    static getInstance(): InferenceEngine {
        if (!InferenceEngine.instance) {
            InferenceEngine.instance = new InferenceEngine();
        }
        return InferenceEngine.instance;
    }

    async setup(): Promise<boolean> {
        try {
            await tf.ready();

            // OPTIMIZATION: Use WebGL or WASM if available for speed
            if (tf.findBackend('wasm')) {
                await tf.setBackend('wasm');
                console.log("⚡ Using WASM Backend");
            } else if (tf.findBackend('webgl')) {
                await tf.setBackend('webgl');
                // Tweak WebGL for mobile performance
                tf.env().set('WEBGL_PACK', false);
                console.log("⚡ Using WebGL Backend");
            }

            await customExtractor.loadConfig();
            this.model = await tf.loadLayersModel('/models/tfjs_model/model.json');

            const res = await fetch('/models/reciters_map.json');
            this.labels = await res.json();

            try {
                const normRes = await fetch('/models/normalization.json');
                if (normRes.ok) {
                    this.normalization = await normRes.json();
                    console.log("📊 Stats:", this.normalization);
                }
            } catch (e) {
                console.warn("Using default stats");
            }

            return !!this.model;
        } catch (e) {
            console.error("Brain Offline:", e);
            return false;
        }
    }

    handleIncomingAudio(chunk: Float32Array) {
        let sum = 0;
        for (let i = 0; i < chunk.length; i++) sum += chunk[i] * chunk[i];
        const rms = Math.sqrt(sum / chunk.length);

        // 🔍 DEBUG LOG: See if your mic is loud enough
        console.log("🎤 Volume (RMS):", rms.toFixed(4));

        if (rms < this.silenceThreshold) {
            this.silenceCounter++;
            if (this.silenceCounter > 25) { // ~1 second of silence
                this.dispatchSilence();
                this.silenceCounter = 0;
                if (this.buffer.isFull) {
                    this.buffer.clear();
                    console.log("ｧｹ Buffer Cleared (Silence)");
                }
            }
            return;
        }

        this.silenceCounter = 0;
        this.buffer.write(chunk);

        if (this.buffer.isFull) {
            const now = Date.now();
            if (now - this.lastPredictionTime > 500) {
                this.lastPredictionTime = now;
                // 🔍 DEBUG LOG: Confirm the AI is firing
                console.log("🧠 Prediction Fired!");
                this.predict();
            }
        }
    }

    /**
     * Helper to tell the UI "Nothing is happening"
     */
    private dispatchSilence() {
        window.dispatchEvent(new CustomEvent('qari-found', {
            detail: {
                winner: {name: STATE_IDLE, score: 0},
                others: []
            }
        }));
    }

    private async predict() {
        if (!this.model) return;

        // FIX: Yield to Main Thread!
        // This tiny pause allows the UI to render "Analyzing..."
        // BEFORE the heavy math freezes the screen.
        await new Promise(resolve => setTimeout(resolve, 10));

        const signal = this.buffer.read();

        const prediction = tf.tidy(() => {
            const input = customExtractor.extractFullClip(signal);
            const batch = input.expandDims(0);

            const mean = this.normalization.mean || -0.77;
            const std = this.normalization.std || 5.20;

            const normalized = batch.sub(mean).div(std);
            return this.model!.predict(normalized) as tf.Tensor;
        });

        const probs = await prediction.data();
        prediction.dispose();

        // 1. Smoothing
        this.recentScores.push(Array.from(probs));
        if (this.recentScores.length > 5) this.recentScores.shift();

        const averagedProbs = new Array(probs.length).fill(0);
        for (const scoreArr of this.recentScores) {
            for (let i = 0; i < scoreArr.length; i++) {
                averagedProbs[i] += scoreArr[i];
            }
        }
        for (let i = 0; i < averagedProbs.length; i++) {
            averagedProbs[i] /= this.recentScores.length;
        }

        const allMatches = this.labels.map((name, i) => ({
            name: name.replace(/_/g, ' ').toUpperCase(),
            score: averagedProbs[i]
        }));

        const sorted = allMatches.sort((a, b) => b.score - a.score);
        const winner = sorted[0];
        const runnerUp = sorted[1];

        const isConfident = winner.score > 0.40;
        const isClear = runnerUp ? (winner.score - runnerUp.score > 0.10) : true;

        let finalWinner;
        if (isConfident && isClear) {
            finalWinner = winner;
        } else {
            // Return IDLE state if unsure, so UI keeps listening
            finalWinner = {name: STATE_IDLE, score: winner.score};
        }

        window.dispatchEvent(new CustomEvent('qari-found', {
            detail: {
                winner: finalWinner,
                others: sorted.slice(0, 3).filter(m => m.score > 0.05)
            }
        }));
    }
}

export const inferenceEngine = InferenceEngine.getInstance();
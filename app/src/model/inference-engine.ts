import * as tf from '@tensorflow/tfjs';
import {customExtractor} from '../features/custom-extractor';
import {RingBuffer} from './ring-buffer';

export class InferenceEngine {
    private static instance: InferenceEngine;
    private model: tf.LayersModel | null = null;
    private labels: string[] = [];
    private normalization = {mean: 0, std: 1}; // Default values
    private buffer: RingBuffer;
    private lastPredictionTime = 0;
    private recentScores: number[][] = [];
    private silenceThreshold = 0.005;

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
            // Ensure TFJS is ready
            await tf.ready();

            // 1. Load Audio Physics
            await customExtractor.loadConfig();

            // 2. Load the Model
            // No cache busting as requested
            this.model = await tf.loadLayersModel('/models/tfjs_model/model.json');

            // 3. Load Labels (Deterministic Array)
            const res = await fetch('/models/reciters_map.json');
            this.labels = await res.json();

            // 4. Load Normalization
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

    // app/src/model/inference-engine.ts

    handleIncomingAudio(chunk: Float32Array) {
        let sum = 0;
        for (let i = 0; i < chunk.length; i++) sum += chunk[i] * chunk[i];
        const rms = Math.sqrt(sum / chunk.length);

        // 🔍 DEBUG LOG: See if your mic is loud enough
        console.log("🎤 Volume (RMS):", rms.toFixed(4));

        if (rms < this.silenceThreshold) {
            if (this.buffer.isFull) {
                this.buffer.clear();
                console.log("🧹 Buffer Cleared due to silence");
            }
            return;
        }

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

    private async predict() {
        if (!this.model) return;

        const signal = this.buffer.read();

        const prediction = tf.tidy(() => {
            const input = customExtractor.extractFullClip(signal);
            const batch = input.expandDims(0);

            // Use loaded normalization (or default to safe values)
            const mean = this.normalization.mean || -0.77;
            const std = this.normalization.std || 5.20;

            const normalized = batch.sub(mean).div(std);
            return this.model!.predict(normalized) as tf.Tensor;
        });

        const probs = await prediction.data();
        prediction.dispose();

        // --- 1. SMOOTHING (Moving Average) ---
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

        // --- 2. SMART DECISION LOGIC ---
        const sorted = allMatches.sort((a, b) => b.score - a.score);
        const winner = sorted[0];
        const runnerUp = sorted[1];

        // Rule A: Minimum Confidence (Must be at least 40% sure)
        const isConfident = winner.score > 0.40;

        // Rule B: Clarity Gap (Must beat the runner-up by at least 10%)
        // Prevents flickering when the AI is "confused" between two similar voices
        const isClear = runnerUp ? (winner.score - runnerUp.score > 0.10) : true;

        let finalWinner;

        if (isConfident && isClear) {
            finalWinner = winner;
        } else {
            // If unsure, show "..." or the top guess but faded
            finalWinner = {name: "Analyzing...", score: winner.score};
        }

        // 3. Dispatch
        window.dispatchEvent(new CustomEvent('qari-found', {
            detail: {
                winner: finalWinner,
                others: sorted.slice(0, 3).filter(m => m.score > 0.05)
            }
        }));
    }
}

export const inferenceEngine = InferenceEngine.getInstance();
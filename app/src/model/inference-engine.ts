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
    private noiseFloor = 0.005;     // adaptive baseline
    private pendingWinner: string | null = null;
    private pendingCount = 0;
    private stableWinner: { name: string; score: number } | null = null;
    private REQUIRED_STABILITY = 2;

    private entropy(probs: number[]): number {
        let h = 0;
        for (const p of probs) {
            if (p > 0) h -= p * Math.log(p); // natural log
        }
        return h;
    }

    private entropyNormalized(probs: number[]): number {
        const n = probs.length;
        if (n <= 1) return 0;
        const h = this.entropy(probs);
        return h / Math.log(n); // 0..1
    }

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
        // RMS
        let sum = 0;
        for (let i = 0; i < chunk.length; i++) sum += chunk[i] * chunk[i];
        const rms = Math.sqrt(sum / chunk.length);

        console.log("🎤 Volume (RMS):", rms.toFixed(4));

        // Adaptive noise floor update (only update when quiet-ish)
        // This makes it work across quiet/loud phones.
        if (rms < this.noiseFloor * 1.5) {
            this.noiseFloor = 0.995 * this.noiseFloor + 0.005 * rms;
        }

        // Dynamic speech gate
        const gate = Math.max(this.silenceThreshold, this.noiseFloor * 3.0);
        const isSpeech = rms >= gate;

        if (!isSpeech) {
            this.silenceCounter++;
            if (this.silenceCounter > 25) {
                this.dispatchSilence();
                this.resetState();
                console.log("ｧｹ Buffer Cleared (Silence)");
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

    private resetState() {
        this.buffer.clear();
        this.recentScores = [];
        this.pendingWinner = null;
        this.pendingCount = 0;
        this.stableWinner = null;
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
        await new Promise(r => setTimeout(r, 0));

        const signal = this.buffer.read();

        const prediction = tf.tidy(() => {
            const input = customExtractor.extractFullClip(signal);
            const batch = input.expandDims(0);
            const mean = this.normalization.mean || -0.77;
            const std = this.normalization.std || 5.20;
            return this.model!.predict(batch.sub(mean).div(std)) as tf.Tensor;
        });

        const probs = Array.from(await prediction.data());
        prediction.dispose();

        // smoothing
        this.recentScores.push(probs);
        if (this.recentScores.length > 5) this.recentScores.shift();

        const avg = new Array(probs.length).fill(0);
        for (const s of this.recentScores) for (let i = 0; i < s.length; i++) avg[i] += s[i];
        for (let i = 0; i < avg.length; i++) avg[i] /= this.recentScores.length;

        // gating inputs
        const ent = this.entropyNormalized(avg);

        const matches = this.labels.map((name, i) => ({
            name: name.replace(/_/g, ' ').toUpperCase(),
            score: avg[i],
        })).sort((a, b) => b.score - a.score);

        const top1 = matches[0];
        const top2 = matches[1];
        const ratio = top2?.score ? (top1.score / top2.score) : Infinity;
        const diff = top2 ? (top1.score - top2.score) : top1.score;

        // tuned gates (good starting points)
        const notConfused = ent < 0.75;          // 0..1
        const strongTop1 = top1.score > 0.45;
        const clearWin = (ratio > 1.25) && (diff > 0.08);

        let winner = {name: STATE_IDLE, score: 0};

        if (notConfused && strongTop1 && clearWin) {
            // stability / hysteresis
            if (this.pendingWinner === top1.name) {
                this.pendingCount++;
            } else {
                this.pendingWinner = top1.name;
                this.pendingCount = 1; // <-- key fix
            }

            if (this.pendingCount >= this.REQUIRED_STABILITY) {
                this.stableWinner = top1;
            }

            winner = this.stableWinner ?? {name: STATE_IDLE, score: top1.score};
        } else {
            // reset pending if confused/weak
            this.pendingWinner = null;
            this.pendingCount = 0;
            winner = this.stableWinner ?? {name: STATE_IDLE, score: 0};
        }

        window.dispatchEvent(new CustomEvent('qari-found', {
            detail: {
                winner,
                others: matches.slice(0, 3).filter(m => m.score > 0.05)
            }
        }));
    }
}

export const inferenceEngine = InferenceEngine.getInstance();
import * as tf from '@tensorflow/tfjs';
import {customExtractor} from '../features/custom-extractor';
import {RingBuffer} from './ring-buffer';
import {EVENTS, type QariMatch} from '../core/events';
import {audioManager} from "../core/audio-manager.ts";
import {forceWasmBackend} from "./tf-backend";

export const STATE_IDLE = 'Analyzing...';

/**
 * Dummy SpecAugment layer for TensorFlow.js.
 */
class SpecAugment extends tf.layers.Layer {
    constructor(config: any) {
        super(config);
    }

    call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor | tf.Tensor[] {
        return Array.isArray(inputs) ? inputs[0] : inputs;
    }

    static get className() {
        return 'SpecAugment';
    }
}

interface EngineConfig {
    // DSP / gains
    targetRms: number;
    minGain: number;
    maxGain: number;

    // Voice gating
    silenceThreshold: number;
    snrOn: number;
    snrOff: number;
    voicedHangMax: number;

    // Timing
    predictionIntervalMs: number;

    // Stability
    stabilityThreshold: number;
    smoothingFrames: number;
    dispatchMinIntervalMs: number;
    noiseAdaptAlpha: number;
}

tf.serialization.registerClass(SpecAugment);

class InferenceEngine {
    private static instance: InferenceEngine;

    // --- Dependencies ---
    private model: tf.GraphModel | null = null;
    private labels: string[] = [];
    private normalization = {
        mean: -0.5885211229324341,
        std: 5.6245293617248535
    };
    private buffer: RingBuffer;

    private gateScratch: Float32Array | null = null;

    private acceptTop1Min = 0.14;

    // --- Debug State ---
    // ✅ NEW: Auto-detect debug mode from URL
    private isDebug = typeof window !== 'undefined' && window.location.search.includes('debug=1');

    // --- Configuration ---
    private config: EngineConfig = {
        // ✅ CHANGED: Unlocked Gain Settings
        targetRms: 0.10,
        minGain: 0.6,
        maxGain: 2.5,

        silenceThreshold: 0.002,
        snrOn: 2.0,
        snrOff: 1.4,
        voicedHangMax: 15,

        predictionIntervalMs: 500,

        stabilityThreshold: 3,
        smoothingFrames: 8,
        dispatchMinIntervalMs: 150,
        noiseAdaptAlpha: 0.005,
    };

    /**
     * Sets the MODEL CONFIDENCE threshold (0.0 - 1.0).
     * How sure must the AI be to trigger a match?
     */
    public setConfidenceThreshold(p: number) {
        if (!Number.isFinite(p)) return;
        // Clamp to avoid accidentally setting it to 0 (which accepts everything)
        this.acceptTop1Min = Math.min(0.99, Math.max(0.10, p));

        if (this.isDebug) {
            console.log(`🎚️ Model Confidence Threshold set to ${(this.acceptTop1Min * 100).toFixed(1)}%`);
        }
    }

    public getConfidenceThreshold() {
        return this.acceptTop1Min;
    }

    /**
     * Sets the AUDIO GATE noise floor (RMS Amplitude).
     * Sounds below this relative level are ignored as silence.
     */
    public setNoiseFloor(noiseFloorRms: number) {
        if (!Number.isFinite(noiseFloorRms)) return;

        // Clamp to sane values (0.0003 is dead silent, 0.05 is a loud coffee shop)
        const nf = Math.min(0.05, Math.max(0.0003, noiseFloorRms));

        // 1. Update the baseline noise floor
        this.gate.noiseFloor = nf;

        this.gate.calibrated = true;

        // 2. Reset the gate counters so we don't get stuck in "Voiced" mode
        this.gate.isVoiced = false;
        this.gate.hangCounter = 0;

        if (this.isDebug) {
            console.log(`🔇 Noise Floor calibrated to RMS: ${this.gate.noiseFloor.toFixed(5)}`);
        }
    }

    // --- Feature flags ---
    private flags = {
        rmsNormalize: true,
        preEmphasis: false,
        cmvn: false,
    };

    public getCurrentRms(): number {
        // You need to store the last calculated RMS in a class property
        // In handleIncomingAudio, assign `this.lastRms = rms;`
        return this.lastRms || 0;
    }

    private lastRms = 0;

    // --- DSP State ---
    private filterState = {x1: 0, y1: 0};
    private readonly HP_COEFF = 0.90;

    // --- Runtime ---
    private gate = {
        noiseFloor: 0.0015,
        calibrated: false,
        isVoiced: false,
        hangCounter: 0,
        silenceCounter: 0,
    };

    private state = {
        isPredicting: false,
        lastPredictionTime: 0,
        recentScores: [] as number[][],
        pendingWinner: null as string | null,
        pendingCount: 0,
        stableWinner: null as QariMatch | null,
        lastDispatchWinner: '',
        lastDispatchTime: 0,
        noWinCount: 0,
    };

    // --- Debug ---
    private debug = {
        written: 0,
        skipped: 0,
        lastGateLogTime: 0,
        lastRateLogTime: 0,
        topLogLast: 0,
        topLogInterval: 1000,
    };

    private constructor() {
        this.buffer = new RingBuffer(2, 22050);
    }

    static getInstance(): InferenceEngine {
        if (!InferenceEngine.instance) InferenceEngine.instance = new InferenceEngine();
        return InferenceEngine.instance;
    }

    // =========================================
    // Setup & Configuration
    // =========================================

    async setup(): Promise<boolean> {
        try {
            await forceWasmBackend(this.isDebug);
            await customExtractor.loadConfig();
            this.model = await tf.loadGraphModel('/models/tfjs_model/model.json');

            const res = await fetch('/models/reciters_map.json');
            if (!res.ok) throw new Error(`Reciters map missing (${res.status})`);
            this.labels = await res.json();

            // 🟢 HARD-FAIL: Ensure normalization is loaded and valid
            const normRes = await fetch('/models/normalization.json');
            if (!normRes.ok) {
                throw new Error(`CRITICAL: normalization.json missing (Status: ${normRes.status})`);
            }

            const stats = await normRes.json();
            if (!Number.isFinite(stats.mean) || !Number.isFinite(stats.std)) {
                throw new Error("CRITICAL: Normalization stats contain non-finite values.");
            }
            this.normalization = stats;

            return !!this.model;
        } catch (e) {
            console.error('Brain Offline:', e);
            return false;
        } finally {
            if (this.isDebug) this.logConfigStatus();
        }
    }

    private logConfigStatus() {
        console.log(`🧪 CMVN: ${this.flags.cmvn ? "ON" : "OFF"}`);
        console.log(`🧪 Far Field Mode: ${audioManager.isFarFieldMode() ? "ON" : "OFF"}`);
        console.log(`🧪 PreEmphasis: ${this.flags.preEmphasis ? "ON" : "OFF"}`);
        console.log(`🧪 RMS Normalized: ${this.flags.rmsNormalize ? "ON" : "OFF"}`);
        console.log(`🧪 acceptTop1Min: ${(this.acceptTop1Min * 100).toFixed(1)}%`);
    }

    public reset() {
        this.resetState();
        this.emitIdle();
        if (this.isDebug) console.log('🧹 Inference Engine State Reset');
    }

    private resetState() {
        this.gate.silenceCounter = 0;
        this.gate.isVoiced = false;
        this.gate.hangCounter = 0;
        this.filterState = {x1: 0, y1: 0};
        this.buffer.clear();
        this.state.recentScores = [];
        this.state.pendingWinner = null;
        this.state.pendingCount = 0;
        this.state.stableWinner = null;
        this.state.lastDispatchWinner = '';
        this.state.lastDispatchTime = 0;
        this.state.noWinCount = 0;
    }

    // =========================================
    // Toggles
    // =========================================

    public isRmsNormalizeEnabled() {
        return this.flags.rmsNormalize;
    }

    public toggleRmsNormalize() {
        this.flags.rmsNormalize = !this.flags.rmsNormalize;
    }

    public isPreEmphasisEnabled() {
        return this.flags.preEmphasis;
    }

    public togglePreEmphasis() {
        this.flags.preEmphasis = !this.flags.preEmphasis;
    }

    public isCmvnEnabled() {
        return this.flags.cmvn;
    }

    public toggleCMVN() {
        this.flags.cmvn = !this.flags.cmvn;
    }

    // =========================================
    // Core DSP Logic
    // =========================================

    private applyHighPassInPlace(chunk: Float32Array) {
        let {x1, y1} = this.filterState;
        const R = this.HP_COEFF;
        for (let i = 0; i < chunk.length; i++) {
            const x0 = chunk[i];
            const y0 = x0 - x1 + R * y1;
            chunk[i] = y0;
            x1 = x0;
            y1 = y0;
        }
        this.filterState = {x1, y1};
    }

    private calculateRms(x: Float32Array): number {
        let sum = 0;
        for (let i = 0; i < x.length; i++) sum += x[i] * x[i];
        return Math.sqrt(sum / Math.max(1, x.length));
    }

    private normalizeSignal(x: Float32Array, floor = 0.002): { y: Float32Array; gain: number; rms: number } {
        const r = this.calculateRms(x);
        if (r < floor) return {y: x, gain: 1, rms: r};

        let g = this.config.targetRms / r;
        g = Math.min(Math.max(g, this.config.minGain), this.config.maxGain);

        if (Math.abs(g - 1) < 1e-3) return {y: x, gain: g, rms: r};

        const y = new Float32Array(x.length);
        for (let i = 0; i < x.length; i++) {
            const v = x[i] * g;
            // tanh-ish soft clip
            y[i] = v / (1 + Math.abs(v));
        }
        return {y, gain: g, rms: r};
    }

    private applyPreEmphasis(x: Float32Array): Float32Array {
        const y = new Float32Array(x.length);
        if (x.length === 0) return y;
        y[0] = x[0];
        const a = 0.95;
        for (let i = 1; i < x.length; i++) y[i] = x[i] - a * x[i - 1];
        return y;
    }

    // =========================================
    // 🧠 Shared Inference Pipeline (The Core Fix)
    // =========================================

    private async runInferencePipeline(signal: Float32Array, log: boolean = false): Promise<number[]> {
        // 1. RMS Normalization
        let processed = signal;
        if (this.flags.rmsNormalize) {
            const n = this.normalizeSignal(processed);
            processed = n.y;
            if (log) console.log(`🎚️ Gain:${n.gain.toFixed(2)}x (RMS:${n.rms.toFixed(4)})`);
        }

        // 2. Pre-emphasis
        if (this.flags.preEmphasis) {
            processed = this.applyPreEmphasis(processed);
        }

        // 3. TF.js Execution
        const prediction = tf.tidy(() => {
            // A. Feature Extraction
            const input = customExtractor.extractFullClip(processed, {cmvn: this.flags.cmvn});

            // 📊 DEBUG: Raw Feature Stats
            if (log) {
                const {mean, variance} = tf.moments(input);
                console.log(`📊 RAW Features: Mean=${mean.dataSync()[0].toFixed(2)} | Std=${Math.sqrt(variance.dataSync()[0]).toFixed(2)}`);
            }

            const batch = input.expandDims(0);

            // B. Normalization (ALWAYS Apply this!)
            const {mean, std} = this.normalization;
            const normalized = batch.sub(mean).div(Math.max(std, 1e-6));

            // 🧠 DEBUG: Model Input Stats (Should be ~0.0 and ~1.0)
            if (log) {
                const {mean: m, variance: v} = tf.moments(normalized);
                console.log(`🧠 MODEL INPUT: Mean=${m.dataSync()[0].toFixed(2)} | Std=${Math.sqrt(v.dataSync()[0]).toFixed(2)}`);
            }

            return this.model!.predict(normalized) as tf.Tensor;
        });

        const probs = Array.from(await prediction.data());
        prediction.dispose();
        return probs;
    }

    // =========================================
    // Live Audio Handling
    // =========================================

    handleIncomingAudio(rawChunk: Float32Array) {
        // ✅ Only log if debug mode is active
        if (this.isDebug && this.debug.lastRateLogTime === 0) {
            console.log(`📦 chunkLen=${rawChunk.length} (expect 4096 @22k)`);
            this.debug.lastRateLogTime = 1;
        }

        // 🛑 TRUST WORKLET: It delivers 22050
        const chunk = rawChunk;

        // 🛑 PARITY: Filter only for GATE, not for FEATURES
        if (!this.gateScratch || this.gateScratch.length !== chunk.length) {
            this.gateScratch = new Float32Array(chunk.length);
        }
        this.gateScratch.set(chunk);
        this.applyHighPassInPlace(this.gateScratch);

        const now = Date.now();
        const rms = this.calculateRms(this.gateScratch);
        this.lastRms = rms;

        // Adaptive Noise Floor
        // Calibration-aware gate
        // Adaptive Noise Floor (two-speed + never drift below intended minimum)
        const minGate = this.config.silenceThreshold;          // ✅ ALWAYS keep this safety floor
        const minNoiseFloor = minGate / this.config.snrOn;     // ✅ ensures gateOn >= silenceThreshold

        const aDown = this.config.noiseAdaptAlpha;             // fast down
        const aUp = aDown * 0.15;                              // slow up (tune 0.10–0.25)

        // Provisional gates from current floor
        let gateOn = Math.max(minGate, this.gate.noiseFloor * this.config.snrOn);
        let gateOff = Math.max(minGate * 0.7, this.gate.noiseFloor * this.config.snrOff);

        // Learn noise floor only from "noise-like" frames
        const noiseLike = (!this.gate.isVoiced) || (rms < gateOff);

        if (noiseLike) {
            const target = rms;
            const alpha = target < this.gate.noiseFloor ? aDown : aUp;
            this.gate.noiseFloor = (1 - alpha) * this.gate.noiseFloor + alpha * target;

            // ✅ clamp to prevent drift-too-low and insane highs
            this.gate.noiseFloor = Math.min(0.05, Math.max(minNoiseFloor, this.gate.noiseFloor));
        } else {
            // ✅ still enforce minimum
            this.gate.noiseFloor = Math.min(0.05, Math.max(minNoiseFloor, this.gate.noiseFloor));
        }

        // Recompute gates after potential update
        gateOn = Math.max(minGate, this.gate.noiseFloor * this.config.snrOn);
        gateOff = Math.max(minGate * 0.7, this.gate.noiseFloor * this.config.snrOff);


        if (this.isDebug && now - this.debug.lastGateLogTime > 2000) {
            console.log(`🎤 RMS:${rms.toFixed(4)} Gate:${gateOn.toFixed(4)} Noise:${this.gate.noiseFloor.toFixed(4)}`);
            this.debug.lastGateLogTime = now;
        }

        // Gate Logic
        if (!this.gate.isVoiced) {
            if (rms < gateOn) {
                this.gate.silenceCounter++;
                if (this.gate.silenceCounter > 25) {
                    this.emitIdle();
                    this.resetState();
                }
                return;
            }
            this.gate.isVoiced = true;
            this.gate.hangCounter = this.config.voicedHangMax;
        } else {
            if (rms < gateOff) {
                this.gate.hangCounter--;
                if (this.gate.hangCounter <= 0) {
                    this.gate.isVoiced = false;
                    this.gate.silenceCounter++;
                }
                return;
            } else {
                this.gate.hangCounter = this.config.voicedHangMax;
            }
        }

        this.gate.silenceCounter = 0;
        this.debug.written++;

        // 🛑 PARITY: Write RAW 22050 to buffer (Unfiltered)
        this.buffer.write(chunk);

        // Trigger Prediction
        if (this.buffer.isFull && !this.state.isPredicting) {
            if (now - this.state.lastPredictionTime > this.config.predictionIntervalMs) {
                this.state.lastPredictionTime = now;
                void this.predict();
            }
        }
    }

    private async predict() {
        if (!this.model || this.state.isPredicting) return;
        this.state.isPredicting = true;

        try {
            await new Promise<void>(r => requestAnimationFrame(() => r()));
            const signal = this.buffer.read();

            // ✅ CALL SHARED PIPELINE (Log enabled only if debug=1)
            const probs = await this.runInferencePipeline(signal, this.isDebug);

            await this.handlePredictionResult(probs);
        } finally {
            this.state.isPredicting = false;
        }
    }

    private async handlePredictionResult(probs: number[]) {
        const now = Date.now();
        const d = this.decideWinnerFromProbs(probs);

        if (this.isDebug) {
            console.log(`🧮 top1=${d.top1.name}:${(d.top1.score * 100).toFixed(1)}% diff=${d.diff.toFixed(3)} ratio=${d.ratio.toFixed(2)} ent=${d.ent.toFixed(3)}`);
        }

        await new Promise<void>(r => requestAnimationFrame(() => r()));

        const winnerChanged = d.winner.name !== this.state.lastDispatchWinner;
        const enoughTime = now - this.state.lastDispatchTime > this.config.dispatchMinIntervalMs;

        if (winnerChanged || enoughTime) {
            this.state.lastDispatchWinner = d.winner.name;
            this.state.lastDispatchTime = now;
            window.dispatchEvent(new CustomEvent(EVENTS.RESULT_FOUND, {
                detail: {winner: d.winner, others: d.others, stable: d.stable}
            }));

        }
    }

    // =========================================
    // Offline / File Test
    // =========================================

    public async predictFromSignal(
        signal: Float32Array,
        opts?: { startSec?: number; windowSec?: number; dispatchToUI?: boolean; log?: boolean; independent?: boolean }
    ) {
        if (!this.model) throw new Error('Model not loaded.');

        const sr = 22050;
        const windowSec = opts?.windowSec ?? 2;
        const win = Math.max(1, Math.floor(windowSec * sr));
        const startSec = Math.max(0, opts?.startSec ?? 0);
        const start = Math.floor(startSec * sr);

        // Windowing
        const windowed = new Float32Array(win);
        if (signal.length >= win) {
            const sliceStart = Math.min(start, Math.max(0, signal.length - win));
            windowed.set(signal.subarray(sliceStart, sliceStart + win));
        } else {
            windowed.set(signal);
        }

        if (opts?.log) {
            console.log(`🧱 windowed.length=${windowed.length} (expected ${win})`);
        }

        // Run model
        const probs = await this.runInferencePipeline(windowed, opts?.log ?? false);

        // 1) RAW diagnostics (no smoothing / no stability)
        const raw = this.analyzeRawProbs(probs);

        if (opts?.log) {
            console.log(
                `🏁 FILETEST RAW | ${raw.top3.map(x => `${x.name}:${(x.score * 100).toFixed(0)}%`).join(' ')}`
            );
            console.log(
                `🧮 RAW top1=${raw.top1.name}:${(raw.top1.score * 100).toFixed(1)}% ` +
                `top2=${raw.top2?.name ?? "-"}:${((raw.top2?.score ?? 0) * 100).toFixed(1)}% ` +
                `diff=${raw.diff.toFixed(3)} ratio=${raw.ratio.toFixed(2)} ent=${raw.ent.toFixed(3)}`
            );
        }

        // 2) DECISION (same logic as live)
        // independent=true means each file window is treated like a fresh clip
        const independent = opts?.independent ?? true;
        if (independent) {
            this.state.recentScores = [];
            this.state.pendingWinner = null;
            this.state.pendingCount = 0;
            this.state.stableWinner = null;
            this.state.noWinCount = 0;
        }

        const decision = this.decideWinnerFromProbs(probs);

        if (opts?.log) {
            console.log(
                `✅ DECISION | winner=${decision.winner.name}:${(decision.winner.score * 100).toFixed(1)}% ` +
                `diff=${decision.diff.toFixed(3)} ratio=${decision.ratio.toFixed(2)} ent=${decision.ent.toFixed(3)}`
            );
        }

        // 3) Dispatch DECISION winner (not raw top1)
        if (opts?.dispatchToUI) {
            window.dispatchEvent(
                new CustomEvent(EVENTS.RESULT_FOUND, {
                    detail: {winner: decision.winner, others: decision.others},
                })
            );
        }

        // Return both so debug tools can choose
        return {probs, raw, decision};
    }

    private emitIdle() {
        requestAnimationFrame(() => {
            window.dispatchEvent(
                new CustomEvent(EVENTS.RESULT_FOUND, {
                    detail: {winner: {name: STATE_IDLE, score: 0}, others: []},
                })
            );
        });
    }

    private getTop3(probs: number[]) {
        let i1 = -1, i2 = -1, i3 = -1;
        let s1 = -1, s2 = -1, s3 = -1;
        for (let i = 0; i < probs.length; i++) {
            const p = probs[i];
            if (p > s1) {
                s3 = s2;
                i3 = i2;
                s2 = s1;
                i2 = i1;
                s1 = p;
                i1 = i;
            } else if (p > s2) {
                s3 = s2;
                i3 = i2;
                s2 = p;
                i2 = i;
            } else if (p > s3) {
                s3 = p;
                i3 = i;
            }
        }
        const result: QariMatch[] = [];
        if (i1 !== -1) result.push({name: this.formatName(this.labels[i1]), score: s1});
        if (i2 !== -1) result.push({name: this.formatName(this.labels[i2]), score: s2});
        if (i3 !== -1) result.push({name: this.formatName(this.labels[i3]), score: s3});
        return result;
    }

    private formatName(raw: string): string {
        return raw ? raw.replace(/_/g, ' ').toUpperCase() : 'UNKNOWN';
    }

    private entropyNormalized(probs: number[]): number {
        const N = probs.length;
        if (N <= 1) return 0;
        let e = 0;
        for (const p of probs) {
            if (p > 0) e -= p * Math.log(p);
        }
        return e / Math.log(N);
    }

    private decideWinnerFromProbs(probs: number[]) {
        this.state.recentScores.push(probs);
        if (this.state.recentScores.length > this.config.smoothingFrames) this.state.recentScores.shift();

        // avg
        const avg = new Array(probs.length).fill(0);
        for (const s of this.state.recentScores) for (let i = 0; i < s.length; i++) avg[i] += s[i];
        for (let i = 0; i < avg.length; i++) avg[i] /= this.state.recentScores.length;

        const ent = this.entropyNormalized(avg);
        const topMatches = this.getTop3(avg);

        const now = Date.now();
        if (this.isDebug && now - this.debug.topLogLast > this.debug.topLogInterval) {
            this.debug.topLogLast = now;
            const fmt = (m?: QariMatch) => (m ? `${m.name}:${(m.score * 100).toFixed(0)}%` : '-');
            console.log(`🏆 TOP3 [Ent:${ent.toFixed(2)}] | ${fmt(topMatches[0])} | ${fmt(topMatches[1])} | ${fmt(topMatches[2])}`);
        }

        const top1 = topMatches[0] ?? {name: STATE_IDLE, score: 0};
        const top2 = topMatches[1];

        const ratio = top2?.score ? top1.score / top2.score : Infinity;
        const diff = top2 ? top1.score - top2.score : top1.score;

        if (this.isDebug) {
            const t1 = top1?.score ?? 0;
            const t2 = top2?.score ?? 0;
            const ratio = t2 > 0 ? t1 / t2 : Infinity;
            const diff = t2 > 0 ? t1 - t2 : t1;
            console.log(
                `🧮 top1=${top1?.name}:${(t1 * 100).toFixed(1)}% ` +
                `top2=${top2?.name ?? "-"}:${(t2 * 100).toFixed(1)}% ` +
                `diff=${diff.toFixed(3)} ratio=${ratio.toFixed(2)}`
            );
        }

        // TEMP: allow high entropy while model is underconfident
        const notConfused = ent < 0.95;
        const strongTop1 = top1.score > this.acceptTop1Min;
        const clearWin = ratio > 1.35 && diff > 0.04;

        let winner: QariMatch = {name: STATE_IDLE, score: 0};

        if (notConfused && strongTop1 && clearWin) {
            this.state.noWinCount = 0;
            if (this.state.pendingWinner === top1.name) this.state.pendingCount++;
            else {
                this.state.pendingWinner = top1.name;
                this.state.pendingCount = 1;
            }

            if (this.state.pendingCount >= this.config.stabilityThreshold) this.state.stableWinner = top1;

            // IMPORTANT: show candidate if not yet stable
            winner = this.state.stableWinner ?? top1;
        } else {
            this.state.pendingWinner = null;
            this.state.pendingCount = 0;
            this.state.noWinCount++;
            if (this.state.noWinCount >= 4) {
                this.state.stableWinner = null;
                this.state.noWinCount = 0;
            }
            winner = this.state.stableWinner ?? {name: STATE_IDLE, score: 0};
        }

        const stable =
            winner.name !== STATE_IDLE &&
            this.state.stableWinner != null &&
            winner.name === this.state.stableWinner.name &&
            notConfused && strongTop1 && clearWin;

        return {
            winner,
            others: topMatches.filter(m => m.score > 0.05),
            ent,
            top1,
            top2,
            diff,
            ratio,
            stable,
        };
    }

    private analyzeRawProbs(probs: number[]) {
        const ent = this.entropyNormalized(probs);
        const top3 = this.getTop3(probs);
        const top1 = top3[0] ?? {name: STATE_IDLE, score: 0};
        const top2 = top3[1];

        const ratio = top2?.score ? top1.score / top2.score : Infinity;
        const diff = top2 ? top1.score - top2.score : top1.score;

        return {ent, top1, top2, top3, diff, ratio};
    }
}

export default InferenceEngine;
export const inferenceEngine = InferenceEngine.getInstance();
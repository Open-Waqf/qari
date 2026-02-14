import * as tf from '@tensorflow/tfjs';
import {customExtractor} from '../features/custom-extractor';
import {RingBuffer} from './ring-buffer';
import {EVENTS, type QariMatch} from '../core/events';
import {audioManager} from "../core/audio-manager.ts";

export const STATE_IDLE = 'Analyzing...';

/**
 * Central configuration for thresholds and gains.
 * Keep all “magic numbers” here so tuning is easy.
 */
interface EngineConfig {
    // DSP / gains
    targetRms: number;
    minGain: number;
    maxGain: number;
    preEmphasisAlpha: number;

    // Voice gating
    silenceThreshold: number; // floor clamp
    snrOn: number;            // enter voiced when rms >= noiseFloor * snrOn
    snrOff: number;           // stay voiced while rms >= noiseFloor * snrOff
    voicedHangMax: number;    // chunks to bridge gaps

    // Timing
    predictionIntervalMs: number;

    // Stability
    stabilityThreshold: number;

    // Smoothing window
    smoothingFrames: number;

    // Dispatch throttling
    dispatchMinIntervalMs: number;

    // Misc
    noiseAdaptAlpha: number; // small = slower adaption (e.g. 0.005)
}

class InferenceEngine {
    private static instance: InferenceEngine;

    // --- Core dependencies ---
    private model: tf.LayersModel | null = null;
    private labels: string[] = [];
    private normalization = {mean: 0, std: 1};
    private buffer: RingBuffer;

    // --- Configuration ---
    private config: EngineConfig = {
        targetRms: 0.10,
        minGain: 0.25,
        maxGain: 6.0,
        preEmphasisAlpha: 0.97,

        silenceThreshold: 0.006,
        snrOn: 2.0,
        snrOff: 1.4,
        voicedHangMax: 15,

        predictionIntervalMs: 500,

        stabilityThreshold: 2,
        smoothingFrames: 5,

        dispatchMinIntervalMs: 150,

        noiseAdaptAlpha: 0.005,
    };

    // --- Feature flags (toggles) ---
    private flags = {
        rmsNormalize: true,  // ON by default for mic robustness
        preEmphasis: false,  // OFF by default
        cmvn: false,
    };

    // --- Runtime: voice gate ---
    private gate = {
        noiseFloor: 0.0015,  // start low; adapts
        isVoiced: false,
        hangCounter: 0,
        silenceCounter: 0,
    };

    // --- Runtime: inference / stability / dispatch ---
    private state = {
        isPredicting: false,
        lastPredictionTime: 0,
        recentScores: [] as number[][],

        pendingWinner: null as string | null,
        pendingCount: 0,
        stableWinner: null as QariMatch | null,

        lastDispatchWinner: '',
        lastDispatchTime: 0,
    };

    // --- Debug / stats ---
    private debug = {
        written: 0,
        skipped: 0,
        lastGateLogTime: 0,
        lastRateLogTime: 0,
        topLogLast: 0,
        topLogInterval: 1000,
    };

    private constructor() {
        this.buffer = new RingBuffer(3, 16000);
    }

    static getInstance(): InferenceEngine {
        if (!InferenceEngine.instance) InferenceEngine.instance = new InferenceEngine();
        return InferenceEngine.instance;
    }

    // =========================================
    // Setup & reset
    // =========================================

    async setup(): Promise<boolean> {
        try {
            await tf.ready();

            // Prefer WASM if available (often better on mobile), else WebGL
            if (tf.findBackend('wasm')) {
                await tf.setBackend('wasm');
            } else if (tf.findBackend('webgl')) {
                await tf.setBackend('webgl');
                tf.env().set('WEBGL_PACK', false);
            }

            await customExtractor.loadConfig();
            this.model = await tf.loadLayersModel('/models/tfjs_model/model.json');

            const res = await fetch('/models/reciters_map.json');
            this.labels = await res.json();

            try {
                const normRes = await fetch('/models/normalization.json');
                if (normRes.ok) this.normalization = await normRes.json();
            } catch {
                console.warn('Using default stats');
            }

            return !!this.model;
        } catch (e) {
            console.error('Brain Offline:', e);
            return false;
        } finally {
            console.log(`🧪 CMVN: ${inferenceEngine.isCmvnEnabled() ? "ON" : "OFF"}`);
            console.log(`🧪 Far Field Mode: ${audioManager.isFarFieldMode() ? "ON" : "OFF"}`);
            console.log(`🧪 PreEmphasis: ${inferenceEngine.isPreEmphasisEnabled() ? "ON" : "OFF"}`);
            console.log(`🧪 RMS Normalized: ${inferenceEngine.isRmsNormalizeEnabled() ? "ON" : "OFF"}`);
        }
    }

    /** Public reset (internal state + UI idle). */
    public reset() {
        this.resetState();
        this.emitIdle();
        console.log('🧹 Inference Engine State Reset');
    }

    /** Internal-only reset (no UI dispatch). */
    private resetState() {
        // Gate
        this.gate.silenceCounter = 0;
        this.gate.isVoiced = false;
        this.gate.hangCounter = 0;

        // Buffer & inference
        this.buffer.clear();
        this.state.recentScores = [];
        this.state.pendingWinner = null;
        this.state.pendingCount = 0;
        this.state.stableWinner = null;

        // Dispatch throttles
        this.state.lastDispatchWinner = '';
        this.state.lastDispatchTime = 0;

        // Stats
        this.debug.written = 0;
        this.debug.skipped = 0;
    }

    // =========================================
    // Toggles & setters
    // =========================================

    public isRmsNormalizeEnabled(): boolean {
        return this.flags.rmsNormalize;
    }

    public toggleRmsNormalize() {
        this.flags.rmsNormalize = !this.flags.rmsNormalize;
        console.log(`🧪 RMS Normalize: ${this.flags.rmsNormalize ? 'ON' : 'OFF'}`);
    }

    public isPreEmphasisEnabled(): boolean {
        return this.flags.preEmphasis;
    }

    public togglePreEmphasis() {
        this.flags.preEmphasis = !this.flags.preEmphasis;
        console.log(`🧪 PreEmphasis: ${this.flags.preEmphasis ? 'ON' : 'OFF'}`);
    }

    public isCmvnEnabled(): boolean {
        return this.flags.cmvn;
    }

    public toggleCMVN() {
        this.flags.cmvn = !this.flags.cmvn;
        console.log(`🧪 CMVN: ${this.flags.cmvn ? 'ON' : 'OFF'}`);
    }

    /**
     * Treat this as a *floor clamp* (don’t let gates go below it).
     * For UI tuning, changing SNR often works better than raw RMS.
     */
    public setThreshold(newThreshold: number) {
        this.config.silenceThreshold = Math.max(newThreshold, 0.002);
    }

    /** Optional tuning: SNR thresholds (recommended over raw RMS). */
    public setSNR(snrOn: number, snrOff?: number) {
        this.config.snrOn = Math.min(6.0, Math.max(1.5, snrOn));
        this.config.snrOff = Math.min(this.config.snrOn, Math.max(1.1, snrOff ?? this.config.snrOn * 0.7));
    }

    // =========================================
    // DSP helpers
    // =========================================

    private calculateRms(x: Float32Array): number {
        let sum = 0;
        for (let i = 0; i < x.length; i++) sum += x[i] * x[i];
        return Math.sqrt(sum / Math.max(1, x.length));
    }

    /**
     * Normalize window loudness to target RMS (linear gain only).
     * - caps gain so we don't explode noise
     * - clamps output to [-1, 1]
     */
    private normalizeSignal(
        x: Float32Array,
        floor = 0.002
    ): { y: Float32Array; gain: number; rms: number } {
        const r = this.calculateRms(x);
        if (r < floor) return {y: x, gain: 1, rms: r};

        let g = this.config.targetRms / r;
        g = Math.min(Math.max(g, this.config.minGain), this.config.maxGain);

        if (Math.abs(g - 1) < 1e-3) return {y: x, gain: g, rms: r};

        const y = new Float32Array(x.length);
        for (let i = 0; i < x.length; i++) {
            const v = x[i] * g;
            y[i] = v > 1 ? 1 : (v < -1 ? -1 : v);
        }
        return {y, gain: g, rms: r};
    }

    private applyPreEmphasis(x: Float32Array): Float32Array {
        const y = new Float32Array(x.length);
        if (x.length === 0) return y;
        y[0] = x[0];
        const a = this.config.preEmphasisAlpha;
        for (let i = 1; i < x.length; i++) y[i] = x[i] - a * x[i - 1];
        return y;
    }

    // =========================================
    // Main audio pipeline
    // =========================================

    handleIncomingAudio(chunk: Float32Array) {
        const now = Date.now();
        const rms = this.calculateRms(chunk);

        // 1) Adaptive noise floor update (only when "quiet-ish")
        if (rms < this.gate.noiseFloor * 1.5) {
            const a = this.config.noiseAdaptAlpha; // e.g. 0.005
            this.gate.noiseFloor = (1 - a) * this.gate.noiseFloor + a * rms;
        }

        // 2) Two-threshold voice gate (hysteresis)
        const gateOn = Math.max(this.config.silenceThreshold, this.gate.noiseFloor * this.config.snrOn);
        const gateOff = Math.max(this.config.silenceThreshold * 0.7, this.gate.noiseFloor * this.config.snrOff);

        // Throttled debug log
        if (now - this.debug.lastGateLogTime > 1000) {
            console.log(
                `🎤 RMS:${rms.toFixed(4)} on:${gateOn.toFixed(4)} off:${gateOff.toFixed(4)} noise:${this.gate.noiseFloor.toFixed(4)} voiced:${this.gate.isVoiced}`
            );
            this.debug.lastGateLogTime = now;
        }

        // 3) Gate state machine
        if (!this.gate.isVoiced) {
            // Require stronger gate to start voicing
            if (rms < gateOn) {
                this.gate.silenceCounter++;

                // Prolonged silence: emit idle + internal reset (NO double-dispatch)
                if (this.gate.silenceCounter > 25) {
                    this.emitIdle();
                    this.resetState();
                }

                this.debug.skipped++;
                return;
            }

            // Enter voiced
            this.gate.isVoiced = true;
            this.gate.hangCounter = this.config.voicedHangMax;
        } else {
            // Already voiced: allow dips (hangover)
            if (rms < gateOff) {
                this.gate.hangCounter--;
                if (this.gate.hangCounter <= 0) {
                    this.gate.isVoiced = false;
                    this.gate.silenceCounter++;
                }
                this.debug.skipped++;
                return;
            } else {
                // Refresh hang window
                this.gate.hangCounter = this.config.voicedHangMax;
            }
        }

        // 4) Accept chunk
        this.gate.silenceCounter = 0;
        this.debug.written++;

        // Write raw chunk (recommended) — normalization happens on full window in predict()
        this.buffer.write(chunk);

        // 5) Trigger prediction: never overlap, throttle in time
        if (this.buffer.isFull && !this.state.isPredicting) {
            if (now - this.state.lastPredictionTime > this.config.predictionIntervalMs) {
                this.state.lastPredictionTime = now;
                void this.predict();
            }
        }

        // Debug rates
        if (now - this.debug.lastRateLogTime > 1000) {
            console.log(`🧪 buffer writes/sec=${this.debug.written} skipped/sec=${this.debug.skipped}`);
            this.debug.written = 0;
            this.debug.skipped = 0;
            this.debug.lastRateLogTime = now;
        }
    }

    private async predict() {
        if (!this.model || this.state.isPredicting) return;
        this.state.isPredicting = true;

        try {
            // Yield so UI can paint before heavy work
            await new Promise<void>(r => requestAnimationFrame(() => r()));

            // A) Acquire signal window
            let signal = this.buffer.read();

            // B) Preprocessing (window-level only)
            if (this.flags.rmsNormalize) {
                const n = this.normalizeSignal(signal);
                signal = n.y;

                // If this is too spammy, throttle it like top3 logs
                console.log(`🎚️ RMSNorm gain=${n.gain.toFixed(2)} rms=${n.rms.toFixed(4)}`);
            }

            const featSignal = this.flags.preEmphasis ? this.applyPreEmphasis(signal) : signal;

            // C) Inference
            const prediction = tf.tidy(() => {
                const input = customExtractor.extractFullClip(featSignal, {cmvn: this.flags.cmvn});
                const batch = input.expandDims(0);

                if (this.flags.cmvn) {
                    return this.model!.predict(batch) as tf.Tensor;
                }

                const mean = this.normalization.mean ?? -0.77;
                const std = this.normalization.std ?? 5.20;
                return this.model!.predict(batch.sub(mean).div(std)) as tf.Tensor;
            });

            const probs = Array.from(await prediction.data());
            prediction.dispose();

            // 🚨 Must await to avoid overlap bugs
            await this.handlePredictionResult(probs);
        } finally {
            this.state.isPredicting = false;
        }
    }

    /**
     * Process raw probabilities -> smoothed + stable winner -> throttled UI dispatch.
     */
    private async handlePredictionResult(probs: number[]) {
        // 1) Smooth (rolling average)
        this.state.recentScores.push(probs);
        if (this.state.recentScores.length > this.config.smoothingFrames) this.state.recentScores.shift();

        const avg = new Array(probs.length).fill(0);
        for (const s of this.state.recentScores) {
            for (let i = 0; i < s.length; i++) avg[i] += s[i];
        }
        for (let i = 0; i < avg.length; i++) avg[i] /= this.state.recentScores.length;

        // 2) Stats + top matches
        const ent = this.entropyNormalized(avg);
        const topMatches = this.getTop3(avg);

        // 3) Throttled top logs
        const now = Date.now();
        if (now - this.debug.topLogLast > this.debug.topLogInterval) {
            this.debug.topLogLast = now;
            const fmt = (m?: QariMatch) => (m ? `${m.name}:${(m.score * 100).toFixed(1)}%` : '-');
            console.log(`🏆 TOP3 | ent=${ent.toFixed(3)} | ${fmt(topMatches[0])} | ${fmt(topMatches[1])} | ${fmt(topMatches[2])}`);
        }

        // 4) Decision logic
        const top1 = topMatches[0] ?? {name: STATE_IDLE, score: 0};
        const top2 = topMatches[1];

        const ratio = top2?.score ? top1.score / top2.score : Infinity;
        const diff = top2 ? top1.score - top2.score : top1.score;

        // Decision gates (tune as needed)
        const notConfused = ent < 0.75;
        const strongTop1 = top1.score > 0.45;
        const clearWin = ratio > 1.25 && diff > 0.08;

        let winner: QariMatch = {name: STATE_IDLE, score: 0};

        if (notConfused && strongTop1 && clearWin) {
            if (this.state.pendingWinner === top1.name) {
                this.state.pendingCount++;
            } else {
                this.state.pendingWinner = top1.name;
                this.state.pendingCount = 1;
            }

            if (this.state.pendingCount >= this.config.stabilityThreshold) {
                this.state.stableWinner = top1;
            }

            // If not stable yet, show “Analyzing…” but carry the current confidence score
            winner = this.state.stableWinner ?? {name: STATE_IDLE, score: top1.score};
        } else {
            // Reset pending streak, keep stable winner if exists
            this.state.pendingWinner = null;
            this.state.pendingCount = 0;
            winner = this.state.stableWinner ?? {name: STATE_IDLE, score: 0};
        }

        // 5) Yield again so compute + dispatch don’t fight the frame
        await new Promise<void>(r => requestAnimationFrame(() => r()));

        // 6) Throttled dispatch: winner change OR minimum interval
        const winnerChanged = winner.name !== this.state.lastDispatchWinner;
        const enoughTime = now - this.state.lastDispatchTime > this.config.dispatchMinIntervalMs;

        if (winnerChanged || enoughTime) {
            this.state.lastDispatchWinner = winner.name;
            this.state.lastDispatchTime = now;

            window.dispatchEvent(
                new CustomEvent(EVENTS.RESULT_FOUND, {
                    detail: {
                        winner,
                        others: topMatches.filter(m => m.score > 0.05),
                    },
                })
            );
        }
    }

    // =========================================
    // File testing / debug
    // =========================================

    public async predictFromSignal(
        signal16k: Float32Array,
        opts?: { startSec?: number; windowSec?: number; dispatchToUI?: boolean; log?: boolean }
    ) {
        if (!this.model) throw new Error('Model not loaded. Call inferenceEngine.setup() first.');

        const sr = 16000;
        const windowSec = opts?.windowSec ?? 3;
        const win = Math.max(1, Math.floor(windowSec * sr));

        const startSec = Math.max(0, opts?.startSec ?? 0);
        const start = Math.floor(startSec * sr);

        // Windowing
        let windowed = new Float32Array(win);
        if (signal16k.length >= win) {
            const sliceStart = Math.min(start, Math.max(0, signal16k.length - win));
            windowed.set(signal16k.subarray(sliceStart, sliceStart + win));
        } else {
            windowed.set(signal16k);
        }

        // Same preprocessing as live
        if (this.flags.rmsNormalize) {
            const n = this.normalizeSignal(windowed);
            windowed = n.y as any;
            console.log(`🎚️ RMSNorm gain=${n.gain.toFixed(2)} rms=${n.rms.toFixed(4)}`);
        }

        const featSignal = this.flags.preEmphasis ? this.applyPreEmphasis(windowed) : windowed;

        const prediction = tf.tidy(() => {
            const input = customExtractor.extractFullClip(featSignal, {cmvn: this.flags.cmvn});
            const batch = input.expandDims(0);

            if (this.flags.cmvn) return this.model!.predict(batch) as tf.Tensor;

            const mean = this.normalization.mean ?? -0.77;
            const std = this.normalization.std ?? 5.20;
            return this.model!.predict(batch.sub(mean).div(std)) as tf.Tensor;
        });

        const probs = Array.from(await prediction.data());
        prediction.dispose();

        const ent = this.entropyNormalized(probs);
        const top3 = this.getTop3(probs);
        const top1 = top3[0] ?? {name: STATE_IDLE, score: 0};

        if (opts?.log) {
            console.log(
                `🏁 FILETEST | start=${startSec.toFixed(2)}s ent=${ent.toFixed(3)} | ` +
                top3.map(x => `${x.name}:${(x.score * 100).toFixed(1)}%`).join(' | ')
            );
        }

        if (opts?.dispatchToUI) {
            window.dispatchEvent(
                new CustomEvent(EVENTS.RESULT_FOUND, {
                    detail: {winner: top1, others: top3},
                })
            );
        }

        return {top1, top3, ent, probs};
    }

    // =========================================
    // Event helpers
    // =========================================

    private emitIdle() {
        requestAnimationFrame(() => {
            window.dispatchEvent(
                new CustomEvent(EVENTS.RESULT_FOUND, {
                    detail: {
                        winner: {name: STATE_IDLE, score: 0},
                        others: [],
                    },
                })
            );
        });
    }

    // =========================================
    // Math / selection helpers
    // =========================================

    /**
     * Top-3 selection without full sort (O(N)).
     */
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

    /**
     * Normalized entropy in [0..1], robust across different class counts.
     */
    private entropyNormalized(probs: number[]): number {
        const N = probs.length;
        if (N <= 1) return 0;

        let e = 0;
        for (const p of probs) {
            if (p > 0) e -= p * Math.log(p);
        }
        return e / Math.log(N);
    }

    /**
     * Installs hotkeys for inference toggles (Shift + N/E/C).
     * @param onAction Callback to run when a toggle changes (receives status message for UI)
     * @returns Cleanup function to remove listeners
     */
    public installHotkeys(onAction: (status: string) => void): () => void {
        const handler = (e: KeyboardEvent) => {
            // Ignore if typing in an input
            const target = e.target as HTMLElement | null;
            if (target?.tagName === 'INPUT' || target?.tagName === 'TEXTAREA' || target?.isContentEditable) return;

            if (!e.shiftKey) return;

            const key = e.key.toLowerCase();
            let msg = "";

            if (key === 'n') {
                this.toggleRmsNormalize();
                msg = this.isRmsNormalizeEnabled() ? "RmsNormalize: ON" : "RmsNormalize: OFF";
            } else if (key === 'e') {
                this.togglePreEmphasis();
                msg = this.isPreEmphasisEnabled() ? "PreEmphasis: ON" : "PreEmphasis: OFF";
            } else if (key === 'c') {
                this.toggleCMVN();
                msg = this.isCmvnEnabled() ? "CMVN: ON" : "CMVN: OFF";
            }

            if (msg) {
                console.log(`🧪 ${msg}`);
                onAction(msg);
            }
        };

        window.addEventListener('keydown', handler);
        console.log("🧪 Inference Hotkeys Installed: Shift + [N]ormalize, [E]mphasis, [C]MVN");

        return () => window.removeEventListener('keydown', handler);
    }
}

export default InferenceEngine;
export const inferenceEngine = InferenceEngine.getInstance();

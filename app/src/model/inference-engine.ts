import * as tf from '@tensorflow/tfjs';
import {customExtractor} from '../features/custom-extractor';
import {RingBuffer} from './ring-buffer';

// CONSTANT for System ID
export const STATE_IDLE = "Analyzing...";

class InferenceEngine {
    private static instance: InferenceEngine;

    private model: tf.LayersModel | null = null;
    private labels: string[] = [];
    private normalization = {mean: 0, std: 1};
    private buffer: RingBuffer;

    // Timing / state
    private lastPredictionTime = 0;
    private recentScores: number[][] = [];
    private isPredicting = false;

    // Performance throttles
    private lastLogTime = 0;
    private lastDispatchTime = 0;
    private lastDispatchWinner = "";

    // Noise / voice gating
    // NOTE: silenceThreshold is a floor clamp, NOT the main gate.
    private silenceThreshold = 0.006;
    private silenceCounter = 0;

    // Start low so quiet rooms don't get treated as noisy at boot.
    private noiseFloor = 0.0015;

    // Hysteresis gate (two thresholds)
    private snrOn = 2.0;    // enter voiced when rms >= noiseFloor*snrOn
    private snrOff = 1.4;   // stay voiced while rms >= noiseFloor*snrOff
    private isVoiced = false;
    private voicedHang = 0; // hangover chunks to bridge gaps
    private readonly VOICED_HANG_MAX = 15; // ~10 chunks

    // Stability state
    private pendingWinner: string | null = null;
    private pendingCount = 0;
    private stableWinner: { name: string; score: number } | null = null;
    private REQUIRED_STABILITY = 2;

    private dbgWritten = 0;
    private dbgSkipped = 0;
    private dbgLastDbgTime = 0;

    private dbgTopLogLast = 0;
    private dbgTopLogEveryMs = 1000;

    private cmvnEnabled = false;

    private preEmphasisEnabled = false;     // start OFF
    private preEmphasisA = 0.97;

    private rmsNormalizeEnabled = true;  // start ON for mic robustness
    private targetRms = 0.10;           // DO NOT chase 0.40; your loopback succeeds at ~0.10
    private minGain = 0.25;
    private maxGain = 6.0;

    private applyRmsNormalize(chunk: Float32Array, rms: number): Float32Array {
        // Guard
        if (rms < 1e-8) return chunk;

        let g = this.targetRms / rms;
        if (g < this.minGain) g = this.minGain;
        if (g > this.maxGain) g = this.maxGain;

        // Apply gain + hard limiter
        const out = new Float32Array(chunk.length);
        for (let i = 0; i < chunk.length; i++) {
            let y = chunk[i] * g;
            if (y > 1) y = 1;
            else if (y < -1) y = -1;
            out[i] = y;
        }
        return out;
    }

    public isRmsNormalizeEnabled(): boolean {
        return this.rmsNormalizeEnabled;
    }

    toggleRmsNormalize() {
        this.rmsNormalizeEnabled = !this.rmsNormalizeEnabled;
        console.log(`🧪 RMS Normalize: ${this.rmsNormalizeEnabled ? "ON" : "OFF"}`);
    }

    public isPreEmphasisEnabled(): boolean {
        return this.preEmphasisEnabled;
    }

    private rms(x: Float32Array) {
        let sum = 0;
        for (let i = 0; i < x.length; i++) sum += x[i] * x[i];
        return Math.sqrt(sum / Math.max(1, x.length));
    }

    /**
     * Normalize window loudness to target RMS (linear gain only).
     * - caps gain so we don't explode noise
     * - clamps output to [-1, 1] (safe for model + WAV debug)
     */
    private normalizeRms(
        x: Float32Array,
        targetRms = 0.10,
        minGain = 0.25,
        maxGain = 8.0,
        floor = 0.002
    ) {
        const r = this.rms(x);
        if (r < floor) return {y: x, gain: 1, rms: r};

        let g = targetRms / r;
        if (g < minGain) g = minGain;
        if (g > maxGain) g = maxGain;

        if (Math.abs(g - 1) < 1e-3) return {y: x, gain: g, rms: r};

        const y = new Float32Array(x.length);
        for (let i = 0; i < x.length; i++) {
            const v = x[i] * g;
            y[i] = v > 1 ? 1 : (v < -1 ? -1 : v);
        }
        return {y, gain: g, rms: r};
    }

    togglePreEmphasis() {
        this.preEmphasisEnabled = !this.preEmphasisEnabled;
        console.log(`🧪 PreEmphasis: ${this.preEmphasisEnabled ? "ON" : "OFF"}`);
    }

    private preEmphasis(x: Float32Array, a = this.preEmphasisA) {
        const y = new Float32Array(x.length);
        if (x.length === 0) return y;
        y[0] = x[0];
        for (let i = 1; i < x.length; i++) y[i] = x[i] - a * x[i - 1];
        return y;
    }

    public isCmvnEnabled(): boolean {
        return this.cmvnEnabled;
    }

    toggleCMVN() {
        this.cmvnEnabled = !this.cmvnEnabled;
        console.log(`🧪 CMVN: ${this.cmvnEnabled ? "ON" : "OFF"}`);
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

            // Prefer WASM on mobile if available; otherwise WebGL
            if (tf.findBackend('wasm')) {
                await tf.setBackend('wasm');
                // console.log("⚡ Using WASM Backend");
            } else if (tf.findBackend('webgl')) {
                await tf.setBackend('webgl');
                tf.env().set('WEBGL_PACK', false);
                // console.log("⚡ Using WebGL Backend");
            }

            await customExtractor.loadConfig();
            this.model = await tf.loadLayersModel('/models/tfjs_model/model.json');

            const res = await fetch('/models/reciters_map.json');
            this.labels = await res.json();

            try {
                const normRes = await fetch('/models/normalization.json');
                if (normRes.ok) this.normalization = await normRes.json();
            } catch {
                console.warn("Using default stats");
            }

            return !!this.model;
        } catch (e) {
            console.error("Brain Offline:", e);
            return false;
        }
    }

    /**
     * Treat this as a *floor clamp* (don’t let gates go below it).
     * If you expose a UI slider, consider mapping it to snrOn instead.
     */
    setThreshold(newThreshold: number) {
        this.silenceThreshold = Math.max(newThreshold, 0.002);
    }

    /**
     * Optional: expose this for tuning in UI (recommended over raw RMS)
     * snrOn should usually be in [1.8..4.0]
     */
    setSNR(snrOn: number, snrOff?: number) {
        this.snrOn = Math.min(6.0, Math.max(1.5, snrOn));
        this.snrOff = Math.min(this.snrOn, Math.max(1.1, snrOff ?? (this.snrOn * 0.7)));
    }

    handleIncomingAudio(chunk: Float32Array) {
        // 1) Calculate RMS
        let sum = 0;
        for (let i = 0; i < chunk.length; i++) sum += chunk[i] * chunk[i];
        const rms = Math.sqrt(sum / chunk.length);

        const now = Date.now();

        // 2) Adaptive noise floor update (only when "quiet-ish")
        // Start low and adapt up/down gently.
        if (rms < this.noiseFloor * 1.5) {
            this.noiseFloor = 0.995 * this.noiseFloor + 0.005 * rms;
        }

        // 3) Two-threshold voice gate (hysteresis)
        const gateOn = Math.max(this.silenceThreshold, this.noiseFloor * this.snrOn);
        const gateOff = Math.max(this.silenceThreshold * 0.7, this.noiseFloor * this.snrOff);

        // Throttled debug log (once per second)
        if (now - this.lastLogTime > 1000) {
            console.log(
                `🎤 RMS:${rms.toFixed(4)} on:${gateOn.toFixed(4)} off:${gateOff.toFixed(4)} noise:${this.noiseFloor.toFixed(4)} voiced:${this.isVoiced}`
            );
            this.lastLogTime = now;
        }

        if (!this.isVoiced) {
            // Not currently in voice: require stronger gate to start
            if (rms < gateOn) {
                this.silenceCounter++;
                if (this.silenceCounter > 25) {
                    this.dispatchSilence();
                    this.resetState();
                }
                this.dbgSkipped++;
                return;
            }
            // Enter voiced state
            this.isVoiced = true;
            this.voicedHang = this.VOICED_HANG_MAX;
        } else {
            // Already in voice: allow dips (hangover)
            if (rms < gateOff) {
                this.voicedHang--;
                if (this.voicedHang <= 0) {
                    this.isVoiced = false;
                    this.silenceCounter++;
                }
                this.dbgSkipped++;
                return;
            } else {
                // Refresh hang window
                this.voicedHang = this.VOICED_HANG_MAX;
            }
        }

        // If we got here, we accept this chunk
        this.silenceCounter = 0;
        this.dbgWritten++;
        const chunkForBuffer =
            this.rmsNormalizeEnabled ? this.applyRmsNormalize(chunk, rms) : chunk;

        this.buffer.write(chunkForBuffer);

        // Fire inference at most every 500ms and never overlap
        if (this.buffer.isFull && !this.isPredicting) {
            if (now - this.lastPredictionTime > 500) {
                this.lastPredictionTime = now;
                this.predict();
            }
        }
        if (now - this.dbgLastDbgTime > 1000) {
            console.log(`🧪 buffer writes/sec=${this.dbgWritten} skipped/sec=${this.dbgSkipped}`);
            this.dbgWritten = 0;
            this.dbgSkipped = 0;
            this.dbgLastDbgTime = now;
        }
    }

    private resetState() {
        this.silenceCounter = 0;
        this.isVoiced = false;
        this.voicedHang = 0;

        this.buffer.clear();
        this.recentScores = [];
        this.pendingWinner = null;
        this.pendingCount = 0;
        this.stableWinner = null;
    }

    private dispatchSilence() {
        // Schedule on next frame (no await needed)
        requestAnimationFrame(() => {
            window.dispatchEvent(
                new CustomEvent('qari-found', {
                    detail: {
                        winner: {name: STATE_IDLE, score: 0},
                        others: []
                    }
                })
            );
        });
    }

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

        const result: { name: string; score: number }[] = [];
        if (i1 !== -1) result.push({name: this.formatName(this.labels[i1]), score: s1});
        if (i2 !== -1) result.push({name: this.formatName(this.labels[i2]), score: s2});
        if (i3 !== -1) result.push({name: this.formatName(this.labels[i3]), score: s3});
        return result;
    }

    private formatName(raw: string): string {
        return raw ? raw.replace(/_/g, ' ').toUpperCase() : "UNKNOWN";
    }

    public reset() {
        this.resetState();
        // 2. Force the UI to go back to "Idle/Listening" immediately
        // This stops the Ring from showing the old Reciter for a split second on restart
        window.dispatchEvent(new CustomEvent('qari-found', {
            detail: {
                winner: {name: STATE_IDLE, score: 0},
                others: []
            }
        }));

        console.log("🧹 Inference Engine State Reset");
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

    private async predict() {
        if (!this.model || this.isPredicting) return;
        this.isPredicting = true;

        try {
            // Yield so UI/visualizer can paint before heavy work
            await new Promise<void>(r => requestAnimationFrame(() => r()));

            let signal = this.buffer.read();
            if (this.isRmsNormalizeEnabled()) {
                const n = this.normalizeRms(signal);
                signal = n.y;
                console.log(`🎚️ RMSNorm gain=${n.gain.toFixed(2)} rms=${n.rms.toFixed(4)}`);
            }
            const featSignal = this.preEmphasisEnabled ? this.preEmphasis(signal) : signal;

            const prediction = tf.tidy(() => {
                const input = customExtractor.extractFullClip(featSignal, {cmvn: this.cmvnEnabled});
                const batch = input.expandDims(0);

                if (this.cmvnEnabled) {
                    // CMVN already normalized per-clip
                    return this.model!.predict(batch) as tf.Tensor;
                }

                // Original global normalization path
                const mean = this.normalization.mean ?? -0.77;
                const std = this.normalization.std ?? 5.20;
                return this.model!.predict(batch.sub(mean).div(std)) as tf.Tensor;
            });

            const probs = Array.from(await prediction.data());
            prediction.dispose();

            // Smoothing: keep last 5
            this.recentScores.push(probs);
            if (this.recentScores.length > 5) this.recentScores.shift();

            const avg = new Array(probs.length).fill(0);
            for (const s of this.recentScores) {
                for (let i = 0; i < s.length; i++) avg[i] += s[i];
            }
            for (let i = 0; i < avg.length; i++) avg[i] /= this.recentScores.length;

            // Entropy on smoothed probs
            const ent = this.entropyNormalized(avg);

            // Top matches
            const topMatches = this.getTop3(avg);
            const nowTop = Date.now();
            if (nowTop - this.dbgTopLogLast > this.dbgTopLogEveryMs) {
                this.dbgTopLogLast = nowTop;

                const fmt = (m?: { name: string; score: number }) =>
                    m ? `${m.name}:${(m.score * 100).toFixed(1)}%` : "-";

                console.log(
                    `🏆 TOP3 | ent=${ent.toFixed(3)} | ` +
                    `${fmt(topMatches[0])} | ${fmt(topMatches[1])} | ${fmt(topMatches[2])}`
                );
            }

            const top1 = topMatches[0] || {name: STATE_IDLE, score: 0};
            const top2 = topMatches[1];

            const ratio = top2?.score ? (top1.score / top2.score) : Infinity;
            const diff = top2 ? (top1.score - top2.score) : top1.score;

            // Decision gates (tune as needed)
            const notConfused = ent < 0.75;
            const strongTop1 = top1.score > 0.45;
            const clearWin = (ratio > 1.25) && (diff > 0.08);

            let winner = {name: STATE_IDLE, score: 0};

            if (notConfused && strongTop1 && clearWin) {
                if (this.pendingWinner === top1.name) {
                    this.pendingCount++;
                } else {
                    this.pendingWinner = top1.name;
                    this.pendingCount = 1;
                }

                if (this.pendingCount >= this.REQUIRED_STABILITY) {
                    this.stableWinner = top1;
                }

                // Show stable if we have it; otherwise "Analyzing..." but with current score
                winner = this.stableWinner ?? {name: STATE_IDLE, score: top1.score};
            } else {
                // Reset pending streak, keep stable winner if exists
                this.pendingWinner = null;
                this.pendingCount = 0;
                winner = this.stableWinner ?? {name: STATE_IDLE, score: 0};
            }

            // Yield again so we don't do heavy compute + UI dispatch in same frame
            await new Promise<void>(r => requestAnimationFrame(() => r()));

            // Throttled dispatch: winner change OR 150ms passed
            const now = Date.now();
            const winnerChanged = winner.name !== this.lastDispatchWinner;

            if (winnerChanged || (now - this.lastDispatchTime > 150)) {
                this.lastDispatchWinner = winner.name;
                this.lastDispatchTime = now;

                window.dispatchEvent(
                    new CustomEvent('qari-found', {
                        detail: {
                            winner,
                            others: topMatches.filter(m => m.score > 0.05)
                        }
                    })
                );
            }
        } finally {
            this.isPredicting = false;
        }
    }

    public async predictFromSignal(
        signal16k: Float32Array,
        opts?: { startSec?: number; windowSec?: number; dispatchToUI?: boolean; log?: boolean }
    ) {
        if (!this.model) throw new Error("Model not loaded. Call inferenceEngine.setup() first.");

        const sr = 16000;
        const windowSec = opts?.windowSec ?? 3;
        const win = Math.max(1, Math.floor(windowSec * sr));

        // pick window
        const startSec = Math.max(0, opts?.startSec ?? 0);
        const start = Math.floor(startSec * sr);

        let windowed = new Float32Array(win);
        if (signal16k.length >= win) {
            const sliceStart = Math.min(start, Math.max(0, signal16k.length - win));
            windowed.set(signal16k.subarray(sliceStart, sliceStart + win));
        } else {
            // pad if too short
            windowed.set(signal16k);
        }

        if (this.isRmsNormalizeEnabled()) {
            const n = this.normalizeRms(windowed);
            windowed = n.y as any;
            console.log(`🎚️ RMSNorm gain=${n.gain.toFixed(2)} rms=${n.rms.toFixed(4)}`);
        }

        const featSignal = this.preEmphasisEnabled ? this.preEmphasis(windowed) : windowed;

        const prediction = tf.tidy(() => {
            const input = customExtractor.extractFullClip(featSignal, {cmvn: this.cmvnEnabled});
            const batch = input.expandDims(0);

            if (this.cmvnEnabled) {
                return this.model!.predict(batch) as tf.Tensor;
            }

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
                top3.map(x => `${x.name}:${(x.score * 100).toFixed(1)}%`).join(" | ")
            );
        }

        if (opts?.dispatchToUI) {
            window.dispatchEvent(new CustomEvent("qari-found", {
                detail: {
                    winner: top1,
                    others: top3
                }
            }));
        }

        return {top1, top3, ent, probs};
    }

}

export default InferenceEngine

export const inferenceEngine = InferenceEngine.getInstance();

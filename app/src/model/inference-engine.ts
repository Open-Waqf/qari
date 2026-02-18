import * as tf from '@tensorflow/tfjs';
import {customExtractor} from '../features/custom-extractor';
import {RingBuffer} from './ring-buffer';
import {EVENTS, type QariMatch} from '../core/events';
import {audioManager} from "../core/audio-manager.ts";
import {forceWasmBackend} from "./tf-backend";

export const STATE_IDLE = '__IDLE__';

type Activity = 'voiced' | 'silence' | 'noise';

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

    lockDelayMs: number;      // Time stable before locking (e.g. 2s)
    switchDelayMs: number;    // Time new winner must persist to break lock (e.g. 3s)
    unlockDelayMs: number;    // Time of silence/noise to release lock (e.g. 3s)
}

tf.serialization.registerClass(SpecAugment);

/**
 * Internal helper: voice gate / VAD state + noise floor adaptation.
 * IMPORTANT: This is a behavior-preserving extraction of the existing logic.
 *
 * - Applies HPF ONLY to gateScratch (gate RMS), NOT to features.
 * - Adapts noise floor with the same two-speed alpha logic.
 * - Maintains isVoiced, hangCounter, silenceCounter exactly as before.
 *
 * The engine remains responsible for:
 * - buffer.clear() on becameVoiced
 * - emitIdle() and decision-memory resets on idleTriggered
 * - voiced warmup timers
 */
class VoiceActivityDetector {
    private gateScratch: Float32Array | null = null;

    private filterState = {x1: 0, y1: 0};
    private readonly HP_COEFF = 0.90;

    private lastRms = 0;

    private gate = {
        noiseFloor: 0.0015,
        calibrated: false,
        isVoiced: false,
        hangCounter: 0,
        silenceCounter: 0,
    };

    constructor(private readonly config: EngineConfig) {
    }

    public reset() {
        this.gate.silenceCounter = 0;
        this.gate.isVoiced = false;
        this.gate.hangCounter = 0;
        this.filterState = {x1: 0, y1: 0};
    }

    public getLastRms(): number {
        return this.lastRms;
    }

    public getNoiseFloor(): number {
        return this.gate.noiseFloor;
    }

    public isVoiced(): boolean {
        return this.gate.isVoiced;
    }

    public isCalibrated(): boolean {
        return this.gate.calibrated;
    }

    public setNoiseFloor(noiseFloorRms: number) {
        if (!Number.isFinite(noiseFloorRms)) return;

        // Clamp to sane values (0.0003 is dead silent, 0.05 is a loud coffee shop)
        // 1. Update the baseline noise floor
        this.gate.noiseFloor = Math.min(0.05, Math.max(0.0003, noiseFloorRms));
        this.gate.calibrated = true;

        // 2. Reset the gate counters so we don't get stuck in "Voiced" mode
        this.gate.isVoiced = false;
        this.gate.hangCounter = 0;
    }

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

    public processFrame(chunk: Float32Array) {
        // 🛑 PARITY: Filter only for GATE, not for FEATURES
        if (!this.gateScratch || this.gateScratch.length !== chunk.length) {
            this.gateScratch = new Float32Array(chunk.length);
        }
        this.gateScratch.set(chunk);
        this.applyHighPassInPlace(this.gateScratch);

        const rms = this.calculateRms(this.gateScratch);
        this.lastRms = rms;

        // Adaptive Noise Floor (two-speed + never drift below intended minimum)
        const minGate = this.config.silenceThreshold;      // ✅ ALWAYS keep this safety floor
        const minNoiseFloor = minGate / this.config.snrOn; // ✅ ensures gateOn >= silenceThreshold

        const aDown = this.config.noiseAdaptAlpha; // fast down
        const aUp = aDown * 0.15;                  // slow up

        // Provisional gates from current floor
        let gateOn = Math.max(minGate, this.gate.noiseFloor * this.config.snrOn);
        let gateOff = Math.max(minGate * 0.7, this.gate.noiseFloor * this.config.snrOff);

        // Learn noise floor only from "noise-like" frames
        const noiseLike = (!this.gate.isVoiced) || (rms < gateOff);

        if (noiseLike) {
            const target = rms;
            let alpha = target < this.gate.noiseFloor ? aDown : aUp;
            if (this.gate.calibrated && target > this.gate.noiseFloor) {
                alpha = 0; // Prevent upward drift
            }
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

        let becameVoiced = false;
        let becameSilent = false;
        let idleTriggered = false;

        if (!this.gate.isVoiced) {
            if (rms < gateOn) {
                this.gate.silenceCounter++;
                if (this.gate.silenceCounter > 25) {
                    idleTriggered = true;
                }
            } else {
                this.gate.isVoiced = true;
                this.gate.hangCounter = this.config.voicedHangMax;
                this.gate.silenceCounter = 0;
                becameVoiced = true;
            }
        } else {
            if (rms < gateOff) {
                this.gate.hangCounter--;
                if (this.gate.hangCounter <= 0) {
                    this.gate.isVoiced = false;
                    this.gate.silenceCounter++;
                    becameSilent = true;
                }
            } else {
                this.gate.hangCounter = this.config.voicedHangMax;
            }
        }

        const voiced = this.gate.isVoiced;

        // activeFrame means: current chunk is strong enough to be treated as real signal
        const activeFrame = voiced ? (rms >= gateOff) : (rms >= gateOn);

        return {
            rms,
            voiced,
            activeFrame,
            gateOn,
            gateOff,
            noiseFloor: this.gate.noiseFloor,
            idleTriggered,
            becameVoiced,
            becameSilent,
            silenceCounter: this.gate.silenceCounter,
        };
    }
}

/**
 * Internal helper: session locking / hysteresis layer.
 * Behavior preserved from updateSessionState().
 */
class SessionLocker {
    private session = {
        lockedWinner: null as QariMatch | null,
        lockedAt: 0,

        // The "Candidate" trying to break the lock
        candidateName: null as string | null,
        candidateFirstSeen: 0,

        // Counter for silence/noise to break lock
        unlockConditionStart: 0,
    };

    public update(
        currentStable: QariMatch | null,
        activity: Activity,
        config: EngineConfig,
        isDebug: boolean
    ): { finalWinner: QariMatch; isLocked: boolean } {
        const now = Date.now();
        const {lockDelayMs, switchDelayMs, unlockDelayMs} = config;

        // 1. HANDLE SILENCE / UNLOCKING
        if (!currentStable || activity !== 'voiced') {
            if (this.session.lockedWinner) {
                if (this.session.unlockConditionStart === 0) {
                    this.session.unlockConditionStart = now;
                } else if (now - this.session.unlockConditionStart > unlockDelayMs) {
                    // 🔓 UNLOCK due to silence
                    if (isDebug) console.log(`🔓 Session Unlocked (Silence > ${unlockDelayMs}ms)`);
                    this.session.lockedWinner = null;
                    this.session.lockedAt = 0;
                    this.session.unlockConditionStart = 0;
                }
            }

            return {
                finalWinner: this.session.lockedWinner ?? {name: STATE_IDLE, score: 0},
                isLocked: !!this.session.lockedWinner
            };
        }

        // We have a VOICED signal and a valid stable winner from the lower layer
        this.session.unlockConditionStart = 0; // Reset silence timer

        // 2. IF NOT LOCKED: Attempt to Lock
        if (!this.session.lockedWinner) {
            if (this.session.candidateName === currentStable.name) {
                if (now - this.session.candidateFirstSeen > lockDelayMs) {
                    this.session.lockedWinner = currentStable;
                    this.session.lockedAt = now;
                    if (isDebug) console.log(`🔒 Session Locked: ${currentStable.name}`);
                }
            } else {
                this.session.candidateName = currentStable.name;
                this.session.candidateFirstSeen = now;
            }

            return {finalWinner: currentStable, isLocked: false};
        }

        // 3. IF LOCKED: Handle Switching (Hysteresis)
        if (currentStable.name === this.session.lockedWinner.name) {
            this.session.candidateName = null;
            this.session.candidateFirstSeen = 0;
            return {finalWinner: this.session.lockedWinner, isLocked: true};
        }

        // A Challenger Appears!
        if (this.session.candidateName === currentStable.name) {
            if (now - this.session.candidateFirstSeen > switchDelayMs) {
                // 🔄 SWITCH LOCK
                if (isDebug) {
                    console.log(`🔄 Session Switched: ${this.session.lockedWinner.name} -> ${currentStable.name}`);
                }
                this.session.lockedWinner = currentStable;
                this.session.lockedAt = now;
                this.session.candidateName = null;
                this.session.candidateFirstSeen = 0;
                return {finalWinner: currentStable, isLocked: true};
            }
        } else {
            this.session.candidateName = currentStable.name;
            this.session.candidateFirstSeen = now;
        }

        return {finalWinner: this.session.lockedWinner, isLocked: true};
    }
}

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

    private acceptTop1Min = 0.14;

    // --- Buffer freshness across silence ---
    private zeroChunk: Float32Array | null = null;
    private voicedStartedAt = 0;
    private voicedChunksSinceStart = 0;

    // Predict only after we have some voiced audio in the window
    private readonly minVoicedMsForPredict = 350;
    private readonly minVoicedChunksForPredict = 2;

    // ✅ Auto-detect debug mode from URL
    private isDebug = typeof window !== 'undefined' && window.location.search.includes('debug=1');

    // --- Configuration ---
    private config: EngineConfig = {
        targetRms: 0.10,
        minGain: 0.6,
        maxGain: 5,

        silenceThreshold: 0.002,
        snrOn: 2.0,
        snrOff: 1.4,
        voicedHangMax: 15,

        predictionIntervalMs: 500,

        stabilityThreshold: 3,
        smoothingFrames: 4,
        dispatchMinIntervalMs: 150,
        noiseAdaptAlpha: 0.005,

        lockDelayMs: 2000,
        switchDelayMs: 3000,
        unlockDelayMs: 4000,
    };

    // --- Internal helpers (same-file refactor; public API unchanged) ---
    private vad = new VoiceActivityDetector(this.config);
    private sessionLocker = new SessionLocker();

    // --- Feature flags ---
    private flags = {
        rmsNormalize: true,
        preEmphasis: false,
        cmvn: false,
    };

    // --- Engine state (unchanged behavior) ---
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

    /**
     * ✅ Expose TFJS backend for Debug Panel
     */
    public getBackend(): string {
        return tf.getBackend();
    }

    /**
     * ✅ Expose last inference timestamp for Debug Panel
     */
    public getLastInferenceTime(): number {
        return this.state.lastPredictionTime;
    }

    /**
     * Backward-compatible alias (safe).
     * If your UI previously called inferenceEngine.calibrate(), it will keep working.
     */
    public calibrate(noiseFloorRms?: number) {
        const nf =
            Number.isFinite(noiseFloorRms as number)
                ? (noiseFloorRms as number)
                : this.getCurrentRms();
        this.setNoiseFloor(nf);
    }

    /**
     * Sets the MODEL CONFIDENCE threshold (0.0 - 1.0).
     * How sure must the AI be to trigger a match?
     */
    public setConfidenceThreshold(p: number) {
        if (!Number.isFinite(p)) return;
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

        this.vad.setNoiseFloor(noiseFloorRms);

        if (this.isDebug) {
            console.log(`🔇 Noise Floor calibrated to RMS: ${this.vad.getNoiseFloor().toFixed(5)}`);
        }
    }

    public getCurrentRms(): number {
        return this.vad.getLastRms() || 0;
    }

    // =========================================
    // Setup & Configuration
    // =========================================

    async setup(): Promise<boolean> {
        if (this.model && this.labels.length > 0) {
            return true;
        }
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
        // preserve previous semantics: gate/filter reset + buffer clear + decision memory reset
        this.vad.reset();
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
    // Core DSP Logic (Feature-path only)
    // =========================================

    private calculateRms(x: Float32Array): number {
        let sum = 0;
        for (let i = 0; i < x.length; i++) sum += x[i] * x[i];
        return Math.sqrt(sum / Math.max(1, x.length));
    }

    private normalizeSignal(x: Float32Array, floor = 0.002): { y: Float32Array; gain: number; rms: number } {
        const r = this.calculateRms(x);

        // 1. Floor Gate (Parity with Python)
        if (r < floor) return {y: x, gain: 1, rms: r};

        // 2. Linear Gain Calculation
        let g = this.config.targetRms / r;
        g = Math.min(Math.max(g, this.config.minGain), this.config.maxGain);

        // Optimization: Skip if gain is effectively 1.0
        if (Math.abs(g - 1) < 1e-3) return {y: x, gain: g, rms: r};

        const y = new Float32Array(x.length);
        for (let i = 0; i < x.length; i++) {
            // 🟢 PARITY FIX: Linear Gain + Hard Clip
            let v = x[i] * g;
            if (v > 1.0) v = 1.0;
            else if (v < -1.0) v = -1.0;
            y[i] = v;
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
    // Shared Inference Pipeline (unchanged)
    // =========================================

    private async runInferencePipeline(signal: Float32Array, log: boolean = false): Promise<number[]> {
        // 1. RMS Normalization
        let processed = signal;
        if (this.flags.rmsNormalize) {
            const dynFloor = Math.max(0.0006, this.vad.getNoiseFloor() * this.config.snrOff);
            const n = this.normalizeSignal(processed, dynFloor);
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
                console.log(
                    `📊 RAW Features: Mean=${mean.dataSync()[0].toFixed(2)} | Std=${Math.sqrt(variance.dataSync()[0]).toFixed(2)}`
                );
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
        if (this.isDebug && this.debug.lastRateLogTime === 0) {
            console.log(`📦 chunkLen=${rawChunk.length} (expect 4096 @22k)`);
            this.debug.lastRateLogTime = 1;
        }

        // 🛑 TRUST WORKLET: It delivers 22050
        const chunk = rawChunk;
        const now = Date.now();

        // NOTE: VAD must ONLY HPF the gateScratch, never the feature buffer (training parity).
        const g = this.vad.processFrame(chunk);

        if (this.isDebug && now - this.debug.lastGateLogTime > 2000) {
            console.log(`🎤 RMS:${g.rms.toFixed(4)} Gate:${g.gateOn.toFixed(4)} Noise:${g.noiseFloor.toFixed(4)}`);
            this.debug.lastGateLogTime = now;
        }

        // Preserve the exact side-effects previously performed in handleIncomingAudio:

        if (g.idleTriggered) {
            this.emitIdle();

            // ✅ Clear decision memory only (keep buffer full)
            this.state.recentScores = [];
            this.state.pendingWinner = null;
            this.state.pendingCount = 0;
            this.state.stableWinner = null;
            this.state.noWinCount = 0;

            // optional: also reset lastDispatchWinner to allow UI update if needed
            this.state.lastDispatchWinner = '';
        }

        if (g.becameVoiced) {
            this.buffer.clear();

            // ✅ mark voiced start (for warmup)
            this.voicedStartedAt = now;
            this.voicedChunksSinceStart = 0;
            if (this.isDebug) console.log(`became Voiced ` + true);
        }

        if (g.becameSilent) {
            // ✅ reset voiced window tracking
            this.voicedStartedAt = 0;
            this.voicedChunksSinceStart = 0;

            // ✅ prevent old pending stability from carrying over
            this.state.pendingWinner = null;
            this.state.pendingCount = 0;

            if (this.isDebug) console.log(`became Silent ` + true);
        }

        const voiced = g.voiced;
        const activeFrame = g.activeFrame;
        const gateOff = g.gateOff;
        const rms = g.rms;

        let writeBuf: Float32Array;

        // ✅ Change: never write ZERO while voiced (avoid poisoning the ring buffer)
        if (voiced) {
            writeBuf = chunk;

            // Keep warmup counter tied to real-energy frames, not hang frames
            if (rms >= gateOff) {
                this.voicedChunksSinceStart++;
            }
        } else {
            // Only flush with zeros when truly not voiced
            if (!this.zeroChunk || this.zeroChunk.length !== chunk.length) {
                this.zeroChunk = new Float32Array(chunk.length);
            } else {
                this.zeroChunk.fill(0);
            }
            writeBuf = this.zeroChunk;
        }

        if (this.isDebug) {
            console.log(`write=${voiced ? "AUDIO" : "ZERO"} voiced=${voiced} chunks=${this.voicedChunksSinceStart}`);
        }

        this.buffer.write(writeBuf);

        // Trigger Prediction (only if voiced AND after short warmup)
        const voicedWarm =
            this.voicedStartedAt > 0 &&
            (now - this.voicedStartedAt) >= this.minVoicedMsForPredict &&
            this.voicedChunksSinceStart >= this.minVoicedChunksForPredict;

        if (voiced && voicedWarm && activeFrame && this.buffer.isFull && !this.state.isPredicting) {
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
            console.log(
                `🧮 top1=${d.top1.name}:${(d.top1.score * 100).toFixed(1)}% diff=${d.diff.toFixed(3)} ratio=${d.ratio.toFixed(2)} ent=${d.ent.toFixed(3)}`
            );
        }

        await new Promise<void>(r => requestAnimationFrame(() => r()));

        const winnerChanged = d.winner.name !== this.state.lastDispatchWinner;
        const enoughTime = now - this.state.lastDispatchTime > this.config.dispatchMinIntervalMs;

        if (winnerChanged || enoughTime) {
            this.state.lastDispatchWinner = d.winner.name;
            this.state.lastDispatchTime = now;
            window.dispatchEvent(new CustomEvent(EVENTS.RESULT_FOUND, {
                detail: {winner: d.winner, others: d.others, stable: d.stable, activity: d.activity}
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
            console.log(`🏁 FILETEST RAW | ${raw.top3.map(x => `${x.name}:${(x.score * 100).toFixed(0)}%`).join(' ')}`);
            console.log(
                `🧮 RAW top1=${raw.top1.name}:${(raw.top1.score * 100).toFixed(1)}% ` +
                `top2=${raw.top2?.name ?? "-"}:${((raw.top2?.score ?? 0) * 100).toFixed(1)}% ` +
                `diff=${raw.diff.toFixed(3)} ratio=${raw.ratio.toFixed(2)} ent=${raw.ent.toFixed(3)}`
            );
        }

        // 2) DECISION (same logic as live)
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
                    detail: {winner: decision.winner, others: decision.others}
                })
            );
        }

        return {probs, raw, decision};
    }

    private emitIdle() {
        requestAnimationFrame(() => {
            window.dispatchEvent(new CustomEvent(EVENTS.RESULT_FOUND, {
                detail: {
                    winner: {name: STATE_IDLE, score: 0},
                    others: [],
                    stable: false,
                    activity: 'silence'
                },
            }));
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
        return raw ? raw.replace(/_/g, ' ').trim().toUpperCase() : 'UNKNOWN';
    }

    private isBackgroundName(name: string): boolean {
        return name.trim().toUpperCase() === 'BACKGROUND';
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
        const nonBg = topMatches.filter(m => !this.isBackgroundName(m.name));
        const bgScore = topMatches.find(m => this.isBackgroundName(m.name))?.score ?? 0;

        const top1 = nonBg[0] ?? {name: STATE_IDLE, score: 0};
        const top2 = nonBg[1];

        const now = Date.now();
        if (this.isDebug && now - this.debug.topLogLast > this.debug.topLogInterval) {
            this.debug.topLogLast = now;
            const fmt = (m?: QariMatch) => (m ? `${m.name}:${(m.score * 100).toFixed(0)}%` : '-');
            console.log(`🏆 TOP3 [Ent:${ent.toFixed(2)}] | ${fmt(topMatches[0])} | ${fmt(topMatches[1])} | ${fmt(topMatches[2])}`);
        }

        const ratio = top2?.score ? top1.score / top2.score : Infinity;
        const diff = top2 ? top1.score - top2.score : top1.score;

        if (this.isDebug) {
            const t1 = top1?.score ?? 0;
            const t2 = top2?.score ?? 0;
            const ratio2 = t2 > 0 ? t1 / t2 : Infinity;
            const diff2 = t2 > 0 ? t1 - t2 : t1;
            console.log(
                `🧮 top1=${top1?.name}:${(t1 * 100).toFixed(1)}% ` +
                `top2=${top2?.name ?? "-"}:${(t2 * 100).toFixed(1)}% ` +
                `diff=${diff2.toFixed(3)} ratio=${ratio2.toFixed(2)}`
            );
        }

        // If we have no non-bg candidate at all, treat as idle/noise
        if (top1.name === STATE_IDLE || top1.score <= 0) {
            this.state.pendingWinner = null;
            this.state.pendingCount = 0;
            this.state.noWinCount++;

            return {
                winner: {name: STATE_IDLE, score: 0},
                others: topMatches.filter(m => m.score > 0.05 && !this.isBackgroundName(m.name)),
                ent,
                top1,
                top2,
                diff,
                ratio,
                stable: false,
                activity: bgScore > 0.7 ? 'noise' : 'voiced'
            };
        }

        const notConfused = ent < 0.95;
        const strongTop1 = top1.score > this.acceptTop1Min;

        // ✅ Make clearWin robust when top2 is missing
        const clearWin = top2
            ? (ratio > 1.35 && diff > 0.04)
            : (top1.score > Math.max(0.45, this.acceptTop1Min));

        let winner: QariMatch = {name: STATE_IDLE, score: 0};
        let stable = false;

        if (notConfused && strongTop1 && clearWin) {
            this.state.noWinCount = 0;
            if (this.state.pendingWinner === top1.name) this.state.pendingCount++;
            else {
                this.state.pendingWinner = top1.name;
                this.state.pendingCount = 1;
            }

            if (this.state.pendingCount >= this.config.stabilityThreshold) {
                this.state.stableWinner = top1;
                stable = true;
            }
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

        stable =
            winner.name !== STATE_IDLE &&
            this.state.stableWinner != null &&
            winner.name === this.state.stableWinner.name &&
            notConfused && strongTop1 && clearWin;

        // We take the "frame-stable" winner and run it through the "session-lock" logic
        const {finalWinner, isLocked} = this.sessionLocker.update(
            stable ? winner : null,
            bgScore > 0.7 ? 'noise' : 'voiced',
            this.config,
            this.isDebug
        );

        return {
            winner: finalWinner,
            others: topMatches.filter(m => m.score > 0.05 && !this.isBackgroundName(m.name)),
            ent,
            top1,
            top2,
            diff,
            ratio,
            stable: isLocked || stable,
            activity: 'voiced' as const,
            locked: isLocked
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
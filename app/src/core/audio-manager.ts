import {platform} from "./platform-service";
import {type AppStatePayload, EVENTS} from "./events";

export class AudioManager {
    private static instance: AudioManager;

    // Nodes
    private _context: AudioContext | null = null;
    public analyser: AnalyserNode | null = null;
    public worklet: AudioWorkletNode | null = null;
    private _stream: MediaStream | null = null;

    // State
    private isModuleLoaded = false;
    public isRunning = false;

    // Configuration
    private config = {
        targetSampleRate: 16000,
        workletPath: "/processors/resampler-processor.js",
        latencyHint: (platform.isNative ? "playback" : "interactive") as AudioContextLatencyCategory
    };

    // Feature Flags
    private flags = {
        farFieldMode: false // Default: OFF / Purist Mode
    };

    private constructor() {
        // Use the Typed Event
        window.addEventListener(EVENTS.APP_STATE_CHANGE, (e: Event) => {
            const {isActive} = (e as CustomEvent<AppStatePayload>).detail;
            if (!isActive) {
                console.log("⏸️ App Backgrounded - Stopping Audio");
                this.stop();
            }
        });
    }

    static getInstance() {
        return this.instance || (this.instance = new AudioManager());
    }

    /**
     * Feature: Toggle Far-Field Mode (AGC/Echo Cancellation)
     */
    public setFarFieldMode(enabled: boolean) {
        this.flags.farFieldMode = enabled;
        console.log(`🎤 Far-Field Mode: ${enabled ? 'ON (AGC Active)' : 'OFF (Raw Audio)'}`);
    }

    public isFarFieldMode(): boolean {
        return this.flags.farFieldMode;
    }

    get context(): AudioContext {
        if (!this._context) {
            const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
            this._context = new AudioContextClass({
                latencyHint: this.config.latencyHint,
            });
            this.analyser = this._context.createAnalyser();
            console.log("🎛️ AudioContext init sampleRate =", this._context.sampleRate);
        }
        return this._context;
    }

    async start(onDataReceived: (data: Float32Array) => void) {
        if (this.isRunning) return;

        // 1. Stream Acquisition (Browser quirks handling)
        // Firefox requires getting the stream BEFORE the context to handle sample rates correctly.
        this._stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: this.flags.farFieldMode,
                noiseSuppression: this.flags.farFieldMode,
                autoGainControl: this.flags.farFieldMode,
                sampleRate: this.config.targetSampleRate as any // Best effort
            }
        });

        const track = this._stream.getAudioTracks()[0];
        const settings = track.getSettings();
        const trackSr = (settings as any).sampleRate as number | undefined;

        console.log("🎛️ Mic Settings:", settings);

        // 2. Context Reconciliation
        // If an old context exists with a different sample rate than the new stream, kill it.
        if (this._context && trackSr && this._context.sampleRate !== trackSr) {
            await this._context.close();
            this._context = null;
            this.isModuleLoaded = false;
        }

        // 3. Context Creation
        if (!this._context) {
            const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
            this._context = new AudioContextClass({
                latencyHint: this.config.latencyHint,
                // Match the mic's hardware rate if known to avoid OS-level resampling glitches
                ...(trackSr ? {sampleRate: trackSr} : {})
            });
            this.analyser = this._context.createAnalyser();
        }

        // 4. Resume (Mobile Requirement)
        if (this._context.state === "suspended") await this._context.resume();

        // 5. Worklet Injection
        if (!this.isModuleLoaded) {
            await this._context.audioWorklet.addModule(this.config.workletPath);
            this.isModuleLoaded = true;
        }

        // 6. Build Graph
        const source = this._context.createMediaStreamSource(this._stream);

        // Reset old worklet node if exists
        if (this.worklet) {
            this.worklet.disconnect();
            this.worklet = null;
        }

        this.worklet = new AudioWorkletNode(this._context, "resampler-processor");
        this.worklet.port.onmessage = (e) => onDataReceived(e.data);

        // Source -> Analyser (Visuals)
        source.connect(this.analyser!);
        // Source -> Worklet (Processing)
        source.connect(this.worklet);

        // Worklet -> Mute -> Destination (Keep graph alive without feedback)
        const mute = this._context.createGain();
        mute.gain.value = 0;
        this.worklet.connect(mute).connect(this._context.destination);

        this.isRunning = true;
        platform.hapticSuccess();
        console.log(`🎤 Mic Active at ${this._context.sampleRate} Hz | Far-Field: ${this.flags.farFieldMode}`);
    }

    stop() {
        if (this.worklet) {
            this.worklet.disconnect();
            this.worklet = null;
        }
        if (this._stream) {
            this._stream.getTracks().forEach(t => t.stop());
            this._stream = null;
        }
        if (this._context) {
            this._context.close();
            this._context = null;
        }

        this.isRunning = false;
        this.isModuleLoaded = false;
        console.log("🛑 Microphone Hardware Released.");
    }
}

export const audioManager = AudioManager.getInstance();
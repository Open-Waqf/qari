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
        targetSampleRate: 22050,
        workletPath: "/processors/resampler-processor.js",
        latencyHint: (platform.isNative ? "playback" : "interactive") as AudioContextLatencyCategory
    };

    // Feature Flags
    private flags = {
        farFieldMode: false // Default: OFF / Purist Mode
    };

    public onDataReceived: ((data: Float32Array) => void) | null = null;

    private constructor() {
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

    get sampleRate(): number {
        return this._context ? this._context.sampleRate : 0;
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

    // Lazy Getter for Context
    get context(): AudioContext {
        if (!this._context) {
            const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
            this._context = new AudioContextClass({
                latencyHint: this.config.latencyHint,
            });
            this.analyser = this._context.createAnalyser();
            console.log("🎛️ AudioContext Lazy Init | Rate:", this._context.sampleRate);
        }
        return this._context;
    }

    async start(onDataReceived: (data: Float32Array) => void) {
        this.onDataReceived = onDataReceived;

        if (this.isRunning) {
            console.log("🔁 AudioManager already running — callback updated");
            return;
        }

        try {
            // 1. Stream Acquisition
            // 🟢 FIX: Removed 'sampleRate' constraint. Let HW run native (44.1/48k).
            // This prevents OverconstrainedError on mobiles.
            this._stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,

                    echoCancellation: this.flags.farFieldMode,
                    noiseSuppression: this.flags.farFieldMode,
                    autoGainControl: this.flags.farFieldMode,

                    // @ts-ignore - Android "Nuclear" Options
                    googEchoCancellation: this.flags.farFieldMode,
                    googAutoGainControl: this.flags.farFieldMode,
                    googNoiseSuppression: this.flags.farFieldMode,
                    googHighpassFilter: this.flags.farFieldMode,
                    googAudioMirroring: false,
                }
            });

            const track = this._stream.getAudioTracks()[0];
            const settings = track.getSettings();
            const trackSr = (settings as any).sampleRate as number | undefined;

            console.log("🎛️ Mic Hardware Settings:", settings);

            // 2. Context Reconciliation
            if (this._context && trackSr && this._context.sampleRate !== trackSr) {
                console.warn(`⚠️ Mismatch! Context: ${this._context.sampleRate}, Mic: ${trackSr}. Recreating...`);
                await this._context.close();
                this._context = null;
                this.isModuleLoaded = false;
            }

            // 3. Context Creation
            if (!this._context) {
                const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
                this._context = new AudioContextClass({
                    latencyHint: this.config.latencyHint,
                    // Match the mic exactly to prevent OS-level resampling glitches
                    ...(trackSr ? {sampleRate: trackSr} : {})
                });
                this.analyser = this._context.createAnalyser();
            }

            if (this._context.state === "suspended") {
                await this._context.resume();
            }

            // 4. Worklet Injection
            if (!this.isModuleLoaded) {
                try {
                    await this._context.audioWorklet.addModule(this.config.workletPath);
                    this.isModuleLoaded = true;
                } catch (e) {
                    console.error(`❌ Failed to load worklet at ${this.config.workletPath}`, e);
                    throw e;
                }
            }

            // 5. Build Graph
            const source = this._context.createMediaStreamSource(this._stream);

            if (this.worklet) {
                this.worklet.disconnect();
                this.worklet = null;
            }

            this.worklet = new AudioWorkletNode(this._context, "resampler-processor");
            this.worklet.port.onmessage = (e) => this.onDataReceived?.(e.data);

            // Graph: Source -> Analyser -> Worklet -> Mute -> Dest
            source.connect(this.analyser!);
            source.connect(this.worklet);

            const mute = this._context.createGain();
            mute.gain.value = 0;
            this.worklet.connect(mute).connect(this._context.destination);

            this.isRunning = true;
            platform.hapticSuccess();
            console.log(`🎤 Mic Active at ${this._context.sampleRate} Hz | Far-Field: ${this.flags.farFieldMode}`);

        } catch (error) {
            console.error("🚨 AudioManager Start Failed:", error);
            this.stop();
        }
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
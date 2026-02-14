import {platform} from "./platform-service";

export class AudioManager {
    private static instance: AudioManager;
    private _context: AudioContext | null = null;
    public analyser: AnalyserNode | null = null;
    public worklet: AudioWorkletNode | null = null;
    private isModuleLoaded = false;
    private _stream: MediaStream | null = null;


    // NEW: Far-Field State (Default: OFF / Purist Mode)
    private _farFieldMode = false;

    public isRunning = false;

    private constructor() {
        window.addEventListener('app-state-change', (e: any) => {
            if (!e.detail.isActive) {
                console.log("⏸️ App Backgrounded - Stopping Audio");
                this.stop();
            }
        });
    }

    static getInstance() {
        return this.instance || (this.instance = new AudioManager());
    }

    /**
     * Step 1 Feature: Toggle Far-Field Mode (AGC)
     * Useful for laptop mics listening to a phone in a room.
     */
    public setFarFieldMode(enabled: boolean) {
        this._farFieldMode = enabled;
        console.log(`🎤 Far-Field Mode: ${enabled ? 'ON (AGC Active)' : 'OFF (Raw Audio)'}`);
    }

    public isFarFieldMode(): boolean {
        return this._farFieldMode;
    }

    get context(): AudioContext {
        if (!this._context) {
            const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
            this._context = new AudioContextClass({
                latencyHint: platform.isNative ? "playback" : "interactive",
                // IMPORTANT: no forced sampleRate here
            });
            this.analyser = this._context.createAnalyser();
            console.log("🎛️ AudioContext sampleRate =", this._context.sampleRate);
        }
        return this._context;
    }

    async start(onDataReceived: (data: Float32Array) => void) {
        if (this.isRunning) return;

        const canRecord = await platform.checkMicPermission();
        if (!canRecord) throw new Error("Microphone permission missing");

        // 1) Get stream FIRST (Firefox requirement)
        this._stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: this._farFieldMode,
                noiseSuppression: this._farFieldMode,
                autoGainControl: this._farFieldMode,
                // you can TRY this; Firefox usually ignores it, but it won't hurt:
                sampleRate: 16000 as any
            }
        });

        const track = this._stream.getAudioTracks()[0];
        const settings = track.getSettings();
        console.log("🎛️ getSettings()", settings);
        console.log("🎛️ getConstraints()", track.getConstraints());

        // 2) If context exists with different SR, recreate it
        const trackSr = (settings as any).sampleRate as number | undefined;
        if (this._context && trackSr && this._context.sampleRate !== trackSr) {
            await this._context.close();
            this._context = null;
            this.isModuleLoaded = false;
        }

        // 3) Create context AFTER stream (and match SR if provided)
        if (!this._context) {
            const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
            const opts: AudioContextOptions = {
                latencyHint: platform.isNative ? "playback" : "interactive",
                ...(trackSr ? {sampleRate: trackSr} : {})
            };
            this._context = new AudioContextClass(opts);
            this.analyser = this._context.createAnalyser();
            console.log("🎛️ AudioContext sampleRate =", this._context.sampleRate);
        }

        // 4) Resume context
        if (this._context.state === "suspended") await this._context.resume();

        // 5) Load worklet
        if (!this.isModuleLoaded) {
            await this._context.audioWorklet.addModule("/processors/resampler-processor.js");
            this.isModuleLoaded = true;
        }

        const source = this._context.createMediaStreamSource(this._stream);

        // 6) Worklet node
        if (this.worklet) {
            this.worklet.disconnect();
            this.worklet = null;
        }
        this.worklet = new AudioWorkletNode(this._context, "resampler-processor");

        // 7) Connect graph
        source.connect(this.analyser!);
        source.connect(this.worklet);

        const mute = this._context.createGain();
        mute.gain.value = 0;
        this.worklet.connect(mute).connect(this._context.destination);

        this.worklet.port.onmessage = (e) => onDataReceived(e.data);

        this.isRunning = true;
        platform.hapticSuccess();
        console.log(`🎤 Mic Active at ${this._context.sampleRate} Hz | Far-Field: ${this._farFieldMode}`);
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
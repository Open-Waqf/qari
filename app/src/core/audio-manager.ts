// app/src/core/audio-manager.ts
import {platform} from "./platform-service"; // .ts extension removed for standard resolution

export class AudioManager {
    private static instance: AudioManager;
    private _context: AudioContext | null = null;
    public analyser: AnalyserNode | null = null;
    public worklet: AudioWorkletNode | null = null;
    private isModuleLoaded = false;

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

    get context(): AudioContext {
        if (!this._context) {
            // iOS/Android WebViews usually need this specific config for best latency
            const options: AudioContextOptions = {
                latencyHint: platform.isNative ? 'playback' : 'interactive',
            };

            const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
            this._context = new AudioContextClass(options);

            this.analyser = this._context.createAnalyser();
        }
        return this._context;
    }

    async start(onDataReceived: (data: Float32Array) => void) {

        const canRecord = await platform.checkMicPermission();
        if (!canRecord) {
            throw new Error("Microphone permission missing");
        }

        // 1. Force Resume
        if (this.context.state === 'suspended') {
            await this.context.resume();
        }

        // 2. Load the processor once
        if (!this.isModuleLoaded) {
            try {
                await this.context.audioWorklet.addModule('/processors/resampler-processor.js');
                this.isModuleLoaded = true;
            } catch (e) {
                console.error("AudioWorklet Load Failed. Are paths correct in dist?", e);
                // Reset context to ensure clean state on retry
                this._context?.close();
                this._context = null;
                throw new Error(`Worklet Load Failed: ${e}`);
            }
        }

        // 3. Request Mic
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: platform.isNative,
                noiseSuppression: platform.isNative,
                autoGainControl: false,
            }
        });

        const source = this.context.createMediaStreamSource(stream);

        // Clean up old worklet
        if (this.worklet) {
            this.worklet.disconnect();
            this.worklet = null;
        }

        // 4. Create Worklet Node
        // If this throws "Unknown AudioWorklet name", it means addModule failed silently above
        try {
            this.worklet = new AudioWorkletNode(this.context, 'resampler-processor');
        } catch (e) {
            throw new Error(`Worklet Registered incorrectly. Did the file load? ${e}`);
        }

        // 5. Connect the Graph
        source.connect(this.analyser!);
        source.connect(this.worklet);

        // Keep worklet alive with muted output
        const mute = this.context.createGain();
        mute.gain.value = 0;
        this.worklet.connect(mute).connect(this.context.destination);

        this.worklet.port.onmessage = (e) => onDataReceived(e.data);

        platform.hapticSuccess();
        console.log(`🎤 Mic Active at ${this.context.sampleRate} Hz`);
    }

    stop() {
        if (this.worklet) {
            this.worklet.disconnect();
            this.worklet = null;
        }
        if (this._context) {
            this._context.close();
            this._context = null;
        }
        this.isModuleLoaded = false;
        console.log("🛑 Microphone Hardware Released.");
    }
}

export const audioManager = AudioManager.getInstance();
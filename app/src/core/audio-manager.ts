export class AudioManager {
    private static instance: AudioManager;
    private _context: AudioContext | null = null;
    public analyser: AnalyserNode | null = null;
    public worklet: AudioWorkletNode | null = null;
    private isModuleLoaded = false; // Prevents double-loading errors

    private constructor() {
    }

    static getInstance() {
        return this.instance || (this.instance = new AudioManager());
    }

    get context(): AudioContext {
        if (!this._context) {
            this._context = new AudioContext();
            this.analyser = this._context.createAnalyser();
        }
        return this._context;
    }

    async start(onDataReceived: (data: Float32Array) => void) {

        if (window.hasOwnProperty('Capacitor')) {
            console.log("📱 Mobile Native Environment Detected");
        }

        // 1. Force Resume (Crucial for Chrome/Edge)
        if (this.context.state === 'suspended') {
            await this.context.resume();
        }

        // 2. Load the processor once
        if (!this.isModuleLoaded) {
            const workletUrl = new URL('/processors/resampler-processor.js', import.meta.url).href;
            await this.context.audioWorklet.addModule(workletUrl);
            this.isModuleLoaded = true;
        }

        // 3. Request Mic
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false,
            } as any
        });

        const source = this.context.createMediaStreamSource(stream);

        // Clean up old worklet if it exists
        if (this.worklet) this.worklet.disconnect();

        this.worklet = new AudioWorkletNode(this.context, 'resampler-processor');

        // 4. Connect the Graph
        source.connect(this.analyser!);
        source.connect(this.worklet);

        // Keep worklet alive with muted output
        const mute = this.context.createGain();
        mute.gain.value = 0;
        this.worklet.connect(mute).connect(this.context.destination);

        this.worklet.port.onmessage = (e) => onDataReceived(e.data);

        console.log("🎤 Microphone and Resampler Active at", this.context.sampleRate, "Hz");
    }

    /**
     * Stops the microphone and releases the hardware stream.
     * Essential for resetting the state between calibration and live mode.
     */
    stop() {
        if (this.worklet) {
            this.worklet.disconnect();
            this.worklet = null;
        }
        if (this._context) {
            // Closing the context fully releases the microphone hardware
            this._context.close();
            this._context = null;
        }
        this.isModuleLoaded = false; // Reset so next start re-loads the processor
        console.log("🛑 Microphone Hardware Released.");
    }
}

export const audioManager = AudioManager.getInstance();
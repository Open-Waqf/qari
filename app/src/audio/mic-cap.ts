export class MicCap16k {
    private buf: Float32Array | null = null;
    private write = 0;
    private sr = 16000;

    public active = false;
    public ready = false;

    start(seconds = 8, sampleRate = 16000) {
        this.sr = sampleRate;
        this.buf = new Float32Array(Math.floor(seconds * this.sr));
        this.write = 0;
        this.active = true;
        this.ready = false;
        console.log(`🎙️ MICCAP START | ${seconds}s @ ${this.sr}Hz`);
    }

    // called from audio loop (expects 16k chunks)
    onChunk(chunk16k: Float32Array) {
        if (!this.active || !this.buf) return;

        const remaining = this.buf.length - this.write;
        if (remaining <= 0) return;

        const n = Math.min(remaining, chunk16k.length);
        this.buf.set(chunk16k.subarray(0, n), this.write);
        this.write += n;

        if (this.write >= this.buf.length) {
            this.active = false;
            this.ready = true;
            console.log(`🎙️ MICCAP FULL | ready to analyze`);
        }
    }

    stopEarly() {
        if (!this.buf) return;
        this.active = false;
        this.ready = true;
        console.log(`🎙️ MICCAP STOP | captured ${(this.write / this.sr).toFixed(2)}s`);
    }

    take(): Float32Array | null {
        if (!this.buf || !this.ready) return null;
        const out = this.buf.subarray(0, this.write); // view, no copy
        this.buf = null;
        this.write = 0;
        this.ready = false;
        return out;
    }
}

// ✅ singleton shared by main + audio pipeline
export const micCap = new MicCap16k();

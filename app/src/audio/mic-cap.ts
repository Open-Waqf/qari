export class MicCap {
    private buf: Float32Array | null = null;
    private write = 0;
    private sr = 22050; // 🟢 Updated

    public active = false;
    public ready = false;

    start(seconds = 8, sampleRate = 22050) { // 🟢 Updated
        this.sr = sampleRate;
        this.buf = new Float32Array(Math.floor(seconds * this.sr));
        this.write = 0;
        this.active = true;
        this.ready = false;
        console.log(`🎙️ MICCAP START | ${seconds}s @ ${this.sr}Hz`);
    }

    // called from audio loop
    onChunk(chunk: Float32Array) {
        if (!this.active || !this.buf) return;

        const remaining = this.buf.length - this.write;
        if (remaining <= 0) return;

        const n = Math.min(remaining, chunk.length);
        this.buf.set(chunk.subarray(0, n), this.write);
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

// Export a singleton instance for global use
export const micCap = new MicCap();
class ResamplerProcessor extends AudioWorkletProcessor {
    constructor() {
        super();

        // ✅ App parity with 22k training
        this.TARGET = 22050;

        // Output buffer
        this.out = new Float32Array(16384);
        this.outLen = 0;

        // FIR settings
        this.TAPS = 63;

        // Specialized 44.1k -> 22.05k state (Decimate by 2)
        this.h44 = null;
        this.hist44 = new Float32Array(this.TAPS);
        this.histPos44 = 0;
        this.sampleCounter44 = 0;

        // Generic lowpass state (for fs != 44.1k)
        this.hLP = null;
        this.hLPFs = 0;
        this.hLPCut = 0;
        this.histLP = new Float32Array(this.TAPS);
        this.histPosLP = 0;

        this.tmpLP = new Float32Array(128);

        // Fractional resample state
        this.frac = 0;
        this.prev = 0;

        this.didLog = false;
    }

    buildFIR(fs, cutoffHz) {
        const cutoff = Math.min(cutoffHz, 0.45 * fs);
        const taps = this.TAPS;
        const M = taps - 1;
        const fc = cutoff / fs;

        const h = new Float32Array(taps);
        let sum = 0;

        for (let i = 0; i < taps; i++) {
            const n = i - M / 2;
            let sinc = (n === 0) ? 2 * fc : Math.sin(2 * Math.PI * fc * n) / (Math.PI * n);
            // Blackman window
            const w = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / M) + 0.08 * Math.cos((4 * Math.PI * i) / M);
            const v = sinc * w;
            h[i] = v;
            sum += v;
        }
        for (let i = 0; i < taps; i++) h[i] /= sum;
        return h;
    }

    pushOut(sample) {
        if (this.outLen >= this.out.length) {
            const bigger = new Float32Array(this.out.length * 2);
            bigger.set(this.out);
            this.out = bigger;
        }
        this.out[this.outLen++] = sample;
    }

    flush4096() {
        while (this.outLen >= 4096) {
            this.port.postMessage(this.out.slice(0, 4096));
            this.out.copyWithin(0, 4096, this.outLen);
            this.outLen -= 4096;
        }
    }

    // ✅ RESTORED: Specialized path for 44.1k -> 22.05k
    firDecimateBy2(inCh) {
        if (!this.h44) this.h44 = this.buildFIR(sampleRate, 10000);

        for (let i = 0; i < inCh.length; i++) {
            this.hist44[this.histPos44] = inCh[i];
            this.histPos44 = (this.histPos44 + 1) % this.TAPS;
            this.sampleCounter44++;

            if (this.sampleCounter44 % 2 !== 0) continue;

            let y = 0;
            let idx = this.histPos44;
            for (let k = 0; k < this.TAPS; k++) {
                idx = (idx - 1 + this.TAPS) % this.TAPS;
                y += this.hist44[idx] * this.h44[k];
            }
            this.pushOut(y);
        }
    }

    firLowpassBlock(inCh, fs) {
        const cutoffHz = 0.45 * Math.min(fs, this.TARGET);
        if (!this.hLP || this.hLPFs !== fs || this.hLPCut !== cutoffHz) {
            this.hLP = this.buildFIR(fs, cutoffHz);
            this.hLPFs = fs;
            this.hLPCut = cutoffHz;
            this.histLP.fill(0);
            this.histPosLP = 0;
            this.frac = 0;
            this.prev = 0;
        }

        if (this.tmpLP.length < inCh.length) this.tmpLP = new Float32Array(inCh.length);

        for (let i = 0; i < inCh.length; i++) {
            this.histLP[this.histPosLP] = inCh[i];
            this.histPosLP = (this.histPosLP + 1) % this.TAPS;
            let y = 0;
            let idx = this.histPosLP;
            for (let k = 0; k < this.TAPS; k++) {
                idx = (idx - 1 + this.TAPS) % this.TAPS;
                y += this.histLP[idx] * this.hLP[k];
            }
            this.tmpLP[i] = y;
        }
        return this.tmpLP.subarray(0, inCh.length);
    }

    linearResample(inCh, ratio) {
        let t = this.frac;
        while (t < inCh.length - 1) {
            const i0 = Math.floor(t);
            const a = t - i0;
            const s0 = (i0 < 0) ? this.prev : inCh[i0];
            const s1 = inCh[i0 + 1];
            this.pushOut(s0 + (s1 - s0) * a);
            t += ratio;
        }
        this.frac = t - inCh.length;
        this.prev = inCh[inCh.length - 1];
    }

    process(inputs) {
        const input = inputs[0];
        if (!input || !input[0]) return true;
        const inCh = input[0];
        const fs = sampleRate;

        if (!this.didLog) {
            console.log(`🎛️ Resampler: ${fs}Hz -> ${this.TARGET}Hz`);
            if (fs === 44100) console.log("✅ Mode: FIR Decimate-by-2");
            else if (fs === this.TARGET) console.log("✅ Mode: Pass-through");
            else console.log("✅ Mode: FIR + Linear Resample");
            this.didLog = true;
        }

        if (fs === this.TARGET) {
            for (let i = 0; i < inCh.length; i++) this.pushOut(inCh[i]);
        } else if (fs === 44100) {
            this.firDecimateBy2(inCh);
        } else {
            const filtered = this.firLowpassBlock(inCh, fs);
            this.linearResample(filtered, fs / this.TARGET);
        }

        this.flush4096();
        return true;
    }
}

registerProcessor("resampler-processor", ResamplerProcessor);
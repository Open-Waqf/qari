class ResamplerProcessor extends AudioWorkletProcessor {
    constructor() {
        super();

        // ✅ App parity with training
        this.TARGET = 22050;

        // Output buffer
        this.out = new Float32Array(16384);
        this.outLen = 0;

        // FIR settings
        this.TAPS = 63; // odd number

        // FIR lowpass state
        this.hLP = null;
        this.hLPFs = 0;
        this.hLPCut = 0;

        this.histLP = new Float32Array(this.TAPS);
        this.histPosLP = 0;
        this.filledLP = 0;

        this.tmpLP = new Float32Array(128); // worklet block size

        // Fractional resample state
        this.frac = 0;
        this.prev = 0;

        this.didLog = false;
    }

    buildFIR(fs, cutoffHz) {
        // Clamp cutoff to input Nyquist to avoid invalid designs
        const cutoff = Math.min(cutoffHz, 0.45 * fs);

        // Windowed-sinc lowpass (Blackman), DC-normalized
        const taps = this.TAPS;
        const M = taps - 1;
        const fc = cutoff / fs; // cycles/sample

        const h = new Float32Array(taps);
        let sum = 0;

        for (let i = 0; i < taps; i++) {
            const n = i - M / 2;

            let sinc;
            if (n === 0) sinc = 2 * fc;
            else sinc = Math.sin(2 * Math.PI * fc * n) / (Math.PI * n);

            // Blackman window
            const w =
                0.42 -
                0.5 * Math.cos((2 * Math.PI * i) / M) +
                0.08 * Math.cos((4 * Math.PI * i) / M);

            const v = sinc * w;
            h[i] = v;
            sum += v;
        }

        // Normalize DC gain to 1.0
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
            const chunk = this.out.slice(0, 4096);
            this.port.postMessage(chunk);

            this.out.copyWithin(0, 4096, this.outLen);
            this.outLen -= 4096;
        }
    }

    // FIR lowpass (anti-alias) for downsampling
    firLowpassBlock(inCh, fs) {
        // ✅ Critical: bandlimit to TARGET Nyquist when downsampling
        // cutoff ≈ 0.45 * min(inputFs, targetFs)
        const cutoffHz = 0.45 * Math.min(fs, this.TARGET);

        if (!this.hLP || this.hLPFs !== fs || this.hLPCut !== cutoffHz) {
            this.hLP = this.buildFIR(fs, cutoffHz);
            this.hLPFs = fs;
            this.hLPCut = cutoffHz;

            this.histLP.fill(0);
            this.histPosLP = 0;
            this.filledLP = 0;

            // Reset fractional resample phase
            this.frac = 0;
            this.prev = 0;
        }

        if (this.tmpLP.length < inCh.length) {
            this.tmpLP = new Float32Array(inCh.length);
        }

        for (let i = 0; i < inCh.length; i++) {
            const x = inCh[i];

            this.histLP[this.histPosLP] = x;
            this.histPosLP = (this.histPosLP + 1) % this.TAPS;
            if (this.filledLP < this.TAPS) this.filledLP++;

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

    // Linear resample (fed with lowpassed audio)
    linearResample(inCh, ratio) {
        if (inCh.length === 0) return;

        let t = this.frac;

        while (t < inCh.length - 1) {
            const i0 = Math.floor(t);
            const a = t - i0;

            const s0 = (i0 < 0) ? this.prev : inCh[i0];
            const s1 = inCh[i0 + 1];

            const y = s0 + (s1 - s0) * a;
            this.pushOut(y);

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
            console.log(`🎛️ Worklet fs=${fs}Hz → target=${this.TARGET}Hz`);
            this.didLog = true;
        }

        // Pass-through if already TARGET
        if (fs === this.TARGET) {
            for (let i = 0; i < inCh.length; i++) this.pushOut(inCh[i]);
            this.flush4096();
            return true;
        }

        // Generic bandlimited resample for all other rates (48k, 44.1k, etc.)
        const filtered = this.firLowpassBlock(inCh, fs);
        const ratio = fs / this.TARGET;
        this.linearResample(filtered, ratio);
        this.flush4096();

        return true;
    }
}

registerProcessor("resampler-processor", ResamplerProcessor);

class ResamplerProcessor extends AudioWorkletProcessor {
    constructor() {
        super();

        this.TARGET = 16000;

        // Output buffer
        this.out = new Float32Array(16384);
        this.outLen = 0;

        // FIR settings
        this.DECIM = 3;
        this.TAPS = 63;          // odd number
        this.CUTOFF_HZ = 7300;   // under 8k Nyquist of 16k

        // --- 48k -> 16k state (decimate by 3) ---
        this.h48 = null;
        this.hist48 = new Float32Array(this.TAPS);
        this.histPos48 = 0;
        this.filled48 = 0;
        this.sampleCounter48 = 0;

        // --- Generic lowpass state (for fs != 48k) ---
        this.hLP = null;
        this.hLPFs = 0;
        this.histLP = new Float32Array(this.TAPS);
        this.histPosLP = 0;
        this.filledLP = 0;

        this.tmpLP = new Float32Array(128); // reused per block

        // Fallback resample state (fractional position)
        this.frac = 0;
        this.prev = 0;

        this.didLog = false;
    }

    buildFIR(fs) {
        // Clamp cutoff to input Nyquist to avoid invalid designs
        const cutoffHz = Math.min(this.CUTOFF_HZ, 0.45 * fs);

        // Windowed-sinc lowpass (Blackman), DC-normalized
        const taps = this.TAPS;
        const M = taps - 1;
        const fc = cutoffHz / fs; // cycles/sample

        const h = new Float32Array(taps);
        let sum = 0;

        for (let i = 0; i < taps; i++) {
            const n = i - M / 2;

            let sinc;
            if (n === 0) {
                sinc = 2 * fc;
            } else {
                sinc = Math.sin(2 * Math.PI * fc * n) / (Math.PI * n);
            }

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
            const chunk = this.out.slice(0, 4096); // copy
            this.port.postMessage(chunk);

            this.out.copyWithin(0, 4096, this.outLen);
            this.outLen -= 4096;
        }
    }

    // --- 48k -> 16k path (unchanged) ---
    firDecimateBy3(inCh) {
        if (!this.h48) this.h48 = this.buildFIR(sampleRate);

        for (let i = 0; i < inCh.length; i++) {
            const x = inCh[i];

            this.hist48[this.histPos48] = x;
            this.histPos48 = (this.histPos48 + 1) % this.TAPS;

            if (this.filled48 < this.TAPS) {
                this.filled48++;
                this.sampleCounter48++;
                continue;
            }

            this.sampleCounter48++;
            if (this.sampleCounter48 % this.DECIM !== 0) continue;

            let y = 0;
            let idx = this.histPos48;
            for (let k = 0; k < this.TAPS; k++) {
                idx = (idx - 1 + this.TAPS) % this.TAPS;
                y += this.hist48[idx] * this.h48[k];
            }

            this.pushOut(y);
        }
    }

    // --- NEW: FIR lowpass for fallback (fs != 48k) ---
    firLowpassBlock(inCh, fs) {
        if (!this.hLP || this.hLPFs !== fs) {
            this.hLP = this.buildFIR(fs);
            this.hLPFs = fs;
            this.histLP.fill(0);
            this.histPosLP = 0;
            this.filledLP = 0;

            // Reset fractional resample state so we don't carry old phase
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

            // Convolution (symmetric FIR, newest sample first)
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

    // Fallback resample (now fed with lowpassed audio)
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
            console.log(`🎛️ Worklet fs=${fs}Hz (target 16k)`);

            if (fs === 48000) {
                console.log("✅ Mode: FIR decimate-by-3 (48k→16k)");
            } else if (fs === this.TARGET) {
                console.log("✅ Mode: Direct pass-through (16k)");
            } else {
                console.log("✅ Mode: FIR lowpass + linear fallback");
            }

            this.didLog = true; // Latch set to true forever
        }

        // Fast path: already 16k
        if (fs === this.TARGET) {
            for (let i = 0; i < inCh.length; i++) this.pushOut(inCh[i]);
            this.flush4096();
            return true;
        }

        // Best path: 48k -> 16k via FIR decimation-by-3
        if (fs === 48000) {
            this.firDecimateBy3(inCh);
            this.flush4096();
            return true;
        }

        // NEW: Bandlimited fallback for all other rates (esp. 44.1k)
        if (this.didLog) console.log("✅ Resample mode: FIR lowpass + linear (generic→16k)");
        const filtered = this.firLowpassBlock(inCh, fs);
        const ratio = fs / this.TARGET;
        this.linearResample(filtered, ratio);
        this.flush4096();
        return true;
    }
}

registerProcessor("resampler-processor", ResamplerProcessor);
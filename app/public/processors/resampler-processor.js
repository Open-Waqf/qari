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

        // ✅ CUBIC STATE: Store last 3 samples of previous block
        // We need 4 points (p0, p1, p2, p3) for interpolation.
        // p1 is "current", p2 is "next". p0 and p(-1) are history.
        this.cubicMem = new Float32Array(3);

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
            this.cubicMem.fill(0); // Reset cubic state
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

    // ✅ REPLACED: Hermite Cubic Interpolation (4-point)
    // Much better passband flatness than Linear, much cheaper than Sinc.
    cubicResample(inCh, ratio) {
        let t = this.frac;
        const len = inCh.length;

        while (t < len) {
            const i1 = Math.floor(t);
            const mu = t - i1;

            // Gather 4 points: y0, y1(current), y2(next), y3
            // Use history (cubicMem) if indices are negative
            let y0, y1, y2, y3;

            // y1 (at i1)
            if (i1 < 0) y1 = this.cubicMem[3 + i1];
            else y1 = inCh[i1];

            // y0 (at i1-1)
            if (i1 - 1 < 0) y0 = this.cubicMem[3 + (i1 - 1)];
            else y0 = inCh[i1 - 1];

            // y2 (at i1+1)
            if (i1 + 1 >= len) break; // Need at least one future point. Wait for next block.
            y2 = inCh[i1 + 1];

            // y3 (at i1+2)
            // If we are at the very edge, linear extrapolate or just dup y2.
            // Duping y2 is safe enough for 1 sample at edge.
            if (i1 + 2 >= len) y3 = y2;
            else y3 = inCh[i1 + 2];

            // Hermite interpolation
            const mu2 = mu * mu;
            const a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
            const a1 = y0 - 2.5 * y1 + 2 * y2 - 0.5 * y3;
            const a2 = -0.5 * y0 + 0.5 * y2;
            const a3 = y1;

            const out = a0 * mu * mu2 + a1 * mu2 + a2 * mu + a3;
            this.pushOut(out);

            t += ratio;
        }

        this.frac = t - len;

        // Save tail for next block's history
        // We need the last 3 samples: len-3, len-2, len-1
        if (len >= 3) {
            this.cubicMem[0] = inCh[len - 3];
            this.cubicMem[1] = inCh[len - 2];
            this.cubicMem[2] = inCh[len - 1];
        } else {
            // Edge case: tiny chunk (shouldn't happen with 128)
            // Shift manually if needed, but standard Web Audio chunks are 128
            for (let i = 0; i < len; i++) {
                this.cubicMem[0] = this.cubicMem[1];
                this.cubicMem[1] = this.cubicMem[2];
                this.cubicMem[2] = inCh[i];
            }
        }
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
            else console.log("✅ Mode: FIR + Cubic Resample");
            this.didLog = true;
        }

        if (fs === this.TARGET) {
            for (let i = 0; i < inCh.length; i++) this.pushOut(inCh[i]);
        } else if (fs === 44100) {
            this.firDecimateBy2(inCh);
        } else {
            const filtered = this.firLowpassBlock(inCh, fs);
            // ✅ Use Cubic
            this.cubicResample(filtered, fs / this.TARGET);
        }

        this.flush4096();
        return true;
    }
}

registerProcessor("resampler-processor", ResamplerProcessor);
class ResamplerProcessor extends AudioWorkletProcessor {
    constructor() {
        super();

        this.TARGET = 16000;

        // Output buffer
        this.out = new Float32Array(16384);
        this.outLen = 0;

        // For 48k -> 16k (decimate by 3)
        this.DECIM = 3;
        this.TAPS = 63;               // odd number
        this.CUTOFF_HZ = 7300;        // keep under 8k Nyquist of 16k

        this.h = null;                // FIR coeffs (built when we know fs)
        this.hist = new Float32Array(this.TAPS);
        this.histPos = 0;
        this.filled = 0;
        this.sampleCounter = 0;

        // Fallback resample state (for fs != 48k)
        this.frac = 0;
        this.prev = 0;

        this.didLog = false;
    }

    buildFIR(fs) {
        // Windowed-sinc lowpass (Blackman), DC-normalized
        const taps = this.TAPS;
        const M = taps - 1;
        const fc = this.CUTOFF_HZ / fs; // cycles/sample

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
        // Expand out buffer if needed (rare)
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

            // shift remaining
            this.out.copyWithin(0, 4096, this.outLen);
            this.outLen -= 4096;
        }
    }

    firDecimateBy3(inCh) {
        // Build FIR once (needs fs)
        if (!this.h) this.h = this.buildFIR(sampleRate);

        for (let i = 0; i < inCh.length; i++) {
            const x = inCh[i];

            // write into circular history buffer
            this.hist[this.histPos] = x;
            this.histPos = (this.histPos + 1) % this.TAPS;

            if (this.filled < this.TAPS) {
                this.filled++;
                this.sampleCounter++;
                continue;
            }

            this.sampleCounter++;

            // output every 3 input samples
            if (this.sampleCounter % this.DECIM !== 0) continue;

            // convolution: newest sample is at histPos-1 (wrapped)
            let y = 0;
            let idx = this.histPos;

            for (let k = 0; k < this.TAPS; k++) {
                idx = (idx - 1 + this.TAPS) % this.TAPS;
                y += this.hist[idx] * this.h[k];
            }

            this.pushOut(y);
        }
    }

    // Minimal fallback (neutral, no lowpass “muffle”)
    linearResample(inCh, ratio) {
        // ratio = fs / 16000
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
            this.didLog = true;
            console.log(`🎛️ Worklet fs=${fs}Hz (target 16k)`);
        }

        // Best path: 48k -> 16k via FIR decimation-by-3
        if (fs === 48000) {
            this.firDecimateBy3(inCh);
            this.flush4096();
            return true;
        }

        // Fallback: linear ratio resample (neutral)
        const ratio = fs / this.TARGET;
        this.linearResample(inCh, ratio);
        this.flush4096();
        return true;
    }
}

registerProcessor("resampler-processor", ResamplerProcessor);
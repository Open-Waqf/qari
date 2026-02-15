class ResamplerProcessor extends AudioWorkletProcessor {
    constructor() {
        super();

        this.TARGET = 16000;

        // Output buffer (we post 4096-sample blocks)
        this.out = new Float32Array(16384);
        this.outLen = 0;

        // 48k -> 16k FIR decimation-by-3
        this.DECIM = 3;
        this.TAPS = 63;          // odd
        this.CUTOFF_HZ = 7300;   // < 8k Nyquist @ 16k

        this.h = null;
        this.hist = new Float32Array(this.TAPS);
        this.histPos = 0;
        this.filled = 0;
        this.sampleCounter = 0;

        // Fallback linear resampler state
        this.frac = 0;
        this.prev = 0;

        // Fallback lowpass biquad state (anti-alias before linear downsample)
        this.lp = {
            fs: 0,
            b0: 0, b1: 0, b2: 0,
            a1: 0, a2: 0,
            z1: 0, z2: 0,
        };
        this.tmp = null;

        // Buffer for stereo downmixing
        this.tmpMix = null;

        this.didLog = false;
    }

    buildFIR(fs) {
        const taps = this.TAPS;
        const M = taps - 1;
        const fc = this.CUTOFF_HZ / fs;

        const h = new Float32Array(taps);
        let sum = 0;

        for (let i = 0; i < taps; i++) {
            const n = i - M / 2;

            let sinc;
            if (n === 0) sinc = 2 * fc;
            else sinc = Math.sin(2 * Math.PI * fc * n) / (Math.PI * n);

            const w =
                0.42 -
                0.5 * Math.cos((2 * Math.PI * i) / M) +
                0.08 * Math.cos((4 * Math.PI * i) / M);

            const v = sinc * w;
            h[i] = v;
            sum += v;
        }

        // Normalize
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
            // ✅ PERF: Transfer the buffer to main thread (zero-copy)
            this.port.postMessage(chunk, [chunk.buffer]);

            this.out.copyWithin(0, 4096, this.outLen);
            this.outLen -= 4096;
        }
    }

    firDecimateBy3(inCh) {
        // Build FIR filter lazily if needed
        if (!this.h) this.h = this.buildFIR(sampleRate);

        for (let i = 0; i < inCh.length; i++) {
            const x = inCh[i];

            this.hist[this.histPos] = x;
            this.histPos = (this.histPos + 1) % this.TAPS;

            if (this.filled < this.TAPS) {
                this.filled++;
                // ✅ FIX: Reset phase counter when buffer fills
                // ensures consistent alignment regardless of warmup time
                if (this.filled === this.TAPS) this.sampleCounter = 0;
                continue;
            }

            this.sampleCounter++;
            if (this.sampleCounter % this.DECIM !== 0) continue;

            let y = 0;
            let idx = this.histPos; // Start at oldest sample

            // Convolution
            for (let k = 0; k < this.TAPS; k++) {
                // Circular buffer read (backwards in time)
                idx = (idx - 1 + this.TAPS) % this.TAPS;
                y += this.hist[idx] * this.h[k];
            }

            this.pushOut(y);
        }
    }

    setLowpass(fs) {
        // RBJ biquad LPF around 7.2kHz
        const fc = Math.min(7200, 0.45 * this.TARGET);
        const Q = 0.7071;

        const w0 = 2 * Math.PI * (fc / fs);
        const c = Math.cos(w0);
        const s = Math.sin(w0);
        const alpha = s / (2 * Q);

        const b0 = (1 - c) / 2;
        const b1 = 1 - c;
        const b2 = (1 - c) / 2;
        const a0 = 1 + alpha;
        const a1 = -2 * c;
        const a2 = 1 - alpha;

        this.lp.fs = fs;
        this.lp.b0 = b0 / a0;
        this.lp.b1 = b1 / a0;
        this.lp.b2 = b2 / a0;
        this.lp.a1 = a1 / a0;
        this.lp.a2 = a2 / a0;

        // Reset state on rate change to avoid weird transients
        this.lp.z1 = 0;
        this.lp.z2 = 0;
    }

    lowpassBlock(x, fs) {
        if (this.lp.fs !== fs) this.setLowpass(fs);
        // Ensure temp buffer is large enough
        if (!this.tmp || this.tmp.length < x.length) this.tmp = new Float32Array(x.length);

        let z1 = this.lp.z1, z2 = this.lp.z2;
        const {b0, b1, b2, a1, a2} = this.lp;

        for (let i = 0; i < x.length; i++) {
            const xn = x[i];
            const y = b0 * xn + z1;
            z1 = b1 * xn - a1 * y + z2;
            z2 = b2 * xn - a2 * y;
            this.tmp[i] = y;
        }

        this.lp.z1 = z1;
        this.lp.z2 = z2;
        // Return slice of valid length
        return this.tmp.subarray(0, x.length);
    }

    linearResample(inCh, ratio) {
        if (inCh.length === 0) return;

        let t = this.frac; // Fractional position from last block

        // ✅ FIX: Loop condition safely avoids OOB access
        while (t < inCh.length - 1) {
            const i0 = Math.floor(t);
            const a = t - i0;

            // Handle boundary (start of block vs middle)
            const s0 = (i0 < 0) ? this.prev : inCh[i0];
            const s1 = inCh[i0 + 1];

            this.pushOut(s0 + (s1 - s0) * a);
            t += ratio;
        }

        // Save state for next block
        this.frac = t - inCh.length;
        this.prev = inCh[inCh.length - 1];
    }

    process(inputs) {
        const input0 = inputs[0];
        if (!input0 || input0.length === 0) return true;

        // ✅ FIX: Downmix to mono (handle stereo mics correctly)
        let inCh;
        if (input0.length === 1) {
            inCh = input0[0];
        } else {
            const N = input0[0].length;
            if (!this.tmpMix || this.tmpMix.length < N) {
                this.tmpMix = new Float32Array(N);
            }

            // Mix
            this.tmpMix.fill(0);
            for (let c = 0; c < input0.length; c++) {
                const ch = input0[c];
                for (let i = 0; i < N; i++) this.tmpMix[i] += ch[i];
            }

            // Normalize
            const inv = 1 / input0.length;
            for (let i = 0; i < N; i++) this.tmpMix[i] *= inv;

            inCh = this.tmpMix.subarray(0, N);
        }

        // In Worklet scope, global 'sampleRate' is the context rate
        const fs = sampleRate;

        if (!this.didLog) {
            this.didLog = true;
            console.log(`🎛️ Worklet fs=${fs}Hz -> 16k (Mode: ${fs === 48000 ? "FIR" : "Fallback"})`);
        }

        // 1. Pass-through (Already 16k)
        if (fs === this.TARGET) {
            for (let i = 0; i < inCh.length; i++) this.pushOut(inCh[i]);
            this.flush4096();
            return true;
        }

        // 2. High-Quality FIR (48k -> 16k)
        if (fs === 48000) {
            this.firDecimateBy3(inCh);
            this.flush4096();
            return true;
        }

        // 3. Fallback (Linear with Lowpass)
        const ratio = fs / this.TARGET;
        // Only apply lowpass if downsampling (fs > 16k) to prevent aliasing
        const src = (fs > this.TARGET) ? this.lowpassBlock(inCh, fs) : inCh;

        this.linearResample(src, ratio);
        this.flush4096();
        return true;
    }
}

registerProcessor("resampler-processor", ResamplerProcessor);
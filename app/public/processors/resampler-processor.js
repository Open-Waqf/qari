class ResamplerProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.TARGET_SAMPLE_RATE = 16000;
        this.buffer = [];

        // 1. FIR Filter for 48k -> 16k (Factor 3)
        // Raw coefficients (Sum ~= 0.86)
        const rawCoeffs = [
            0.0037, -0.0107, -0.0069, 0.0736, 0.2227,
            0.2952,
            0.2227, 0.0736, -0.0069, -0.0107, 0.0037
        ];

        // FIX: Normalize to ensure Gain = 1.0 (prevents volume drop)
        const sum = rawCoeffs.reduce((a, b) => a + b, 0);
        this.coefficients = rawCoeffs.map(c => c / sum);
        this.filterLen = this.coefficients.length;

        // Buffer for FIR path
        this.inputBuffer = new Float32Array(2048);
        this.inputBufferLen = 0;

        // 2. State for Fallback Path (Biquad + Linear)
        this.remainder = 0;
        this.lastSample = 0; // Bridges gap between chunks
        this.lp = null;      // Lazy init
        this.lp2 = null;
    }

    // --- Biquad Filter Helper (Lowpass) ---
    makeLowpass(fs, cutoffHz = 7200, Q = 0.707) {
        const w0 = 2 * Math.PI * (cutoffHz / fs);
        const cosw0 = Math.cos(w0);
        const sinw0 = Math.sin(w0);
        const alpha = sinw0 / (2 * Q);

        const a0 = 1 + alpha;
        return {
            b0: ((1 - cosw0) / 2) / a0,
            b1: (1 - cosw0) / a0,
            b2: ((1 - cosw0) / 2) / a0,
            a1: (-2 * cosw0) / a0,
            a2: (1 - alpha) / a0,
            x1: 0, x2: 0, y1: 0, y2: 0
        };
    }

    biquadStep(st, x) {
        const y = st.b0 * x + st.b1 * st.x1 + st.b2 * st.x2 - st.a1 * st.y1 - st.a2 * st.y2;
        st.x2 = st.x1;
        st.x1 = x;
        st.y2 = st.y1;
        st.y1 = y;
        return y;
    }

    process(inputs) {
        const input = inputs[0];
        if (!input || !input[0]) return true;

        const inCh = input[0];
        const fs = sampleRate;

        // Path A: Optimized 48kHz (Native Android/Most PC)
        if (fs === 48000) {
            this.firDecimateBy3(inCh);
            this.flushBuffer();
            return true;
        }

        // Path B: Universal Fallback (44.1kHz, etc.)

        // 1. Initialize Filters if needed
        if (!this.lp) {
            this.lp = this.makeLowpass(fs, 7200);
            this.lp2 = this.makeLowpass(fs, 7200);
        }

        // 2. Filter In-Place
        const filtered = new Float32Array(inCh.length);
        for (let i = 0; i < inCh.length; i++) {
            let x = inCh[i];
            x = this.biquadStep(this.lp, x);
            x = this.biquadStep(this.lp2, x);
            filtered[i] = x;
        }

        // 3. Robust Linear Resampling
        this.linearResample(filtered, fs / this.TARGET_SAMPLE_RATE);
        this.flushBuffer();
        return true;
    }

    firDecimateBy3(inputChannel) {
        // Append new data
        const totalLen = this.inputBufferLen + inputChannel.length;
        if (totalLen > this.inputBuffer.length) {
            const newBuf = new Float32Array(totalLen + 1024);
            newBuf.set(this.inputBuffer.subarray(0, this.inputBufferLen));
            this.inputBuffer = newBuf;
        }
        this.inputBuffer.set(inputChannel, this.inputBufferLen);
        this.inputBufferLen += inputChannel.length;

        // Decimate
        const ratio = 3;
        const outputLen = Math.floor((this.inputBufferLen - this.filterLen) / ratio);

        for (let i = 0; i < outputLen; i++) {
            const offset = i * ratio;
            let sum = 0;
            for (let k = 0; k < this.filterLen; k++) {
                sum += this.inputBuffer[offset + k] * this.coefficients[k];
            }
            this.buffer.push(sum);
        }

        // Shift buffer
        const consumed = outputLen * ratio;
        this.inputBuffer.set(this.inputBuffer.subarray(consumed, this.inputBufferLen));
        this.inputBufferLen -= consumed;
    }

    linearResample(input, ratio) {
        // Guard: Tiny buffers can cause OOB errors
        if (input.length < 2) {
            if (input.length > 0) this.lastSample = input[input.length - 1];
            return;
        }

        let inputIdx = this.remainder;

        // Loop until we need a sample from the NEXT chunk
        while (inputIdx < input.length - 1) {
            const prevIdx = Math.floor(inputIdx);
            const fraction = inputIdx - prevIdx;

            // Bridge logic:
            // If prevIdx is -1, we interpolate between 'lastSample' (prev chunk) and input[0].
            const p0 = (prevIdx < 0) ? this.lastSample : input[prevIdx];

            // Simplified p1: since inputIdx < input.length - 1,
            // prevIdx + 1 is always valid within this chunk (>= 0 and < length).
            const p1 = input[prevIdx + 1];

            const sample = p0 + (p1 - p0) * fraction;
            this.buffer.push(sample);

            inputIdx += ratio;
        }

        // Store negative remainder relative to the END of this chunk.
        this.remainder = inputIdx - input.length;

        // Save the very last FILTERED sample to use as p0 for the next chunk
        this.lastSample = input[input.length - 1];
    }

    flushBuffer() {
        while (this.buffer.length >= 4096) {
            const chunk = new Float32Array(this.buffer.slice(0, 4096));
            this.port.postMessage(chunk);
            this.buffer = this.buffer.slice(4096);
        }
    }
}

registerProcessor('resampler-processor', ResamplerProcessor);
class ResamplerProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.TARGET_SAMPLE_RATE = 16000;
        this.buffer = [];

        // FIR Filter Coefficients (Anti-Aliasing for 48k -> 16k)
        this.coefficients = [
            0.0037, -0.0107, -0.0069, 0.0736, 0.2227,
            0.2952,
            0.2227, 0.0736, -0.0069, -0.0107, 0.0037
        ];
        this.filterLen = this.coefficients.length;

        // Buffers
        this.inputBuffer = new Float32Array(2048);
        this.inputBufferLen = 0;

        // State for Fallback Resampler (Crucial for preventing clicks)
        this.remainder = 0;
        this.lastSample = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input[0]) return true;

        const inputChannel = input[0];
        const inputSampleRate = sampleRate;

        // 1. Smart Ratio Detection
        const exactRatio = inputSampleRate / this.TARGET_SAMPLE_RATE;
        const roundedRatio = Math.round(exactRatio);
        const isIntegerRatio = Math.abs(exactRatio - roundedRatio) < 0.1;

        // 2. Decide Path
        // If ratio is messy (e.g. 44.1k -> 16k = 2.756), use Fallback
        if (!isIntegerRatio || roundedRatio < 1) {
            this.linearResample(inputChannel, exactRatio);
            return true;
        }

        // 3. High-Quality FIR Path (For 48k/96k devices)
        const ratio = roundedRatio;

        // Append new data
        const totalLen = this.inputBufferLen + inputChannel.length;
        if (totalLen > this.inputBuffer.length) {
            const newBuf = new Float32Array(totalLen + 1024);
            newBuf.set(this.inputBuffer.subarray(0, this.inputBufferLen));
            this.inputBuffer = newBuf;
        }
        this.inputBuffer.set(inputChannel, this.inputBufferLen);
        this.inputBufferLen += inputChannel.length;

        // Convolve & Decimate
        const outputLen = Math.floor((this.inputBufferLen - this.filterLen) / ratio);
        if (outputLen > 0) {
            for (let i = 0; i < outputLen; i++) {
                const offset = i * ratio;
                let sum = 0;
                for (let k = 0; k < this.filterLen; k++) {
                    sum += this.inputBuffer[offset + k] * this.coefficients[k];
                }
                this.buffer.push(sum);
            }

            // Slide remaining samples
            const consumed = outputLen * ratio;
            this.inputBuffer.set(this.inputBuffer.subarray(consumed, this.inputBufferLen));
            this.inputBufferLen -= consumed;
        }

        this.flushBuffer();
        return true;
    }

    /**
     * FALLBACK: Linear Interpolation
     * Handles 44.1kHz -> 16kHz smoothly without aliasing clicks.
     */
    linearResample(input, ratio) {
        // Start from the fractional offset left by the previous chunk
        let inputIdx = this.remainder;

        while (inputIdx < input.length) {
            const prevIdx = Math.floor(inputIdx);
            const nextIdx = Math.ceil(inputIdx);
            const fraction = inputIdx - prevIdx;

            // Get samples (handle boundary with lastSample state)
            const p0 = (prevIdx < 0) ? this.lastSample : input[prevIdx];
            const p1 = (nextIdx >= input.length) ? input[input.length - 1] : input[nextIdx];

            // Linear Interpolation: y = y0 + (y1-y0) * fraction
            const sample = p0 + (p1 - p0) * fraction;
            this.buffer.push(sample);

            inputIdx += ratio;
        }

        // Save state for next chunk
        this.remainder = inputIdx - input.length;
        this.lastSample = input[input.length - 1];

        this.flushBuffer();
    }

    /**
     * Unified sender to ensure consistent chunk sizes
     */
    flushBuffer() {
        if (this.buffer.length >= 4096) {
            const chunk = new Float32Array(this.buffer.slice(0, 4096));
            this.port.postMessage(chunk);
            this.buffer = this.buffer.slice(4096);
        }
    }
}

registerProcessor('resampler-processor', ResamplerProcessor);
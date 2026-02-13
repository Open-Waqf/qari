// app/public/processors/resampler-processor.js

class ResamplerProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.TARGET_SR = 16000;
        this.buffer = [];
    }

    process(inputs) {
        const input = inputs[0];
        if (!input || !input[0]) return true;

        const channel = input[0];
        const ratio = sampleRate / this.TARGET_SR;

        for (let i = 0; i < channel.length; i += ratio) {
            const index = Math.floor(i);
            this.buffer.push(channel[index]);
        }

        // Send a chunk when we have ~1/4 second of audio
        if (this.buffer.length >= 4096) {
            this.port.postMessage(new Float32Array(this.buffer));
            this.buffer = [];
        }

        return true;
    }
}

registerProcessor('resampler-processor', ResamplerProcessor);
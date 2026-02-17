import {runMicCapTest} from "./miccap-test";
import {inferenceEngine} from "../model/inference-engine";

async function decodeToBufferInContext(file: File, ctx: AudioContext): Promise<AudioBuffer> {
    const arrayBuf = await file.arrayBuffer();
    return await ctx.decodeAudioData(arrayBuf.slice(0));
}

function toMonoBuffer(ctx: AudioContext, buf: AudioBuffer): AudioBuffer {
    if (buf.numberOfChannels === 1) return buf;
    const mono = ctx.createBuffer(1, buf.length, buf.sampleRate);
    const out = mono.getChannelData(0);
    const c0 = buf.getChannelData(0);
    const c1 = buf.getChannelData(1);
    for (let i = 0; i < out.length; i++) out[i] = (c0[i] + c1[i]) * 0.5;
    return mono;
}

function concatChunks(chunks: Float32Array[]): Float32Array {
    const total = chunks.reduce((s, c) => s + c.length, 0);
    const out = new Float32Array(total);
    let o = 0;
    for (const c of chunks) {
        out.set(c, o);
        o += c.length;
    }
    return out;
}

export async function runFileLoopback(file: File) {
    await inferenceEngine.setup();

    // 1. Force 48k context to simulate the live environment mismatch
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    const ctx = new AudioContextClass({sampleRate: 48000});

    await ctx.audioWorklet.addModule("/processors/resampler-processor.js");
    const worklet = new AudioWorkletNode(ctx, "resampler-processor");

    const buf = await decodeToBufferInContext(file, ctx);
    const monoBuf = toMonoBuffer(ctx, buf);

    const src = ctx.createBufferSource();
    src.buffer = monoBuf;

    const chunks: Float32Array[] = [];

    let first = true;
    worklet.port.onmessage = (e) => {
        const c = e.data as Float32Array;
        if (first) {
            first = false;
            console.log("🔎 first chunk len=", c.length);
        }
        chunks.push(c);
    };

    const mute = ctx.createGain();
    mute.gain.value = 0;

    src.connect(worklet).connect(mute).connect(ctx.destination);

    console.log(`🔁 LOOPBACK start: "${file.name}" ctxSR=${ctx.sampleRate} (Simulating Live 48k)`);
    src.start();

    await new Promise<void>((resolve) => {
        src.onended = () => resolve();
    });
    await new Promise(r => setTimeout(r, 50)); // small flush time

    await ctx.close();

    // 2. Collect the Raw Output (Now 48k because worklet is passthrough)
    const rawSignal = concatChunks(chunks);
    console.log(`🔁 Raw Collected: ${rawSignal.length} samples (Rate: ${ctx.sampleRate})`);

    // 3. DOWNSAMPLE (The Fix!)
    // We must manually convert 48k -> 22k, just like the live engine does.
    const sig = rawSignal;

    console.log(`✅ worklet output dur=${(sig.length / 22050).toFixed(2)}s @22k`);
    await runMicCapTest(sig);

    console.log(`📉 Downsampled to: ${sig.length} samples @ 22k`);

    await runMicCapTest(sig);
    console.log("✅ LOOPBACK done.");
}
import {inferenceEngine} from "../model/inference-engine";
import {downsampleBuffer} from "../features/custom-extractor.ts";

async function decodeFileToAudioBuffer(file: File): Promise<AudioBuffer> {
    const arrayBuf = await file.arrayBuffer();
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    const ctx = new AudioContextClass();
    try {
        return await ctx.decodeAudioData(arrayBuf.slice(0));
    } finally {
        await ctx.close().catch(() => {
        });
    }
}

function toMonoFloat32(buf: AudioBuffer): Float32Array {
    const ch = buf.numberOfChannels;
    if (ch === 1) return buf.getChannelData(0).slice();

    const len = buf.length;
    const out = new Float32Array(len);
    const c0 = buf.getChannelData(0);
    const c1 = buf.getChannelData(1);
    for (let i = 0; i < len; i++) out[i] = (c0[i] + c1[i]) * 0.5;
    return out;
}

async function resampleTo16k(mono: Float32Array, inSR: number): Promise<Float32Array> {
    if (inSR === 16000) return mono;

    const length16k = Math.ceil(mono.length * 16000 / inSR);
    const offline = new OfflineAudioContext(1, length16k, 16000);

    const srcBuf = offline.createBuffer(1, mono.length, inSR);
    srcBuf.copyToChannel(mono as any, 0);

    const src = offline.createBufferSource();
    src.buffer = srcBuf;
    src.connect(offline.destination);
    src.start();

    const rendered = await offline.startRendering();
    return rendered.getChannelData(0).slice();
}

export async function runFileTest(file: File) {
    const ok = await inferenceEngine.setup();
    if (!ok) throw new Error("Model setup failed.");

    // 1. Decode File
    const buf = await decodeFileToAudioBuffer(file);
    const mono = toMonoFloat32(buf);

    // 2. Downsample (Unified Method)
    let sig16k: Float32Array;
    try {
        sig16k = await resampleTo16k(mono, buf.sampleRate);
    } catch (e) {
        console.warn("OfflineAudioContext resample failed, falling back to linear resample:", e);
        sig16k = downsampleBuffer(mono, buf.sampleRate, 16000); // fallback
    }

    const dur = sig16k.length / 16000;
    console.log(`📁 FILETEST loaded: "${file.name}" | inSR=${buf.sampleRate}Hz | dur=${dur.toFixed(2)}s`);

    // 3. Predict
    const scanAll = confirm("Scan the whole file (every 1.0s window)?\nCancel = test 3 windows only.");

    if (scanAll) {
        const step = 1.0;
        for (let s = 0; s + 3 <= dur; s += step) {
            await inferenceEngine.predictFromSignal(sig16k, {startSec: s, windowSec: 3, log: true, dispatchToUI: true});
        }
    } else {
        const mid = Math.max(0, dur / 2 - 1.5);
        const end = Math.max(0, dur - 3);
        for (const s of [0, mid, end]) {
            await inferenceEngine.predictFromSignal(sig16k, {startSec: s, windowSec: 3, log: true, dispatchToUI: true});
        }
    }

    console.log("✅ FILETEST done.");

}
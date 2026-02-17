import {inferenceEngine} from "../model/inference-engine";
import {downsampleBuffer} from "../features/custom-extractor.ts"; // Ensure this function handles generic targets

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

// 🟢 Updated to 22050
async function resampleToTarget(mono: Float32Array, inSR: number): Promise<Float32Array> {
    const TARGET = 22050;
    if (inSR === TARGET) return mono;

    const dur = mono.length / inSR;
    const len = Math.floor(dur * TARGET);
    const offlineCtx = new OfflineAudioContext(1, len, TARGET);

    const buf = offlineCtx.createBuffer(1, mono.length, inSR);
    buf.copyToChannel(mono as any, 0);

    const src = offlineCtx.createBufferSource();
    src.buffer = buf;
    src.connect(offlineCtx.destination);
    src.start();

    const rendered = await offlineCtx.startRendering();
    return rendered.getChannelData(0);
}

export async function runFileTest(file: File) {
    console.log(`📂 FILETEST: Reading ${file.name} (${file.size} bytes)...`);

    const buf = await decodeFileToAudioBuffer(file);
    const mono = toMonoFloat32(buf);

    let sigTarget: Float32Array;
    try {
        sigTarget = await resampleToTarget(mono, buf.sampleRate);
    } catch (e) {
        console.warn("OfflineAudioContext resample failed, falling back to manual resample:", e);
        // Assuming downsampleBuffer in custom-extractor can handle target rate
        sigTarget = downsampleBuffer(mono, buf.sampleRate, 22050);
    }

    const dur = sigTarget.length / 22050;
    console.log(`📁 FILETEST loaded: "${file.name}" | inSR=${buf.sampleRate}Hz | dur=${dur.toFixed(2)}s`);

    await runScan(sigTarget);
    console.log("✅ FILETEST done.");

    async function runScan(signal: Float32Array) {
        const dur = signal.length / 22050;
        const scanAll = confirm("Scan the whole file (every 1.0s window)?\nCancel = test 3 windows only.");

        if (scanAll) {
            const step = 1.0;
            for (let s = 0; s + 2 <= dur; s += step) { // +2 for 2.0s window
                await inferenceEngine.predictFromSignal(signal, {
                    startSec: s,
                    windowSec: 2, // 🟢 2.0s window
                    log: true,
                    dispatchToUI: true
                });
            }
        } else {
            const mid = Math.max(0, dur / 2 - 1);
            const windows = [0, mid, Math.max(0, dur - 2)];
            for (const s of windows) {
                await inferenceEngine.predictFromSignal(signal, {
                    startSec: s,
                    windowSec: 2, // 🟢 2.0s window
                    log: true,
                    dispatchToUI: true
                });
            }
        }
    }
}
import {inferenceEngine} from "../model/inference-engine";
import {downsampleBuffer} from "../features/custom-extractor.ts";
import {decodePcmWav} from "./wav.ts";

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

async function resampleToTarget(mono: Float32Array, inSR: number): Promise<Float32Array> {
    const TARGET = 22050; // 🟢 22k
    if (inSR === TARGET) return mono;

    const lengthTarget = Math.ceil(mono.length * TARGET / inSR);
    const offline = new OfflineAudioContext(1, lengthTarget, TARGET);

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
    // 🟢 FIX: Restore Setup Check
    const ok = await inferenceEngine.setup();
    if (!ok) throw new Error("Model setup failed.");

    // 🟢 FIX: Header-Based Fast-Path (Try PCM Decode First)
    // This allows ANY 16-bit WAV to bypass WebAudio filters, regardless of filename.
    try {
        const {samples, sampleRate} = await decodePcmWav(file);

        let signal: Float32Array;
        if (sampleRate !== 22050) {
            console.log(`⚠️ WAV is ${sampleRate}Hz. Resampling to 22050Hz...`);
            signal = await resampleToTarget(samples, sampleRate);
        } else {
            signal = samples;
        }

        const dur = signal.length / 22050;
        console.log(`📁 FILETEST (Fast-Path): "${file.name}" | dur=${dur.toFixed(2)}s`);
        await runScan(signal);
        console.log("✅ FILETEST done.");
        return; // Success, exit early
    } catch (e) {
        // If decodePcmWav failed (MP3, AAC, 24-bit WAV, etc.), just log debug info and continue
        if (file.name.toLowerCase().endsWith(".wav")) {
            console.warn("Fast-path PCM decode skipped (using WebAudio):", e);
        }
    }

    // Standard Path (WebAudio Decode)
    const buf = await decodeFileToAudioBuffer(file);
    const mono = toMonoFloat32(buf);

    let sigTarget: Float32Array;
    try {
        sigTarget = await resampleToTarget(mono, buf.sampleRate);
    } catch (e) {
        console.warn("Offline resample failed, using fallback:", e);
        sigTarget = downsampleBuffer(mono, buf.sampleRate, 22050);
    }

    console.log(`📁 FILETEST loaded: "${file.name}" @ 22050Hz`);
    await runScan(sigTarget);
    console.log("✅ FILETEST done.");

    async function runScan(signal: Float32Array) {
        const SR = 22050;
        const dur = signal.length / SR;
        const scanAll = confirm("Scan the whole file (every 1.0s window)?");

        const step = 1.0;
        const windowSec = 2; // 🟢 Matches 2.0s Model

        if (scanAll) {
            inferenceEngine.reset();
            for (let s = 0; s + windowSec <= dur; s += step) {
                await inferenceEngine.predictFromSignal(signal, {
                    startSec: s,
                    windowSec: windowSec,
                    log: true,
                    dispatchToUI: true,
                    independent: false,
                });
            }
        } else {
            const mid = Math.max(0, dur / 2 - (windowSec / 2));
            const end = Math.max(0, dur - windowSec);
            for (const s of [0, mid, end]) {
                await inferenceEngine.predictFromSignal(signal, {
                    startSec: s,
                    windowSec: windowSec,
                    log: true,
                    dispatchToUI: true
                });
            }
        }
    }
}

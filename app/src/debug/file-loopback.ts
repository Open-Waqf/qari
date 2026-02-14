// debug/file-loopback.ts
import {runMicCapTest} from "./miccap-test";
import {inferenceEngine} from "../model/inference-engine";
import {audioManager} from "../core/audio-manager.ts";

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

async function runFileLoopback(file: File) {
    console.log(`🧪 FILETEST CMVN: ${inferenceEngine.isCmvnEnabled() ? "ON" : "OFF"}`);
    console.log(`🧪 FILETEST Far Field Mode: ${audioManager.isFarFieldMode() ? "ON" : "OFF"}`);
    console.log(`🧪 FILETEST PreEmphasis: ${inferenceEngine.isPreEmphasisEnabled() ? "ON" : "OFF"}`);
    console.log(`🧪 FILETEST RMS Normalized: ${inferenceEngine.isRmsNormalizeEnabled() ? "ON" : "OFF"}`);

    await inferenceEngine.setup();

    // Force 48k context so your FIR decimate-by-3 path is exercised
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    const ctx = new AudioContextClass({sampleRate: 48000});

    await ctx.audioWorklet.addModule("/processors/resampler-processor.js");
    const worklet = new AudioWorkletNode(ctx, "resampler-processor");

    const buf = await decodeToBufferInContext(file, ctx);
    const monoBuf = toMonoBuffer(ctx, buf);

    const src = ctx.createBufferSource();
    src.buffer = monoBuf;

    const chunks: Float32Array[] = [];
    worklet.port.onmessage = (e) => chunks.push(e.data as Float32Array);

    const mute = ctx.createGain();
    mute.gain.value = 0;

    src.connect(worklet).connect(mute).connect(ctx.destination);

    console.log(`🔁 LOOPBACK start: "${file.name}" ctxSR=${ctx.sampleRate}, bufSR=${monoBuf.sampleRate}`);
    src.start();

    await new Promise<void>((resolve) => {
        src.onended = () => resolve();
    });
    await new Promise(r => setTimeout(r, 50)); // small flush time

    await ctx.close();

    const sig16k = concatChunks(chunks);
    console.log(`🔁 LOOPBACK collected ${(sig16k.length / 16000).toFixed(2)}s @16k, chunks=${chunks.length}`);

    await runMicCapTest(sig16k);
    console.log("✅ LOOPBACK done.");
}

export function installFileLoopbackHotkey() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "audio/*";
    input.style.display = "none";
    document.body.appendChild(input);

    input.addEventListener("change", async () => {
        const file = input.files?.[0];
        input.value = "";
        if (!file) return;
        try {
            await runFileLoopback(file);
        } catch (e) {
            console.error("❌ LOOPBACK error:", e);
            alert(`LOOPBACK error: ${(e as any)?.message ?? e}`);
        }
    });

    window.addEventListener("keydown", (e) => {
        if (e.shiftKey && (e.key === "L" || e.key === "l")) {
            e.preventDefault();
            input.click();
        }
    });

    console.log("🧪 LOOPBACK hotkey installed: Shift+L");
}

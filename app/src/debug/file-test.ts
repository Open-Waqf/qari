import {inferenceEngine} from "../model/inference-engine";
import {audioManager} from "../core/audio-manager.ts";

async function decodeFileToAudioBuffer(file: File): Promise<AudioBuffer> {
    const arrayBuf = await file.arrayBuffer();
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    const ctx = new AudioContextClass();
    try {
        return await ctx.decodeAudioData(arrayBuf.slice(0));
    } finally {
        // close to free resources (safe in modern browsers)
        await ctx.close().catch(() => {
        });
    }
}

function toMonoFloat32(buf: AudioBuffer): Float32Array {
    const ch = buf.numberOfChannels;
    const len = buf.length;
    if (ch === 1) return buf.getChannelData(0).slice();

    const out = new Float32Array(len);
    const c0 = buf.getChannelData(0);
    const c1 = buf.getChannelData(1);
    for (let i = 0; i < len; i++) out[i] = (c0[i] + c1[i]) * 0.5;
    return out;
}

async function resampleTo16k(mono: Float32Array, inRate: number): Promise<Float32Array> {
    if (inRate === 16000) return mono;

    const durationSec = mono.length / inRate;
    const outLen = Math.max(1, Math.round(durationSec * 16000));

    const oac = new OfflineAudioContext(1, outLen, 16000);

    // create a buffer at ORIGINAL rate, then OfflineAudioContext will resample when rendering at 16k
    const tmp = oac.createBuffer(1, mono.length, inRate);
    tmp.copyToChannel(mono as any, 0);

    const src = oac.createBufferSource();
    src.buffer = tmp;
    src.connect(oac.destination);
    src.start();

    const rendered = await oac.startRendering();
    return rendered.getChannelData(0).slice();
}

async function runFileTest(file: File) {

    console.log(`🧪 FILETEST CMVN: ${inferenceEngine.isCmvnEnabled() ? "ON" : "OFF"}`);
    console.log(`🧪 FILETEST Far Field Mode: ${audioManager.isFarFieldMode() ? "ON" : "OFF"}`);
    console.log(`🧪 FILETEST PreEmphasis: ${inferenceEngine.isPreEmphasisEnabled() ? "ON" : "OFF"}`);
    console.log(`🧪 FILETEST RMS Normalized: ${inferenceEngine.isRmsNormalizeEnabled() ? "ON" : "OFF"}`);

    // Ensure model is ready (won't regress your flow; it just loads if needed)
    const ok = await inferenceEngine.setup();
    if (!ok) throw new Error("Model setup failed.");

    const buf = await decodeFileToAudioBuffer(file);
    const mono = toMonoFloat32(buf);
    const sig16k = await resampleTo16k(mono, buf.sampleRate);

    const dur = sig16k.length / 16000;
    console.log(`📁 FILETEST loaded: "${file.name}" | inSR=${buf.sampleRate}Hz | dur=${dur.toFixed(2)}s`);

    // Ask whether to scan whole file or just sample 3 windows
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

export function installFileTestHotkey() {
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
            await runFileTest(file);
        } catch (e) {
            console.error("❌ FILETEST error:", e);
            alert(`FILETEST error: ${(e as any)?.message ?? e}`);
        }
    });

    window.addEventListener("keydown", (e) => {
        // Shift+T
        if (e.shiftKey && (e.key === "T" || e.key === "t")) {
            e.preventDefault();
            input.click();
        }
    });

    console.log("🧪 FILETEST hotkey installed: Shift+T");
}

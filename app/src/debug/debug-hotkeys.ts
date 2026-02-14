import {inferenceEngine} from "../model/inference-engine";
import {EVENTS} from "../core/events";
import {audioManager} from "../core/audio-manager";
import {runFileLoopback} from "./file-loopback";
import {micCap} from "../audio/mic-cap.ts";
import {downloadBlob, encodeWav16k} from "./wav";
import {runMicCapTest} from "./miccap-test";
import {runFileTest} from "./file-test";

export function installInterferenceHotkey() {
    window.addEventListener("keydown", (e) => {
        // Ignore inputs
        const target = e.target as HTMLElement | null;
        if (target?.tagName === 'INPUT' || target?.tagName === 'TEXTAREA' || target?.isContentEditable) return;

        if (!e.shiftKey) return;
        const key = e.key.toLowerCase();
        let msg = "";

        if (key === 'n') {
            inferenceEngine.toggleRmsNormalize();
            msg = inferenceEngine.isRmsNormalizeEnabled() ? 'RmsNormalize: ON' : 'RmsNormalize: OFF';
        } else if (key === 'e') {
            inferenceEngine.togglePreEmphasis();
            msg = inferenceEngine.isPreEmphasisEnabled() ? 'PreEmphasis: ON' : 'PreEmphasis: OFF';
        } else if (key === 'c') {
            inferenceEngine.toggleCMVN();
            msg = inferenceEngine.isCmvnEnabled() ? 'CMVN: ON' : 'CMVN: OFF';
        }

        if (msg) {
            console.log(`🧪 ${msg}`);
            // Dispatch event so QariApp can show it in the pill
            window.dispatchEvent(new CustomEvent(EVENTS.DEBUG_MESSAGE, {
                detail: {message: msg}
            }));
        }
    });

    console.log("🧪 Inference Hotkeys Installed: Shift + [N]ormalize, [E]mphasis, [C]MVN");
}

export function installFarFieldHotkey() {
    window.addEventListener("keydown", (e) => {
        const target = e.target as HTMLElement | null;
        if (target?.tagName === 'INPUT' || target?.tagName === 'TEXTAREA' || target?.isContentEditable) return;

        if (!e.shiftKey) return;

        if (e.key.toLowerCase() === 'f') {
            // 1. Toggle State
            const isNowOn = !audioManager.isFarFieldMode();
            audioManager.setFarFieldMode(isNowOn);

            const msg = isNowOn ? 'Far Field AGC: ON' : 'Far Field AGC: OFF';

            // 2. Dispatch Message (for Status Pill)
            window.dispatchEvent(new CustomEvent(EVENTS.DEBUG_MESSAGE, {
                detail: {message: msg}
            }));

            // 3. Dispatch Restart Request (if engine is running)
            if (audioManager.isRunning) {
                console.log("🔄 Requesting Engine Restart...");
                window.dispatchEvent(new CustomEvent(EVENTS.REQUEST_RESTART));
            }
        }
    });

    console.log("🧪 Far Field Hotkeys Installed: Shift + F");
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


export function installMicCapHotkey() {
    window.addEventListener("keydown", async (e) => {
        if (!(e.shiftKey && e.key.toLowerCase() === "m")) return;
        e.preventDefault();

        // Ensure model ready (prevents weird early runs)
        await inferenceEngine.setup().catch(() => {
        });

        // 1) start capture
        if (!micCap.active && !micCap.ready) {
            micCap.start(8, 16000);
            return;
        }

        // 2) stop early if still active
        if (micCap.active) micCap.stopEarly();

        // 3) analyze if ready
        const signal = micCap.take();
        if (signal) {
            const wav = encodeWav16k(signal, 16000);
            downloadBlob(wav, `miccap_16k_${Date.now()}.wav`);
            await runMicCapTest(signal);
        }

    });

    console.log("🧪 MICCAP hotkey installed: Shift+M");
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

export function installRawMicRecordHotkey() {
    window.addEventListener("keydown", async (e) => {
        if (!(e.shiftKey && e.key.toLowerCase() === "r")) return;
        e.preventDefault();

        // Engine must be started so _stream exists
        const am: any = audioManager as any;
        const stream: MediaStream | null = am._stream ?? null;
        if (!stream) {
            alert("Start engine first (Initialize Engine), then press Shift+R.");
            return;
        }

        const candidates = [
            "audio/ogg;codecs=opus",
            "audio/webm;codecs=opus",
            "audio/ogg",
            "audio/webm",
        ];

        const mimeType =
            candidates.find(t => (window as any).MediaRecorder?.isTypeSupported?.(t)) || "";

        const rec = new MediaRecorder(stream, mimeType ? {mimeType} : undefined);

        const chunks: BlobPart[] = [];
        rec.ondataavailable = (ev) => {
            if (ev.data.size) chunks.push(ev.data);
        };

        rec.onstop = () => {
            const type = mimeType || rec.mimeType || "audio/webm";
            const blob = new Blob(chunks, {type});
            const ext = type.includes("ogg") ? "ogg" : "webm";
            downloadBlob(blob, `raw_mic_${Date.now()}.${ext}`);
            console.log(`✅ RAW MIC saved (${type})`);
        };

        console.log(`🎙️ RAW MIC REC start 8s mime="${mimeType || rec.mimeType}"`);
        rec.start();
        setTimeout(() => rec.stop(), 8000);
    });

    console.log("🧪 Raw mic record hotkey installed: Shift+R");
}
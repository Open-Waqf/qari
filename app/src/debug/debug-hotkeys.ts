import {inferenceEngine} from "../model/inference-engine";
import {EVENTS} from "../core/events";
import {audioManager} from "../core/audio-manager";
import {runFileLoopback} from "./file-loopback";
import {micCap} from "../audio/mic-cap";
import {downloadBlob, encodeWav} from "./wav";
import {runMicCapTest} from "./miccap-test";
import {runFileTest} from "./file-test";

/**
 * 🛠️ RESTORED: Centralized Keydown Guard
 */
function isInputFocused(e: KeyboardEvent): boolean {
    const target = e.target as HTMLElement | null;
    return !!(target?.tagName === 'INPUT' || target?.tagName === 'TEXTAREA' || target?.isContentEditable);
}

export function installInterferenceHotkey() {
    window.addEventListener("keydown", (e) => {
        if (isInputFocused(e) || !e.shiftKey) return;
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
            window.dispatchEvent(new CustomEvent(EVENTS.DEBUG_MESSAGE, {detail: {message: msg}}));
        }
    });
}

export function installFarFieldHotkey() {
    window.addEventListener("keydown", (e) => {
        if (isInputFocused(e) || !e.shiftKey || e.key.toLowerCase() !== 'f') return;

        const isNowOn = !audioManager.isFarFieldMode();
        audioManager.setFarFieldMode(isNowOn);
        const msg = isNowOn ? 'Far Field AGC: ON' : 'Far Field AGC: OFF';

        window.dispatchEvent(new CustomEvent(EVENTS.DEBUG_MESSAGE, {detail: {message: msg}}));

        if (audioManager.isRunning) {
            console.log("🔄 Requesting Engine Restart for AGC change...");
            window.dispatchEvent(new CustomEvent(EVENTS.REQUEST_RESTART));
        }
    });

    console.log("🧪 Far Field Hotkeys Installed: Shift + F");
}

export function installFileLoopbackHotkey() {
    window.addEventListener("keydown", (e) => {
        if (!isInputFocused(e) && e.shiftKey && e.key.toLowerCase() === 'l') {
            const input = document.createElement("input");
            input.type = "file";
            input.onchange = () => {
                if (input.files?.[0]) runFileLoopback(input.files[0]);
            };
            input.click();
        }
    });

    console.log("🧪 LOOPBACK hotkey installed: Shift+L");
}

/**
 * 🎙️ FIXED: MICCAP Logic (Shift+M)
 * Restored the 3-state logic: Start -> Stop -> Analyze
 */
export function installMicCapHotkey() {
    window.addEventListener("keydown", async (e) => {
        if (isInputFocused(e) || !e.shiftKey || e.key.toLowerCase() !== 'm') return;
        e.preventDefault();

        // 🟢 REGRESSION FIX: Ensure engine ready
        await inferenceEngine.setup().catch(() => {
        });

        // 1. Start Capture if idle
        if (!micCap.active && !micCap.ready) {
            console.log("🎙️ Starting Mic Capture (5s) @ 22050Hz...");
            micCap.start(5.0, 22050);
            return;
        }

        // 2. Stop early if still recording
        if (micCap.active) {
            micCap.stopEarly();
            // Fall through to analyze if stopEarly marks it ready
        }

        // 3. 🟢 REGRESSION FIX: Use the destructive take/getBuffer
        const signal = micCap.take(); // Use the fixed getBuffer() we discussed
        if (signal) {
            console.log("🧪 Analyzing Capture...");
            const wav = encodeWav(signal, 22050);
            downloadBlob(wav, `miccap_22k_${Date.now()}.wav`);
            await runMicCapTest(signal);
        }
    });

    console.log("🧪 MICCAP hotkey installed: Shift+M");
}

/**
 * 📁 RESTORED: File Test (Shift+T) with Input reuse
 */
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
        if (!isInputFocused(e) && e.shiftKey && e.key.toLowerCase() === 't') {
            e.preventDefault();
            input.click();
        }
    });

    console.log("🧪 FILETEST hotkey installed: Shift+T");
}

/**
 * 🔴 RESTORED: Raw Mic Record (Shift+R) with 8s duration
 */
export function installRawMicRecordHotkey() {
    window.addEventListener("keydown", (e) => {
        if (isInputFocused(e) || !e.shiftKey || e.key.toLowerCase() !== 'r') return;
        e.preventDefault();

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

        const mimeType = candidates.find(t => (window as any).MediaRecorder?.isTypeSupported?.(t)) || "";
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
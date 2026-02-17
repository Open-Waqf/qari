import {inferenceEngine} from "../model/inference-engine";
import {EVENTS} from "../core/events";
import {audioManager} from "../core/audio-manager";
import {runFileLoopback} from "./file-loopback";
import {micCap} from "../audio/mic-cap"; // 🟢 Updated import
import {downloadBlob, encodeWav} from "./wav"; // 🟢 Updated import
import {runMicCapTest} from "./miccap-test";
import {runFileTest} from "./file-test";

export function installInterferenceHotkey() {
    window.addEventListener("keydown", (e) => {
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
        } else {
            return;
        }

        console.log(`🧪 ${msg}`);
        const event = new CustomEvent(EVENTS.DEBUG_MESSAGE, {detail: {message: msg, type: 'info'}});
        window.dispatchEvent(event);
    });
    console.log("🧪 Inference Hotkeys Installed: Shift + [N]ormalize, [E]mphasis, [C]MVN");
}

export function installFarFieldHotkey() {
    window.addEventListener("keydown", (e) => {
        if (e.shiftKey && e.key.toLowerCase() === 'f') {
            const am: any = audioManager as any;
            if (am.flags) {
                am.flags.farFieldMode = !am.flags.farFieldMode;
                console.log(`🧪 Far Field Mode: ${am.flags.farFieldMode ? "ON" : "OFF"}`);
                const event = new CustomEvent(EVENTS.DEBUG_MESSAGE, {
                    detail: {
                        message: `Far Field: ${am.flags.farFieldMode}`,
                        type: 'info'
                    }
                });
                window.dispatchEvent(event);

                // Restart to apply
                if (am.isRunning) {
                    am.stop();
                    setTimeout(() => am.start(), 200);
                }
            }
        }
    });
    console.log("🧪 Far Field Hotkeys Installed: Shift + F");
}

export function installFileLoopbackHotkey() {
    window.addEventListener("keydown", (e) => {
        if (e.shiftKey && e.key.toLowerCase() === "l") {
            e.preventDefault();
            const input = document.createElement("input");
            input.type = "file";
            input.accept = "audio/*";
            input.onchange = async () => {
                if (input.files?.[0]) {
                    console.log("🧪 Starting loopback...");
                    await runFileLoopback(input.files[0]);
                }
            };
            input.click();
        }
    });
    console.log("🧪 LOOPBACK hotkey installed: Shift+L");
}

export function installMicCapHotkey() {
    window.addEventListener("keydown", async (e) => {
        if (e.shiftKey && e.key.toLowerCase() === "m") {
            e.preventDefault();
            if (micCap.active) {
                // Stop early
                micCap.stopEarly();
                const buf = micCap.getBuffer();
                if (buf) {
                    // Download raw 22k wav
                    const blob = encodeWav(buf, 22050); // 🟢 Updated
                    downloadBlob(blob, `mic_cap_${Date.now()}.wav`);

                    // Run test
                    await runMicCapTest(buf);
                }
            } else {
                // Start
                console.log("🎙️ Starting Mic Capture (5s)...");
                micCap.start(5.0, 22050); // 🟢 Updated
            }
        }
    });
    console.log("🧪 MICCAP hotkey installed: Shift+M");
}

export function installFileTestHotkey() {
    window.addEventListener("keydown", (e) => {
        if (e.shiftKey && e.key.toLowerCase() === "t") {
            e.preventDefault();
            const input = document.createElement("input");
            input.type = "file";
            input.accept = "audio/*";
            input.onchange = async () => {
                if (input.files?.[0]) {
                    await runFileTest(input.files[0]);
                }
            };
            input.click();
        }
    });
    console.log("🧪 FILETEST hotkey installed: Shift+T");
}

export function installRawMicRecordHotkey() {
    window.addEventListener("keydown", (e) => {
        if (!(e.shiftKey && e.key.toLowerCase() === "r")) return;
        e.preventDefault();

        const am: any = audioManager as any;
        const stream: MediaStream | null = am._stream ?? null;
        if (!stream) {
            alert("Start engine first, then press Shift+R.");
            return;
        }

        const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg"];
        const mimeType = candidates.find(t => (window as any).MediaRecorder?.isTypeSupported?.(t)) || "";
        const rec = new MediaRecorder(stream, mimeType ? {mimeType} : undefined);
        const chunks: BlobPart[] = [];

        rec.ondataavailable = (ev) => {
            if (ev.data.size) chunks.push(ev.data);
        };
        rec.onstop = () => {
            const type = mimeType || rec.mimeType || "audio/webm";
            const blob = new Blob(chunks, {type});
            downloadBlob(blob, `raw_mic_${Date.now()}.webm`);
            console.log(`✅ RAW MIC saved (${type})`);
        };

        console.log("🔴 Recording 4s raw mic...");
        rec.start();
        setTimeout(() => rec.stop(), 4000);
    });
    console.log("🧪 Raw mic record hotkey installed: Shift+R");
}
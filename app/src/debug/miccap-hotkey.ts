import {micCap} from "../audio/mic-cap";
import {runMicCapTest} from "./miccap-test";
import {inferenceEngine} from "../model/inference-engine";
import {downloadBlob, encodeWav16k} from "./wav.ts";

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

// debug/raw-mic-record.ts
import {audioManager} from "../core/audio-manager";
import {downloadBlob} from "./wav"; // reuse your helper

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

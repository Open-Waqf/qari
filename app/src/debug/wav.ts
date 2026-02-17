/**
 * Encodes a Float32Array into a 16-bit PCM WAV Blob.
 * Defaults to 22050Hz to match the new model requirements.
 */
export function encodeWav(signal: Float32Array, sampleRate = 22050): Blob {
    const numChannels = 1;
    const bitsPerSample = 16;

    const pcm = new Int16Array(signal.length);
    for (let i = 0; i < signal.length; i++) {
        const s = Math.max(-1, Math.min(1, signal[i]));
        pcm[i] = (s < 0 ? s * 0x8000 : s * 0x7fff) | 0;
    }

    const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
    const blockAlign = numChannels * (bitsPerSample / 8);
    const dataSize = pcm.length * 2;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    let o = 0;
    const wstr = (s: string) => {
        for (let i = 0; i < s.length; i++) view.setUint8(o++, s.charCodeAt(i));
    };
    const w32 = (v: number) => {
        view.setUint32(o, v, true);
        o += 4;
    };
    const w16 = (v: number) => {
        view.setUint16(o, v, true);
        o += 2;
    };

    wstr("RIFF");
    w32(36 + dataSize);
    wstr("WAVE");
    wstr("fmt ");
    w32(16);
    w16(1);                 // Audio Format: PCM
    w16(numChannels);
    w32(sampleRate);
    w32(byteRate);
    w16(blockAlign);
    w16(bitsPerSample);
    wstr("data");
    w32(dataSize);

    // Optimized PCM data copy
    const pcmBytes = new Uint8Array(pcm.buffer);
    const body = new Uint8Array(buffer, 44);
    body.set(pcmBytes);

    return new Blob([buffer], {type: "audio/wav"});
}

export function downloadBlob(blob: Blob, filename: string) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    // Revoke after a delay to ensure the browser has started the download
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Decodes a raw PCM WAV file.
 * Required for the 'miccap' Fast-Path to detect if a file is 16k or 22k.
 */
export async function decodePcmWav(file: File): Promise<{ samples: Float32Array; sampleRate: number }> {
    const buf = await file.arrayBuffer();
    const view = new DataView(buf);

    const readStr = (o: number, n: number) =>
        Array.from({length: n}, (_, i) => String.fromCharCode(view.getUint8(o + i))).join("");

    if (readStr(0, 4) !== "RIFF" || readStr(8, 4) !== "WAVE") {
        throw new Error("Not a RIFF/WAVE file");
    }

    let fmtOff = -1, dataOff = -1, dataSize = 0;
    let sampleRate = 0, bits = 0, channels = 0, audioFormat = 0;

    let o = 12;
    while (o + 8 <= view.byteLength) {
        const id = readStr(o, 4);
        const sz = view.getUint32(o + 4, true);
        const chunk = o + 8;

        if (id === "fmt ") {
            fmtOff = chunk;
            audioFormat = view.getUint16(chunk + 0, true);
            channels = view.getUint16(chunk + 2, true);
            sampleRate = view.getUint32(chunk + 4, true);
            bits = view.getUint16(chunk + 14, true);
        } else if (id === "data") {
            dataOff = chunk;
            dataSize = sz;
            break;
        }

        o = chunk + sz + (sz % 2); // word align
    }

    if (fmtOff < 0 || dataOff < 0) throw new Error("Missing fmt/data chunk");
    if (audioFormat !== 1) throw new Error(`Unsupported WAV format ${audioFormat} (need PCM)`);
    if (bits !== 16) throw new Error(`Unsupported bitsPerSample=${bits} (need 16)`);

    const numSamples = dataSize / (channels * (bits / 8));
    const out = new Float32Array(numSamples);

    let p = dataOff;
    const stride = channels * 2; // 2 bytes per sample
    for (let i = 0; i < numSamples; i++, p += stride) {
        // We take the first channel if stereo, or the only channel if mono
        const s = view.getInt16(p, true);
        out[i] = s / 32768.0;
    }

    return {samples: out, sampleRate};
}
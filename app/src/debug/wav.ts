export function downloadBlob(blob: Blob, name: string) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
}

export function decodePcmWav(view: DataView): Float32Array {
    // Check RIFF/WAVE header
    const readStr = (o: number, len: number) => {
        let s = "";
        for (let i = 0; i < len; i++) s += String.fromCharCode(view.getUint8(o + i));
        return s;
    };

    if (readStr(0, 4) !== "RIFF" || readStr(8, 4) !== "WAVE") {
        throw new Error("Not a RIFF/WAVE file");
    }

    let fmtOff = -1, dataOff = -1, dataSize = 0;
    let bits = 0, channels = 0, audioFormat = 0;

    let o = 12;
    while (o + 8 <= view.byteLength) {
        const id = readStr(o, 4);
        const sz = view.getUint32(o + 4, true);
        const chunk = o + 8;

        if (id === "fmt ") {
            fmtOff = chunk;
            audioFormat = view.getUint16(chunk + 0, true);
            channels = view.getUint16(chunk + 2, true);
            bits = view.getUint16(chunk + 14, true);
        } else if (id === "data") {
            dataOff = chunk;
            dataSize = sz;
            break;
        }

        o = chunk + sz + (sz % 2); // word align
    }

    if (fmtOff < 0 || dataOff < 0) throw new Error("Missing fmt/data chunk");
    if (audioFormat !== 1) throw new Error(`Unsupported compression format ${audioFormat}`);

    // We strictly want to handle data, resampling is caller's job usually,
    // but here we just decode raw values.

    const numSamples = dataSize / (channels * (bits / 8));
    const out = new Float32Array(numSamples);

    // Simple 16-bit mono/stereo decoder
    let p = dataOff;
    const stride = channels * (bits / 8);
    for (let i = 0; i < numSamples; i++) {
        const int16 = view.getInt16(p, true);
        out[i] = int16 / 32768.0;
        p += stride;
    }

    return out;
}

export function encodeWav(signal: Float32Array, sampleRate = 22050): Blob { // 🟢 Updated
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
    w16(1); // PCM
    w16(numChannels);
    w32(sampleRate);
    w32(byteRate);
    w16(blockAlign);
    w16(bitsPerSample);
    wstr("data");
    w32(dataSize);

    // Copy PCM data
    const pcmBytes = new Uint8Array(pcm.buffer);
    const body = new Uint8Array(buffer, 44);
    body.set(pcmBytes);

    return new Blob([buffer], {type: "audio/wav"});
}
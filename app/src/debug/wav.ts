// debug/wav.ts
export function encodeWav16k(signal16k: Float32Array, sampleRate = 16000): Blob {
    const numChannels = 1;
    const bitsPerSample = 16;

    const pcm = new Int16Array(signal16k.length);
    for (let i = 0; i < signal16k.length; i++) {
        const s = Math.max(-1, Math.min(1, signal16k[i]));
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
    w16(1);                 // PCM
    w16(numChannels);
    w32(sampleRate);
    w32(byteRate);
    w16(blockAlign);
    w16(bitsPerSample);
    wstr("data");
    w32(dataSize);

    // PCM data
    for (let i = 0; i < pcm.length; i++, o += 2) view.setInt16(o, pcm[i], true);

    return new Blob([buffer], {type: "audio/wav"});
}

export function downloadBlob(blob: Blob, filename: string) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

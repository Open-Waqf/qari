import {inferenceEngine} from "../model/inference-engine";

function rmsOf(x: Float32Array) {
    let sum = 0;
    for (let i = 0; i < x.length; i++) sum += x[i] * x[i];
    return Math.sqrt(sum / x.length);
}

export async function runMicCapTest(signal16k: Float32Array) {
    const SR = 16000;
    const WIN = 3 * SR;   // 3s
    const HOP = 1 * SR;   // 1s

    console.log(`🏁 MICCAP | dur=${(signal16k.length / SR).toFixed(2)}s`);

    const candidates: { startSec: number; rms: number; ent: number; top3: any[] }[] = [];

    for (let start = 0; start + WIN <= signal16k.length; start += HOP) {
        const startSec = start / SR;
        const window = signal16k.subarray(start, start + WIN);
        const r = rmsOf(window);

        // Skip near-silent windows (tune if needed)
        if (r < 0.01) {
            console.log(`🏁 MICCAP | start=${startSec.toFixed(2)}s rms=${r.toFixed(4)} SKIP(silent)`);
            continue;
        }

        const {ent, top3} = await inferenceEngine.predictFromSignal(window);
        console.log(
            `🏁 MICCAP | start=${startSec.toFixed(2)}s rms=${r.toFixed(4)} ent=${ent.toFixed(3)} | ` +
            top3.map((t: any) => `${t.name}:${(t.score * 100).toFixed(1)}%`).join(" | ")
        );

        candidates.push({startSec, rms: r, ent, top3});

        await new Promise<void>(r => requestAnimationFrame(() => r()));
    }

    if (candidates.length === 0) {
        console.log("🏁 MICCAP FINAL | no non-silent windows found (increase volume / move closer)");
        return;
    }

    // Pick best by (lowest entropy, then highest top1 score)
    candidates.sort((a, b) => {
        const a1 = a.top3?.[0]?.score ?? 0;
        const b1 = b.top3?.[0]?.score ?? 0;
        if (a.ent !== b.ent) return a.ent - b.ent;
        return b1 - a1;
    });

    const best = candidates[0];
    const bestTop1 = best.top3?.[0];
    console.log(
        `🏁 MICCAP FINAL | bestStart=${best.startSec.toFixed(2)}s rms=${best.rms.toFixed(4)} ent=${best.ent.toFixed(3)} | ` +
        `${bestTop1?.name ?? "?"}:${((bestTop1?.score ?? 0) * 100).toFixed(1)}%`
    );
}
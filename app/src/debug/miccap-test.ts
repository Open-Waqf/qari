import {inferenceEngine} from "../model/inference-engine";

function rmsOf(x: Float32Array) {
    let sum = 0;
    for (let i = 0; i < x.length; i++) sum += x[i] * x[i];
    return Math.sqrt(sum / Math.max(1, x.length));
}

function entropyNormalized(probs: number[]) {
    const N = probs.length;
    if (N <= 1) return 0;
    let e = 0;
    for (const p of probs) if (p > 0) e -= p * Math.log(p);
    return e / Math.log(N);
}

export async function runMicCapTest(signal: Float32Array) {
    const SR = 22050; // 🟢 Updated
    const WIN = 2 * SR; // 🟢 2.0s window
    const HOP = 1 * SR; // 1.0s hop

    console.log(`🏁 MICCAP | dur=${(signal.length / SR).toFixed(2)}s`);

    let sumProbs: number[] | null = null;
    let wSum = 0;

    for (let start = 0; start + WIN <= signal.length; start += HOP) {
        const startSec = start / SR;
        const window = signal.subarray(start, start + WIN);
        const r = rmsOf(window);

        // IMPORTANT: mic often comes in quieter than you think
        if (r < 0.005) continue;

        // 🟢 Using 2s window
        const result = await inferenceEngine.predictFromSignal(window, {
            startSec: 0,
            windowSec: 2,
            log: false,
            dispatchToUI: false,
            independent: true,
        });

        const probs = result.probs;
        const ent = result.raw.ent;
        const top3 = result.raw.top3;

        console.log(
            `🏁 MICCAP | start=${startSec.toFixed(2)}s rms=${r.toFixed(4)} ent=${ent.toFixed(3)} | ` +
            top3.map(t => `${t.name}:${(t.score * 100).toFixed(1)}%`).join(" | ")
        );

        // weight: louder + more confident windows count more
        const w = r * (1 - ent);
        if (!sumProbs) sumProbs = new Array(probs.length).fill(0);
        for (let i = 0; i < probs.length; i++) sumProbs[i] += probs[i] * w;
        wSum += w;

        await new Promise<void>(rr => requestAnimationFrame(() => rr()));
    }

    if (!sumProbs || wSum <= 0) {
        console.log("🏁 MICCAP FINAL | no non-silent windows found");
        return;
    }

    const avg = sumProbs.map(v => v / wSum);
    const entFinal = entropyNormalized(avg);

    console.log(`✅ MICCAP SEQUENCE DONE | Avg Entropy: ${entFinal.toFixed(3)}`);
}
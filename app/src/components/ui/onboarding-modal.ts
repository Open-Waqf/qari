import {css, html, LitElement} from 'lit';
import {customElement, property, state} from 'lit/decorators.js';
import {inferenceEngine} from "../../model/inference-engine.ts";
import {i18n} from "../../core/i18n.ts";
import {audioManager} from "../../core/audio-manager.ts";

@customElement('onboarding-modal')
export class OnboardingModal extends LitElement {
    @property({type: Boolean}) showDetails = false;

    @state() isCalibrating = false;

    @property({type: Object}) dict = i18n.t;

    static styles = css`
        :host {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(2, 4, 10, 0.95);
            z-index: 999;
            display: flex;
            justify-content: center;
            align-items: center;
            backdrop-filter: blur(10px);
            transition: opacity 0.3s ease;

            /* ✅ FIX: Force initial opacity to 1 */
            opacity: 1;
            padding: 24px;
            padding-bottom: env(safe-area-inset-bottom);
            box-sizing: border-box;
        }

        :host([hidden]) {
            display: none;
        }

        :host(.closing) {
            opacity: 0;
            pointer-events: none;
        }

        .card {
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.01));
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 32px;
            width: 100%;
            max-width: 340px;
            text-align: center;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
        }

        h2 {
            margin: 0 0 16px;
            font-size: 1.5rem;
            letter-spacing: -0.5px;
        }

        p {
            color: var(--text-dim, #8b949e);
            line-height: 1.5;
            margin-bottom: 24px;
            font-size: 0.95rem;
        }

        .btn {
            background: #0077ff;
            color: white;
            border: none;
            padding: 16px;
            width: 100%;
            border-radius: 14px;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: transform 0.2s;
        }

        .btn:active {
            transform: scale(0.98);
        }

        .sub-btn {
            background: transparent;
            color: var(--text-dim, #8b949e);
            border: none;
            margin-top: 16px;
            font-size: 0.85rem;
            cursor: pointer;
            text-decoration: underline;
        }

        .calibration-loader {
            width: 40px;
            height: 40px;
            border: 3px solid rgba(0, 119, 255, 0.3);
            border-radius: 50%;
            border-top-color: #0077ff;
            animation: spin 1s ease-in-out infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            to {
                transform: rotate(360deg);
            }
        }
    `;

    private async _handleStart() {
        // Just triggers the unified flow
        await this.runCalibration();
        this.dispatchEvent(new CustomEvent('onboard-complete'));
    }

    async runCalibration(): Promise<number> {
        this.hidden = false;
        this.classList.remove("closing");
        this.isCalibrating = true;

        const samples: number[] = [];

        const wasRunning = audioManager.isRunning;
        const prevCb = audioManager.onDataReceived;

        const warmupMs = 350;        // allow worklet/resampler to “wake up”
        const minDurationMs = 2000;  // your UX target
        const maxDurationMs = 3500;  // safety if chunk delivery is sparse
        const minSamples = 10;       // chunk-based; 2s @185ms ≈ 10–11 chunks

        const t0 = performance.now();

        // Collect once per chunk (not by polling)
        const onChunk = (chunk: Float32Array) => {
            // Update lastRms using your exact gate RMS path
            inferenceEngine.handleIncomingAudio(chunk);

            if (performance.now() - t0 < warmupMs) return;

            const rms = inferenceEngine.getCurrentRms();
            if (Number.isFinite(rms) && rms > 0) samples.push(rms);
        };

        try {
            await audioManager.start(onChunk);
        } catch (e) {
            console.warn("⚠️ Calibration: could not start mic", e);

            // Hard fallback (and SAVE it so startEngine restores something sane)
            const fallback = 0.001;
            inferenceEngine.setNoiseFloor(fallback);
            localStorage.setItem("qari_noise_floor", String(fallback));

            this._finish();
            return fallback;
        }

        // Wait until we have enough data or hit max time
        const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));
        while (true) {
            const dt = performance.now() - t0;
            const enoughTime = dt >= minDurationMs;
            const enoughSamples = samples.length >= minSamples;
            if ((enoughTime && enoughSamples) || dt >= maxDurationMs) break;
            await sleep(50);
        }

        let noiseFloor: number;

        if (samples.length < 5) {
            // ✅ fallback should NOT blindly reuse a noisy saved value
            const saved = Number(localStorage.getItem("qari_noise_floor"));
            const savedOk = Number.isFinite(saved) && saved > 0;

            // If saved exists, cap it so we don’t lock the gate shut on quiet mics
            noiseFloor = savedOk ? Math.min(saved, 0.005) : 0.001;

            console.warn(`⚠️ Calibration fallback used (samples=${samples.length}) → ${noiseFloor}`);
        } else {
            samples.sort((a, b) => a - b);

            // Same idea you had (high percentile + headroom),
            // but now it’s fed by true chunk updates.
            const p90 = samples[Math.floor(samples.length * 0.90)] ?? samples[samples.length - 1];
            noiseFloor = p90 * 1.15;

            // ✅ clamp to sane floor: avoid 0.0003 “always open gate”
            noiseFloor = Math.max(0.001, Math.min(0.05, noiseFloor));

            console.log(`✅ Calibrated Noise Floor: ${noiseFloor} (samples=${samples.length})`);
        }

        inferenceEngine.setNoiseFloor(noiseFloor);
        localStorage.setItem("qari_noise_floor", String(noiseFloor));

        // Restore previous callback if mic was already running
        if (wasRunning) {
            audioManager.onDataReceived = prevCb ?? null;
        } else {
            // If calibration started the mic, stop it here.
            // This prevents startEngine() from needing to stop it again.
            audioManager.stop();
        }

        this._finish();
        return noiseFloor;
    }

    // Helper to close UI cleanly
    private _finish() {
        this.isCalibrating = false;
        this.classList.add('closing');

        setTimeout(() => {
            this.hidden = true;
            this.classList.remove('closing'); // Reset for next time
        }, 300);
    }

    render() {
        return html`
            <div class="card">
                ${this.isCalibrating ? html`
                    <div class="calibration-loader"></div>
                    <h2>${this.dict.calibTitle || "Calibrating..."}</h2>
                    <p>${this.dict.calibDesc || "Please stay silent for a moment."}</p>
                ` : this.showDetails ? html`
                    <h2>${this.dict.scienceTitle || "How it works"}</h2>
                    <p>${this.dict.scienceDesc || "We analyze audio features..."}</p>
                    <button class="sub-btn" @click="${() => this.showDetails = false}">${this.dict.back || "Back"}
                    </button>
                ` : html`
                    <h2>${this.dict.welcomeTitle || "Welcome"}</h2>
                    <p>${this.dict.welcomeDesc || "Let's calibrate your microphone."}</p>
                    <button class="btn" @click="${this._handleStart}">${this.dict.getStarted || "Start"}</button>
                `}
            </div>
        `;
    }
}
import {css, html, LitElement} from 'lit';
import {customElement, property, state} from 'lit/decorators.js';
import {inferenceEngine} from "../../model/inference-engine.ts";
import {i18n} from "../../core/i18n.ts";

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
        // ✅ FIX: Force visibility + Spinner state immediately
        this.hidden = false;
        this.classList.remove('closing');
        this.isCalibrating = true;

        return new Promise((resolve) => {
            const samples: number[] = [];

            const readRms = () => inferenceEngine.getCurrentRms();

            // Collect samples
            const interval = window.setInterval(() => {
                try {
                    const v = readRms();
                    if (Number.isFinite(v)) samples.push(Math.max(0, v));
                } catch (e) { /* ignore */
                }
            }, 50);

            // Finish after 2s
            const timeout = window.setTimeout(() => {
                window.clearInterval(interval);

                // Fallback Logic
                if (samples.length < 10) {
                    const saved = Number(localStorage.getItem('qari_noise_floor'));
                    const fallback = (Number.isFinite(saved) && saved > 0) ? saved : 0.001;
                    inferenceEngine.setNoiseFloor(fallback);
                    this._finish();
                    resolve(fallback);
                    return;
                }

                // P90 Logic
                samples.sort((a, b) => a - b);
                const p90 = samples[Math.floor(samples.length * 0.90)] ?? 0;
                let noiseFloor = p90 * 1.15;
                noiseFloor = Math.max(0.0003, Math.min(0.05, noiseFloor));

                inferenceEngine.setNoiseFloor(noiseFloor);
                localStorage.setItem('qari_noise_floor', String(noiseFloor));

                console.log('✅ Calibrated Noise Floor:', noiseFloor);

                this._finish();
                resolve(noiseFloor);
            }, 2000);

            // Safety cleanup
            const stop = () => {
                window.clearInterval(interval);
                window.clearTimeout(timeout);
                this.isCalibrating = false;
            };
            this.addEventListener('disconnected', stop, {once: true} as any);
        });
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
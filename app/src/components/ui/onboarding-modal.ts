import {css, html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';
import {audioManager} from "../../core/audio-manager.ts";
import {inferenceEngine} from "../../model/inference-engine.ts";

@customElement('onboarding-modal')
export class OnboardingModal extends LitElement {
    @property({type: Boolean}) open = true;
    @property({type: Boolean}) showDetails = false;
    @property({type: Boolean}) isCalibrating = false;

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
        }

        :host([hidden]) {
            display: none;
        }

        .card {
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.01));
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 32px;
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

        .features {
            display: flex;
            justify-content: space-around;
            margin-bottom: 32px;
        }

        .feat {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
            font-size: 0.8rem;
            color: #fff;
        }

        .icon {
            font-size: 1.5rem;
            background: rgba(255, 255, 255, 0.1);
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
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
    `;

    private _handleStart() {
        this.style.opacity = '0';
        setTimeout(() => {
            this.hidden = true;
            this.dispatchEvent(new CustomEvent('onboard-complete'));
        }, 300);
    }

    async runCalibration() {
        this.hidden = false;
        this.style.opacity = '1';
        this.open = true;
        this.isCalibrating = true;

        const samples: number[] = [];

        return new Promise((resolve) => {
            const checkNoise = (chunk: Float32Array) => {
                let sum = 0;
                for (let i = 0; i < chunk.length; i++) sum += chunk[i] * chunk[i];
                samples.push(Math.sqrt(sum / chunk.length));
            };

            audioManager.start(checkNoise);

            setTimeout(() => {
                audioManager.stop();
                const peakNoise = Math.max(...samples);
                const safeThreshold = peakNoise * 1.5;

                inferenceEngine.setThreshold(safeThreshold);

                this.isCalibrating = false;
                this.hidden = true; // Hide it again when done
                resolve(safeThreshold);
            }, 2000);
        });
    }

    render() {
        return html`
            <div class="card">
                ${this.isCalibrating ? html`
                    <div class="calibration-loader"></div>
                    <h2>Calibrating Ears</h2>
                    <p>We are adjusting to your room's noise levels. Please stay silent...</p>
                ` : this.showDetails ? html`
                    <h2>Science of Sound</h2>
                    <p>Local neural networks analyze frequency patterns without recording audio.</p>
                    <button class="sub-btn" @click="${() => this.showDetails = false}">Back</button>
                ` : html`
                    <h2>Welcome to Qari Finder</h2>
                    <p>Identify reciters instantly and privately.</p>
                    <button class="btn" @click="${this._handleStart}">Get Started</button>
                `}
            </div>
        `;
    }
}
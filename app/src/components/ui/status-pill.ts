import {css, html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';

export type StatusState = 'idle' | 'listening' | 'stabilizing' | 'match' | 'quiet';

@customElement('status-pill')
export class StatusPill extends LitElement {
    @property() state: StatusState = 'idle';
    @property() text = '';

    static styles = css`
        :host {
            display: inline-block;
            margin-bottom: 20px;
        }

        .pill {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 8px 16px;
            border-radius: 30px;
            font-size: 0.85rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
            transition: all 0.3s ease;
        }

        /* State Colors */

        .listening {
            border-color: #0077ff;
            color: #fff;
        }

        .match {
            background: rgba(0, 255, 136, 0.1);
            border-color: #00ff88;
            color: #00ff88;
        }

        .quiet {
            border-color: #ffaa00;
            color: #ffaa00;
        }

        /* Dot Animation */

        .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
        }

        .listening .dot {
            animation: pulse 1s infinite;
        }

        @keyframes pulse {
            0% {
                opacity: 1;
            }
            50% {
                opacity: 0.4;
            }
            100% {
                opacity: 1;
            }
        }
    `;

    render() {
        return html`
            <div class="pill ${this.state}">
                <div class="dot"></div>
                <span>${this.text || 'Ready'}</span>
            </div>
        `;
    }
}
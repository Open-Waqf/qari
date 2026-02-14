import {css, html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';

@customElement('glass-card')
export class GlassCard extends LitElement {
    @property({type: String}) name = '';
    @property({type: Boolean}) visible = false;

    // Translation Props
    @property({type: String}) listenBtnText = 'Listen on YouTube';
    @property({type: String}) dismissBtnText = 'Dismiss';

    static styles = css`
        :host {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            z-index: 100;
            transform: translateY(110%);
            transition: transform 0.5s cubic-bezier(0.32, 0.72, 0, 1);
            pointer-events: none; /* Don't block clicks when hidden */
        }

        :host([visible]) {
            transform: translateY(0);
            pointer-events: auto;
        }

        .card {
            background: rgba(20, 20, 30, 0.9);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-top: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 24px 24px 0 0;
            padding: 32px 24px 48px 24px;
            box-shadow: 0 -10px 40px rgba(0, 0, 0, 0.5);
            text-align: center;
        }

        .handle {
            width: 40px;
            height: 4px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 2px;
            margin: 0 auto 24px auto;
        }

        h2 {
            margin: 0;
            font-size: 2rem;
            color: white;
            letter-spacing: -0.5px;
            background: linear-gradient(to right, #fff, #aaa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 24px;
        }

        .actions {
            display: flex;
            gap: 12px;
            justify-content: center;
            flex-wrap: wrap;
        }

        button {
            background: rgba(255, 255, 255, 0.1);
            border: none;
            color: white;
            padding: 14px 24px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: background 0.2s;
            font-family: inherit;
        }

        button:active {
            transform: scale(0.98);
        }

        button.primary {
            background: #0077ff;
        }
    `;

    render() {
        return html`
            <div class="card">
                <div class="handle"></div>
                <h2>${this.name}</h2>
                <div class="actions">
                    <button class="primary">${this.listenBtnText}</button>
                    <button @click="${() => this.dispatchEvent(new CustomEvent('close'))}">
                        ${this.dismissBtnText}
                    </button>
                </div>
            </div>
        `;
    }
}
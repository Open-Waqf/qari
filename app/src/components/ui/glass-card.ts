import {css, html, LitElement} from 'lit';
import {customElement} from 'lit/decorators.js';

@customElement('glass-card')
export class GlassCard extends LitElement {
    static styles = css`
        :host {
            display: block;
            box-sizing: border-box;
            width: 100%;
            background: rgba(20, 25, 35, 0.6); /* Slightly darker for contrast */
            backdrop-filter: blur(24px) saturate(180%); /* iOS-style blur */
            -webkit-backdrop-filter: blur(24px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-top: 1px solid rgba(255, 255, 255, 0.15); /* Top highlight */
            border-radius: 32px;
            padding: 32px;
            box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.5),
            inset 0 0 0 1px rgba(255, 255, 255, 0.05); /* Inner stroke */
            margin-top: 20px;
        }
    `;

    render() {
        return html`
            <slot></slot>`;
    }
}
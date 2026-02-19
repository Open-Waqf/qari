import {css, html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';

@customElement('match-history')
export class MatchHistory extends LitElement {
    // FIX: Removed duplicate @property decorator
    @property({type: Array}) qariMatches: any[] = [];
    @property({type: Object}) dict: any = null;

    static styles = css`
        :host {
            display: block;
            margin-top: 25px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding-top: 20px;
        }

        .title {
            font-size: 0.7rem;
            text-transform: uppercase;
            color: var(--text-dim);
            letter-spacing: 1.5px;
            margin-bottom: 15px;
            display: block;
            font-weight: 700;
        }

        .title.rtl-mode {
            letter-spacing: 0 !important;
            font-size: unset;
        }

        .match-card {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 12px;
            padding: 12px 16px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-left: 2px solid var(--secondary);
            transition: transform 0.2s ease;
        }

        .match-card:hover {
            transform: translateX(5px);
            background: rgba(255, 255, 255, 0.04);
        }

        .match-name {
            font-weight: 600;
            font-size: 0.85rem;
            color: #fff;
        }

        .match-score {
            color: var(--accent);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
        }

        .match-card {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 12px;
            padding: 12px 16px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-left: 2px solid var(--secondary);
            /* Make the transition bouncy */
            transition: transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1), background 0.2s ease;
        }

        /* Desktop Hover */
        @media (hover: hover) {
            .match-card:hover {
                transform: translateX(5px);
                background: rgba(255, 255, 255, 0.06);
                border-left: 4px solid var(--accent); /* Changes color on hover! */
            }
        }

        /* Mobile Tap */

        .match-card:active {
            transform: scale(0.96); /* Squish inward when tapped */
            background: rgba(255, 255, 255, 0.08);
        }
    `;

    render() {
        if (this.qariMatches.length === 0) return html``;
        // Use dictionary for title if available
        const title = this.dict ? this.dict.historyTitle : 'Recent Matches';
        const isRtl = document.documentElement.dir === 'rtl';

        return html`
            <span class="title ${isRtl ? 'rtl-mode' : ''}">${title}</span>
            ${this.qariMatches.slice(0, 5).map(m => html`
                <div class="match-card">
                    <span class="match-name">${m.name}</span>
                    <span class="match-score">${(m.score * 100).toFixed(0)}%</span>
                </div>
            `)}
        `;
    }
}
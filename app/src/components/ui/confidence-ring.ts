import {css, html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';

@customElement('confidence-ring')
export class ConfidenceRing extends LitElement {
    @property({type: Object}) dict: any = null;
    @property({type: Number}) score = 0;
    @property({type: String}) label = '';

    static styles = css`
        :host {
            display: block;
            width: 240px;
            height: 240px;
        }

        svg {
            width: 100%;
            height: 100%;
            direction: ltr; /* Force LTR for geometry */
        }

        .track {
            fill: none;
            stroke: rgba(255, 255, 255, 0.1);
            stroke-width: 6;
        }

        .progress {
            fill: none;
            stroke: #00d2ff;
            stroke-width: 6;
            stroke-linecap: round;
            transform: rotate(-90deg);
            transform-origin: 50% 50%;
            transition: stroke-dashoffset 0.6s cubic-bezier(0.22, 1, 0.36, 1);
            filter: drop-shadow(0 0 15px rgba(0, 210, 255, 0.3));
        }

        .content {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 10;
            padding: 20px; /* Internal padding to keep text safe */
            box-sizing: border-box;
        }

        /* Responsive Typography */

        .score {
            font-size: 3.5rem;
            font-weight: 800;
            color: #fff;
            line-height: 1;
            transition: font-size 0.3s ease;
        }

        /* When we have a name, shrink score slightly to fit the name */

        .score.has-label {
            font-size: 2.8rem;
        }

        .label {
            color: rgba(255, 255, 255, 0.9);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 8px;
            text-align: center;

            /* WRAPPING LOGIC FOR LONG NAMES */
            white-space: normal; /* Allow wrapping */
            word-wrap: break-word;
            font-size: 1rem;
            font-weight: 600;
            line-height: 1.2;

            /* Clamp to 2 lines max */
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;

            max-width: 180px; /* Ensure it doesn't hit the ring edges */
        }

        .label.analyzing {
            color: rgba(255, 255, 255, 0.5);
            font-size: 0.8rem;
            font-weight: 400;
            letter-spacing: 2px;
        }

        .rtl-mode .label.analyzing {
            letter-spacing: 0; /* Fixes disconnected Arabic letters */
        }
    `;

    render() {
        const r = 100;
        const c = 2 * Math.PI * r;
        const offset = c - (this.score * c);

        const analyzingText = this.dict ? this.dict.analyzing : 'LISTENING...';
        // If score is basically 0, we are analyzing
        const isAnalyzing = this.score <= 0.01;

        const displayLabel = !isAnalyzing ? this.label : analyzingText;
        const strokeColor = this.score > 0.8 ? '#10b981' : '#00d2ff';

        const isRtl = document.documentElement.dir === 'rtl';

        return html`
            <svg>
                <circle class="track" cx="120" cy="120" r="${r}"></circle>
                <circle class="progress" cx="120" cy="120" r="${r}"
                        stroke-dasharray="${c}" stroke-dashoffset="${offset}"
                        style="stroke: ${strokeColor}">
                </circle>
            </svg>
            <div class="content ${isRtl ? 'rtl-mode' : ''}">
                <div class="score ${!isAnalyzing ? 'has-label' : ''}">
                    ${Math.round(this.score * 100)}<span style="font-size:50%">%</span>
                </div>
                <div class="label ${isAnalyzing ? 'analyzing' : ''}">
                    ${displayLabel}
                </div>
            </div>
        `;
    }
}
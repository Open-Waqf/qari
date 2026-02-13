// app/src/components/ui/confidence-meter.ts
import {css, html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';

@customElement('confidence-meter')
export class ConfidenceMeter extends LitElement {

    @property({type: Object}) dict: any = null; // Store translations
    @property({type: String}) label = '';
    @property({type: Number}) score = 0;

    static styles = css`
        .container {
            margin-top: 24px;
        }

        .label-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }

        .name {
            font-weight: 800;
            color: #fff;
            letter-spacing: 0.5px;
        }

        .percent {
            color: #0077ff;
            font-mono: true;
            font-weight: bold;
        }

        .track {
            width: 100%;
            height: 6px;
            background: #161b22;
            border-radius: 10px;
            overflow: hidden;
        }

        .fill {
            height: 100%;
            background: linear-gradient(90deg, #0077ff, #7000ff);
            transition: width 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
            box-shadow: 0 0 15px rgba(0, 119, 255, 0.4);
        }
    `;

    render() {
        const p = Math.round(this.score * 100);
        const analyzingText = this.dict ? this.dict.analyzing : 'ANALYZING...';
        return html`
            <div class="container">
                <div class="label-row">
                    <span class="name">${this.score > 0 ? this.label : analyzingText}</span>
                    <span class="percent">${p}%</span>
                </div>
                <div class="track">
                    <div class="fill" style="width: ${p}%"></div>
                </div>
            </div>
        `;
    }
}
import {css, html, LitElement} from 'lit';
import {customElement, property, query} from 'lit/decorators.js';
import {audioManager} from '../../core/audio-manager';

@customElement('dna-visualizer')
export class DnaVisualizer extends LitElement {
    @query('canvas') canvas!: HTMLCanvasElement;
    @property({type: String}) mode: 'wave' | 'heatmap' = 'wave';

    private ctx: CanvasRenderingContext2D | null = null;
    private tempCanvas = document.createElement('canvas');
    private tempCtx = this.tempCanvas.getContext('2d');

    static styles = css`
        :host {
            display: block;
            width: 100%;
            height: 180px;
            cursor: pointer;
        }

        canvas {
            width: 100%;
            height: 100%;
            border-radius: 12px;
        }
    `;

    firstUpdated() {
        this.ctx = this.canvas.getContext('2d');
        // Ensure temp canvas matches size
        this.tempCanvas.width = this.canvas.width;
        this.tempCanvas.height = this.canvas.height;

        this.addEventListener('click', () => {
            this.mode = this.mode === 'wave' ? 'heatmap' : 'wave';
        });
        this.loop();
    }

    loop() {
        requestAnimationFrame(() => this.loop());
        if (!this.ctx || !audioManager.analyser) return;

        const width = this.canvas.width;
        const height = this.canvas.height;

        if (this.mode === 'wave') {
            this.drawWave(width, height);
        } else {
            this.drawHeatmap(width, height);
        }
    }

    private drawWave(w: number, h: number) {
        const buffer = audioManager.analyser!.frequencyBinCount;
        const data = new Uint8Array(buffer);
        audioManager.analyser!.getByteTimeDomainData(data);

        this.ctx!.fillStyle = '#0d1117';
        this.ctx!.fillRect(0, 0, w, h);

        this.ctx!.lineWidth = 3;
        this.ctx!.strokeStyle = '#0077ff';
        this.ctx!.beginPath();

        const sliceWidth = w / buffer;
        let x = 0;
        for (let i = 0; i < buffer; i++) {
            const v = data[i] / 128.0;
            const y = (v * h) / 2;
            i === 0 ? this.ctx!.moveTo(x, y) : this.ctx!.lineTo(x, y);
            x += sliceWidth;
        }
        this.ctx!.stroke();
    }

    private drawHeatmap(w: number, h: number) {
        const buffer = audioManager.analyser!.frequencyBinCount;
        const data = new Uint8Array(buffer);
        audioManager.analyser!.getByteFrequencyData(data);

        // 1. Copy the current canvas to the temp buffer
        this.tempCtx!.drawImage(this.canvas, 0, 0);

        // 2. Draw the temp buffer back to main canvas, shifted 2px left
        this.ctx!.drawImage(this.tempCanvas, -2, 0);

        // 3. Draw the NEW frequency data on the far right edge (2px wide)
        const barWidth = 2;
        const barHeight = h / (buffer / 4); // Focus on lower frequencies where voice lives

        for (let i = 0; i < buffer / 4; i++) {
            const value = data[i];
            // Cyber Blue to Indigo palette
            const hue = 200 + (value / 255) * 60;
            const saturation = 100;
            const lightness = (value / 255) * 50;

            this.ctx!.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
            // We flip 'i' so low frequencies are at the bottom
            this.ctx!.fillRect(w - barWidth, h - (i * barHeight), barWidth, barHeight);
        }
    }

    render() {
        return html`
            <canvas></canvas>`;
    }
}
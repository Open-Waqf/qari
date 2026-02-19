import {css, html, LitElement} from 'lit';
import {customElement} from 'lit/decorators.js';
import {audioManager} from '../../core/audio-manager';

@customElement('halo-visualizer')
export class HaloVisualizer extends LitElement {
    private canvas!: HTMLCanvasElement;
    private ctx!: CanvasRenderingContext2D;
    private resizeObserver!: ResizeObserver;

    // Cache dimensions to avoid DOM reads in the loop
    private w: number = 0;
    private h: number = 0;

    static styles = css`
        :host {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            pointer-events: none;
            opacity: 0.6;
        }

        canvas {
            width: 100%;
            height: 100%;
        }
    `;

    firstUpdated() {
        this.canvas = this.shadowRoot!.querySelector('canvas')!;
        this.ctx = this.canvas.getContext('2d')!;

        this.resizeObserver = new ResizeObserver(entries => {
            for (let entry of entries) {
                this.w = this.canvas.width = Math.floor(entry.contentRect.width);
                this.h = this.canvas.height = Math.floor(entry.contentRect.height);
            }
        });
        this.resizeObserver.observe(this);

        this.loop();
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        if (this.resizeObserver) this.resizeObserver.disconnect();
    }

    loop = () => {
        requestAnimationFrame(this.loop);

        if (this.w === 0 || this.h === 0 || !audioManager.analyser) return;

        const data = new Uint8Array(audioManager.analyser.frequencyBinCount);
        audioManager.analyser.getByteFrequencyData(data);

        let sum = 0;
        for (let i = 10; i < 100; i++) sum += data[i];
        const energy = sum / 90;

        // 🟢 FIX: Clear the old frame explicitly since we are no longer resetting canvas.width
        this.ctx.clearRect(0, 0, this.w, this.h);

        const radius = 100 + (energy * 1.5);
        const gradient = this.ctx.createRadialGradient(this.w / 2, this.h / 2, radius * 0.2, this.w / 2, this.h / 2, radius);
        gradient.addColorStop(0, 'rgba(0, 210, 255, 0)');
        gradient.addColorStop(0.5, `rgba(0, 210, 255, ${energy / 800})`);
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.w, this.h);
    }

    render() {
        return html`
            <canvas></canvas>`;
    }
}
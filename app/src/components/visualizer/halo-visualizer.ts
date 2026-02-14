import {css, html, LitElement} from 'lit';
import {customElement} from 'lit/decorators.js';
import {audioManager} from '../../core/audio-manager';

@customElement('halo-visualizer')
export class HaloVisualizer extends LitElement {
    private canvas!: HTMLCanvasElement;
    private ctx!: CanvasRenderingContext2D;

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
        this.loop();
    }

    loop() {
        requestAnimationFrame(() => this.loop());

        const w = this.canvas.width = this.offsetWidth;
        const h = this.canvas.height = this.offsetHeight;

        if (!audioManager.analyser) return;

        const data = new Uint8Array(audioManager.analyser.frequencyBinCount);
        audioManager.analyser.getByteFrequencyData(data);

        // Calculate "Energy" (Volume)
        let sum = 0;
        // Focus on voice frequencies (indexes 10 to 100)
        for (let i = 10; i < 100; i++) sum += data[i];
        const energy = sum / 90; // Average volume 0-255

        // Draw the "Halo"
        const radius = 100 + (energy * 1.5); // Expands with voice

        const gradient = this.ctx.createRadialGradient(w / 2, h / 2, radius * 0.2, w / 2, h / 2, radius);
        gradient.addColorStop(0, 'rgba(0, 210, 255, 0)');
        gradient.addColorStop(0.5, `rgba(0, 210, 255, ${energy / 800})`); // Faint glow
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, w, h);
    }

    render() {
        return html`
            <canvas></canvas>`;
    }
}
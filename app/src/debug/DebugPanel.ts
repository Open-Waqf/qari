import {audioManager} from "../core/audio-manager";
import {inferenceEngine} from "../model/inference-engine";
import {runFileLoopback} from "./file-loopback";
import {checkAudioParity} from "./debug-extractor";
import {downloadBlob} from "./wav";
import {runFileTest} from "./file-test";

export class DebugPanel {
    private container: HTMLDivElement;
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private logContainer: HTMLDivElement;
    private isOpen = false;
    private animationFrame = 0;

    private btnRms: HTMLButtonElement | null = null;
    private btnPre: HTMLButtonElement | null = null;
    private btnCmvn: HTMLButtonElement | null = null;
    private btnFar: HTMLButtonElement | null = null;

    constructor() {
        this.container = document.createElement("div");
        this.container.id = "qari-debug-panel";
        this.applyStyles();

        const fab = document.createElement("button");
        fab.innerText = "🐞";
        fab.className = "debug-fab";
        fab.onclick = () => this.toggle();
        this.container.appendChild(fab);

        const content = document.createElement("div");
        content.className = "debug-content";

        // ✅ FIX 1: Added explicit fields for Backend (TF), Resampler (Res), and Latency
        const stats = document.createElement("div");
        stats.className = "debug-stats";
        stats.innerHTML = `
            <div><strong>TF:</strong> <span id="dbg-tf" style="color:#f0f">--</span></div>
            <div><strong>Res:</strong> <span id="dbg-res" style="color:#0ff">--</span></div>
            <div><strong>Lat:</strong> <span id="dbg-lat" style="color:#ff0">--</span> ms</div>
            <div><strong>RMS:</strong> <span id="dbg-rms">0.00</span></div>
            <div><strong>Floor:</strong> <span id="dbg-floor">--</span></div>
        `;
        content.appendChild(stats);

        this.canvas = document.createElement("canvas");
        this.canvas.width = 280;
        this.canvas.height = 40;
        this.ctx = this.canvas.getContext("2d")!;
        content.appendChild(this.canvas);

        const dspGrid = document.createElement("div");
        dspGrid.className = "debug-grid";
        this.btnRms = this.createToggle("RMS", () => {
            inferenceEngine.toggleRmsNormalize();
            this.log(`RMS: ${inferenceEngine.isRmsNormalizeEnabled()}`)
        });
        this.btnPre = this.createToggle("Pre", () => {
            inferenceEngine.togglePreEmphasis();
            this.log(`Pre: ${inferenceEngine.isPreEmphasisEnabled()}`)
        });
        this.btnCmvn = this.createToggle("CMVN", () => {
            inferenceEngine.toggleCMVN();
            this.log(`CMVN: ${inferenceEngine.isCmvnEnabled()}`);
        });
        this.btnFar = this.createToggle("Far", () => {
            const am = audioManager as any;
            if (am.flags) {
                am.flags.farFieldMode = !am.flags.farFieldMode;
                this.log(`FarField: ${am.flags.farFieldMode}`);

                if (am.isRunning) {
                    const cb = am.onDataReceived;
                    am.stop();
                    setTimeout(() => {
                        if (cb) am.start(cb);
                    }, 300);
                }
            }
        });
        dspGrid.append(this.btnRms, this.btnPre, this.btnCmvn, this.btnFar);
        content.appendChild(dspGrid);

        const actionGrid = document.createElement("div");
        actionGrid.className = "debug-grid";
        actionGrid.append(
            this.createButton("Loop", () => this.triggerFilePicker(runFileLoopback)),
            this.createButton("Test", () => this.triggerFilePicker(runFileTest)),
            this.createButton("Wav", () => this.captureRawMic()),
            this.createButton("Math", async () => {
                this.log("Running Parity...");
                const dummy = new Array(512).fill(0).map((_, i) => Math.sin(i * 0.1));
                const res = await checkAudioParity(dummy);
                this.log(`Math: [${res[0].toFixed(2)}, ${res[1].toFixed(2)}, ${res[2].toFixed(2)}]`);
            })
        );
        content.appendChild(actionGrid);

        this.logContainer = document.createElement("div");
        this.logContainer.className = "debug-log";
        content.appendChild(this.logContainer);

        this.container.appendChild(content);
        document.body.appendChild(this.container);
        this.startLoop();
    }

    public log(msg: string) {
        const entry = document.createElement("div");
        entry.innerText = `> ${msg}`;
        this.logContainer.prepend(entry);
        if (this.logContainer.childNodes.length > 8) this.logContainer.lastChild?.remove();
        console.log(`[DEBUG] ${msg}`);
    }

    private captureRawMic() {
        const am = audioManager as any;
        const stream = am._stream;
        if (!stream) return this.log("Error: No Mic Stream");

        try {
            const rec = new MediaRecorder(stream);
            const chunks: Blob[] = [];
            rec.ondataavailable = e => chunks.push(e.data);
            rec.onstop = () => {
                downloadBlob(new Blob(chunks, {type: "audio/webm"}), `mic_${Date.now()}.webm`);
                this.log("Wav Saved");
            };
            rec.start();
            this.log("Recording 4s...");
            setTimeout(() => rec.stop(), 4000);
        } catch (e) {
            this.log("Recorder Error");
        }
    }

    private startLoop() {
        const loop = () => {
            if (this.isOpen) {
                this.updateUI();
            }
            this.animationFrame = requestAnimationFrame(loop);
        };
        loop();
    }

    public stop() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = 0;
        }
    }

    private updateUI() {
        const am = audioManager as any;
        const eng = inferenceEngine as any;

        // ✅ FIX 2: Connect to new getters (safely)
        // Note: These rely on the changes to inference-engine.ts and audio-manager.ts
        const tfBackend = eng.getBackend ? eng.getBackend() : '??';
        const resMode = am.getResamplerMode ? am.getResamplerMode() : '??';
        const lastTime = eng.getLastInferenceTime ? eng.getLastInferenceTime() : 0;
        const latency = Date.now() - lastTime;

        document.getElementById("dbg-tf")!.innerText = tfBackend;
        document.getElementById("dbg-res")!.innerText = resMode;

        // Only show latency if we actually have a recent prediction
        const latText = (lastTime > 0 && latency < 5000) ? `${latency}` : "--";
        document.getElementById("dbg-lat")!.innerText = latText;

        document.getElementById("dbg-floor")!.innerText =
            eng.gate?.noiseFloor != null ? eng.gate.noiseFloor.toFixed(4) : "--";

        // ✅ FIX 3: Correct RMS calculation using Float32 and full FFT size
        if (audioManager.analyser) {
            // Use fftSize (2048 or 4096) for full time domain, not frequencyBinCount (which is half)
            const len = audioManager.analyser.fftSize;
            const data = new Float32Array(len);
            audioManager.analyser.getFloatTimeDomainData(data);

            let sum = 0;
            for (let i = 0; i < len; i++) {
                sum += data[i] * data[i];
            }
            const rms = Math.sqrt(sum / len);
            document.getElementById("dbg-rms")!.innerText = rms.toFixed(4);

            this.ctx.fillStyle = "#111";
            this.ctx.fillRect(0, 0, 280, 40);

            // Draw bar based on RMS
            this.ctx.fillStyle = rms > 0.05 ? "#0f0" : "#0077ff";
            this.ctx.fillRect(0, 10, Math.min(1, rms * 10) * 280, 20);

            // Optional: Draw simple waveform line for better feedback
            this.ctx.strokeStyle = "#333";
            this.ctx.beginPath();
            const slice = 280 / len;
            let x = 0;
            for (let i = 0; i < len; i += 8) { // skip pixels for speed
                const v = 20 + (data[i] * 20);
                if (i === 0) this.ctx.moveTo(x, v);
                else this.ctx.lineTo(x, v);
                x += slice * 8;
            }
            this.ctx.stroke();
        }

        this.setToggleState(this.btnRms, inferenceEngine.isRmsNormalizeEnabled());
        this.setToggleState(this.btnPre, inferenceEngine.isPreEmphasisEnabled());
        this.setToggleState(this.btnCmvn, inferenceEngine.isCmvnEnabled());
        this.setToggleState(this.btnFar, am.flags?.farFieldMode || false);
    }

    private setToggleState(btn: HTMLButtonElement | null, isActive: boolean) {
        if (!btn) return;
        btn.style.borderColor = isActive ? "#0f0" : "#444";
        btn.style.color = isActive ? "#0f0" : "#888";
    }

    private createButton(text: string, onClick: () => void) {
        const btn = document.createElement("button");
        btn.innerText = text;
        btn.className = "debug-btn";
        btn.onclick = onClick;
        return btn;
    }

    private createToggle(text: string, onClick: () => void) {
        const btn = document.createElement("button");
        btn.innerText = text;
        btn.className = "debug-btn";
        btn.onclick = onClick;
        return btn;
    }

    private toggle() {
        this.isOpen = !this.isOpen;
        this.container.classList.toggle("open", this.isOpen);
    }

    private triggerFilePicker(cb: (f: File) => void) {
        const i = document.createElement("input");
        i.type = "file";
        i.onchange = (e) => {
            const f = (e.target as HTMLInputElement).files?.[0];
            if (f) cb(f);
        };
        i.click();
    }

    private applyStyles() {
        const style = document.createElement("style");
        style.innerHTML = `
            #qari-debug-panel { position: fixed; bottom: 15px; right: 15px; z-index: 10000; font-family: 'JetBrains Mono', monospace; }
            .debug-fab { width: 44px; height: 44px; border-radius: 50%; background: #111; border: 1px solid #0f0; color: white; cursor: pointer; font-size: 20px; }
            .debug-content { display: none; background: #050505; padding: 12px; border-radius: 8px; width: 260px; border: 1px solid #0f0; position: absolute; bottom: 55px; right: 0; box-shadow: 0 0 20px rgba(0,255,0,0.2); }
            #qari-debug-panel.open .debug-content { display: block; }
            .debug-grid { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 4px; margin-top: 8px; }
            .debug-btn { background: #000; color: #888; border: 1px solid #444; font-size: 9px; padding: 6px 0; cursor: pointer; text-transform: uppercase; }
            .debug-log { background: #000; height: 70px; overflow-y: auto; font-size: 10px; margin-top: 10px; padding: 6px; color: #0f0; border: 1px solid #1a1a1a; }
            .debug-stats { font-size: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 4px; margin-bottom: 8px; color: #aaa; }
        `;
        document.head.appendChild(style);
    }
}
import './components/visualizer/halo-visualizer';
import './components/ui/confidence-ring';
import './components/ui/glass-card';
import './components/ui/match-history';
import './components/ui/onboarding-modal';
import './components/ui/status-pill';
import {installFileTestHotkey} from "./debug/file-test";

import {audioManager} from './core/audio-manager';
import {inferenceEngine, STATE_IDLE} from './model/inference-engine';
import {i18n} from './core/i18n';
import {platform} from "./core/platform-service";
import {micCap} from './audio/mic-cap';
import {installMicCapHotkey} from "./debug/miccap-hotkey.ts";
import {installFileLoopbackHotkey} from "./debug/file-loopback.ts";
import {installRawMicRecordHotkey} from "./debug/raw-mic-record.ts";

class QariApp {
    private ui: any;
    private deferredPrompt: any = null;
    private lastHistoryKey = "";

    constructor(private root: HTMLElement) {
        this.renderShell();
        this.cacheUI();
        this.checkOnboardingStatus();
        this.bindEvents();
        this.updateUIText(i18n.t);
    }

    private renderShell() {
        const appVersion = typeof __APP_VERSION__ !== 'undefined' ? __APP_VERSION__ : '0.0.0';
        // Fallback if that fails
        const vDisplay = 'v' + appVersion || 'v1.0';

        this.root.innerHTML = `
        <onboarding-modal id="onboard"></onboarding-modal>
        
        <div class="app-layout">
          
          <header class="app-header">
             <div class="top-nav">
                <button id="lang-btn" class="text-btn">عربي</button>
                <button id="recalibrate-btn" class="icon-btn">
                   <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83"/>
                   </svg>
                </button>
             </div>
             <div class="header-content">
                 <h1 class="logo-title">
                    <span id="t-title">Qari</span><span class="accent" id="t-accent">Finder</span>
                 </h1>
                 <p class="subtitle">
                    <span id="t-subtitle">Neural Voice Identification</span>
                    <span class="version-tag">${vDisplay}</span>
                 </p>
             </div>
          </header>
  
          <main class="stage">
             <div class="orb-container">
                <div class="halo-layer">
                    <halo-visualizer></halo-visualizer>
                </div>
                <div class="ring-layer">
                    <confidence-ring id="confidence-ring"></confidence-ring>
                </div>
             </div>

             <div class="pill-container">
                <status-pill id="status-pill"></status-pill>
             </div>
          </main>

          <footer class="app-footer">
             <div id="control-layer">
                <button id="start-btn" class="primary-btn">Initialize Engine</button>
             </div>
             <match-history id="match-history"></match-history>
             <button id="install-btn" class="secondary-btn" style="display: none;">📥 Install App</button>
          </footer>

          <glass-card id="result-card"></glass-card>
        </div>
        `;
    }

    // ... (Keep existing methods: cacheUI, checkOnboardingStatus, bindEvents, etc.) ...

    private cacheUI() {
        this.ui = {
            startBtn: document.getElementById('start-btn'),
            langBtn: document.getElementById('lang-btn'),
            installBtn: document.getElementById('install-btn'),
            ring: document.getElementById('confidence-ring'),
            card: document.getElementById('result-card'),
            history: document.getElementById('match-history'),
            pill: document.getElementById('status-pill'),
            onboard: document.getElementById('onboard'),
            titles: {
                main: document.getElementById('t-title'),
                accent: document.getElementById('t-accent'),
                sub: document.getElementById('t-subtitle'),
            }
        };
    }

    private checkOnboardingStatus() {
        const hasOnboarded = localStorage.getItem('qari_has_onboarded');
        if (hasOnboarded && this.ui.onboard) {
            this.ui.onboard.hidden = true;
        }
    }

    private bindEvents() {
        this.ui.startBtn?.addEventListener('click', () => this.startEngine());
        this.ui.langBtn?.addEventListener('click', () => this.handleLangToggle());
        this.ui.onboard?.addEventListener('onboard-complete', () => {
            localStorage.setItem('qari_has_onboarded', 'true');
            this.startEngine();
        });
        this.ui.card?.addEventListener('close', () => this.resetScanState());

        window.addEventListener('beforeinstallprompt', (e) => this.handleBeforeInstall(e));
        this.ui.installBtn?.addEventListener('click', () => this.handleInstallClick());

        window.addEventListener('lang-change', (e: any) => this.updateUIText(e.detail.t));
        window.addEventListener('qari-found', (e: any) => this.handleAIResult(e.detail));

        document.getElementById('recalibrate-btn')?.addEventListener('click', () => {
            const onboard = document.getElementById('onboard') as any;
            onboard.runCalibration().then(() => this.startEngine());
        });

        window.addEventListener('app-state-change', (e: any) => this.handleAppState(e.detail.isActive));

        window.addEventListener('keydown', async (e) => {

            if (e.key.toLowerCase() === 'n' && e.shiftKey) {
                inferenceEngine.toggleRmsNormalize();
                const isRmsNormalizeEnabled = inferenceEngine.isRmsNormalizeEnabled();
                this.ui.pill.text = isRmsNormalizeEnabled ? "RmsNormalize: ON" : "RmsNormalize: OFF";
            }

            if (e.key.toLowerCase() === 'e' && e.shiftKey) {
                inferenceEngine.togglePreEmphasis();
                const isPreEmphasisEnabled = inferenceEngine.isPreEmphasisEnabled();
                this.ui.pill.text = isPreEmphasisEnabled ? "PreEmphasis: ON" : "PreEmphasis: OFF";
            }

            if (e.key.toLowerCase() === 'c' && e.shiftKey) {
                inferenceEngine.toggleCMVN();
                const isCmvnEnabled = inferenceEngine.isCmvnEnabled();
                this.ui.pill.text = isCmvnEnabled ? "CMNV: ON" : "CMNV: OFF";
            }

            if (e.key.toLowerCase() === 'f' && e.shiftKey) {
                const isNowOn = !audioManager.isFarFieldMode();
                audioManager.setFarFieldMode(isNowOn);

                // UX Feedback
                if (this.ui.pill) {
                    this.ui.pill.state = 'stabilizing'; // Just to show color change
                    this.ui.pill.text = isNowOn ? "AGC: ON" : "AGC: OFF";
                }

                // Restart Engine if running
                if (audioManager.isRunning) {
                    console.log("🔄 Rebooting audio for Far-Field change...");
                    await audioManager.stop();
                    // Slight delay to ensure clean hardware release
                    setTimeout(() => this.startEngine(), 200);
                }
            }
        });
    }

    private handleAppState(isActive: boolean) {
        if (!isActive) {
            if (this.ui.pill) {
                this.ui.pill.state = 'idle';
                this.ui.pill.text = i18n.t.paused;
            }
            if (this.ui.startBtn) {
                this.ui.startBtn.innerText = i18n.t.resume;
                this.ui.startBtn.removeAttribute('disabled');
                if (this.ui.startBtn.parentElement) this.ui.startBtn.parentElement.style.display = 'block';
            }
        }
    }

    private async startEngine() {
        if (!this.ui.startBtn) return;
        this.ui.startBtn.innerText = i18n.t.loading;
        this.ui.startBtn.setAttribute('disabled', 'true');
        this.ui.startBtn.classList.remove('error-btn');

        inferenceEngine.reset();
        this.resetScanState();

        try {
            const isOnline = await inferenceEngine.setup();

            if (isOnline) {
                if (this.ui.startBtn.parentElement) this.ui.startBtn.parentElement.style.display = 'none';
            } else {
                throw new Error("Model Failed");
            }

            await audioManager.start((chunk) => {
                if (chunk.length !== 4096) console.warn("Unexpected chunk size:", chunk.length);
                console.log("🎯 post-worklet chunk", chunk.length, "samples (expect 4096 @16k)");
                inferenceEngine.handleIncomingAudio(chunk);
                micCap.onChunk(chunk);
            });

        } catch (err: any) {
            let msg = "Error";
            if (err.message.includes("Permission") || err.name === "NotAllowedError") msg = i18n.t.micDenied;
            else if (err.message.includes("Worklet")) msg = i18n.t.missingFile;

            this.ui.startBtn.innerText = msg;
            this.ui.startBtn.removeAttribute('disabled');
            this.ui.startBtn.classList.add('error-btn');
        }
    }

    private resetScanState() {
        if (this.ui.pill) {
            this.ui.pill.state = 'listening';
            this.ui.pill.text = i18n.t.analyzing;
        }
        if (this.ui.ring) {
            this.ui.ring.score = 0;
            this.ui.ring.label = i18n.t.analyzing;
        }
        if (this.ui.card) this.ui.card.visible = false;
    }

    private updateHistoryDeferred(others: any[]) {
        const key = others.map(o => `${o.name}:${Math.round(o.score * 100)}`).join("|");
        if (key === this.lastHistoryKey) return;
        this.lastHistoryKey = key;
        requestAnimationFrame(() => {
            if (this.ui.history) this.ui.history.qariMatches = others;
        });
    }

    private handleAIResult(detail: { winner: any, others: any[] }) {
        const {winner, others} = detail;
        this.updateHistoryDeferred(others);

        if (winner.name === STATE_IDLE) {
            this.ui.pill.state = 'listening';
            this.ui.pill.text = i18n.t.analyzing;
            this.ui.ring.score = 0;
            this.ui.ring.label = i18n.t.analyzing;

        } else if (winner.score < 0.75) {
            this.ui.pill.state = 'stabilizing';
            this.ui.pill.text = i18n.t.stabilizing;
            this.ui.ring.score = winner.score;
            this.ui.ring.label = winner.name;

        } else {
            if (this.ui.pill.state !== 'match') platform.hapticSuccess();

            this.ui.pill.state = 'match';
            this.ui.pill.text = i18n.t.confirmed;
            this.ui.ring.score = winner.score;
            this.ui.ring.label = winner.name;

            if (this.ui.card && !this.ui.card.visible) {
                this.ui.card.name = winner.name;
                this.ui.card.visible = true;
            }
        }
    }

    private handleLangToggle() {
        const newLang = i18n.toggle();
        if (this.ui.langBtn) {
            this.ui.langBtn.innerText = newLang === 'ar' ? 'عربي' : 'English';
        }
    }

    private updateUIText(t: any) {
        if (this.ui.titles.main) this.ui.titles.main.innerText = t.appTitle;
        if (this.ui.titles.accent) this.ui.titles.accent.innerText = t.appAccent;
        if (this.ui.titles.sub) this.ui.titles.sub.innerText = t.subtitle;

        if (this.ui.startBtn && !this.ui.startBtn.hasAttribute('disabled')) {
            this.ui.startBtn.innerText = t.initBtn;
        }
        if (this.ui.installBtn) this.ui.installBtn.innerText = t.install;

        if (this.ui.history) this.ui.history.dict = t;
        if (this.ui.onboard) this.ui.onboard.dict = t;

        if (this.ui.ring) {
            this.ui.ring.dict = t;
            if (this.ui.ring.score <= 0.01) this.ui.ring.label = t.analyzing;
        }

        if (this.ui.card) {
            this.ui.card.listenBtnText = t.listenBtn;
            this.ui.card.dismissBtnText = t.dismiss;
        }

        if (this.ui.pill) {
            const key = this.ui.pill.state === 'listening' ? 'analyzing'
                : this.ui.pill.state === 'stabilizing' ? 'stabilizing'
                    : 'confirmed';
            this.ui.pill.text = t[key];
        }
    }

    private handleBeforeInstall(e: Event) {
        e.preventDefault();
        this.deferredPrompt = e;
        if (this.ui.installBtn) this.ui.installBtn.style.display = 'block';
    }

    private async handleInstallClick() {
        if (!this.deferredPrompt) return;
        this.deferredPrompt.prompt();
        const {outcome} = await this.deferredPrompt.userChoice;
        if (outcome === 'accepted') {
            if (this.ui.installBtn) this.ui.installBtn.style.display = 'none';
        }
        this.deferredPrompt = null;
    }
}

const appRoot = document.querySelector<HTMLDivElement>('#app');
if (appRoot) new QariApp(appRoot);
if (location.search.includes("debug=1")) {
    installFileTestHotkey();       // Shift+T
    installMicCapHotkey();         // Shift+M
    installFileLoopbackHotkey();   // Shift+L
    installRawMicRecordHotkey();   // Shift+R
}
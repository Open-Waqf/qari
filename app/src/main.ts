import './components/visualizer/dna-visualizer';
import './components/ui/confidence-meter';
import './components/ui/glass-card';
import './components/ui/match-history';
import './components/ui/onboarding-modal';
import './components/ui/status-pill';

import {audioManager} from './core/audio-manager';
import {inferenceEngine} from './model/inference-engine';
import {i18n} from './core/i18n';
import {platform} from "./core/platform-service.ts";

/**
 * Main Application Controller
 * Handles the "Shell" logic, UI state management, and Event coordination.
 */
class QariApp {
    // Cache for DOM elements to avoid repeated queries
    private ui: {
        startBtn: HTMLButtonElement | null;
        langBtn: HTMLButtonElement | null;
        installBtn: HTMLButtonElement | null;
        statusDot: HTMLElement | null;
        meter: any; // LitComponent
        history: any; // LitComponent
        pill: any; // LitComponent
        onboard: any; // LitComponent
        titles: {
            main: HTMLElement | null;
            accent: HTMLElement | null;
            sub: HTMLElement | null;
            acoustic: HTMLElement | null;
        }
    };

    private deferredPrompt: any = null;

    constructor(private root: HTMLElement) {
        // 1. Render the initial HTML Shell
        this.renderShell();

        // 2. Cache all element references once
        this.ui = {
            startBtn: document.getElementById('start-btn') as HTMLButtonElement,
            langBtn: document.getElementById('lang-btn') as HTMLButtonElement,
            installBtn: document.getElementById('install-btn') as HTMLButtonElement,
            statusDot: document.getElementById('status-dot'),
            meter: document.getElementById('result-meter'),
            history: document.getElementById('match-history'),
            pill: document.getElementById('status-pill'),
            onboard: document.getElementById('onboard'),
            titles: {
                main: document.getElementById('t-title'),
                accent: document.getElementById('t-accent'),
                sub: document.getElementById('t-subtitle'),
                acoustic: document.getElementById('t-acoustic'),
            }
        };

        this.checkOnboardingStatus();
        // 3. Initialize Listeners
        this.bindEvents();
    }

    private checkOnboardingStatus() {
        const hasOnboarded = localStorage.getItem('qari_has_onboarded');

        if (hasOnboarded && this.ui.onboard) {
            this.ui.onboard.hidden = true;
        }
    }

    /**
     * Renders the base HTML structure of the application.
     */
    private renderShell() {
        const appVersion = typeof __APP_VERSION__ !== 'undefined' ? __APP_VERSION__ : '0.0.0';
        this.root.innerHTML = `
        <onboarding-modal id="onboard"></onboarding-modal>
        
        <div class="fade-in">
          <div class="top-bar">
            <button id="lang-btn" class="lang-toggle">عربي</button>
            <button id="recalibrate-btn" class="icon-btn" title="Recalibrate">
              <svg class="recalibrate-svg" viewBox="0 0 24 24">
                <path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83"/>
              </svg>
            </button>
          </div>
          
          <header>
            <h1 class="title">
              <span id="t-title">Qari</span><span class="accent" id="t-accent">Finder</span>
            </h1>
            <p class="subtitle" id="t-subtitle">Neural Voice Identification</p>
          </header>
    
          <glass-card>
            <div style="text-align: center;">
                <status-pill id="status-pill"></status-pill>
            </div>
            
            <div class="status-header">
              <span class="label" id="t-acoustic">Acoustic Signature</span>
              <div id="status-dot" class="dot"></div>
            </div>
            
            <dna-visualizer></dna-visualizer>
            
            <confidence-meter id="result-meter"></confidence-meter>
            <match-history id="match-history"></match-history>
    
            <button id="install-btn" class="secondary-btn" style="display: none; margin-top: 15px;">
                📥 Install App
            </button>
          </glass-card>
    
          <div id="control-layer" class="control-area">
            <button id="start-btn" class="primary-btn">Initialize Engine</button>
          </div>
          <div class="version-tag">v${appVersion}</div>
        </div>
        `;
    }

    /**
     * Binds all event listeners for the app lifecycle.
     */
    private bindEvents() {
        // --- Core Interactions ---
        this.ui.startBtn?.addEventListener('click', () => this.startEngine());
        this.ui.langBtn?.addEventListener('click', () => this.handleLangToggle());
        this.ui.onboard?.addEventListener('onboard-complete', () => this.startEngine());

        // --- PWA Installation ---
        window.addEventListener('beforeinstallprompt', (e) => this.handleBeforeInstall(e));
        this.ui.installBtn?.addEventListener('click', () => this.handleInstallClick());

        // --- Custom Application Events ---
        window.addEventListener('lang-change', (e: any) => this.updateUIText(e.detail.t));
        window.addEventListener('qari-found', (e: any) => this.handleAIResult(e.detail));

        this.ui.onboard?.addEventListener('onboard-complete', () => {
            // Save the status so they don't see it again next time
            localStorage.setItem('qari_has_onboarded', 'true');
            this.startEngine();
        });

        const recalibrateBtn = document.getElementById('recalibrate-btn');
        recalibrateBtn?.addEventListener('click', () => {
            const onboard = document.getElementById('onboard') as any;
            onboard.runCalibration().then(() => {
                // Restart the engine automatically after calibration
                this.startEngine();
            });
        });

        window.addEventListener('app-state-change', (e: any) => this.handleAppState(e.detail.isActive));
    }

    /**
     * NEW: Handle UI updates when App goes to Background or Foreground
     */
    private handleAppState(isActive: boolean) {
        if (!isActive) {
            console.log("💤 UI Detected Background Mode");

            // A. Update the Status Pill
            if (this.ui.pill) {
                this.ui.pill.state = 'idle';
                this.ui.pill.text = "Paused"; // or use i18n.t.paused
            }

            // B. Bring back the "Start/Resume" button
            if (this.ui.startBtn) {
                this.ui.startBtn.innerText = "▶ Resume";
                this.ui.startBtn.removeAttribute('disabled');

                // Un-hide the control layer
                if (this.ui.startBtn.parentElement) {
                    this.ui.startBtn.parentElement.style.display = 'block';
                }
            }

            // C. Turn off the "Online" light
            this.ui.statusDot?.classList.remove('online');
            this.ui.statusDot?.classList.remove('active-pulse');
        }
    }

    /**
     * Boot Sequence: Connects Audio -> Visualizer -> Inference Engine
     */
    private async startEngine() {
        if (!this.ui.startBtn) return;

        this.ui.startBtn.innerText = i18n.t.loading;
        this.ui.startBtn.setAttribute('disabled', 'true');

        if (this.ui.pill) {
            this.ui.pill.state = 'listening';
            this.ui.pill.text = i18n.t.analyzing; // "Listening..."
        }
        // Reset the meter too
        if (this.ui.meter) {
            this.ui.meter.score = 0;
            this.ui.meter.label = "";
        }

        try {
            // 1. Start Audio (This will now work because it's triggered by a click)
            await audioManager.start((chunk) => {
                inferenceEngine.handleIncomingAudio(chunk);
            });

            // 2. Load Model & check if it actually worked
            const isOnline = await inferenceEngine.setup();

            if (isOnline) {
                if (this.ui.startBtn.parentElement) this.ui.startBtn.parentElement.style.display = 'none';
                this.ui.statusDot?.classList.add('online');
            } else {
                this.ui.startBtn.innerText = "🚨 Brain Offline (Reload)";
                this.ui.startBtn.removeAttribute('disabled');
                this.ui.startBtn.classList.add('error-btn');
            }
        } catch (err: any) {
            console.error("🔥 Engine Start Error:", err);

            // FIX: Show the specific error message to the user
            let msg = "❌ Error";
            if (err.message.includes("Worklet")) msg = "❌ Missing File (404)";
            else if (err.message.includes("Permission")) msg = "❌ Mic Denied";
            else if (err.name === "NotAllowedError") msg = "❌ Mic Denied";
            else msg = `❌ ${err.name || "Error"}`;

            this.ui.startBtn.innerText = msg;
            this.ui.startBtn.removeAttribute('disabled');
            this.ui.startBtn.classList.add('error-btn');
        }
    }

    /**
     * Logic for updating the UI based on AI prediction confidence.
     */
    private handleAIResult(detail: { winner: any, others: any[] }) {
        const {winner, others} = detail;
        const pill = this.ui.pill;
        const meter = this.ui.meter;
        const history = this.ui.history;

        // 1. Update Match History (Always show candidates)
        if (history) history.qariMatches = others;

        // 2. Handle State Logic (Idle / Stabilizing / Match)
        if (winner.name === "Analyzing...") {
            // State: Listening / Analyzing
            pill.state = 'listening';
            pill.text = i18n.t.analyzing;
            if (meter) meter.score = 0; // Reset meter while thinking
        } else if (winner.score < 0.6) {
            // State: Unsure / Stabilizing
            pill.state = 'stabilizing';
            pill.text = i18n.t.stabilizing;
        } else {
            if (this.ui.pill.state !== 'match') {
                platform.hapticSuccess(); // 📳 Bzzzt!
            }
            // State: Confirmed Match
            pill.state = 'match';
            pill.text = i18n.t.confirmed;

            if (meter) {
                meter.label = winner.name;
                meter.score = winner.score;
            }
            this.ui.statusDot?.classList.add('active-pulse');
            setTimeout(() => this.ui.statusDot?.classList.remove('active-pulse'), 1000);
        }
    }

    /**
     * Switches language and updates the toggle button text.
     */
    private handleLangToggle() {
        const newLang = i18n.toggle();
        if (this.ui.langBtn) {
            this.ui.langBtn.innerText = newLang === 'en' ? 'عربي' : 'English';
        }
    }

    /**
     * Updates all static text on the page from the dictionary.
     */
    private updateUIText(t: any) {
        // Text Elements
        if (this.ui.titles.main) this.ui.titles.main.innerText = t.appTitle;
        if (this.ui.titles.accent) this.ui.titles.accent.innerText = t.appAccent;
        if (this.ui.titles.sub) this.ui.titles.sub.innerText = t.subtitle;
        if (this.ui.titles.acoustic) this.ui.titles.acoustic.innerText = t.acousticLabel;

        // Buttons
        if (this.ui.startBtn && !this.ui.startBtn.hasAttribute('disabled')) {
            this.ui.startBtn.innerText = t.initBtn;
        }
        if (this.ui.installBtn) {
            this.ui.installBtn.innerText = t.install;
        }

        // Pass dictionary to Web Components
        if (this.ui.meter) this.ui.meter.dict = t;
        if (this.ui.history) this.ui.history.dict = t;
        if (this.ui.onboard) this.ui.onboard.dict = t;

        // Re-trigger the current pill state to update its text
        if (this.ui.pill) {
            // Quick hack to refresh current state text
            const currentStateKey = this.ui.pill.state === 'listening' ? 'analyzing'
                : this.ui.pill.state === 'stabilizing' ? 'stabilizing'
                    : 'confirmed';
            this.ui.pill.text = t[currentStateKey];
        }
    }

    /**
     * Captures the PWA install prompt.
     */
    private handleBeforeInstall(e: Event) {
        e.preventDefault();
        this.deferredPrompt = e;
        if (this.ui.installBtn) this.ui.installBtn.style.display = 'block';
    }

    /**
     * Triggers the PWA install prompt.
     */
    private async handleInstallClick() {
        if (!this.deferredPrompt) return;

        this.deferredPrompt.prompt();
        const {outcome} = await this.deferredPrompt.userChoice;

        if (outcome === 'accepted') {
            console.log('🚀 User accepted the install');
            if (this.ui.installBtn) this.ui.installBtn.style.display = 'none';
        }
        this.deferredPrompt = null;
    }
}

// --- Application Entry Point ---
const appRoot = document.querySelector<HTMLDivElement>('#app');
if (appRoot) {
    new QariApp(appRoot);
}
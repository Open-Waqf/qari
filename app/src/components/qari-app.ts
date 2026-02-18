import {html, LitElement} from 'lit';
import {customElement, state} from 'lit/decorators.js';

import {
    type AppStatePayload,
    type DebugMessagePayload,
    EVENTS,
    type LangChangePayload,
    type QariMatch,
    type QariResultPayload
} from '../core/events';
import {i18n} from '../core/i18n';
import {audioManager} from '../core/audio-manager';
import {inferenceEngine, STATE_IDLE} from '../model/inference-engine';
import {platform} from '../core/platform-service';
import {micCap} from '../audio/mic-cap';

// Child components
import './visualizer/halo-visualizer';
import './ui/confidence-ring';
import './ui/glass-card';
import './ui/match-history';
import './ui/onboarding-modal';
import './ui/status-pill';

// Minimal TS type for the PWA install prompt event (not in standard lib.dom typings)
interface BeforeInstallPromptEvent extends Event {
    prompt: () => Promise<void>;
    userChoice: Promise<{ outcome: 'accepted' | 'dismissed'; platform?: string }>;
}

type PillState = 'idle' | 'listening' | 'stabilizing' | 'match';

@customElement('qari-app')
export class QariApp extends LitElement {
    // --- UI / i18n ---
    @state() private t = i18n.t;
    @state() private pillState: PillState = 'idle';
    @state() private pillText = i18n.t.paused;

    // --- Onboarding ---
    @state() private showOnboard = false;

    // --- Engine / inference display ---
    @state() private winner: QariMatch = {name: STATE_IDLE, score: 0};
    @state() private others: QariMatch[] = [];

    // --- Controls / UX ---
    @state() private isStarting = false; // show loading + disable button
    @state() private hasStarted = false; // hide controls after success
    @state() private isPaused = false;   // show "Resume" label
    @state() private errorMsg = '';

    // --- PWA ---
    @state() private isInstallable = false;
    private deferredPrompt: BeforeInstallPromptEvent | null = null;

    // --- History throttling ---
    private lastHistoryKey = '';
    private historyRaf = 0;
    private pendingOthers: QariMatch[] | null = null;

    @state() private lastStable: QariMatch | null = null;
    private lastStableAt = 0;
    private readonly stableHoldMs = 12000; // allow long reciter pauses

    private noiseStreak = 0;
    private noiseUiOn = false;
    private lastNoiseAt = 0;

// Tune
    private readonly noiseEnterCount = 2; // need 2 consecutive noise events
    private readonly noiseExitMs = 1200;  // clear noise UI after 1.2s without noise

    // Use Light DOM so global CSS applies
    createRenderRoot() {
        return this;
    }

    connectedCallback() {
        super.connectedCallback();
        this.checkOnboarding();
        this.bindGlobalEvents();
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        this.unbindGlobalEvents();
        if (this.historyRaf) cancelAnimationFrame(this.historyRaf);
        // Optional cleanup:
        void audioManager.stop();
    }

    private bindGlobalEvents() {
        window.addEventListener(EVENTS.LANG_CHANGE, this.handleLangChange);
        window.addEventListener(EVENTS.RESULT_FOUND, this.handleResult);
        window.addEventListener(EVENTS.APP_STATE_CHANGE, this.handleAppState);
        window.addEventListener('beforeinstallprompt', this.handleBeforeInstall as EventListener);
        window.addEventListener(EVENTS.DEBUG_MESSAGE, this.handleDebugMessage);
        window.addEventListener(EVENTS.REQUEST_RESTART, this.handleRestartRequest);
    }

    private unbindGlobalEvents() {
        window.removeEventListener(EVENTS.LANG_CHANGE, this.handleLangChange);
        window.removeEventListener(EVENTS.RESULT_FOUND, this.handleResult);
        window.removeEventListener(EVENTS.APP_STATE_CHANGE, this.handleAppState);
        window.removeEventListener('beforeinstallprompt', this.handleBeforeInstall as EventListener);
        window.removeEventListener(EVENTS.DEBUG_MESSAGE, this.handleDebugMessage);
        window.removeEventListener(EVENTS.REQUEST_RESTART, this.handleRestartRequest);
    }

    private handleRestartRequest = async () => {
        if (audioManager.isRunning) {
            await audioManager.stop();
            // Small safety buffer to ensure context is fully closed
            setTimeout(() => this.startEngine(), 200);
        }
    };

    private handleDebugMessage = (e: Event) => {
        const {message} = (e as CustomEvent<DebugMessagePayload>).detail;
        this.pillText = message;
        // Optional: Reset pill to "Listening" after 2 seconds?
    };

    // --- Global event handlers (arrow functions so `this` is stable) ---

    private handleLangChange = (e: Event) => {
        const detail = (e as CustomEvent<LangChangePayload>).detail;
        this.t = detail.t;

        // Keep pill text consistent with current state
        if (this.pillState === 'idle') this.pillText = this.t.paused;
        else if (this.pillState === 'listening') this.pillText = this.t.analyzing;
        else if (this.pillState === 'stabilizing') this.pillText = this.t.stabilizing;
        else if (this.pillState === 'match') this.pillText = this.t.confirmed;
    };

    private handleResult = (e: Event) => {
        const {winner, others, stable, activity} = (e as CustomEvent<QariResultPayload>).detail;
        const act = activity ?? (winner.name === STATE_IDLE ? 'silence' : 'voiced');
        const now = Date.now();

        if (act === 'noise') {
            this.noiseStreak++;
            this.lastNoiseAt = now;
        } else {
            this.noiseStreak = 0;
        }

        // enter noise UI only after streak
        if (!this.noiseUiOn && this.noiseStreak >= this.noiseEnterCount) {
            this.noiseUiOn = true;
        }

        // exit noise UI if no noise recently
        if (this.noiseUiOn && (now - this.lastNoiseAt) > this.noiseExitMs) {
            this.noiseUiOn = false;
        } else {
            this.noiseStreak = 0;
            if (act === 'voiced') this.noiseUiOn = false; // immediate exit
        }

        // Throttle history updates (parity with old requestAnimationFrame batching)
        const filteredOthers = others.filter(o => o.name.trim().toUpperCase() !== 'BACKGROUND');
        const key = filteredOthers.map(o => `${o.name}:${Math.round(o.score * 100)}`).join('|');
        if (key !== this.lastHistoryKey) {
            this.lastHistoryKey = key;
            this.pendingOthers = filteredOthers;
            if (!this.historyRaf) {
                this.historyRaf = requestAnimationFrame(() => {
                    this.historyRaf = 0;
                    if (this.pendingOthers) this.others = this.pendingOthers;
                    this.pendingOthers = null;
                });
            }
        }

        // Track last stable reciter (only real reciters)
        if (stable && winner.name !== STATE_IDLE) {
            this.lastStable = winner;
            this.lastStableAt = Date.now();
        }

        // Expire the held winner after long silence/noise
        if ((act === 'silence' || act === 'noise') && this.lastStable && (Date.now() - this.lastStableAt > this.stableHoldMs)) {
            this.lastStable = null;
            this.lastStableAt = 0;
        }

        // Background noise: don’t treat as match, don’t overwrite ring winner
        if (this.noiseUiOn) {
            this.pillState = 'listening';
            this.pillText = this.t.backgroundNoise ?? "Background noise";
            this.winner = {name: STATE_IDLE, score: 0};
            return;
        }

        // Silence
        if (winner.name === STATE_IDLE) {
            this.pillState = 'listening';
            this.pillText = this.t.analyzing;
            this.winner = {name: STATE_IDLE, score: 0};
            return;
        }

        // Voiced but not stable yet
        if (!stable) {
            this.pillState = 'stabilizing';
            this.pillText = this.t.stabilizing;
            this.winner = winner;
            return;
        }

        // Stable confirmed match
        if (this.pillState !== 'match') platform.hapticSuccess();
        this.pillState = 'match';
        this.pillText = this.t.confirmed;
        this.winner = winner;

        const card = this.querySelector('glass-card') as any;
        if (card && !card.visible) {
            card.name = winner.name;
            card.visible = true;
        }
    };

    private handleAppState = (e: Event) => {
        const {isActive} = (e as CustomEvent<AppStatePayload>).detail;

        if (!isActive) {
            // Parity with old behavior: show resume UI + paused pill
            this.isPaused = true;
            this.hasStarted = false;
            this.isStarting = false;

            this.pillState = 'idle';
            this.pillText = this.t.paused;
        }
    };

    private handleBeforeInstall = (e: BeforeInstallPromptEvent) => {
        e.preventDefault();
        this.deferredPrompt = e;
        this.isInstallable = true;
    };

    // --- Actions ---

    public async startEngine() {
        if (this.isStarting) return;

        this.isStarting = true;
        this.errorMsg = '';
        this.isPaused = false;

        // Reset state before boot
        inferenceEngine.reset();
        this.resetScanState();

        try {
            // If we’re restarting, make sure audio is cleanly stopped
            if (audioManager.isRunning) await audioManager.stop();

            const isOnline = await inferenceEngine.setup();
            const savedNoise = Number(localStorage.getItem('qari_noise_floor'));
            if (Number.isFinite(savedNoise) && savedNoise > 0) {
                console.log('📂 Restoring saved noise floor:', savedNoise);
                inferenceEngine.setNoiseFloor(savedNoise);
            }
            if (!isOnline) throw new Error('Model Failed');

            await audioManager.start((chunk) => {
                inferenceEngine.handleIncomingAudio(chunk);
                micCap.onChunk(chunk);
            });

            // Hide control layer after successful start (parity with old app)
            this.hasStarted = true;
        } catch (err: any) {
            let msg = 'Error';
            if (err?.message?.includes('Permission') || err?.name === 'NotAllowedError') msg = this.t.micDenied;
            else if (err?.message?.includes('Worklet')) msg = this.t.missingFile;

            this.errorMsg = msg;
            this.pillText = msg;

            this.hasStarted = false;
        } finally {
            this.isStarting = false;
        }
    }

    private resetScanState() {
        this.pillState = 'listening';
        this.pillText = this.t.analyzing;
        this.winner = {name: STATE_IDLE, score: 0};

        const card = this.querySelector('glass-card') as any;
        if (card) card.visible = false;
    }

    private toggleLang = () => {
        i18n.toggle(); // LANG_CHANGE event updates state
    };

    private recalibrate = () => {
        const onboard = this.querySelector('onboarding-modal') as any;
        onboard?.runCalibration?.().then(() => void this.startEngine());
    };

    private installApp = async () => {
        if (!this.deferredPrompt) return;
        await this.deferredPrompt.prompt();
        const {outcome} = await this.deferredPrompt.userChoice;
        if (outcome === 'accepted') this.isInstallable = false;
        this.deferredPrompt = null;
    };

    private checkOnboarding() {
        const hasOnboarded = localStorage.getItem('qari_has_onboarded');
        this.showOnboard = !hasOnboarded;
    }

    private handleOnboardComplete = () => {
        localStorage.setItem('qari_has_onboarded', 'true');
        this.showOnboard = false;
        void this.startEngine();
    };

    private onCardClose = () => {
        this.resetScanState();
    };

    private getVersionDisplay() {
        const v =
            (typeof (window as any).__APP_VERSION__ !== 'undefined' && (window as any).__APP_VERSION__) ||
            (typeof (globalThis as any).__APP_VERSION__ !== 'undefined' && (globalThis as any).__APP_VERSION__) ||
            '0.0.0';

        const s = String(v);
        return s.startsWith('v') ? s : `v${s}`;
    }

    // --- Render ---

    render() {
        const vDisplay = this.getVersionDisplay();

        const held = this.lastStable?.name;
        const ringLabel = this.winner.name === STATE_IDLE ? (held ?? this.t.analyzing) : this.winner.name;
        const ringScore = this.winner.name === STATE_IDLE ? 0 : this.winner.score;

        const showControls = !this.hasStarted || !!this.errorMsg || this.isPaused;
        const startLabel = this.errorMsg
            ? this.errorMsg
            : this.isStarting
                ? this.t.loading
                : this.isPaused
                    ? this.t.resume
                    : this.t.initBtn;

        return html`
            <onboarding-modal
                    id="onboard"
                    .dict=${this.t}
                    ?hidden=${!this.showOnboard}
                    @onboard-complete=${this.handleOnboardComplete}
            ></onboarding-modal>

            <div class="app-layout">
                <header class="app-header">
                    <div class="top-nav">
                        <button id="lang-btn" class="text-btn" @click=${this.toggleLang}>
                            ${i18n.currentLang === 'ar' ? 'عربي' : 'English'}
                        </button>

                        <button id="recalibrate-btn" class="icon-btn" @click=${this.recalibrate}>
                            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor"
                                 stroke-width="2">
                                <path
                                        d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83"
                                />
                            </svg>
                        </button>
                    </div>

                    <div class="header-content">
                        <h1 class="logo-title">
                            <span>${this.t.appTitle}</span><span class="accent">${this.t.appAccent}</span>
                        </h1>
                        <p class="subtitle">
                            <span>${this.t.subtitle}</span>
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
                            <confidence-ring
                                    id="confidence-ring"
                                    .dict=${this.t}
                                    .score=${ringScore}
                                    .label=${ringLabel}
                            ></confidence-ring>
                        </div>
                    </div>

                    <div class="pill-container">
                        <status-pill id="status-pill" .state=${this.pillState} .text=${this.pillText}></status-pill>
                    </div>
                </main>

                <footer class="app-footer">
                    <div id="control-layer" style="display: ${showControls ? 'block' : 'none'}">
                        <button
                                id="start-btn"
                                class="primary-btn ${this.errorMsg ? 'error-btn' : ''}"
                                @click=${() => void this.startEngine()}
                                ?disabled=${this.isStarting}
                        >
                            ${startLabel}
                        </button>
                    </div>

                    <match-history id="match-history" .dict=${this.t} .qariMatches=${this.others}></match-history>

                    <button
                            id="install-btn"
                            class="secondary-btn"
                            style="display: ${this.isInstallable ? 'block' : 'none'};"
                            @click=${() => void this.installApp()}
                    >
                        ${this.t.install}
                    </button>
                </footer>

                <glass-card
                        id="result-card"
                        .listenBtnText=${this.t.listenBtn}
                        .dismissBtnText=${this.t.dismiss}
                        @close=${this.onCardClose}
                ></glass-card>
            </div>
        `;
    }
}
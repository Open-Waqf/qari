import {html, LitElement} from 'lit';
import {customElement, property, state} from 'lit/decorators.js';

interface BeforeInstallPromptEvent extends Event {
    prompt: () => Promise<void>;
    userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>;
}

@customElement('pwa-install-banner')
export class PwaInstallBanner extends LitElement {
    @property({type: Object}) dict: any = {};

    @state() private show = false;
    private deferredPrompt: BeforeInstallPromptEvent | null = null;

    private readonly KEY = 'qari_install_dismissed';
    private readonly COOLDOWN = 7 * 24 * 60 * 60 * 1000;

    createRenderRoot() {
        return this;
    } // Use light DOM to share global styles

    connectedCallback() {
        super.connectedCallback();
        window.addEventListener('beforeinstallprompt', this.handlePrompt as EventListener);
    }

    disconnectedCallback() {
        window.removeEventListener('beforeinstallprompt', this.handlePrompt as EventListener);
    }

    private handlePrompt = (e: BeforeInstallPromptEvent) => {
        e.preventDefault();
        this.deferredPrompt = e;

        const last = localStorage.getItem(this.KEY);
        if (last && (Date.now() - parseInt(last) < this.COOLDOWN)) return;

        this.show = true;
    };

    private async onInstall() {
        if (!this.deferredPrompt) return;
        this.show = false;
        await this.deferredPrompt.prompt();
        const {outcome} = await this.deferredPrompt.userChoice;
        if (outcome === 'accepted') localStorage.setItem(this.KEY, Date.now().toString());
        this.deferredPrompt = null;
    }

    private onDismiss() {
        this.show = false;
        localStorage.setItem(this.KEY, Date.now().toString());
    }

    render() {
        if (!this.show) return html``;

        return html`
            <div class="install-floating-card">
                <p>${this.dict.installPrompt ?? 'Install for offline use'}</p>
                <div class="install-actions">
                    <button class="primary-btn small" @click=${this.onInstall}>
                        ${this.dict.install}
                    </button>
                    <button class="text-btn small" @click=${this.onDismiss}>
                        ${this.dict.notNow ?? 'Not Now'}
                    </button>
                </div>
            </div>
        `;
    }
}
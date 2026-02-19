import './index.css'
import './components/qari-app';
import {
    installFarFieldHotkey,
    installFileLoopbackHotkey,
    installFileTestHotkey,
    installInterferenceHotkey,
    installMicCapHotkey,
    installRawMicRecordHotkey
} from "./debug/debug-hotkeys.ts";
import {checkAudioParity} from "./debug/debug-extractor";
import {DebugPanel} from "./debug/DebugPanel.ts";
import {platform} from './core/platform-service'
import {registerSW} from 'virtual:pwa-register'

async function disableServiceWorkerEverywhere() {
    if (!('serviceWorker' in navigator)) return

    const regs = await navigator.serviceWorker.getRegistrations()
    if (!regs.length) return

    await Promise.all(regs.map(r => r.unregister()))

    if ('caches' in window) {
        const keys = await caches.keys()
        await Promise.all(keys.map(k => caches.delete(k)))
    }

    if (!sessionStorage.getItem('sw_purged_once')) {
        sessionStorage.setItem('sw_purged_once', '1')
        location.reload()
    }
}

if (platform.isWeb) {
    // Web/PWA: enable SW + controlled update
    registerSW({
        onNeedRefresh() {
            // show a small toast/button: “Update available”
            // when user accepts: updateSW(true)
        },
        onOfflineReady() {
            // optional: show “App ready offline”
        },
    })
} else {
    // Native (Capacitor): kill SW so it can’t interfere
    disableServiceWorkerEverywhere()
}

const appRoot = document.querySelector<HTMLDivElement>('#app');

if (appRoot && !appRoot.querySelector('qari-app')) {
    appRoot.appendChild(document.createElement('qari-app'));
}

// 🐛 DEBUG MODE
// Activates on localhost OR if ?debug=1 is in URL
const isDebug = location.search.includes("debug=1") || location.hostname === "localhost" || location.hostname === "127.0.0.1";

if (isDebug) {
    console.log("🐛 Debug Mode Enabled: Hotkeys + Panel");

    // 1. Install Keyboard Hotkeys (Shift+T, Shift+M, etc.)
    installFileTestHotkey();
    installMicCapHotkey();
    installFileLoopbackHotkey();
    installRawMicRecordHotkey();
    installInterferenceHotkey();
    installFarFieldHotkey();

    // 2. Install Bridge for Python
    (window as any).checkParity = checkAudioParity;

    // 3. Mount the Mobile Visual Panel
    new DebugPanel();
}
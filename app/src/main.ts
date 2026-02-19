import './index.css'
import './components/qari-app';
import {platform} from './core/platform-service'
import {registerSW} from 'virtual:pwa-register'

const hasDebugUrl = location.search.includes("debug=1");
const hasDebugStorage = localStorage.getItem("qari_debug_mode") === "true";
const isDebug = hasDebugUrl || hasDebugStorage;

async function disableServiceWorkerEverywhere() {
    if (!('serviceWorker' in navigator)) return;

    const regs = await navigator.serviceWorker.getRegistrations();
    if (!regs.length) return;

    await Promise.all(regs.map(r => r.unregister()));

    if ('caches' in window) {
        const keys = await caches.keys();
        await Promise.all(keys.map(k => caches.delete(k)));
    }

    if (!sessionStorage.getItem('sw_purged_once')) {
        sessionStorage.setItem('sw_purged_once', '1');
        location.reload();
    }
}

if (platform.isWeb) {
    const pwaToast = document.getElementById('pwa-toast');
    const pwaRefreshBtn = document.getElementById('pwa-refresh');
    const pwaCloseBtn = document.getElementById('pwa-close');

    const updateSW = registerSW({
        onNeedRefresh() {
            if (isDebug) console.log("🔄 New App Version Found! Showing update toast.");
            if (pwaToast) {
                pwaToast.classList.add('show');
            }
        },
        onOfflineReady() {
            if (isDebug) console.log("📶 App is ready to work offline.");
        },
    });

    if (pwaRefreshBtn && pwaCloseBtn && pwaToast) {
        pwaRefreshBtn.addEventListener('click', () => {
            // Tell the Service Worker to skip waiting and activate the new code
            updateSW(true);
        });

        pwaCloseBtn.addEventListener('click', () => {
            // Hide the toast (user can refresh later)
            pwaToast.classList.remove('show');
        });
    }
} else {
    // Native (Capacitor): kill SW so it can’t interfere
    disableServiceWorkerEverywhere();
}

const appRoot = document.querySelector<HTMLDivElement>('#app');

if (appRoot && !appRoot.querySelector('qari-app')) {
    appRoot.appendChild(document.createElement('qari-app'));
}

// -----------------------------------------------------
// 🐛 DEBUG
// Activate via URL (?debug=1) OR LocalStorage (APK trick)
// -----------------------------------------------------

if (isDebug) {
    console.warn("⚠️ Debug Mode Enabled: Injecting Debug Modules...");

    // Use DYNAMIC IMPORTS: This tells Vite to split debug tools into a
    // separate file so it doesn't slow down the app for normal users.
    Promise.all([
        import('./debug/debug-hotkeys.ts'),
        import('./debug/debug-extractor.ts'),
        import('./debug/DebugPanel.ts')
    ]).then(([hotkeys, extractor, panel]) => {
        hotkeys.installFileTestHotkey();
        hotkeys.installMicCapHotkey();
        hotkeys.installFileLoopbackHotkey();
        hotkeys.installRawMicRecordHotkey();
        hotkeys.installInterferenceHotkey();
        hotkeys.installFarFieldHotkey();

        (window as any).checkParity = extractor.checkAudioParity;
        new panel.DebugPanel();
    }).catch(err => console.error("Failed to load debug modules", err));
}

// 🤫 SECRET DEVELOPER MENU (For Android APKs)
// Tap the Logo/Title 5 times quickly to toggle debug mode
let tapCount = 0;
let tapTimeout: any;

document.addEventListener('click', (e) => {
    const target = e.target as HTMLElement;
    // Check if user clicked on the header title
    if (target.closest('.logo-title')) {
        tapCount++;
        clearTimeout(tapTimeout);

        // Reset count if they stop tapping for 1 second
        tapTimeout = setTimeout(() => tapCount = 0, 1000);

        if (tapCount >= 5) {
            tapCount = 0;
            const currentState = localStorage.getItem("qari_debug_mode") === "true";
            const newState = !currentState;
            localStorage.setItem("qari_debug_mode", newState.toString());

            // Provide haptic feedback if available
            platform.hapticSuccess?.();

            alert(`Developer Mode ${newState ? 'ENABLED' : 'DISABLED'}. App will reload.`);
            location.href = location.pathname; // Clean reload without query params
        }
    }
});
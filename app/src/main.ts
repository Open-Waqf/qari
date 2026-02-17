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
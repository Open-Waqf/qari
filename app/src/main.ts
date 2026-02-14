import './components/qari-app';
import {
    installFarFieldHotkey,
    installFileLoopbackHotkey,
    installFileTestHotkey,
    installInterferenceHotkey,
    installMicCapHotkey,
    installRawMicRecordHotkey
} from "./debug/debug-hotkeys.ts";

const appRoot = document.querySelector<HTMLDivElement>('#app');

if (appRoot && !appRoot.querySelector('qari-app')) {
    appRoot.appendChild(document.createElement('qari-app'));
}

// Debug Initialization
if (location.search.includes("debug=1")) {
    console.log("🐛 Debug Mode Enabled");
    installFileTestHotkey();       // Shift+T
    installMicCapHotkey();         // Shift+M
    installFileLoopbackHotkey();   // Shift+L
    installRawMicRecordHotkey();   // Shift+R
    installInterferenceHotkey(); // Shift+N/E/C
    installFarFieldHotkey(); // Shift+F
}
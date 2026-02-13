// src/core/platform-service.ts
import {Capacitor} from '@capacitor/core';
import {Haptics, ImpactStyle, NotificationType} from '@capacitor/haptics';
import {App} from '@capacitor/app';

export class PlatformService {
    private static instance: PlatformService;

    // Detect environment
    public isNative = Capacitor.isNativePlatform();
    public isAndroid = Capacitor.getPlatform() === 'android';
    public isIOS = Capacitor.getPlatform() === 'ios';
    public isWeb = !this.isNative;

    private constructor() {
        this.initializeListeners();
    }

    static getInstance() {
        return this.instance || (this.instance = new PlatformService());
    }

    /**
     * Initializes platform-specific listeners (Back button, App State)
     */
    private initializeListeners() {
        if (this.isNative) {
            // Handle Android Hardware Back Button
            App.addListener('backButton', ({canGoBack}) => {
                if (!canGoBack) {
                    App.exitApp();
                } else {
                    window.history.back();
                }
            });

            // Handle App Background/Foreground (Save Battery)
            App.addListener('appStateChange', ({isActive}) => {
                window.dispatchEvent(new CustomEvent('app-state-change', {
                    detail: {isActive}
                }));
            });
        } else {
            // Web Fallback for Visibility
            document.addEventListener('visibilitychange', () => {
                const isActive = document.visibilityState === 'visible';
                window.dispatchEvent(new CustomEvent('app-state-change', {
                    detail: {isActive}
                }));
            });
        }
    }

    /**
     * Unified Haptic Feedback
     * Fails silently on Web if not supported
     */
    async hapticSuccess() {
        try {
            // FIX 1: Use NotificationType Enum instead of raw number
            await Haptics.notification({type: NotificationType.Success});
        } catch (e) {
            if (navigator.vibrate) navigator.vibrate(50);
        }
    }

    async hapticLight() {
        try {
            await Haptics.impact({style: ImpactStyle.Light});
        } catch (e) {
            if (navigator.vibrate) navigator.vibrate(10);
        }
    }

    /**
     * Check and Request Microphone Permissions
     */
    async checkMicPermission(): Promise<boolean> {
        // FIX 2: Remove deprecated Capacitor.Plugins.Permissions
        // In modern WebViews (Capacitor), getUserMedia triggers the native prompt automatically.
        // We can optionally check via the standard Web API.
        if (this.isNative) {
            try {
                // Check if the permission API is available in the WebView
                if (navigator.permissions && navigator.permissions.query) {
                    // @ts-ignore - 'microphone' is a valid name but TS sometimes misses it in specific envs
                    const status = await navigator.permissions.query({name: 'microphone'});
                    return status.state === 'granted' || status.state === 'prompt';
                }
            } catch (e) {
                // Fallback for iOS/older WebViews: assume true and let getUserMedia handle the error
                return true;
            }
        }
        return true;
    }
}

export const platform = PlatformService.getInstance();
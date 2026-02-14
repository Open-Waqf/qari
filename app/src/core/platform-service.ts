import {Capacitor} from '@capacitor/core';
import {Haptics, ImpactStyle, NotificationType} from '@capacitor/haptics';
import {App} from '@capacitor/app';
import {EVENTS} from './events';

export class PlatformService {
    private static instance: PlatformService;

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

    private initializeListeners() {
        if (this.isNative) {
            App.addListener('backButton', ({canGoBack}) => {
                if (!canGoBack) App.exitApp();
                else window.history.back();
            });

            App.addListener('appStateChange', ({isActive}) => {
                this.dispatchAppState(isActive);
            });
        } else {
            document.addEventListener('visibilitychange', () => {
                this.dispatchAppState(document.visibilityState === 'visible');
            });
        }
    }

    private dispatchAppState(isActive: boolean) {
        window.dispatchEvent(new CustomEvent(EVENTS.APP_STATE_CHANGE, {
            detail: {isActive}
        }));
    }

    async hapticSuccess() {
        try {
            await Haptics.notification({type: NotificationType.Success});
        } catch {
            if (navigator.vibrate) navigator.vibrate(50);
        }
    }

    async hapticLight() {
        try {
            await Haptics.impact({style: ImpactStyle.Light});
        } catch {
            if (navigator.vibrate) navigator.vibrate(10);
        }
    }

    /**
     * Simplified Permission Check.
     * We trust getUserMedia to prompt the user.
     * Pre-checking permissions is flaky on web/iOS and deprecated in Capacitor logic.
     */
    async checkMicPermission(): Promise<boolean> {
        return true;
    }
}

export const platform = PlatformService.getInstance();
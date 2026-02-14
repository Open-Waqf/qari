import {Haptics, ImpactStyle, NotificationType} from '@capacitor/haptics';

export const haptics = {
    // Light tick for UI interactions
    tick: async () => {
        try {
            await Haptics.impact({style: ImpactStyle.Light});
        } catch {
        }
    },
    // Heavy thud for "Match Found"
    success: async () => {
        try {
            await Haptics.notification({type: NotificationType.Success});
        } catch {
        }
    },
    // Subtle vibration for "Scanning..." heartbeat
    heartbeat: async () => {
        try {
            await Haptics.impact({style: ImpactStyle.Medium});
        } catch {
        }
    }
};
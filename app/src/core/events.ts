export const EVENTS = {
    /** Fired when the inference engine has a prediction result */
    RESULT_FOUND: 'qari-found',
    /** Fired when the app goes background/foreground */
    APP_STATE_CHANGE: 'app-state-change',
    /** Fired when the language is toggled */
    LANG_CHANGE: 'lang-change',
    /** Fired when the user finishes the onboarding flow */
    ONBOARD_COMPLETE: 'onboard-complete',
    DEBUG_MESSAGE: 'debug-message',
    REQUEST_RESTART: 'request-restart',
} as const;

export type EventName = (typeof EVENTS)[keyof typeof EVENTS];

export interface QariMatch {
    name: string;
    score: number;
}

export interface QariResultPayload {
    winner: QariMatch;
    others: QariMatch[];
    stable?: boolean;
}

export interface AppStatePayload {
    isActive: boolean;
}

export interface LangChangePayload {
    lang: string;
    // Typed as 'any' to avoid circular dependency with i18n.ts
    t: any;
}

export interface DebugMessagePayload {
    message: string;
}

/** Optional helper if you want strongly-typed event dispatching */
export function emit<T>(name: EventName, detail: T, target: EventTarget = window) {
    target.dispatchEvent(new CustomEvent<T>(name, {detail}));
}